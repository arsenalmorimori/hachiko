using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Hachiko;

public class Detection {
    public float X, Y, Width, Height, Confidence;
    public int ClassId;
    public string Label;
}

public class YoloInferenceService : IDisposable {
    private readonly InferenceSession _session;

    private const int InputSize = 640;
    private const float ConfThreshold = 0.40f; // YOLOv8 scores are already in [0,1]
    private const float NmsThreshold = 0.45f;

    // YOLOv8 layout: [cx, cy, w, h, class0, class1, ...]
    // NO objectness channel — classes start at index 4
    private const int ClassOffset = 4;

    private static readonly string[] Labels =
 {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

    public YoloInferenceService(byte[] modelBytes) {
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

#if ANDROID
        try { options.AppendExecutionProvider_Nnapi(); } catch { }
#elif IOS
        try { options.AppendExecutionProvider_CoreML(); } catch { }
#endif

        _session = new InferenceSession(modelBytes, options);
    }

    public List<Detection> Detect(float[] input, int origW, int origH) {
        var tensor = new DenseTensor<float>(input, new[] { 1, 3, InputSize, InputSize });

        using var results = _session.Run(new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("images", tensor)
        });

        var output = results.First().AsTensor<float>();
        return ParseOutput(output, origW, origH);
    }

    private List<Detection> ParseOutput(Tensor<float> output, int origW, int origH) {
        var dims = output.Dimensions.ToArray();

        // YOLOv8 standard ONNX export shape: [1, 84, 8400]
        // dim1 = attributes (4 box + N classes), dim2 = proposals
        int dim1 = dims[1];
        int dim2 = dims[2];

        bool isCHW = dim1 < dim2; // true = [1, attrs, proposals]

        int numDet = isCHW ? dim2 : dim1;
        int numAttr = isCHW ? dim1 : dim2;

        float scaleX = (float)origW / InputSize;
        float scaleY = (float)origH / InputSize;

        var boxes = new List<Detection>();

        for (int i = 0; i < numDet; i++) {
            // Box coords — already in pixel space at InputSize scale
            float cx = isCHW ? output[0, 0, i] : output[0, i, 0];
            float cy = isCHW ? output[0, 1, i] : output[0, i, 1];
            float w = isCHW ? output[0, 2, i] : output[0, i, 2];
            float h = isCHW ? output[0, 3, i] : output[0, i, 3];

            // Class scores — already sigmoid-activated by the model export
            // DO NOT apply sigmoid again — that causes everything to collapse to ~0.5
            float bestScore = 0f;
            int bestClass = 0;

            for (int c = ClassOffset; c < numAttr; c++) {
                float score = isCHW ? output[0, c, i] : output[0, i, c];

                if (score > bestScore) {
                    bestScore = score;
                    bestClass = c - ClassOffset;
                }
            }

            if (bestScore < ConfThreshold) continue;

            boxes.Add(new Detection {
                X = (cx - w / 2f) * scaleX,
                Y = (cy - h / 2f) * scaleY,
                Width = w * scaleX,
                Height = h * scaleY,
                Confidence = bestScore,
                ClassId = bestClass,
                Label = bestClass < Labels.Length ? Labels[bestClass] : $"cls{bestClass}"
            });
        }

        return ApplyNms(boxes);
    }

    private List<Detection> ApplyNms(List<Detection> dets) {
        var result = new List<Detection>();
        foreach (var group in dets.GroupBy(d => d.ClassId)) {
            var sorted = group.OrderByDescending(d => d.Confidence).ToList();
            var removed = new bool[sorted.Count];
            for (int i = 0; i < sorted.Count; i++) {
                if (removed[i]) continue;
                result.Add(sorted[i]);
                for (int j = i + 1; j < sorted.Count; j++)
                    if (!removed[j] && IoU(sorted[i], sorted[j]) > NmsThreshold)
                        removed[j] = true;
            }
        }
        return result;
    }

    private static float IoU(Detection a, Detection b) {
        float x1 = Math.Max(a.X, b.X);
        float y1 = Math.Max(a.Y, b.Y);
        float x2 = Math.Min(a.X + a.Width, b.X + b.Width);
        float y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);

        float inter = Math.Max(0f, x2 - x1) * Math.Max(0f, y2 - y1);
        float union = a.Width * a.Height + b.Width * b.Height - inter;

        return union <= 0f ? 0f : inter / union;
    }

    public void Dispose() => _session.Dispose();
}