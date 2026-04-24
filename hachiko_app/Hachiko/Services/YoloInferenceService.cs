using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Hachiko.Services;

// OBJECT PER OBJECT DETECTED
public class Detection {
    public float X, Y, Width, Height, Confidence;
    public int ClassId;
    public string Label;
}

// HANDLER FOR YOLO ONNX PROCESS
public class YoloInferenceService : IDisposable {
    private readonly InferenceSession _session;
    private readonly string _inputName;

    // YOLO CONFIGS — change InputSize here and everything else follows
    public const int InputSize = 320;
    private const int NumProposals = 2100;   // 320×320 → 2100 proposals
    private const float ConfThreshold = 0.75f;
    private const float NmsThreshold = 0.45f;
    private const int ClassOffset = 4;

    // Preallocated tensor buffer — reused every frame
    private readonly float[] _tensorData;
    private readonly DenseTensor<float> _tensor;
    private readonly List<NamedOnnxValue> _inputList;

    // Reusable scratch arrays — sized to NumProposals
    private readonly float[] _scores = new float[NumProposals];
    private readonly int[] _classes = new int[NumProposals];
    private readonly float[] _cx = new float[NumProposals];
    private readonly float[] _cy = new float[NumProposals];
    private readonly float[] _bw = new float[NumProposals];
    private readonly float[] _bh = new float[NumProposals];
    private readonly int[] _indices = new int[NumProposals];
    private readonly bool[] _suppressed = new bool[NumProposals];

    // COCO labels for hachiko_1 model
    private static readonly string[] Labels = {
        "1", "10", "100", "1000", "20", "200", "5", "50", "500", "airplane", "animal", "apple", "backpack", "banana", "bed", "bench", "bike", "boat", "book", "bottle", "bowl", "broccoli", "cake", "carrot", "cell phone", "chair", "clock", "couch", "crosswalk", "cup", "desk", "dobby", "donut", "door", "fire hydrant", "fork", "handbag", "hazard-sign", "keyboard", "kite", "knife", "laptop", "microwave", "mouse", "orange", "oven", "parking meter", "person", "pizza", "potted plant", "refrigerator", "remote", "sandwich", "scissors", "sink", "spoon", "stairs", "stop sign", "teddy bear", "toaster", "toilet", "traffic light", "train", "tv", "umbrella", "vehicle"

    };

    

    public YoloInferenceService(byte[] modelBytes) {
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        options.InterOpNumThreads = 1;
        options.IntraOpNumThreads = 2;

#if ANDROID
        try {
            options.AppendExecutionProvider_Nnapi();
            System.Diagnostics.Debug.WriteLine("[YOLO] NNAPI enabled");
        } catch (Exception ex) {
            System.Diagnostics.Debug.WriteLine($"[YOLO] NNAPI failed: {ex.Message}");
        }
#endif

        _session = new InferenceSession(modelBytes, options);
        _inputName = _session.InputMetadata.Keys.First();

        // Log actual model output shape so we can verify NumProposals
        System.Diagnostics.Debug.WriteLine(
            $"[YOLO] Input: {_inputName}, InputSize constant: {InputSize}");

        int tensorLen = 3 * InputSize * InputSize;
        _tensorData = new float[tensorLen];
        _tensor = new DenseTensor<float>(_tensorData, new[] { 1, 3, InputSize, InputSize });
        _inputList = new List<NamedOnnxValue>(1) {
            NamedOnnxValue.CreateFromTensor(_inputName, _tensor)
        };
    }

    // Caller fills TensorData directly, then calls Detect()
    public float[] TensorData => _tensorData;

    public List<Detection> Detect(int origW, int origH) {
        using var results = _session.Run(_inputList);
        var output = results.First().AsTensor<float>();

        // Log output shape once to verify model matches NumProposals
        var dims = output.Dimensions.ToArray();
        System.Diagnostics.Debug.WriteLine(
            $"[YOLO] Output dims: [{string.Join(", ", dims.ToArray())}]");

        return ParseOutput(output, origW, origH);
    }

    private List<Detection> ParseOutput(Tensor<float> t, int origW, int origH) {
        var dims = t.Dimensions.ToArray();
        int dim1 = dims[1], dim2 = dims[2];

        // YOLOv8 exports as [1, 84, 2100] (CHW) or [1, 2100, 84] (HWC)
        bool isCHW = dim1 < dim2;
        int numDet = isCHW ? dim2 : dim1;
        int numAttr = isCHW ? dim1 : dim2;
        int numClasses = numAttr - ClassOffset;

        float scaleX = (float)origW / InputSize;
        float scaleY = (float)origH / InputSize;

        // Guard: if model still outputs 8400, scratch arrays are too small
        if (numDet > NumProposals) {
            System.Diagnostics.Debug.WriteLine(
                $"[YOLO] WARNING: model outputs {numDet} proposals but NumProposals={NumProposals}. " +
                "Re-export at imgsz=320 or increase NumProposals to 8400.");
            numDet = NumProposals; // clamp to avoid out-of-bounds
        }

        // Pass 1: filter by confidence, fill scratch arrays
        int candidateCount = 0;
        for (int i = 0; i < numDet; i++) {
            float bestScore = 0f;
            int bestClass = 0;
            for (int c = 0; c < numClasses; c++) {
                float s = isCHW ? t[0, c + ClassOffset, i] : t[0, i, c + ClassOffset];
                if (s > bestScore) { bestScore = s; bestClass = c; }
            }
            if (bestScore < ConfThreshold) continue;

            _scores[candidateCount] = bestScore;
            _classes[candidateCount] = bestClass;
            _cx[candidateCount] = isCHW ? t[0, 0, i] : t[0, i, 0];
            _cy[candidateCount] = isCHW ? t[0, 1, i] : t[0, i, 1];
            _bw[candidateCount] = isCHW ? t[0, 2, i] : t[0, i, 2];
            _bh[candidateCount] = isCHW ? t[0, 3, i] : t[0, i, 3];
            _indices[candidateCount] = candidateCount;
            candidateCount++;
        }

        // Pass 2: sort descending by score
        InsertionSortDesc(_indices, _scores, candidateCount);

        // Pass 3: class-aware NMS
        Array.Clear(_suppressed, 0, candidateCount);
        var result = new List<Detection>(candidateCount);

        for (int a = 0; a < candidateCount; a++) {
            int i = _indices[a];
            if (_suppressed[i]) continue;

            float ax = (_cx[i] - _bw[i] / 2f) * scaleX;
            float ay = (_cy[i] - _bh[i] / 2f) * scaleY;
            float aw = _bw[i] * scaleX;
            float ah = _bh[i] * scaleY;

            result.Add(new Detection {
                X = ax, Y = ay, Width = aw, Height = ah,
                Confidence = _scores[i],
                ClassId = _classes[i],
                Label = _classes[i] < Labels.Length ? Labels[_classes[i]] : $"cls{_classes[i]}"
            });

            for (int b = a + 1; b < candidateCount; b++) {
                int j = _indices[b];
                if (_suppressed[j] || _classes[j] != _classes[i]) continue;

                float bx = (_cx[j] - _bw[j] / 2f) * scaleX;
                float by = (_cy[j] - _bh[j] / 2f) * scaleY;
                float bw = _bw[j] * scaleX;
                float bhh = _bh[j] * scaleY;

                if (IoU(ax, ay, aw, ah, bx, by, bw, bhh) > NmsThreshold)
                    _suppressed[j] = true;
            }
        }

        return result;
    }

    private static void InsertionSortDesc(int[] idx, float[] scores, int n) {
        for (int i = 1; i < n; i++) {
            int key = idx[i];
            float keyScore = scores[key];
            int j = i - 1;
            while (j >= 0 && scores[idx[j]] < keyScore) {
                idx[j + 1] = idx[j];
                j--;
            }
            idx[j + 1] = key;
        }
    }

    private static float IoU(float ax, float ay, float aw, float ah,
                              float bx, float by, float bw, float bh) {
        float x1 = MathF.Max(ax, bx), y1 = MathF.Max(ay, by);
        float x2 = MathF.Min(ax + aw, bx + bw);
        float y2 = MathF.Min(ay + ah, by + bh);
        float inter = MathF.Max(0f, x2 - x1) * MathF.Max(0f, y2 - y1);
        float union = aw * ah + bw * bh - inter;
        return union <= 0f ? 0f : inter / union;
    }

    public void Dispose() => _session.Dispose();
}