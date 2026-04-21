using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Hachiko;

// OBJECT PER OBJECT DETECTED
public class Detection {
    public float X, Y, Width, Height, Confidence;
    public int ClassId;
    public string Label;
}


//HANDLER FOR YOLO ONNX PROCESS
public class YoloInferenceService : IDisposable {
    private readonly InferenceSession _session;
    private readonly string _inputName;

    //YOLO CONFIGS
    private const int InputSize = 640;
    private const float ConfThreshold = 0.60f; // Confidence level minimum
    private const float NmsThreshold = 0.45f; // for overlap
    private const int ClassOffset = 4;
    private const int NumProposals = 8400;

    // RAW OBEJCT DETECTED DATA OF TENSOR/YOLO
    private readonly float[] _tensorData;   // 1×3×640×640
    private readonly DenseTensor<float> _tensor;
    private readonly List<NamedOnnxValue> _inputList;

    // REUSABLE MEMORY
    private readonly float[] _scores = new float[NumProposals];
    private readonly int[] _classes = new int[NumProposals];
    private readonly float[] _cx = new float[NumProposals];
    private readonly float[] _cy = new float[NumProposals];
    private readonly float[] _bw = new float[NumProposals];
    private readonly float[] _bh = new float[NumProposals];
    private readonly int[] _indices = new int[NumProposals];
    private readonly bool[] _suppressed = new bool[NumProposals];

    // hachiko_1 model onnx
    private static readonly string[] Labels = {
    "airplane","apple","backpack","banana","bed","bench","bicycle","boat","book","bottle",
    "bowl","broccoli","bus","cake","car","carrot","cat","cell phone","chair","clock",
    "couch","cup","dining table","dog","donut","fire hydrant","fork","handbag","hot dog","keyboard",
    "kite","knife","laptop","microwave","motorcycle","mouse","orange","oven","parking meter","person",
    "pizza","potted plant","refrigerator","remote","sandwich","scissors","sink","spoon","stop sign","teddy bear",
    "toaster","toilet","toothbrush","traffic light","train","truck","tv","umbrella"
};
    public YoloInferenceService(byte[] modelBytes) {
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL; // single-threaded is faster on mobile
        options.InterOpNumThreads = 1;
        options.IntraOpNumThreads = 2; // keep 2 for GEMM parallelism

#if ANDROID
        try { options.AppendExecutionProvider_Nnapi(); } catch { }
#endif

        _session = new InferenceSession(modelBytes, options);
        _inputName = _session.InputMetadata.Keys.First();

        // Preallocate tensor once — reused every frame
        int tensorLen = 3 * InputSize * InputSize;
        _tensorData = new float[tensorLen];
        _tensor = new DenseTensor<float>(_tensorData, new[] { 1, 3, InputSize, InputSize });
        _inputList = new List<NamedOnnxValue>(1) {
            NamedOnnxValue.CreateFromTensor(_inputName, _tensor)
        };
    }

    // ── Public: fills _tensorData in-place, caller must not touch it concurrently ──

    public List<Detection> Detect(int origW, int origH) {
        // _tensorData already filled by the caller (DecodeAndPreprocess writes directly into it)
        using var results = _session.Run(_inputList);
        var output = results.First().AsTensor<float>();
        return ParseOutput(output, origW, origH);
    }

    // ── ParseOutput: zero LINQ, zero heap allocs in hot path ─────────────────

    private List<Detection> ParseOutput(Tensor<float> t, int origW, int origH) {
        var dims = t.Dimensions.ToArray();
        int dim1 = dims[1], dim2 = dims[2];
        bool isCHW = dim1 < dim2;
        int numDet = isCHW ? dim2 : dim1;
        int numAttr = isCHW ? dim1 : dim2;

        float scaleX = (float)origW / InputSize;
        float scaleY = (float)origH / InputSize;
        int numClasses = numAttr - ClassOffset;

        // Pass 1: score every proposal, fill flat scratch arrays
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

        // Pass 2: sort indices descending by score (insertion sort is fast for small N)
        InsertionSortDesc(_indices, _scores, candidateCount);

        // Pass 3: class-aware NMS over sorted indices
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

    public float[] TensorData => _tensorData; // caller writes directly here

    public void Dispose() => _session.Dispose();
}