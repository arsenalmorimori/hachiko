using System.Globalization;
using Camera.MAUI;
using SkiaSharp;
using SkiaSharp.Views.Maui;

namespace Hachiko;

public partial class MainPage : ContentPage {
    private YoloInferenceService _yolo;
    private List<Detection> _detections = new();
    private bool _processing;
    private CancellationTokenSource _cts;
    private readonly Dictionary<string, DateTime> _lastSeen = new();

    private int _globalFrameId = 0;

    private int _frameCount = 0;
    private int _lastW = 640;
    private int _lastH = 640;
    private int _inferenceCount = 0;
    private DateTime _fpsTimer = DateTime.Now;

    private float _currentZoom = 1.0f;
    private bool _cameraReady = false;

    // ── NEW: speech + distance ──────────────────────────────────────────────
    private SpeechQueue _speech;

    // How many pixels tall the inference frame is (set once we know camera res)
    // We use _lastH which is the *original* camera frame height.
    // DistanceEstimator expects the pixel height in the space where box coords live.

    public MainPage() {
        InitializeComponent();
        _speech = new SpeechQueue();
        LoadModel();
    }

    // ─── Model Loading ────────────────────────────────────────────────────────

    async void LoadModel() {
        try {
            using var stream = await FileSystem.OpenAppPackageFileAsync("hachiko_1.onnx");
            using var ms = new MemoryStream();
            await stream.CopyToAsync(ms);
            _yolo = new YoloInferenceService(ms.ToArray());
        } catch (Exception ex) {
            await DisplayAlert("Model Error", $"Failed to load model: {ex.Message}", "OK");
        }
    }

    // ─── Lifecycle ────────────────────────────────────────────────────────────

    protected override void OnAppearing() {
        base.OnAppearing();
        _cts = new CancellationTokenSource();
        _ = RunInferenceLoop(_cts.Token);
    }

    protected override void OnDisappearing() {
        base.OnDisappearing();
        _cameraReady = false;
        _cts?.Cancel();
        _cts?.Dispose();
        _cts = null;
    }

    // ─── Camera Setup ─────────────────────────────────────────────────────────

    void CamView_CamerasLoaded(object s, EventArgs e) {
        var backCameras = CamView.Cameras
            .Where(c => c.Position == CameraPosition.Back)
            .ToList();

        CameraInfo selectedCamera =
            backCameras.OrderBy(c => c.MinZoomFactor).FirstOrDefault()
            ?? CamView.Cameras.FirstOrDefault();

        if (selectedCamera == null) return;

        CamView.Camera = selectedCamera;

        MainThread.BeginInvokeOnMainThread(async () => {
            var result = await CamView.StartCameraAsync();
            if (result == CameraResult.Success) {
                _cameraReady = true;
                await ApplyZoomAsync(0.5f);
            }

            CamView.AutoSnapShotFormat = Camera.MAUI.ImageFormat.JPEG;
            CamView.AutoSnapShotSeconds = 0.05f;
            CamView.AutoSnapShotAsImageSource = false;
        });
    }

    // ─── Zoom ─────────────────────────────────────────────────────────────────

    async Task ApplyZoomAsync(float zoom) {
        _currentZoom = zoom;
        if (_cameraReady) {
            try {
                CamView.ZoomFactor = zoom;
                await Task.Delay(80);
                CamView.ZoomFactor = zoom;
            } catch { }
        }
    }

    // ─── Inference Loop ───────────────────────────────────────────────────────

    async Task RunInferenceLoop(CancellationToken token) {
        while (!token.IsCancellationRequested) {
            try {
                _frameCount++;
                _globalFrameId++;

                // Run inference on every other frame (was every 3rd)
                if (_frameCount % 2 != 0 || _processing || _yolo == null) {
                    await Task.Delay(16, token);
                    continue;
                }

                var streamRef = CamView.SnapShotStream;
                if (streamRef == null || streamRef.Length == 0) {
                    await Task.Delay(16, token);
                    continue;
                }

                _processing = true;

                byte[] bytes;
                try {
                    using var ms = new MemoryStream();
                    streamRef.Position = 0;
                    await streamRef.CopyToAsync(ms);
                    bytes = ms.ToArray();
                } catch {
                    _processing = false;
                    await Task.Delay(16, token);
                    continue;
                }

                (bool ok, int imgW, int imgH) = await Task.Run(() => DecodeIntoYolo(bytes));
                if (ok) {
                    var currentDetections = await Task.Run(() => _yolo.Detect(imgW, imgH));
                    _lastW = imgW;
                    _lastH = imgH;

                    _detections = currentDetections;
                    _inferenceCount++;

                    // ── FPS / count HUD ──────────────────────────────────────
                    var now = DateTime.Now;
                    double elapsed = (now - _fpsTimer).TotalSeconds;
                    if (elapsed >= 1.0) {
                        int fps = (int)(_inferenceCount / elapsed);
                        int objCount = _detections.Count;
                        _inferenceCount = 0;
                        _fpsTimer = now;

                        MainThread.BeginInvokeOnMainThread(() => {
                            FpsLabel.Text = fps.ToString();
                            ObjLabel.Text = objCount.ToString();
                        });
                    }

                    // ── Announce detections ───────────────────────────────────
                    AnnounceDetections(_detections, imgW, imgH);

                    MainThread.BeginInvokeOnMainThread(() => OverlayCanvas.InvalidateSurface());
                }
            } catch (OperationCanceledException) { break; } catch { /* skip bad frame */ } finally {
                _processing = false;
                await Task.Delay(16, token);
            }
        }
    }

    // ─── Speech Announcer ─────────────────────────────────────────────────────

    // ─── Movable object categories ────────────────────────────────────────────────

    private static readonly HashSet<string> MovableLabels = new(StringComparer.OrdinalIgnoreCase) {
    "person", "car", "motorcycle", "bicycle", "bus", "truck",
    "cat", "dog", "bird", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe"
};

    // How long (ms) an object must be absent before we forget it
    private const int StaleThresholdMs = 500;

    // ─── Announce ─────────────────────────────────────────────────────────────────

    // ─── Announce ─────────────────────────────────────────────────────────────────

    void AnnounceDetections(List<Detection> dets, int frameW, int frameH) {
        if (dets == null || dets.Count == 0) return;

        var now = DateTime.Now;
        var activeKeys = new HashSet<string>();

        // ── 1. Split movable vs static ───────────────────────────────────────────
        var movable = dets.Where(d => MovableLabels.Contains(d.Label)).ToList();
        var toProcess = movable.Count > 0 ? movable : dets;
        bool hasMovable = movable.Count > 0;
        bool manyObjects = toProcess.Count >= 2;

        // ── 2. Deduplicate same-label objects by merging into one announcement ───
        //    e.g. 3 "person" → one "3 people" entry (closest one sets the position)
        var grouped = toProcess
            .GroupBy(d => d.Label)
            .Select(g => {
                var closest = g.OrderBy(d => DistanceEstimator.EstimateMetres(d, frameH) ?? float.MaxValue).First();
                return (label: g.Key, count: g.Count(), rep: closest);
            })
            .ToList();

        // ── 3. Build candidate phrases, sorted closest-first ─────────────────────
        var candidates = new List<(string key, string phrase, float dist, bool isMovable)>();

        foreach (var (label, count, rep) in grouped) {
            float? distM = DistanceEstimator.EstimateMetres(rep, frameH);
            float sortDist = distM ?? float.MaxValue;

            // Stable spatial key (80px grid cells)
            int cx = (int)(rep.X + rep.Width / 2f) / 80;
            int cy = (int)(rep.Y + rep.Height / 2f) / 80;
            string key = $"{label}:{cx}:{cy}";

            activeKeys.Add(key);
            _lastSeen[key] = now;

            bool isMovObj = MovableLabels.Contains(label);
            string phrase;

            if (manyObjects) {
                // 2+ objects of any kind → label (+ count) only — no distance noise
                string displayLabel = count > 1
                    ? $"{count} {PluralLabel(label)}"
                    : Capitalise(label);
                phrase = displayLabel;
            } else if (distM.HasValue) {
                // Single object with distance → full detail
                string dir = DistanceEstimator.GetDirection(rep, frameW);
                string dist = DistanceEstimator.FormatDistance(distM.Value);
                phrase = $"{Capitalise(label)} {dir}, {dist}";
            } else {
                // Single object, no distance estimate
                string dir = DistanceEstimator.GetDirection(rep, frameW);
                phrase = $"{Capitalise(label)} {dir}";
            }

            candidates.Add((key, phrase, sortDist, isMovObj));
        }

        // ── 4. Expire stale keys ─────────────────────────────────────────────────
        var expired = _lastSeen
            .Where(kv => (now - kv.Value).TotalMilliseconds > StaleThresholdMs)
            .Select(kv => kv.Key).ToList();
        foreach (var k in expired) _lastSeen.Remove(k);

        _speech.PurgeMissing(activeKeys);

        // ── 5. Announce closest-first; movable objects get the priority slot ─────
        foreach (var (key, phrase, _, isMovObj) in candidates.OrderBy(c => c.dist))
            _speech.Speak(key, phrase, isMovable: isMovObj);
    }

    // Naive English plural for common COCO labels
    static string PluralLabel(string label) => label switch {
        "person" => "people",
        "bus" => "buses",
        "bicycle" => "bicycles",
        "sheep" => "sheep",
        "deer" => "deer",
        _ => label + "s"
    };

    static string Capitalise(string s) =>
        string.IsNullOrEmpty(s) ? s : char.ToUpper(s[0]) + s[1..];
    // ─── Preprocessing ────────────────────────────────────────────────────────

    (bool ok, int w, int h) DecodeIntoYolo(byte[] imgBytes) {
        const int size = 640;

        using var bmp = SKBitmap.Decode(imgBytes);
        if (bmp == null) return (false, 0, 0);

        int origW = bmp.Width;
        int origH = bmp.Height;

        // Decode directly into Rgb888x at target size — one allocation, no intermediate copy
        var info = new SKImageInfo(size, size, SKColorType.Rgb888x, SKAlphaType.Opaque);
        using var resized = bmp.Resize(info, SKFilterQuality.Low); // Low = bilinear, ~2× faster than None on ARM
        if (resized == null) return (false, 0, 0);

        // Write directly into the preallocated tensor buffer
        float[] dst = _yolo.TensorData;
        int planeSize = size * size;

        unsafe {
            byte* ptr = (byte*)resized.GetPixels().ToPointer();
            for (int i = 0; i < planeSize; i++) {
                int px = i * 4;
                dst[i] = ptr[px] * (1f / 255f);
                dst[planeSize + i] = ptr[px + 1] * (1f / 255f);
                dst[planeSize * 2 + i] = ptr[px + 2] * (1f / 255f);
            }
        }

        return (true, origW, origH);
    }
    // ─── Overlay Drawing ──────────────────────────────────────────────────────

    void OverlayCanvas_PaintSurface(object s, SKPaintSurfaceEventArgs e) {
        var canvas = e.Surface.Canvas;
        canvas.Clear();

        if (_detections == null || _detections.Count == 0) return;

        float scaleX = (float)e.Info.Width / _lastW;
        float scaleY = (float)e.Info.Height / _lastH;

        float fontSize = e.Info.Width * 0.03f;
        float boxStroke = e.Info.Width * 0.004f;
        float labelH = fontSize + 12f;

        using var boxPaint = new SKPaint {
            Color = new SKColor(0x55, 0x55, 0xff),
            Style = SKPaintStyle.Stroke,
            StrokeWidth = boxStroke,
            IsAntialias = true
        };
        using var textPaint = new SKPaint {
            Color = SKColors.White,
            TextSize = fontSize,
            IsAntialias = true,
            FakeBoldText = true
        };
        using var bgPaint = new SKPaint {
            Color = new SKColor(0x55, 0x55, 0xff, 200),
            Style = SKPaintStyle.Fill
        };

        foreach (var det in _detections) {
            float x = det.X * scaleX;
            float y = det.Y * scaleY;
            float w = det.Width * scaleX;
            float h = det.Height * scaleY;

            canvas.DrawRect(x, y, w, h, boxPaint);

            // ── Label: "person ahead 2.3m  87%" ─────────────────────────────
            string distPart = "";
            float? distM = DistanceEstimator.EstimateMetres(det, _lastH);
            if (distM.HasValue) {
                string dir = DistanceEstimator.GetDirection(det, _lastW);
                distPart = $" | {dir} | {distM.Value:F1}m";
            }

            //string label = $"{det.Label}{distPart}  {det.Confidence:P0}";
            string label = $"{det.Label}{distPart}";
            float tw = textPaint.MeasureText(label);
            float labelY = y > labelH + 4 ? y - labelH : y + h + 2;

            canvas.DrawRoundRect(x, labelY, tw + 16, labelH, 6, 6, bgPaint);
            canvas.DrawText(label, x + 8, labelY + labelH - 6, textPaint);
        }
    }

    // ─── Dispose ──────────────────────────────────────────────────────────────

    protected override void OnHandlerChanging(HandlerChangingEventArgs args) {
        if (args.NewHandler == null) {
            _speech?.Dispose();
            _speech = null;
        }
        base.OnHandlerChanging(args);
    }
}