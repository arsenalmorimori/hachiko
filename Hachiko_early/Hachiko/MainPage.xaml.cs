using Camera.MAUI;
using SkiaSharp;
using SkiaSharp.Views.Maui;

namespace Hachiko;

public partial class MainPage : ContentPage {

    // ── State ─────────────────────────────────────────────────────────────────
    private YoloInferenceService _yolo;
    private List<Detection> _detections = new();
    private bool _processing;
    private CancellationTokenSource _cts;
    private readonly Dictionary<string, DateTime> _lastSeen = new();

    private int _frameCount = 0;
    private int _lastW = 320;   // match InputSize
    private int _lastH = 320;
    private int _inferenceCount = 0;
    private DateTime _fpsTimer = DateTime.Now;

    private float _currentZoom = 1.0f;
    private bool _cameraReady = false;

    // Frame buffer — reused every frame (1 MB is enough for compressed JPEG/PNG)
    private readonly byte[] _frameBuffer = new byte[2 * 1024 * 1024]; // 2 MB

    // Throttle: run inference at most every 200 ms (~5 fps)
    private DateTime _lastInferenceTime = DateTime.MinValue;
    private const int InferenceIntervalMs = 200;

    // Speech
    private SpeechQueue _speech;

    // ── Constructor ───────────────────────────────────────────────────────────
    public MainPage() {
        InitializeComponent();
        _speech = new SpeechQueue();
        LoadModel();
    }

    // ── Model Loading ─────────────────────────────────────────────────────────
    async void LoadModel() {
        try {
            using var stream = await FileSystem.OpenAppPackageFileAsync("hachiko_2.onnx");
            using var ms = new MemoryStream();
            await stream.CopyToAsync(ms);
            _yolo = new YoloInferenceService(ms.ToArray());
            System.Diagnostics.Debug.WriteLine("[App] Model loaded OK");
        } catch (Exception ex) {
            await DisplayAlert("Model Error", $"Failed to load model:\n{ex.Message}", "OK");
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────
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

    // ── Camera Setup ──────────────────────────────────────────────────────────
    void CamView_CamerasLoaded(object s, EventArgs e) {
        var backCam = CamView.Cameras
            .Where(c => c.Position == CameraPosition.Back)
            .OrderBy(c => c.MinZoomFactor)
            .FirstOrDefault()
            ?? CamView.Cameras.FirstOrDefault();

        if (backCam == null) return;
        CamView.Camera = backCam;

        MainThread.BeginInvokeOnMainThread(async () => {
            var result = await CamView.StartCameraAsync();
            if (result == CameraResult.Success) {
                _cameraReady = true;
                await ApplyZoomAsync(0.5f);
            }
            // JPEG is faster to decode than PNG on mobile
            CamView.AutoSnapShotFormat = Camera.MAUI.ImageFormat.JPEG;
            CamView.AutoSnapShotSeconds = 0.05f;
            CamView.AutoSnapShotAsImageSource = false;
        });
    }

    // ── Zoom ──────────────────────────────────────────────────────────────────
    async Task ApplyZoomAsync(float zoom) {
        _currentZoom = zoom;
        if (!_cameraReady) return;
        try {
            CamView.ZoomFactor = zoom;
            await Task.Delay(80);
            CamView.ZoomFactor = zoom; // apply twice — Camera.MAUI quirk
        } catch { }
    }

    // ── Inference Loop ────────────────────────────────────────────────────────
    async Task RunInferenceLoop(CancellationToken token) {
        while (!token.IsCancellationRequested) {
            try {
                _frameCount++;

                // Throttle + guard
                bool tooSoon = (DateTime.Now - _lastInferenceTime).TotalMilliseconds < InferenceIntervalMs;
                if (tooSoon || _processing || _yolo == null) {
                    await Task.Delay(16, token);
                    continue;
                }

                var streamRef = CamView.SnapShotStream;
                if (streamRef == null || streamRef.Length == 0) {
                    await Task.Delay(16, token);
                    continue;
                }

                _processing = true;
                _lastInferenceTime = DateTime.Now;

                // ── Read frame into reusable buffer ───────────────────────────
                int frameLen;
                try {
                    streamRef.Position = 0;
                    frameLen = await streamRef.ReadAsync(_frameBuffer, 0, _frameBuffer.Length, token);
                } catch {
                    _processing = false;
                    await Task.Delay(16, token);
                    continue;
                }

                if (frameLen == 0) {
                    _processing = false;
                    await Task.Delay(16, token);
                    continue;
                }

                // ── Decode + preprocess on background thread ──────────────────
                // Pass frameLen so we only decode the valid bytes
                (bool ok, int imgW, int imgH) = await Task.Run(
                    () => DecodeIntoYolo(_frameBuffer, frameLen), token);

                if (!ok) {
                    _processing = false;
                    await Task.Delay(16, token);
                    continue;
                }

                // ── Run ONNX inference on background thread ───────────────────
                var currentDetections = await Task.Run(() => _yolo.Detect(imgW, imgH), token);
                _lastW = imgW;
                _lastH = imgH;
                _detections = currentDetections;
                _inferenceCount++;

                // ── FPS / count HUD ───────────────────────────────────────────
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

                // ── Speech + overlay ──────────────────────────────────────────
                AnnounceDetections(_detections, imgW, imgH);
                MainThread.BeginInvokeOnMainThread(() => OverlayCanvas.InvalidateSurface());

            } catch (OperationCanceledException) {
                break;
            } catch (Exception ex) {
                System.Diagnostics.Debug.WriteLine($"[Loop] Frame error: {ex.Message}");
            } finally {
                _processing = false;
                await Task.Delay(16, token);
            }
        }
    }

    // ── Preprocessing ─────────────────────────────────────────────────────────
    // FIXED: uses YoloInferenceService.InputSize (320), not hardcoded 640
    (bool ok, int w, int h) DecodeIntoYolo(byte[] imgBytes, int length) {
        int size = YoloInferenceService.InputSize; // 320

        // Decode only the valid bytes
        using var bmp = SKBitmap.Decode(imgBytes.AsSpan(0, length).ToArray());
        if (bmp == null) return (false, 0, 0);

        int origW = bmp.Width;
        int origH = bmp.Height;

        // Resize directly to target resolution — single allocation
        var info = new SKImageInfo(size, size, SKColorType.Rgb888x, SKAlphaType.Opaque);
        using var resized = bmp.Resize(info, SKFilterQuality.None); // nearest-neighbour = fastest
        if (resized == null) return (false, 0, 0);

        // Write normalised RGB planes directly into preallocated tensor buffer
        float[] dst = _yolo.TensorData;
        int planeSize = size * size;

        unsafe {
            byte* ptr = (byte*)resized.GetPixels().ToPointer();
            for (int i = 0; i < planeSize; i++) {
                int px = i * 4;                               // Rgb888x = 4 bytes/pixel
                dst[i] = ptr[px] * (1f / 255f); // R
                dst[planeSize + i] = ptr[px + 1] * (1f / 255f); // G
                dst[planeSize * 2 + i] = ptr[px + 2] * (1f / 255f); // B
            }
        }

        return (true, origW, origH);
    }

    // ── Speech ────────────────────────────────────────────────────────────────
    private static readonly HashSet<string> MovableLabels = new(StringComparer.OrdinalIgnoreCase) {
        "person","car","motorcycle","bicycle","bus","truck",
        "cat","dog","bird","horse","sheep","cow","elephant","bear","zebra","giraffe"
    };

    private const int StaleThresholdMs = 500;

    void AnnounceDetections(List<Detection> dets, int frameW, int frameH) {
        if (dets == null || dets.Count == 0) return;

        var now = DateTime.Now;
        var activeKeys = new HashSet<string>();

        var movable = dets.Where(d => MovableLabels.Contains(d.Label)).ToList();
        var toProcess = movable.Count > 0 ? movable : dets;
        bool manyObjects = toProcess.Count >= 2;

        var grouped = toProcess
            .GroupBy(d => d.Label)
            .Select(g => {
                var closest = g.OrderBy(d => DistanceEstimator.EstimateMetres(d, frameH) ?? float.MaxValue).First();
                return (label: g.Key, count: g.Count(), rep: closest);
            })
            .ToList();

        var candidates = new List<(string key, string phrase, float dist, bool isMovable)>();

        foreach (var (label, count, rep) in grouped) {
            float? distM = DistanceEstimator.EstimateMetres(rep, frameH);
            float sortDist = distM ?? float.MaxValue;

            int cx = (int)(rep.X + rep.Width / 2f) / 80;
            int cy = (int)(rep.Y + rep.Height / 2f) / 80;
            string key = $"{label}:{cx}:{cy}";

            activeKeys.Add(key);
            _lastSeen[key] = now;

            string phrase;
            if (manyObjects) {
                phrase = count > 1 ? $"{count} {PluralLabel(label)}" : Capitalise(label);
            } else if (distM.HasValue) {
                string dir = DistanceEstimator.GetDirection(rep, frameW);
                string dist = DistanceEstimator.FormatDistance(distM.Value);
                phrase = $"{Capitalise(label)} {dir}, {dist}";
            } else {
                phrase = $"{Capitalise(label)} {DistanceEstimator.GetDirection(rep, frameW)}";
            }

            candidates.Add((key, phrase, sortDist, MovableLabels.Contains(label)));
        }

        // Expire stale keys
        var expired = _lastSeen
            .Where(kv => (now - kv.Value).TotalMilliseconds > StaleThresholdMs)
            .Select(kv => kv.Key).ToList();
        foreach (var k in expired) _lastSeen.Remove(k);

        _speech.PurgeMissing(activeKeys);

        foreach (var (key, phrase, _, isMovObj) in candidates.OrderBy(c => c.dist))
            _speech.Speak(key, phrase, isMovable: isMovObj);
    }

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

    // ── Overlay Drawing ───────────────────────────────────────────────────────
    void OverlayCanvas_PaintSurface(object s, SKPaintSurfaceEventArgs e) {
        var canvas = e.Surface.Canvas;
        canvas.Clear();

        if (_detections == null || _detections.Count == 0) return;

        float scaleX = (float)e.Info.Width / _lastW;
        float scaleY = (float)e.Info.Height / _lastH;

        float fontSize = e.Info.Width * 0.02f;
        float boxStroke = e.Info.Width * 0.002f;
        float labelH = fontSize + 9f;

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

            string distPart = "";
            float? distM = DistanceEstimator.EstimateMetres(det, _lastH);
            if (distM.HasValue) {
                string dir = DistanceEstimator.GetDirection(det, _lastW);
                distPart = $" | {dir} | {distM.Value:F1}m";
            }

            string label = $"{det.Label}{distPart}";
            float tw = textPaint.MeasureText(label);
            float labelY = y > labelH + 4 ? y - labelH : y + h + 2;

            canvas.DrawRoundRect(x, labelY, tw + 16, labelH, 6, 6, bgPaint);
            canvas.DrawText(label, x + 8, labelY + labelH - 6, textPaint);
        }
    }

    // ── Dispose ───────────────────────────────────────────────────────────────
    protected override void OnHandlerChanging(HandlerChangingEventArgs args) {
        if (args.NewHandler == null) {
            _speech?.Dispose();
            _speech = null;
        }
        base.OnHandlerChanging(args);
    }
}