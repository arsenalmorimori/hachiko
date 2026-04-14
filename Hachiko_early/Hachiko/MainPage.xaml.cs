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
    private int _frameCount = 0;
    private int _lastW = 640;
    private int _lastH = 640;

    private int _inferenceCount = 0;
    private DateTime _fpsTimer = DateTime.Now;

    private float _currentZoom = 1.0f;
    private bool _cameraReady = false;   // guard — don't set zoom until camera is running

    public MainPage() {
        InitializeComponent();
        LoadModel();
    }

    // ─── Model Loading ────────────────────────────────────────────────────────

    async void LoadModel() {
        try {
            using var stream = await FileSystem.OpenAppPackageFileAsync("yolov8s.onnx");
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
        // Pick the back camera with the widest FOV (lowest MinZoomFactor)
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
                await ApplyZoomAsync(0.5f);
            }

            CamView.AutoSnapShotFormat = Camera.MAUI.ImageFormat.JPEG;
            CamView.AutoSnapShotSeconds = 0.05f;
            CamView.AutoSnapShotAsImageSource = false;
        });
    }

    // ─── Zoom ────────────────────────────────────────────────────────
    async Task ApplyZoomAsync(float zoom) {
        _currentZoom = zoom;

        // Only drive the camera property once it is fully started
        if (_cameraReady) {
            try {
                // hjam40/Camera.MAUI: ZoomFactor is a bindable property on CameraView
                CamView.ZoomFactor = zoom;

                // Small delay then re-apply — some Android devices ignore the first set
                await Task.Delay(80);
                CamView.ZoomFactor = zoom;
            } catch { /* ignore if not supported */ }
        }


    }



    // ─── Inference Loop ───────────────────────────────────────────────────────

    async Task RunInferenceLoop(CancellationToken token) {
        while (!token.IsCancellationRequested) {
            try {
                _frameCount++;

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

                (float[] tensor, int imgW, int imgH) =
                    await Task.Run(() => DecodeAndPreprocess(bytes));

                if (tensor != null) {
                    _lastW = imgW;
                    _lastH = imgH;

                    _detections = await Task.Run(() => _yolo.Detect(tensor, imgW, imgH));
                    _inferenceCount++;

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

                    MainThread.BeginInvokeOnMainThread(() => OverlayCanvas.InvalidateSurface());
                }
            } catch (OperationCanceledException) { break; } catch { /* skip bad frame */ } finally {
                _processing = false;
                await Task.Delay(16, token);
            }
        }
    }

    // ─── Preprocessing ────────────────────────────────────────────────────────

    (float[] tensor, int w, int h) DecodeAndPreprocess(byte[] imgBytes) {
        const int size = 640;

        using var bmp = SKBitmap.Decode(imgBytes);
        if (bmp == null) return (null, 0, 0);

        int origW = bmp.Width;
        int origH = bmp.Height;

        using var rgb = bmp.Copy(SKColorType.Rgb888x);
        if (rgb == null) return (null, 0, 0);

        using var resized = rgb.Resize(
            new SKImageInfo(size, size, SKColorType.Rgb888x),
            SKFilterQuality.None);
        if (resized == null) return (null, 0, 0);

        float[] tensor = new float[3 * size * size];
        int planeSize = size * size;

        unsafe {
            byte* ptr = (byte*)resized.GetPixels().ToPointer();
            for (int i = 0; i < planeSize; i++) {
                tensor[0 * planeSize + i] = ptr[i * 4 + 0] / 255f;
                tensor[1 * planeSize + i] = ptr[i * 4 + 1] / 255f;
                tensor[2 * planeSize + i] = ptr[i * 4 + 2] / 255f;
            }
        }

        return (tensor, origW, origH);
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

            string label = $"{det.Label} {det.Confidence:P0}";
            float tw = textPaint.MeasureText(label);
            float labelY = y > labelH + 4 ? y - labelH : y + h + 2;

            canvas.DrawRoundRect(x, labelY, tw + 16, labelH, 6, 6, bgPaint);
            canvas.DrawText(label, x + 8, labelY + labelH - 6, textPaint);
        }
    }
}
