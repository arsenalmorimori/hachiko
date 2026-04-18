using Microsoft.Maui.Controls;
using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace LlavaVision;

public partial class MainPage : ContentPage {
    // ── Change this IP to your PC's hotspot IP (check `ipconfig` on Windows)
    private const string ServerUrl = "http://192.168.1.13:8000/analyze";

    // Reuse a single HttpClient for speed (no repeated handshake overhead)
    private static readonly HttpClient _http = new HttpClient {
        Timeout = TimeSpan.FromMinutes(5)   // LLaVA can be slow on first run
    };

    public MainPage() {
        InitializeComponent();
    }

    private async void OnCameraClicked(object sender, EventArgs e) {
        // ── 1. Open default camera ──────────────────────────────────────────
        if (!MediaPicker.Default.IsCaptureSupported) {
            await DisplayAlert("Unsupported", "Camera capture is not supported on this device.", "OK");
            return;
        }

        FileResult? photo;
        try {
            photo = await MediaPicker.Default.CapturePhotoAsync();
        } catch (Exception ex) {
            await DisplayAlert("Camera Error", ex.Message, "OK");
            return;
        }

        if (photo is null) return;   // user cancelled

        // ── 2. Show preview ─────────────────────────────────────────────────
        string localPath = Path.Combine(FileSystem.CacheDirectory, photo.FileName);
        await using (var stream = await photo.OpenReadAsync())
        await using (var fs = File.OpenWrite(localPath))
            await stream.CopyToAsync(fs);

        PreviewImage.Source = ImageSource.FromFile(localPath);
        ImageCard.IsVisible = true;
        ResultCard.IsVisible = false;
        StatusRow.IsVisible = true;
        CameraButton.IsEnabled = false;

        // ── 3. Send to Python server ─────────────────────────────────────────
        try {
            string result = await SendImageAsync(localPath);
            ResultLabel.Text = result;
            ResultCard.IsVisible = true;
        } catch (Exception ex) {
            ResultLabel.Text = $"⚠ Error: {ex.Message}\n\nMake sure the Python server is running and ServerUrl is correct.";
            ResultCard.IsVisible = true;
        } finally {
            StatusRow.IsVisible = false;
            CameraButton.IsEnabled = true;
        }
    }

    /// <summary>
    /// Sends the image as base64 JSON — no multipart overhead, single round-trip.
    /// </summary>
    private static async Task<string> SendImageAsync(string imagePath) {
        byte[] imageBytes = await File.ReadAllBytesAsync(imagePath);
        string base64 = Convert.ToBase64String(imageBytes);

        // Detect mime type from extension
        string ext = Path.GetExtension(imagePath).ToLowerInvariant();
        string mimeType = ext == ".png" ? "image/png" : "image/jpeg";

        var payload = new {
            image = base64,
            mime_type = mimeType
        };

        string json = JsonSerializer.Serialize(payload);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        HttpResponseMessage response = await _http.PostAsync(ServerUrl, content);
        response.EnsureSuccessStatusCode();

        string responseJson = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(responseJson);
        return doc.RootElement.GetProperty("result").GetString() ?? "(empty response)";
    }
}