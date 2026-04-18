using System.Net.Http;
using System.Text;
using System.Text.Json;

namespace LlavaVision;

public class VisionService {
    // ── Change this if your laptop IP changes ──────────────────────────────────
    private const string ServerBaseUrl = "http://192.168.1.13:5000";
    // ──────────────────────────────────────────────────────────────────────────

    private readonly HttpClient _http;

    public VisionService() {
        _http = new HttpClient {
            Timeout = TimeSpan.FromSeconds(120)
        };
    }

    /// <summary>Check whether the Python server + Ollama are reachable.</summary>
    public async Task<(bool ok, string message)> CheckHealthAsync() {
        try {
            var response = await _http.GetAsync($"{ServerBaseUrl}/health");
            var json = await response.Content.ReadAsStringAsync();
            var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            bool ollamaOk = root.GetProperty("ollama").GetString() == "running";
            bool llavaOk = root.GetProperty("llava_available").GetBoolean();

            if (!ollamaOk) return (false, "Ollama is not running on the server.");
            if (!llavaOk) return (false, "LLaVA model not found. Run: ollama pull llava");

            return (true, "Server ready ✓");
        } catch (HttpRequestException) {
            return (false, $"Cannot reach server at {ServerBaseUrl}.\nMake sure the Python server is running.");
        } catch (TaskCanceledException) {
            return (false, "Connection timed out.");
        } catch (Exception ex) {
            return (false, $"Error: {ex.Message}");
        }
    }

    /// <summary>Send a JPEG byte array to the server for LLaVA analysis.</summary>
    public async Task<(bool success, string result)> AnalyzeImageAsync(byte[] imageBytes) {
        try {
            using var content = new MultipartFormDataContent();

            var fileContent = new ByteArrayContent(imageBytes);
            fileContent.Headers.ContentType =
                new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");

            content.Add(fileContent, "file", "image.jpg");

            var response = await _http.PostAsync($"{ServerBaseUrl}/analyze", content);
            var json = await response.Content.ReadAsStringAsync();

            var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            bool ok = root.GetProperty("success").GetBoolean();
            string text = root.GetProperty("result").GetString() ?? "";

            return (ok, text);
        } catch (Exception ex) {
            return (false, ex.Message);
        }
    }
}