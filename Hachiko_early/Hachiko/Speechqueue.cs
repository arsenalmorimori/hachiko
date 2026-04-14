using System.Collections.Concurrent;

namespace Hachiko;

/// <summary>
/// Debounced speech queue.
/// - Same label is suppressed for <CooldownSeconds> after it was last spoken.
/// - Only one utterance plays at a time; new ones queue behind it.
/// - Queue is capped so stale items are dropped when inference outruns speech.
/// </summary>
public class SpeechQueue : IDisposable {
    private const double CooldownSeconds = 4.0;   // silence same object for N seconds
    private const int MaxQueueDepth = 3;      // drop older items beyond this

    private readonly ConcurrentQueue<string> _queue = new();
    private readonly Dictionary<string, DateTime> _lastSpoken = new();
    private readonly SemaphoreSlim _signal = new(0, int.MaxValue);
    private readonly CancellationTokenSource _cts = new();
    private bool _disposed;

    public SpeechQueue() => Task.Run(DrainLoop);

    /// <summary>Enqueue a phrase if it is not currently on cooldown.</summary>
    public void Speak(string phrase) {
        if (string.IsNullOrWhiteSpace(phrase)) return;

        lock (_lastSpoken) {
            if (_lastSpoken.TryGetValue(phrase, out var last) &&
                (DateTime.Now - last).TotalSeconds < CooldownSeconds)
                return;

            _lastSpoken[phrase] = DateTime.Now;
        }

        // Trim queue depth — drop the oldest if full
        while (_queue.Count >= MaxQueueDepth)
            _queue.TryDequeue(out _);

        _queue.Enqueue(phrase);
        _signal.Release();
    }

    private async Task DrainLoop() {
        var token = _cts.Token;
        while (!token.IsCancellationRequested) {
            try {
                await _signal.WaitAsync(token);

                if (_queue.TryDequeue(out var text)) {
                    var settings = new SpeechOptions {
                        Pitch = 1.0f,
                        Volume = 1.0f
                    };
                    await TextToSpeech.Default.SpeakAsync(text, settings, token);
                }
            } catch (OperationCanceledException) { break; } catch { /* ignore TTS errors */ }
        }
    }

    public void Dispose() {
        if (_disposed) return;
        _disposed = true;
        _cts.Cancel();
        _cts.Dispose();
        _signal.Dispose();
    }
}