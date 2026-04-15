using System.Collections.Concurrent;

namespace Hachiko;

/// <summary>
/// Interrupt-capable speech queue for assistive navigation.
/// • Cancels current speech immediately when a higher-priority phrase arrives.
/// • Two slots: priority (movable objects) always preempts regular.
/// • Per-key cooldown — shorter for movable objects (1.2s vs 2.0s).
/// • Queue depth = 1 per slot: only the freshest item ever plays.
/// </summary>
public class SpeechQueue : IDisposable {
    private const double CooldownMovable = 1.2;
    private const double CooldownStatic = 2.0;

    private readonly Dictionary<string, DateTime> _lastSpoken = new();
    private readonly SemaphoreSlim _signal = new(0, int.MaxValue);
    private readonly CancellationTokenSource _cts = new();

    // Two slots — priority wins
    private volatile string _priorityPending;  // movable objects
    private volatile string _regularPending;   // static objects

    // CTS for the currently running TTS call — cancelled when we interrupt
    private CancellationTokenSource _currentTts;

    private bool _disposed;

    public SpeechQueue() => Task.Run(DrainLoop);

    // ── Public API ───────────────────────────────────────────────────────────

    public void Speak(string key, string phrase, bool isMovable = false) {
        if (string.IsNullOrWhiteSpace(phrase)) return;

        double cooldown = isMovable ? CooldownMovable : CooldownStatic;

        lock (_lastSpoken) {
            if (_lastSpoken.TryGetValue(key, out var last) &&
                (DateTime.Now - last).TotalSeconds < cooldown)
                return;

            _lastSpoken[key] = DateTime.Now;
        }

        if (isMovable) {
            _priorityPending = phrase;
            // Interrupt current speech if we have something urgent
            Interlocked.Exchange(ref _currentTts, null)?.Cancel();
        } else {
            // Only fill regular slot if nothing priority is pending
            if (_priorityPending == null)
                _regularPending = phrase;
        }

        _signal.Release();
    }

    public void PurgeMissing(IEnumerable<string> activeKeys) {
        var active = new HashSet<string>(activeKeys);
        lock (_lastSpoken) {
            var stale = _lastSpoken.Keys.Where(k => !active.Contains(k)).ToList();
            foreach (var k in stale) _lastSpoken.Remove(k);
        }
    }

    // ── Drain loop ───────────────────────────────────────────────────────────

    private async Task DrainLoop() {
        var token = _cts.Token;
        while (!token.IsCancellationRequested) {
            try {
                await _signal.WaitAsync(token);

                // Drain extra signals
                while (_signal.CurrentCount > 0) _signal.Wait(0);

                // Priority slot wins; fall back to regular
                var text = Interlocked.Exchange(ref _priorityPending, null)
                        ?? Interlocked.Exchange(ref _regularPending, null);

                if (string.IsNullOrEmpty(text)) continue;

                using var ttsCts = CancellationTokenSource.CreateLinkedTokenSource(token);
                Interlocked.Exchange(ref _currentTts, ttsCts);

                try {
                    await TextToSpeech.Default.SpeakAsync(text,
                        new SpeechOptions { Pitch = 1.05f, Volume = 1.0f },
                        ttsCts.Token);
                } catch (OperationCanceledException) {
                    // Interrupted by a priority phrase — that's expected, loop continues
                }

                Interlocked.CompareExchange(ref _currentTts, null, ttsCts);
            } catch (OperationCanceledException) { break; } catch { /* ignore TTS engine errors */ }
        }
    }

    public void Dispose() {
        if (_disposed) return;
        _disposed = true;
        _currentTts?.Cancel();
        _cts.Cancel();
        _cts.Dispose();
        _signal.Dispose();
    }
}