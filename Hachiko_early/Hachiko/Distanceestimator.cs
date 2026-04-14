namespace Hachiko;

/// <summary>
/// Rough monocular distance estimate using the apparent bounding-box height.
///
/// Formula (pinhole model):
///   distance = (realHeight * focalLength) / pixelHeight
///
/// focalLength is back-calculated from an assumed reference distance.
/// This is approximate — it breaks with extreme zoom / lens distortion —
/// but gives useful "near / mid / far" guidance without any depth sensor.
/// </summary>
public static class DistanceEstimator {
    // Assumed vertical FOV ~65° for a typical phone rear camera → focal length
    // for a 640-px-tall inference frame:
    //   f = (frameH / 2) / tan(vFOV/2) ≈ 320 / tan(32.5°) ≈ 506 px
    private const float FocalLengthPx = 506f;

    // Approximate real-world heights in metres for common COCO classes.
    // Classes not listed fall back to DefaultHeight.
    private static readonly Dictionary<string, float> RealHeightM = new(StringComparer.OrdinalIgnoreCase)
    {
        { "person",          1.75f },
        { "bicycle",         1.00f },
        { "car",             1.50f },
        { "motorcycle",      1.10f },
        { "bus",             3.20f },
        { "truck",           3.00f },
        { "train",           4.00f },
        { "airplane",        5.00f },
        { "boat",            2.00f },
        { "traffic light",   0.80f },
        { "fire hydrant",    0.60f },
        { "stop sign",       0.75f },
        { "bench",           0.85f },
        { "bird",            0.20f },
        { "cat",             0.25f },
        { "dog",             0.45f },
        { "horse",           1.60f },
        { "sheep",           0.80f },
        { "cow",             1.40f },
        { "elephant",        2.80f },
        { "bear",            1.20f },
        { "chair",           0.90f },
        { "couch",           0.85f },
        { "bed",             0.60f },
        { "dining table",    0.75f },
        { "toilet",          0.70f },
        { "tv",              0.60f },
        { "laptop",          0.25f },
        { "bottle",          0.25f },
        { "cup",             0.12f },
        { "backpack",        0.50f },
        { "umbrella",        1.00f },
        { "cell phone",      0.15f },
        { "clock",           0.30f },
        { "vase",            0.30f },
        { "potted plant",    0.45f },
        { "refrigerator",    1.80f },
        { "microwave",       0.30f },
        { "oven",            0.60f },
        { "sink",            0.50f },
        { "book",            0.22f },
    };

    private const float DefaultHeight = 0.50f; // fallback for unlisted classes

    /// <summary>
    /// Returns estimated distance in metres, or null if box height is too small
    /// to give a meaningful estimate.
    /// </summary>
    public static float? EstimateMetres(Detection det, int frameHeightPx) {
        if (det.Height < 4f) return null; // box too small — unreliable

        float realH = RealHeightM.TryGetValue(det.Label, out var h) ? h : DefaultHeight;
        float dist = (realH * FocalLengthPx) / det.Height;

        // Clamp to a sensible range [0.3 m … 50 m]
        return Math.Clamp(dist, 0.3f, 50f);
    }

    /// <summary>Returns a human-readable distance string like "1.2 m" or "8 m".</summary>
    public static string FormatDistance(float metres)
        => metres < 10f
            ? $"{metres:F1} metres"
            : $"{(int)Math.Round(metres)} metres";

    /// <summary>
    /// Returns the horizontal direction label based on box-centre X
    /// relative to frame width (thirds split).
    /// </summary>
    public static string GetDirection(Detection det, int frameWidthPx) {
        float cx = det.X + det.Width / 2f;
        float third = frameWidthPx / 3f;

        if (cx < third) return "to your left";
        if (cx > third * 2f) return "to your right";
        return "ahead";
    }
}