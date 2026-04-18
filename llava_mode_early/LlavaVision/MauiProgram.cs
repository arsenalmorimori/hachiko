using Microsoft.Extensions.Logging;
using Camera.MAUI;

namespace LlavaVision;

public static class MauiProgram {
    public static MauiApp CreateMauiApp() {
        var builder = MauiApp.CreateBuilder();
        builder
            .UseMauiApp<App>()
            .UseMauiCameraView()   // Camera.MAUI
            .ConfigureFonts(fonts => {
                fonts.AddFont("SpaceMono-Regular.ttf", "SpaceMono");
                fonts.AddFont("SpaceMono-Bold.ttf", "SpaceMonoBold");
            });

        builder.Services.AddSingleton<MainPage>();
        builder.Services.AddSingleton<VisionService>();



        return builder.Build();
    }
}