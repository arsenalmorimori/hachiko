using Camera.MAUI;
using Microsoft.Extensions.Logging;
using Camera.MAUI;
using SkiaSharp.Views.Maui.Controls.Hosting;

namespace Hachiko
{
    public static class MauiProgram
    {
        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .UseMauiCameraView()       // Camera.MAUI
                .UseSkiaSharp()            // ← this is what was missing
                .ConfigureFonts(fonts => {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                });

#if DEBUG
            builder.Logging.AddDebug();
#endif


            return builder.Build();
        }
    }
}
