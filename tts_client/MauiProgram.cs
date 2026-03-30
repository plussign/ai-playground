using TtsClient.Services;
using TtsClient.ViewModels;

namespace TtsClient;

public static class MauiProgram
{
    public static MauiApp CreateMauiApp()
    {
        var builder = MauiApp.CreateBuilder();

        builder
            .UseMauiApp<App>()
            .ConfigureFonts(fonts =>
            {
            });

        builder.Services.AddSingleton<SettingsService>();
        builder.Services.AddSingleton<OpenAiTtsService>();
        builder.Services.AddSingleton<AudioPlaybackService>();
        builder.Services.AddSingleton<MainPageViewModel>();
        builder.Services.AddSingleton<MainPage>();

        return builder.Build();
    }
}