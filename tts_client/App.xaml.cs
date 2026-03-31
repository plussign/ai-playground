using Microsoft.Extensions.DependencyInjection;

namespace TtsClient;

public partial class App : Application
{
    private readonly IServiceProvider _serviceProvider;

    public App(IServiceProvider serviceProvider)
    {
        InitializeComponent();
        _serviceProvider = serviceProvider;
    }

    protected override Window CreateWindow(IActivationState? activationState)
    {
        var mainPage = _serviceProvider.GetRequiredService<MainPage>();

        var window = new Window(mainPage)
        {
            Title = "TTS客户端"
        };

        return window;
    }
}