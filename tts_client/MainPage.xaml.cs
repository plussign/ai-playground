using TtsClient.ViewModels;

namespace TtsClient;

public partial class MainPage : ContentPage
{
    private bool _isInitialized;

    public MainPage(MainPageViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = viewModel;
        Loaded += OnLoaded;
    }

    private async void OnLoaded(object? sender, EventArgs e)
    {
        if (_isInitialized || BindingContext is not MainPageViewModel viewModel)
        {
            return;
        }

        _isInitialized = true;
        await viewModel.InitializeAsync();
    }
}