using System.Text;

namespace TtsClient.WinUI;

public partial class App : MauiWinUIApplication
{
    public App()
    {
        InitializeComponent();
        UnhandledException += OnUnhandledException;
    }

    protected override MauiApp CreateMauiApp()
    {
        return MauiProgram.CreateMauiApp();
    }

    private static void OnUnhandledException(object sender, Microsoft.UI.Xaml.UnhandledExceptionEventArgs e)
    {
        try
        {
            var logPath = Path.Combine(AppContext.BaseDirectory, "startup-crash.log");
            var builder = new StringBuilder();
            builder.AppendLine($"Time: {DateTime.Now:O}");
            builder.AppendLine($"Message: {e.Message}");
            builder.AppendLine($"Exception: {e.Exception}");
            builder.AppendLine(new string('-', 80));
            File.AppendAllText(logPath, builder.ToString());
        }
        catch
        {
        }
    }
}