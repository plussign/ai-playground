namespace TtsClient.Models;

public sealed class AppSettings
{
    public string BaseUrl { get; set; } = string.Empty;

    public string ApiKey { get; set; } = string.Empty;

    public string ModelName { get; set; } = string.Empty;
}