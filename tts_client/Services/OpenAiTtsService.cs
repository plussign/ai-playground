using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace TtsClient.Services;

public sealed class OpenAiTtsService
{
    private readonly HttpClient _httpClient = new();

    public async Task<string> SynthesizeAsync(string baseUrl, string apiKey, string modelName, string inputText, CancellationToken cancellationToken)
    {
        var normalizedBaseUrl = NormalizeBaseUrl(baseUrl);
        var requestUri = new Uri($"{normalizedBaseUrl}/v1/audio/speech", UriKind.Absolute);

        using var request = new HttpRequestMessage(HttpMethod.Post, requestUri);
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("audio/wav"));

        var payload = new
        {
            model = modelName,
            input = inputText,
            voice = "alloy",
            response_format = "wav"
        };

        request.Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

        using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            var details = await ReadErrorDetailsAsync(response, cancellationToken);
            throw new InvalidOperationException($"TTS 请求失败: {(int)response.StatusCode} {response.ReasonPhrase} {details}".Trim());
        }

        var contentType = response.Content.Headers.ContentType?.MediaType;
        var rawBytes = await response.Content.ReadAsByteArrayAsync(cancellationToken);
        var wavBytes = NormalizeToWavBytes(rawBytes, contentType);

        var outputFilePath = Path.Combine(AppContext.BaseDirectory, $"tts-output-{DateTime.Now:yyyyMMdd-HHmmss}.wav");
        await File.WriteAllBytesAsync(outputFilePath, wavBytes, cancellationToken);
        return outputFilePath;
    }

    private static byte[] NormalizeToWavBytes(byte[] responseBytes, string? mediaType)
    {
        if (IsWav(responseBytes))
        {
            return responseBytes;
        }

        if (TryExtractWavFromJson(responseBytes, out var wavFromJson))
        {
            return wavFromJson;
        }

        var type = string.IsNullOrWhiteSpace(mediaType) ? "unknown" : mediaType;
        throw new InvalidOperationException($"接口未返回可播放 WAV 音频，响应类型: {type}。请检查模型是否支持 wav 或兼容接口实现是否正确。\n如果服务返回的是 mp3，请改为 wav 或扩展播放器支持。" );
    }

    private static bool TryExtractWavFromJson(byte[] responseBytes, out byte[] wavBytes)
    {
        wavBytes = Array.Empty<byte>();

        try
        {
            using var doc = JsonDocument.Parse(responseBytes);
            if (TryGetBase64Field(doc.RootElement, "audio", out var audioBase64) ||
                TryGetBase64Field(doc.RootElement, "data", out audioBase64) ||
                TryGetBase64Field(doc.RootElement, "audio_base64", out audioBase64))
            {
                var decoded = Convert.FromBase64String(audioBase64);
                if (IsWav(decoded))
                {
                    wavBytes = decoded;
                    return true;
                }
            }
        }
        catch
        {
        }

        return false;
    }

    private static bool TryGetBase64Field(JsonElement element, string propertyName, out string value)
    {
        value = string.Empty;

        if (element.ValueKind != JsonValueKind.Object)
        {
            return false;
        }

        if (!element.TryGetProperty(propertyName, out var property) || property.ValueKind != JsonValueKind.String)
        {
            return false;
        }

        var raw = property.GetString();
        if (string.IsNullOrWhiteSpace(raw))
        {
            return false;
        }

        value = raw;
        return true;
    }

    private static bool IsWav(byte[] bytes)
    {
        if (bytes.Length < 12)
        {
            return false;
        }

        return bytes[0] == (byte)'R' &&
               bytes[1] == (byte)'I' &&
               bytes[2] == (byte)'F' &&
               bytes[3] == (byte)'F' &&
               bytes[8] == (byte)'W' &&
               bytes[9] == (byte)'A' &&
               bytes[10] == (byte)'V' &&
               bytes[11] == (byte)'E';
    }

    private static string NormalizeBaseUrl(string baseUrl)
    {
        var trimmed = baseUrl.Trim().TrimEnd('/');

        if (trimmed.EndsWith("/v1", StringComparison.OrdinalIgnoreCase))
        {
            trimmed = trimmed[..^3];
        }

        return trimmed;
    }

    private static async Task<string> ReadErrorDetailsAsync(HttpResponseMessage response, CancellationToken cancellationToken)
    {
        try
        {
            var contentType = response.Content.Headers.ContentType?.MediaType;

            if (contentType is not null &&
                (contentType.StartsWith("text/", StringComparison.OrdinalIgnoreCase) ||
                 contentType.Contains("json", StringComparison.OrdinalIgnoreCase)))
            {
                var body = await response.Content.ReadAsStringAsync(cancellationToken);
                return string.IsNullOrWhiteSpace(body) ? string.Empty : $"- {body}";
            }
        }
        catch
        {
        }

        return string.Empty;
    }
}