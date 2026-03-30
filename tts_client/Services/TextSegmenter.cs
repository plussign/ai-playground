namespace TtsClient.Services;

public static class TextSegmenter
{
    private static readonly HashSet<char> SplitChars =
    [
        '，', ',', '。', '.', '！', '!', '？', '?',
        '；', ';', '：', ':', '、', '\n', '\r'
    ];

    public static List<string> Split(string text, int maxLength = 100)
    {
        var segments = new List<string>();

        if (string.IsNullOrWhiteSpace(text))
        {
            return segments;
        }

        var remaining = text.AsSpan();

        while (remaining.Length > 0)
        {
            remaining = remaining.TrimStart();

            if (remaining.Length == 0)
            {
                break;
            }

            if (remaining.Length <= maxLength)
            {
                var trimmed = remaining.Trim();
                if (trimmed.Length > 0)
                {
                    segments.Add(trimmed.ToString());
                }

                break;
            }

            var splitIndex = -1;
            for (var i = maxLength - 1; i >= 0; i--)
            {
                if (SplitChars.Contains(remaining[i]))
                {
                    splitIndex = i;
                    break;
                }
            }

            if (splitIndex < 0)
            {
                splitIndex = maxLength - 1;
            }

            var segment = remaining[..(splitIndex + 1)].Trim();
            if (segment.Length > 0)
            {
                segments.Add(segment.ToString());
            }

            remaining = remaining[(splitIndex + 1)..];
        }

        return segments;
    }
}
