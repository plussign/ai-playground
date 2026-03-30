using NAudio.Wave;

namespace TtsClient.Services;

public sealed class AudioPlaybackService : IDisposable
{
    private WaveOutEvent? _outputDevice;
    private AudioFileReader? _audioFileReader;

    public void Play(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("找不到要播放的 WAV 文件。", filePath);
        }

        StopAndDisposePlayback();

        try
        {
            _audioFileReader = new AudioFileReader(filePath);
            _outputDevice = new WaveOutEvent();
            _outputDevice.Init(_audioFileReader);
            _outputDevice.Play();
        }
        catch (Exception ex)
        {
            StopAndDisposePlayback();
            throw new InvalidOperationException($"音频播放失败: {ex.Message}", ex);
        }
    }

    public void Dispose()
    {
        StopAndDisposePlayback();
    }

    private void StopAndDisposePlayback()
    {
        if (_outputDevice is not null)
        {
            _outputDevice.Stop();
            _outputDevice.Dispose();
            _outputDevice = null;
        }

        _audioFileReader?.Dispose();
        _audioFileReader = null;
    }
}