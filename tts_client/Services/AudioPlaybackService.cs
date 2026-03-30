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

    public async Task PlayAsync(string filePath, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("找不到要播放的 WAV 文件。", filePath);
        }

        StopAndDisposePlayback();

        var tcs = new TaskCompletionSource();
        var reader = new AudioFileReader(filePath);
        var device = new WaveOutEvent();

        _audioFileReader = reader;
        _outputDevice = device;

        try
        {
            device.Init(reader);

            device.PlaybackStopped += (_, e) =>
            {
                if (cancellationToken.IsCancellationRequested)
                    tcs.TrySetCanceled(cancellationToken);
                else if (e.Exception is not null)
                    tcs.TrySetException(e.Exception);
                else
                    tcs.TrySetResult();
            };

            using var registration = cancellationToken.Register(() => device.Stop());

            device.Play();
            await tcs.Task;
        }
        finally
        {
            device.Dispose();
            reader.Dispose();

            if (ReferenceEquals(_outputDevice, device))
            {
                _outputDevice = null;
            }

            if (ReferenceEquals(_audioFileReader, reader))
            {
                _audioFileReader = null;
            }
        }
    }

    public void Stop()
    {
        StopAndDisposePlayback();
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