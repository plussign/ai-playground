using System.ComponentModel;
using System.Runtime.CompilerServices;
using Microsoft.Maui.Graphics;

namespace TtsClient.Models;

public sealed class TextSegment : INotifyPropertyChanged
{
    private readonly TaskCompletionSource _readySignal = new(TaskCreationOptions.RunContinuationsAsynchronously);

    private string _status = "等待中";
    private string? _filePath;
    private bool _isPlaying;

    public required string Text { get; init; }

    public Task WaitUntilReadyAsync(CancellationToken cancellationToken)
    {
        return _readySignal.Task.WaitAsync(cancellationToken);
    }

    public void MarkReady()
    {
        _readySignal.TrySetResult();
    }

    public string Status
    {
        get => _status;
        set => SetProperty(ref _status, value);
    }

    public string? FilePath
    {
        get => _filePath;
        set => SetProperty(ref _filePath, value);
    }

    public bool IsPlaying
    {
        get => _isPlaying;
        set
        {
            if (SetProperty(ref _isPlaying, value))
            {
                OnPropertyChanged(nameof(BackgroundColor));
                OnPropertyChanged(nameof(TextColor));
            }
        }
    }

    public Color BackgroundColor => IsPlaying ? Color.FromArgb("#2E5941") : Colors.Transparent;

    public Color TextColor => IsPlaying ? Colors.White : Color.FromArgb("#000000");

    public event PropertyChangedEventHandler? PropertyChanged;

    private bool SetProperty<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }

        field = value;
        OnPropertyChanged(propertyName);
        return true;
    }

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
