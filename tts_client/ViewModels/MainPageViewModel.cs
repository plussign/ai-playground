using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using Microsoft.Maui.Graphics;
using TtsClient.Models;
using TtsClient.Services;

namespace TtsClient.ViewModels;

public sealed class MainPageViewModel : INotifyPropertyChanged
{
    private static readonly Color DisabledButtonColor = Color.FromArgb("#A7B2A1");
    private static readonly Color EnabledPlayButtonColor = Color.FromArgb("#7B4C24");

    private readonly SettingsService _settingsService;
    private readonly OpenAiTtsService _ttsService;
    private readonly AudioPlaybackService _audioPlaybackService;

    private string _baseUrl = string.Empty;
    private string _apiKey = string.Empty;
    private string _modelName = string.Empty;
    private string _inputText = string.Empty;
    private string _statusMessage = "请输入配置和文本，然后开始推理。";
    private bool _isBusy;
    private bool _isPlaying;
    private int _playbackGeneration;
    private CancellationTokenSource? _playbackCts;

    public MainPageViewModel(SettingsService settingsService, OpenAiTtsService ttsService, AudioPlaybackService audioPlaybackService)
    {
        _settingsService = settingsService;
        _ttsService = ttsService;
        _audioPlaybackService = audioPlaybackService;

        InferCommand = new Command(async () => await RunInferenceAsync(), () => CanInfer);
        PlayCommand = new Command(async () => await PlayOrStopAsync(), () => CanPlay);
        PlaySegmentCommand = new Command<TextSegment>(async segment => await PlaySingleSegmentAsync(segment));
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    public ICommand InferCommand { get; }

    public ICommand PlayCommand { get; }

    public ICommand PlaySegmentCommand { get; }

    public ObservableCollection<TextSegment> Segments { get; } = [];

    public string BaseUrl
    {
        get => _baseUrl;
        set
        {
            if (SetProperty(ref _baseUrl, value))
            {
                PersistSettings();
                NotifyCommandStates();
            }
        }
    }

    public string ApiKey
    {
        get => _apiKey;
        set
        {
            if (SetProperty(ref _apiKey, value))
            {
                PersistSettings();
                NotifyCommandStates();
            }
        }
    }

    public string ModelName
    {
        get => _modelName;
        set
        {
            if (SetProperty(ref _modelName, value))
            {
                PersistSettings();
                NotifyCommandStates();
            }
        }
    }

    public string InputText
    {
        get => _inputText;
        set
        {
            if (SetProperty(ref _inputText, value))
            {
                NotifyCommandStates();
            }
        }
    }

    public string StatusMessage
    {
        get => _statusMessage;
        private set => SetProperty(ref _statusMessage, value);
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            if (SetProperty(ref _isBusy, value))
            {
                NotifyCommandStates();
            }
        }
    }

    public bool IsPlaying
    {
        get => _isPlaying;
        private set
        {
            if (SetProperty(ref _isPlaying, value))
            {
                OnPropertyChanged(nameof(PlayButtonText));
                NotifyCommandStates();
            }
        }
    }

    public bool CanInfer =>
        !IsBusy &&
        !string.IsNullOrWhiteSpace(BaseUrl) &&
        !string.IsNullOrWhiteSpace(ApiKey) &&
        !string.IsNullOrWhiteSpace(ModelName) &&
        !string.IsNullOrWhiteSpace(InputText);

    public bool CanPlay => IsPlaying || Segments.Count > 0;

    public string PlayButtonText => IsPlaying ? "停止朗读" : "开始朗读";

    public Color PlayButtonColor => CanPlay ? EnabledPlayButtonColor : DisabledButtonColor;

    public Task InitializeAsync()
    {
        var settings = _settingsService.Load();
        _baseUrl = settings.BaseUrl;
        _apiKey = settings.ApiKey;
        _modelName = settings.ModelName;

        OnPropertyChanged(nameof(BaseUrl));
        OnPropertyChanged(nameof(ApiKey));
        OnPropertyChanged(nameof(ModelName));
        NotifyCommandStates();

        if (!string.IsNullOrWhiteSpace(settings.BaseUrl) ||
            !string.IsNullOrWhiteSpace(settings.ApiKey) ||
            !string.IsNullOrWhiteSpace(settings.ModelName))
        {
            StatusMessage = $"已从 {_settingsService.GetSettingsFilePath()} 加载上次配置。";
        }

        return Task.CompletedTask;
    }

    public void PersistSettings()
    {
        _settingsService.Save(new AppSettings
        {
            BaseUrl = BaseUrl.Trim(),
            ApiKey = ApiKey.Trim(),
            ModelName = ModelName.Trim()
        });
    }

    private async Task RunInferenceAsync()
    {
        if (!CanInfer)
        {
            StatusMessage = "请先完整填写 Base URL、API Key、模型名和待合成文本。";
            return;
        }

        try
        {
            IsBusy = true;
            CancelPlayback();

            var texts = TextSegmenter.Split(InputText);
            Segments.Clear();

            foreach (var text in texts)
            {
                Segments.Add(new TextSegment { Text = text });
            }

            NotifyCommandStates();

            // Auto-start playback concurrently
            _ = PlayAllSegmentsAsync();

            for (var i = 0; i < Segments.Count; i++)
            {
                var segment = Segments[i];
                segment.Status = "转换中...";

                if (!IsPlaying)
                {
                    StatusMessage = $"正在转换第 {i + 1}/{Segments.Count} 段...";
                }

                try
                {
                    var filePath = await _ttsService.SynthesizeAsync(
                        BaseUrl, ApiKey, ModelName, segment.Text, CancellationToken.None);
                    segment.FilePath = filePath;
                    segment.Status = "✓ 已完成";
                }
                catch (Exception ex)
                {
                    segment.Status = "✗ 失败";

                    if (!IsPlaying)
                    {
                        StatusMessage = $"第 {i + 1} 段转换失败: {ex.Message}";
                    }
                }

                segment.MarkReady();
                NotifyCommandStates();
            }

            var completed = 0;
            foreach (var s in Segments)
            {
                if (!string.IsNullOrWhiteSpace(s.FilePath))
                {
                    completed++;
                }
            }

            if (!IsPlaying)
            {
                StatusMessage = $"推理完成，共 {Segments.Count} 段，成功 {completed} 段。";
            }
        }
        catch (Exception ex)
        {
            StatusMessage = ex.Message;
        }
        finally
        {
            // Mark any remaining segments as ready so playback won't hang
            foreach (var s in Segments)
            {
                s.MarkReady();
            }

            IsBusy = false;
            NotifyCommandStates();
        }
    }

    private async Task PlayOrStopAsync()
    {
        if (IsPlaying)
        {
            CancelPlayback();
            StatusMessage = "已停止朗读。";
            return;
        }

        await PlayAllSegmentsAsync();
    }

    private async Task PlayAllSegmentsAsync()
    {
        CancelPlayback();
        _playbackCts = new CancellationTokenSource();
        var ct = _playbackCts.Token;
        var generation = ++_playbackGeneration;

        IsPlaying = true;

        try
        {
            foreach (var segment in Segments)
            {
                if (ct.IsCancellationRequested)
                {
                    break;
                }

                // Wait for this segment's inference to finish (success or failure)
                try
                {
                    await segment.WaitUntilReadyAsync(ct);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                // Skip segments that failed or have no audio file
                if (string.IsNullOrWhiteSpace(segment.FilePath) || !File.Exists(segment.FilePath))
                {
                    continue;
                }

                segment.IsPlaying = true;
                StatusMessage = $"正在朗读: {TruncateText(segment.Text, 40)}";

                try
                {
                    await _audioPlaybackService.PlayAsync(segment.FilePath, ct);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    StatusMessage = $"播放失败: {ex.Message}";
                }
                finally
                {
                    segment.IsPlaying = false;
                }
            }

            if (!ct.IsCancellationRequested)
            {
                StatusMessage = "朗读完成。";
            }
        }
        finally
        {
            if (_playbackGeneration == generation)
            {
                IsPlaying = false;
                ClearPlayingStates();
            }
        }
    }

    private async Task PlaySingleSegmentAsync(TextSegment? segment)
    {
        if (segment?.FilePath is null || !File.Exists(segment.FilePath))
        {
            return;
        }

        CancelPlayback();
        _playbackCts = new CancellationTokenSource();
        var ct = _playbackCts.Token;
        var generation = ++_playbackGeneration;

        ClearPlayingStates();
        segment.IsPlaying = true;
        IsPlaying = true;

        try
        {
            StatusMessage = $"正在朗读: {TruncateText(segment.Text, 40)}";
            await _audioPlaybackService.PlayAsync(segment.FilePath, ct);

            if (!ct.IsCancellationRequested)
            {
                StatusMessage = "朗读完成。";
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            StatusMessage = $"播放失败: {ex.Message}";
        }
        finally
        {
            segment.IsPlaying = false;

            if (_playbackGeneration == generation)
            {
                IsPlaying = false;
            }
        }
    }

    private void CancelPlayback()
    {
        if (_playbackCts is not null)
        {
            _playbackCts.Cancel();
            _playbackCts.Dispose();
            _playbackCts = null;
        }

        _audioPlaybackService.Stop();
        ClearPlayingStates();
    }

    private void ClearPlayingStates()
    {
        foreach (var s in Segments)
        {
            s.IsPlaying = false;
        }
    }

    private static string TruncateText(string text, int maxLength)
    {
        return text.Length <= maxLength ? text : text[..maxLength] + "...";
    }

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

    private void NotifyCommandStates()
    {
        OnPropertyChanged(nameof(CanInfer));
        OnPropertyChanged(nameof(CanPlay));
        OnPropertyChanged(nameof(PlayButtonColor));

        if (InferCommand is Command inferCommand)
        {
            inferCommand.ChangeCanExecute();
        }

        if (PlayCommand is Command playCommand)
        {
            playCommand.ChangeCanExecute();
        }
    }

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}