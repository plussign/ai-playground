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
    private string? _lastOutputFilePath;
    private bool _isBusy;

    public MainPageViewModel(SettingsService settingsService, OpenAiTtsService ttsService, AudioPlaybackService audioPlaybackService)
    {
        _settingsService = settingsService;
        _ttsService = ttsService;
        _audioPlaybackService = audioPlaybackService;

        InferCommand = new Command(async () => await RunInferenceAsync(), () => CanInfer);
        PlayCommand = new Command(PlayAudio, () => CanPlay);
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    public ICommand InferCommand { get; }

    public ICommand PlayCommand { get; }

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

    public bool CanInfer =>
        !IsBusy &&
        !string.IsNullOrWhiteSpace(BaseUrl) &&
        !string.IsNullOrWhiteSpace(ApiKey) &&
        !string.IsNullOrWhiteSpace(ModelName) &&
        !string.IsNullOrWhiteSpace(InputText);

    public bool CanPlay => !IsBusy && !string.IsNullOrWhiteSpace(_lastOutputFilePath) && File.Exists(_lastOutputFilePath);

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
            StatusMessage = "正在请求 TTS 接口并等待 WAV 返回...";

            var outputFilePath = await _ttsService.SynthesizeAsync(BaseUrl, ApiKey, ModelName, InputText, CancellationToken.None);
            _lastOutputFilePath = outputFilePath;

            StatusMessage = $"推理完成，WAV 已保存到 {outputFilePath}";
        }
        catch (Exception ex)
        {
            _lastOutputFilePath = null;
            StatusMessage = ex.Message;
        }
        finally
        {
            IsBusy = false;
            NotifyCommandStates();
        }
    }

    private void PlayAudio()
    {
        if (!CanPlay || string.IsNullOrWhiteSpace(_lastOutputFilePath))
        {
            StatusMessage = "当前没有可播放的 WAV 文件，请先完成推理。";
            return;
        }

        try
        {
            _audioPlaybackService.Play(_lastOutputFilePath);
            StatusMessage = $"正在通过系统默认音频设备播放 {_lastOutputFilePath}";
        }
        catch (Exception ex)
        {
            StatusMessage = ex.Message;
        }
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