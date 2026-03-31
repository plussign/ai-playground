# TTS Client 项目架构说明

## 概述

TTS Client 是一个基于 **.NET 9 MAUI** 的 Windows 桌面应用，用于调用兼容 OpenAI 格式的 TTS（文本转语音）接口，将长文本分段合成为 WAV 音频并逐段朗读。

- **框架**: .NET 9 + MAUI (WinUI 3)
- **音频库**: NAudio 2.2.1
- **架构模式**: MVVM（手动实现 `INotifyPropertyChanged`，无第三方 MVVM 框架）

---

## 目录结构

```
TtsClient/
├── App.xaml / App.xaml.cs            # 应用入口与全局资源
├── MauiProgram.cs                    # DI 容器配置
├── MainPage.xaml / MainPage.xaml.cs  # 主界面视图
├── Models/
│   ├── AppSettings.cs                # 配置数据模型
│   └── TextSegment.cs                # 文本分段模型
├── ViewModels/
│   └── MainPageViewModel.cs          # 主界面 ViewModel
├── Services/
│   ├── OpenAiTtsService.cs           # TTS API 调用服务
│   ├── AudioPlaybackService.cs       # WAV 音频播放服务
│   ├── SettingsService.cs            # 本地配置持久化服务
│   └── TextSegmenter.cs             # 文本分段工具
└── Platforms/
    └── Windows/
        └── App.xaml.cs               # WinUI 平台入口
```

---

## 模块说明

### 1. 应用入口与 DI 配置

#### `MauiProgram.cs`

应用的启动配置入口。使用 MAUI 的依赖注入容器注册所有服务和页面，均以 **Singleton** 生命周期注册：

| 注册类型 | 说明 |
|---|---|
| `SettingsService` | 配置读写 |
| `OpenAiTtsService` | TTS 接口调用 |
| `AudioPlaybackService` | 音频播放 |
| `MainPageViewModel` | 主页 ViewModel |
| `MainPage` | 主页视图 |

#### `App.xaml` / `App.xaml.cs`

- **App.xaml**: 定义全局颜色资源（`WindowBackgroundColor`、`AccentColor` 等），统一 UI 色彩风格。
- **App.xaml.cs**: 通过 DI 解析 `MainPage`，创建应用窗口，设置窗口标题为"TTS客户端"。

#### `Platforms\Windows\App.xaml.cs`

WinUI 平台入口，继承 `MauiWinUIApplication`。包含全局未处理异常捕获（`UnhandledException`），防止应用崩溃时无提示。

---

### 2. Models — 数据模型

#### `AppSettings.cs`

存储用户配置的简单 POCO 类：

| 属性 | 说明 |
|---|---|
| `BaseUrl` | TTS 服务的 Base URL |
| `ApiKey` | API 鉴权密钥 |
| `ModelName` | 使用的模型名称 |

#### `TextSegment.cs`

代表一个文本分段的模型，实现 `INotifyPropertyChanged` 以支持 UI 绑定。

**核心属性：**

| 属性 | 说明 |
|---|---|
| `Text` | 分段文本内容（只读） |
| `Status` | 当前状态：`等待中` → `转换中...` → `✓ 已完成` / `✗ 失败` |
| `FilePath` | 合成成功后的 WAV 文件路径 |
| `IsPlaying` | 当前是否正在朗读此段 |
| `BackgroundColor` | 朗读中为深绿色 `#2E5941`，否则透明 |
| `TextColor` | 朗读中为白色，否则黑色 |

**并发同步机制：**

- 内部持有 `TaskCompletionSource _readySignal`，用于推理与播放的并发协调。
- `MarkReady()`: 推理完成时调用，通知播放端该段已就绪。
- `WaitUntilReadyAsync(CancellationToken)`: 播放端调用，阻塞等待直到该段推理完成或取消。

---

### 3. ViewModels — 视图模型

#### `MainPageViewModel.cs`

应用的核心逻辑所在，负责整个推理 + 朗读流程的编排。

**绑定属性：**

| 属性 | 说明 |
|---|---|
| `BaseUrl` / `ApiKey` / `ModelName` | 配置输入，变更时自动持久化 |
| `InputText` | 待合成的原始文本 |
| `StatusMessage` | 底部状态栏文本 |
| `IsBusy` | 是否正在推理中 |
| `IsPlaying` | 是否正在朗读中 |
| `Segments` | 分段列表（`ObservableCollection<TextSegment>`） |
| `CanInfer` | 推理按钮可用条件：不忙且配置/文本非空 |
| `CanPlay` | 朗读按钮可用条件：正在播放 或 有分段存在 |
| `PlayButtonText` | 按钮文字，朗读中显示"停止朗读"，否则"开始朗读" |
| `PlayButtonColor` | 按钮颜色，可用时为棕色，不可用时为灰色 |

**命令：**

| 命令 | 触发方式 | 行为 |
|---|---|---|
| `InferCommand` | 点击"开始推理" | 分段文本、逐段调用 TTS、自动开始朗读 |
| `PlayCommand` | 点击"开始/停止朗读" | 切换：如在播放则停止，否则从头开始顺序朗读 |
| `PlaySegmentCommand` | 点击分段列表某行 | 停止当前朗读，播放该单段 |

**推理 + 朗读并发流程（`RunInferenceAsync`）：**

```
用户点击"开始推理"
  │
  ├─ 清空旧分段，文本按标点分段（≤100 字/段）
  ├─ 并发启动 PlayAllSegmentsAsync()    ← 朗读协程
  └─ 逐段调用 TTS 接口
       │
       ├─ 成功 → 设 FilePath + Status="✓ 已完成" → MarkReady()
       └─ 失败 → Status="✗ 失败" → MarkReady()
                                        │
PlayAllSegmentsAsync() 内部：            │
  foreach segment:                      │
    await segment.WaitUntilReadyAsync() ← 等待此段就绪
    if 失败 → 跳过
    if 成功 → 播放 WAV → 等播完 → 下一段
```

**播放代际计数器（`_playbackGeneration`）：**

用于解决异步竞态问题。每次启动新的播放会递增 `_playbackGeneration`，旧播放任务的 `finally` 块检查代际是否匹配，仅当前代才更新 `IsPlaying` 状态，避免旧的异步续体覆盖新状态。

---

### 4. Services — 服务层

#### `OpenAiTtsService.cs`

调用兼容 OpenAI 格式的 `/v1/audio/speech` 接口合成语音。

**核心方法：**
- `SynthesizeAsync(baseUrl, apiKey, modelName, inputText, cancellationToken)` → 返回生成的 WAV 文件路径。

**关键逻辑：**
- 自动规范化 Base URL（去尾部斜杠、去重复 `/v1`）。
- 请求参数：`model`、`input`、`voice="alloy"`、`response_format="wav"`。
- 响应处理兼容多种返回格式：
  - 直接返回 WAV 二进制流。
  - 返回 JSON 包裹 Base64 编码的 WAV（支持 `audio`、`data`、`audio_base64` 字段）。
- WAV 格式校验：检查 RIFF/WAVE 文件头魔数。
- 输出文件保存到 `AppContext.BaseDirectory`，文件名含时间戳。

#### `AudioPlaybackService.cs`

基于 **NAudio** 的音频播放服务，支持同步与异步播放。

| 方法 | 说明 |
|---|---|
| `Play(filePath)` | 同步播放（即发即忘） |
| `PlayAsync(filePath, cancellationToken)` | 异步播放，返回 `Task` 在播放完成时完成 |
| `Stop()` | 停止当前播放并释放资源 |

**`PlayAsync` 实现细节：**
- 使用 `TaskCompletionSource` 将 NAudio 的 `PlaybackStopped` 事件转换为可 `await` 的 `Task`。
- 通过 `CancellationToken.Register(() => device.Stop())` 支持取消。
- `finally` 块中使用 `ReferenceEquals` 检查，确保只释放自己创建的设备实例，避免与后续新播放任务的竞态冲突。

#### `SettingsService.cs`

将 `AppSettings` 以 JSON 格式持久化到本地文件。

| 方法 | 说明 |
|---|---|
| `Load()` | 从 `tts-client-settings.json` 加载配置，文件不存在或解析失败返回默认值 |
| `Save(settings)` | 序列化并写入 JSON 文件 |
| `GetSettingsFilePath()` | 返回配置文件完整路径，用于状态提示 |

文件位置：`{AppContext.BaseDirectory}/tts-client-settings.json`。

#### `TextSegmenter.cs`

纯静态工具类，将长文本按标点符号智能分段。

**分段规则：**
- 每段最多 100 个字符（可配置 `maxLength` 参数）。
- 在以下标点处优先断句：`，,。.！!？?；;：:、\n\r`。
- 从第 100 个字符向前搜索最近的标点作为断点。
- 若 100 字符内无标点，则强制在第 100 字符处截断。
- 自动去除首尾空白。

---

### 5. View — 视图层

#### `MainPage.xaml`

主界面布局，采用 5 行 `Grid`：

| 行 | 内容 | 高度 |
|---|---|---|
| Row 0 | 配置栏：Base URL、API Key、Model Name 三个输入框 | Auto |
| Row 1 | 文本输入区：多行 `Editor`（200px 高） | Auto |
| Row 2 | 分段列表：`CollectionView`，含表头（状态/文本）、每行可点击播放 | `*`（填满剩余空间） |
| Row 3 | 底部操作栏：状态文字 + "开始推理"按钮 + "开始朗读/停止朗读"按钮 | Auto |
| Row 4 | 加载指示器：`ActivityIndicator`（推理时旋转） | Auto |

**分段列表特性：**
- 使用编译绑定（`x:DataType`）绑定到 `TextSegment`。
- 每行背景色和文字色绑定到 `TextSegment.BackgroundColor` / `TextColor`，朗读时自动高亮为深绿底白字。
- 通过 `TapGestureRecognizer` + `x:Reference` 回调 ViewModel 的 `PlaySegmentCommand`。
- 空状态显示提示文本。

#### `MainPage.xaml.cs`

极简 code-behind，仅负责：
- 构造时通过 DI 接收 `MainPageViewModel` 并设为 `BindingContext`。
- `Loaded` 事件中调用 `viewModel.InitializeAsync()` 加载持久化配置（仅首次）。

---

## 数据流总结

```
用户输入文本 → [开始推理]
    │
    ▼
TextSegmenter.Split()      将文本按标点分成 ≤100 字的段落
    │
    ▼
ObservableCollection<TextSegment>    段落列表显示在 UI
    │
    ├──────────────────────────┐
    ▼ (并发)                   ▼ (并发)
推理循环                     朗读循环
  for each segment:            for each segment:
    OpenAiTtsService             WaitUntilReadyAsync() ← 等推理
      .SynthesizeAsync()         AudioPlaybackService
    segment.MarkReady() ───→       .PlayAsync()
                                 segment.IsPlaying ← UI 高亮
```
