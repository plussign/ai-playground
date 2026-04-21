# Qwen3-ASR-Test.py

基于 Qwen3-ASR 的语音识别工具，支持音频/视频文件转录为 SRT 字幕或 LRC 歌词格式。

## 功能特性

- **多格式支持**：支持 `.mp4`、`.mkv`、`.avi`、`.wav`、`.mp3` 等音视频格式
- **长音频处理**：使用 Silero VAD 进行语音活动检测，自动将长音频分段处理（默认 3 分钟）
- **多语言支持**：中文 (ch)、英文 (en)、日语 (jp)
- **多设备支持**：CUDA、XPU、CPU 自动检测
- **并发转录**：支持多线程并发处理多个音频片段
- **音频预处理**：支持高通滤波（去除低频噪声）和自动增益控制（AGC）
- **智能分句**：基于标点符号和时间间隙自动断句
- **输出格式**：
  - 视频文件 → SRT 字幕格式
  - 音频文件 → LRC 歌词格式

## 依赖安装

```bash
# 安装 Qwen-ASR
pip install -U qwen-asr pydub torch scipy

# 通过 ModelScope 下载模型（推荐国内用户）
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./Qwen/Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./Qwen/Qwen3-ForcedAligner-0.6B
```

## 使用方法

```bash
python Qwen3-ASR-Test.py [选项] <输入文件>

# 基本选项
-d, --device       设备选择：cuda / xpu / cpu（默认自动检测）
-l, --language     语言：ch（中文）/ en（英文）/ jp（日语）
-c, --concurrency  并发转录任务数（默认 2）
-m, --max_chunk    VAD 分段的最大时长（秒，默认 10）
-v, --vad-thres    VAD 语音检测阈值（默认 0.3）

# 音频预处理选项（默认启用）
--no-preprocess    禁用音频预处理（高通滤波 + AGC）
--highpass 80      高通滤波器截止频率（Hz，默认 80，设为 0 禁用）
--no-agc           禁用自动增益控制（AGC）

# 调试选项
--debug            调试模式：保留临时文件并生成 chunk 级别的 LRC

# 示例
# 转录中文音频
python Qwen3-ASR-Test.py -l ch input.wav

# 使用 GPU 转录英文视频
python Qwen3-ASR-Test.py -d cuda -l en video.mp4

# 使用 8 个并发任务
python Qwen3-ASR-Test.py -c 8 -l ch long_audio.wav

# 禁用音频预处理
python Qwen3-ASR-Test.py --no-preprocess -l ch clean_audio.wav

# 自定义高通截止频率为 100Hz
python Qwen3-ASR-Test.py --highpass 100 input.wav

# 只使用高通滤波，不使用 AGC
python Qwen3-ASR-Test.py --no-agc input.wav
```

## 工作流程

1. **视频处理**：若是视频文件，自动提取音频
2. **音频预处理（可选）**：
   - 高通滤波：去除低频背景噪声（默认 80Hz）
   - 自动增益控制：标准化音频音量
3. **VAD 分段**：长音频（>3 分钟）自动使用 Silero VAD 分段
4. **ASR 转录**：调用 Qwen3-ASR 模型进行语音识别，返回带时间戳的文本
5. **文本对齐**：将完整文本与时间戳对齐，处理标点符号
6. **分句生成**：基于标点符号和时间间隙智能断句
7. **格式输出**：生成 SRT 字幕或 LRC 歌词文件

## 核心函数

| 函数 | 功能 |
|------|------|
| `preprocess_audio()` | 音频预处理：高通滤波 + 自动增益控制 |
| `apply_highpass_filter()` | 应用高通滤波器去除低频噪声 |
| `split_audio_with_vad()` | 使用 Silero VAD 分段长音频 |
| `transcribe_chunks()` | 并发转录多个音频片段 |
| `build_sentence_segments()` | 智能分句，构建时间轴 |
| `write_srt()` / `write_lrc()` | 输出字幕/歌词文件 |
| `extend_segment_end_times()` | 调整字幕时间轴，使播放更流畅 |

## 音频预处理说明

音频预处理默认启用（高通滤波 + AGC）。当音频质量较好不需要预处理时，可以使用 `--no-preprocess` 参数禁用：

- **高通滤波**：移除 80Hz 以下的低频噪声（如空调声、电流声等），可通过 `--highpass` 调整截止频率
- **自动增益控制**：自动将音频音量标准化到合适水平，避免音量忽大忽小

**VAD 参数说明**：
- `-v, --vad-thres` 默认值为 0.3，值越大越容易切分（更容易检测到静音），值越小越不容易切分

## 输出示例

**SRT 格式**：
```
1
00:00:00,000 --> 00:00:03,500
你好，欢迎使用语音识别

2
00:00:04,000 --> 00:00:07,200
这是一段测试音频
```

**LRC 格式**：
```
[00:00.00]你好，欢迎使用语音识别
[00:04.00]这是一段测试音频
```