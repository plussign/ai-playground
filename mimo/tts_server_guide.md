# MiMo TTS Server 说明文档

`tts_server.py` 是一个基于 FastAPI 实现的文本转语音（TTS）服务接口，它通过调用 MiMo TTS API 将文本转换为 WAV 格式的音频流。

## 核心功能

该服务器充当一个代理，将兼容 OpenAI 格式的 TTS 请求转发给 MiMo TTS 后端服务。其主要特点包括：

- **语音合成**：支持将输入文本转换为高质量的 WAV 音频。
- **兼容性接口**：提供了类似于 OpenAI TTS 的 API 路径（如 `/v1/audio/speech`）。
- **容错处理**：当合成过程中出现异常或 API 返回失败时，服务器会返回一段“空音频”数据而非直接抛出 HTTP 错误，以确保调用方流程的连续性。
- **基础管理接口**：提供健康检查及模型/语音列表查询接口。

## 接口定义

### 1. 语音合成 (Speech Synthesis)
- **路径**: `POST /v1/audio/speech`
- **请求体**: `OpenAITTSRequest`
  - `input` (string, 必填): 要合成的文本。
  - `model` (string, 可选): 模型名称，默认为 `mimo-v2-tts`。
  - `voice` (string, 可选): 语音名称，默认为 `mimo_default`。
  - `response_format` (string, 可选): 响应格式，默认 `wav`。
  - `speed` (float, 可选): 语速，默认 `1.0`。
- **响应**: 返回 `audio/wav` 格式的二进制音频流。

### 2. 信息查询接口
- `GET /health`: 健康检查，返回 `{"status": "ok"}`。
- `GET /v1/models`: 获取可用模型列表。
- `GET /v1/audio/models`: 获取可用音频模型列表。
- `GET /v1/audio/voices`: 获取可用语音列表。

## 配置与运行

### 环境变量
服务器支持通过以下环境变量进行配置：
- `MIMO_API_KEY`: MiMo API 密钥（若未设置，则使用代码中的默认值）。
- `MIMO_BASE_URL`: MiMo API 基础地址（默认为 `https://api.xiaomimimo.com/v1`）。

### 启动命令
可以使用 Python 直接运行该脚本：
```bash
python tts_server.py
```
启动后，服务将运行在 `http://0.0.0.0:8000`。

## 代码逻辑要点
- **音频生成**: 使用 `OpenAI` 客户端发送聊天请求，通过 `audio` 参数指定合成格式和语音。
- **Base64 解码**: API 返回的音频数据经过 Base64 编码，服务器在返回前会将其解码为原始字节流。
- **空音频兜底**: 定义了 `empty_wav_bytes()` 函数，在出错时生成一个符合 WAV 格式的标准空文件，防止客户端播放器崩溃。