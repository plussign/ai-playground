# Open-WebUI TTS 服务

基于小米 MIMO TTS API 的文本转语音服务，提供 OpenAI 兼容的 TTS 接口，可用于 Open-WebUI 集成。

## 功能特性

- ✅ 兼容 OpenAI TTS API 格式
- ✅ 支持异步处理，适用于高并发场景
- ✅ 支持 WAV 和 MP3 音频格式
- ✅ 提供健康检查端点
- ✅ 支持环境变量配置
- ✅ 自动生成 API 文档

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量（可选）

创建 `.env` 文件或设置环境变量：

```bash
# MIMO API 配置
MIMO_API_KEY=sk-your-api-key
MIMO_BASE_URL=https://api.xiaomimimo.com/v1
MIMO_MODEL=mimo-v2-tts

# 默认配置
DEFAULT_VOICE=mimo_default
DEFAULT_FORMAT=wav
MAX_TEXT_LENGTH=4096

# 服务配置
HOST=0.0.0.0
PORT=8000
```

### 3. 启动服务

```bash
python webui_tts.py
```

或者使用 uvicorn：

```bash
uvicorn webui_tts:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后，访问 http://localhost:8000/docs 查看 API 文档。

## API 端点

### 1. 生成语音 (POST /v1/audio/speech)

兼容 OpenAI TTS API 的语音生成端点。

**请求示例：**

```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "你好，世界！",
    "voice": "mimo_default",
    "response_format": "wav"
  }' \
  --output speech.wav
```

**请求参数：**

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| model | string | 否 | tts-1 | 模型名称（兼容字段） |
| input | string | 是 | - | 要转换为语音的文本 |
| voice | string | 否 | mimo_default | 语音音色 |
| response_format | string | 否 | wav | 音频格式：wav 或 mp3 |
| speed | float | 否 | 1.0 | 语速（兼容字段） |

**响应：**

返回音频流（WAV 或 MP3 格式）。

### 2. 列出模型 (GET /v1/models)

列出可用的 TTS 模型。

```bash
curl "http://localhost:8000/v1/models"
```

### 3. 列出语音 (GET /v1/audio/voices)

列出可用的语音音色。

```bash
curl "http://localhost:8000/v1/audio/voices"
```

### 4. 健康检查 (GET /health)

检查服务和 MIMO API 连接状态。

```bash
curl "http://localhost:8000/health"
```

### 5. 服务状态 (GET /)

获取服务基本信息。

```bash
curl "http://localhost:8000/"
```

## Open-WebUI 集成

### 配置步骤

1. 启动 TTS 服务，确保服务在 `http://your-host:8000` 运行

2. 在 Open-WebUI 中配置 TTS：
   - 进入设置 → 文本转语音
   - 选择 API 类型：OpenAI
   - API 端点：`http://your-host:8000/v1`
   - API Key：任意值（服务不验证）
   - 模型：tts-1
   - 语音：mimo_default

3. 保存配置并测试

### 使用 Python 客户端

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",  # 服务不验证
    base_url="http://localhost:8000/v1"
)

response = client.audio.speech.create(
    model="tts-1",
    input="你好，这是一段测试文本。",
    voice="mimo_default",
    response_format="wav"
)

response.stream_to_file("output.wav")
```

### 使用 curl

```bash
# 生成 WAV 音频
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input": "你好，世界！", "voice": "mimo_default"}' \
  --output hello.wav

# 生成 MP3 音频
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input": "你好，世界！", "response_format": "mp3"}' \
  --output hello.mp3
```

## 错误处理

服务返回标准 HTTP 错误码：

- `400 Bad Request`：请求参数错误（如空文本、文本过长、不支持的格式）
- `500 Internal Server Error`：MIMO API 调用失败或内部错误

错误响应格式：

```json
{
  "detail": "错误描述信息"
}
```

## 常见问题

### 1. MIMO API 连接失败

检查：
- API Key 是否正确
- 网络连接是否正常
- MIMO_BASE_URL 是否正确

### 2. 音频生成失败

检查：
- 文本长度是否超过限制
- 语音名称是否有效
- MIMO API 服务是否正常

### 3. 端口被占用

修改 `PORT` 环境变量或使用其他端口：

```bash
PORT=8001 python webui_tts.py
```

## 开发说明

### 项目结构

```
.
├── webui_tts.py           # 主服务文件
├── requirements.txt       # Python 依赖
├── WEBUI_TTS_README.md    # 使用说明
└── .env                   # 环境变量配置（可选）
```

### 添加新的语音音色

在 `AVAILABLE_VOICES` 列表中添加新音色：

```python
AVAILABLE_VOICES: List[str] = [
    "mimo_default",
    "voice_1",
    "voice_2",
]
```

### 扩展功能

可以通过修改 `webui_tts.py` 添加更多功能：
- 语音克隆支持
- 语速控制
- 情感控制
- 多语言支持

## 许可证

MIT License