"""
Open-WebUI TTS 服务
基于小米 MIMO TTS API，提供 OpenAI 兼容的 TTS 接口
支持异步处理，适用于高并发场景
"""

import os
import base64
import io
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import uvicorn
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置
MIMO_API_KEY = os.getenv("MIMO_API_KEY", "sk-cfbsyiff06wjmifljcxdln82fsf3p17db54e68w8hbngbq11")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
MIMO_MODEL = os.getenv("MIMO_MODEL", "mimo-v2-tts")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "mimo_default")
DEFAULT_FORMAT = os.getenv("DEFAULT_FORMAT", "wav")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "4096"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print(f"启动 Open-WebUI TTS 服务...")
    print(f"MIMO API: {MIMO_BASE_URL}")
    print(f"服务地址: http://{HOST}:{PORT}")
    print(f"API 文档: http://{HOST}:{PORT}/docs")
    yield
    print("关闭 TTS 服务...")


# 创建 FastAPI 应用
app = FastAPI(
    title="Open-WebUI TTS Service",
    description="基于小米 MIMO 的 TTS 服务，兼容 OpenAI TTS API",
    version="1.1.0",
    lifespan=lifespan
)

# 初始化异步 MIMO 客户端
client = AsyncOpenAI(
    api_key=MIMO_API_KEY,
    base_url=MIMO_BASE_URL
)


# 请求模型 - 兼容 OpenAI TTS API 格式
class TTSRequest(BaseModel):
    model: Optional[str] = Field(default="tts-1", description="模型名称（兼容字段）")
    input: str = Field(..., description="要转换为语音的文本")
    voice: Optional[str] = Field(default=DEFAULT_VOICE, description="语音音色")
    response_format: Optional[str] = Field(default=DEFAULT_FORMAT, description="音频格式: wav, mp3")
    speed: Optional[float] = Field(default=1.0, description="语速（兼容字段）")


# 可用的语音列表
AVAILABLE_VOICES: List[str] = [
    "mimo_default",
    # 可以根据 MIMO API 支持的音色添加更多选项
]


def validate_voice(voice: str) -> str:
    """验证语音参数"""
    if voice and voice not in AVAILABLE_VOICES:
        # 暂时允许任何语音名称，MIMO API 会处理无效的语音
        pass
    return voice


@app.get("/")
async def root():
    """服务状态检查"""
    return {
        "service": "Open-WebUI TTS Service",
        "status": "running",
        "api_base": MIMO_BASE_URL,
        "model": MIMO_MODEL,
        "version": "1.1.0"
    }


@app.get("/v1/models")
async def list_models():
    """列出可用模型（兼容 OpenAI API）"""
    return {
        "object": "list",
        "data": [
            {
                "id": "mimo-v2-tts",
                "object": "model",
                "created": 1677610600,
                "owned_by": "mimo"
            }
        ]
    }


@app.get("/v1/audio/voices")
async def list_voices():
    """列出可用语音"""
    return {
        "voices": AVAILABLE_VOICES,
        "default": DEFAULT_VOICE
    }


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    生成语音 - 兼容 OpenAI TTS API
    
    请求示例:
    {
        "model": "tts-1",
        "input": "你好，世界！",
        "voice": "mimo_default",
        "response_format": "wav"
    }
    """
    try:
        # 验证输入
        if not request.input or len(request.input.strip()) == 0:
            raise HTTPException(status_code=400, detail="输入文本不能为空")
        
        if len(request.input) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=400, detail=f"输入文本过长，最大支持{MAX_TEXT_LENGTH}字符")
        
        # 验证语音参数
        voice = validate_voice(request.voice or DEFAULT_VOICE)
        
        # 验证格式参数
        audio_format = request.response_format or DEFAULT_FORMAT
        if audio_format not in ["wav", "mp3"]:
            raise HTTPException(status_code=400, detail="不支持的音频格式，仅支持 wav 和 mp3")
        
        # 构建 MIMO API 请求
        # 根据 tts_test.py 的格式，需要提供 messages
        messages = [
            {
                "role": "user",
                "content": "请朗读以下文本。"
            },
            {
                "role": "assistant", 
                "content": request.input
            }
        ]
        
        # 调用 MIMO TTS API (异步)
        completion = await client.chat.completions.create(
            model=MIMO_MODEL,
            messages=messages,
            audio={
                "format": audio_format,
                "voice": voice
            }
        )
        
        # 提取音频数据
        message = completion.choices[0].message
        if not hasattr(message, 'audio') or not message.audio:
            raise HTTPException(status_code=500, detail="MIMO API 未返回音频数据")
        
        # 解码 base64 音频数据
        audio_bytes = base64.b64decode(message.audio.data)
        
        # 确定内容类型
        content_type = "audio/wav"
        if audio_format == "mp3":
            content_type = "audio/mpeg"
        
        # 返回音频流
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{audio_format}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"TTS 生成错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS 生成失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 测试 MIMO API 连接
        test_completion = await client.chat.completions.create(
            model=MIMO_MODEL,
            messages=[
                {"role": "user", "content": "测试"},
                {"role": "assistant", "content": "测试"}
            ],
            audio={"format": "wav", "voice": DEFAULT_VOICE},
            max_tokens=10
        )
        return {
            "status": "healthy",
            "mimo_api": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "mimo_api": "disconnected",
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(
        "webui_tts:app",
        host=HOST,
        port=PORT,
        log_level="info",
        reload=True
    )