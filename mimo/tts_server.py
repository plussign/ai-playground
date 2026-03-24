# coding=utf-8
import base64
import io
import os
import wave
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import Response
from openai import OpenAI
from pydantic import BaseModel, Field

# MiMo TTS API 配置
MIMO_API_KEY = os.getenv("MIMO_API_KEY", "sk-cfbsyiff06wjmifljcxdln82fsf3p17db54e68w8hbngbq11")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
DEFAULT_MODEL = "mimo-v2-tts"
DEFAULT_VOICE = "mimo_default"

SAMPLE_RATE = 22050
CHANNELS = 1
SAMPLE_WIDTH = 2


class OpenAITTSRequest(BaseModel):
    model: Optional[str] = Field(default=None)
    input: str = Field(default="")
    voice: Optional[str] = Field(default=None)
    response_format: Optional[str] = Field(default="wav")
    speed: Optional[float] = Field(default=1.0)


def empty_wav_bytes() -> bytes:
    """生成空的 WAV 音频数据"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"")
    return buf.getvalue()


def synthesize_to_wav(text: str, model_name: Optional[str], voice_name: Optional[str]) -> bytes:
    """
    调用 MiMo TTS API 生成音频
    
    Args:
        text: 要合成的文本
        model_name: 模型名称
        voice_name: 语音名称
    
    Returns:
        WAV 格式的音频数据，失败时返回空音频
    """
    try:
        client = OpenAI(
            api_key=MIMO_API_KEY,
            base_url=MIMO_BASE_URL
        )
        
        completion = client.chat.completions.create(
            model=model_name or DEFAULT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "请朗读以下内容。"
                },
                {
                    "role": "assistant",
                    "content": text
                }
            ],
            audio={
                "format": "wav",
                "voice": voice_name or DEFAULT_VOICE
            }
        )
        
        message = completion.choices[0].message
        
        # 检查是否有音频数据
        if not message.audio or not message.audio.data:
            return empty_wav_bytes()
        
        # 解码 base64 音频数据
        audio_bytes = base64.b64decode(message.audio.data)
        return audio_bytes
        
    except Exception as e:
        # 发生任何异常都返回空音频，不抛出 HTTP 错误
        print(f"TTS synthesis error: {e}")
        return empty_wav_bytes()


app = FastAPI(title="MiMo TTS Service", version="1.0.0")


@app.get("/health")
def health() -> dict:
    """健康检查接口"""
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict:
    """返回可用模型列表"""
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "xiaomi-mimo",
            }
        ],
    }


@app.get("/v1/audio/models")
def list_audio_models() -> dict:
    """返回音频模型列表"""
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "xiaomi-mimo",
            }
        ],
    }


@app.get("/v1/audio/voices")
def list_audio_voices() -> dict:
    """返回可用语音列表"""
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_VOICE,
                "object": "voice",
                "model": DEFAULT_MODEL,
            }
        ],
    }


@app.post("/v1/audio/speech")
def create_speech(req: OpenAITTSRequest) -> Response:
    """
    创建语音合成请求
    
    注意：当模型服务返回失败时，不会返回 HTTP 错误码，
    而是返回空音频，以便继续后续内容的生成。
    """
    wav_data = synthesize_to_wav(req.input, req.model, req.voice)
    return Response(content=wav_data, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("tts_server:app", host="0.0.0.0", port=8000, reload=False)