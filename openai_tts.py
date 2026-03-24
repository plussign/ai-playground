# coding=utf-8
import io
import os
import threading
import wave
from typing import List, Optional

import dashscope
from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Configure DashScope from environment variables.
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.base_websocket_api_url = os.getenv(
    "DASHSCOPE_WS_URL",
    "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
)

DEFAULT_MODEL = "cosyvoice-v3.5-plus"
DEFAULT_VOICE = "cosyvoice-v3.5-plus-theresa-cd03f9694851482282d20399a89c681b"

SAMPLE_RATE = 22050
CHANNELS = 1
SAMPLE_WIDTH = 2


class OpenAITTSRequest(BaseModel):
    model: Optional[str] = Field(default=None)
    input: str = Field(default="")
    voice: Optional[str] = Field(default=None)
    response_format: Optional[str] = Field(default="wav")
    speed: Optional[float] = Field(default=1.0)


class PCMCollectorCallback(ResultCallback):
    def __init__(self) -> None:
        self._chunks: List[bytes] = []
        self._done = threading.Event()
        self.error: Optional[str] = None

    def on_open(self):
        pass

    def on_complete(self):
        self._done.set()

    def on_error(self, message: str):
        self.error = message
        self._done.set()

    def on_close(self):
        self._done.set()

    def on_event(self, message):
        pass

    def on_data(self, data: bytes) -> None:
        if data:
            self._chunks.append(data)

    def wait(self, timeout: float = 60.0) -> bool:
        return self._done.wait(timeout=timeout)

    @property
    def pcm_bytes(self) -> bytes:
        if not self._chunks:
            return b""
        return b"".join(self._chunks)


def pcm_to_wav_bytes(pcm_data: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def empty_wav_bytes() -> bytes:
    return pcm_to_wav_bytes(b"")


def synthesize_to_wav(text: str, model_name: Optional[str], voice_name: Optional[str]) -> bytes:
    callback = PCMCollectorCallback()
    synthesizer = SpeechSynthesizer(
        model=model_name or DEFAULT_MODEL,
        voice=voice_name or DEFAULT_VOICE,
        format=AudioFormat.PCM_22050HZ_MONO_16BIT,
        callback=callback,
    )
    synthesizer.call(text)
    callback.wait(timeout=90.0)
    if callback.error is not None:
        return empty_wav_bytes()
    return pcm_to_wav_bytes(callback.pcm_bytes)


app = FastAPI(title="OpenAI Compatible TTS", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict:
    # Minimal model list for OpenAI-compatible clients.
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "dashscope",
            }
        ],
    }


@app.get("/v1/audio/models")
def list_audio_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "dashscope",
            }
        ],
    }


@app.get("/v1/audio/voices")
def list_audio_voices() -> dict:
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
    # Never raise HTTP errors for synthesis failures; always return WAV bytes.
    try:
        wav_data = synthesize_to_wav(req.input, req.model, req.voice)
    except Exception:
        wav_data = empty_wav_bytes()
    return Response(content=wav_data, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("openai_tts:app", host="0.0.0.0", port=8000, reload=False)
