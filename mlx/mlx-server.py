import argparse
import audioop
import io
import os
import tempfile
import threading
import uuid
import wave
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import Response
import lameenc
import uvicorn
from modelscope import snapshot_download

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model


MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
MODELSCOPE_CACHE_DIR = Path("/Users/zhenghao/.omlx/models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for OpenWebUI using local mlx_audio model"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8002,
        help="Port to bind",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default="/Users/zhenghao/.omlx/al.wav",
        help="Reference audio path for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default="哎呀，真是有趣的设计呢。偶尔尝试一下新风格也不错，你说呢？粉色妖精小姐。嗯。",
        help="Reference text aligned with reference audio",
    )
    parser.add_argument(
        "--lang-code",
        type=str,
        default="zh",
        help="Language code passed to generate_audio",
    )
    return parser.parse_args()


def build_silent_wav(duration_seconds: float = 0.2, sample_rate: int = 24000) -> bytes:
    frame_count = max(1, int(duration_seconds * sample_rate))
    silent_pcm = b"\x00\x00" * frame_count

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silent_pcm)
    return buffer.getvalue()


def encode_wav_bytes_to_mp3(wav_bytes: bytes, bitrate_kbps: int = 64) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        pcm_frames = wav_file.readframes(wav_file.getnframes())

    if not pcm_frames:
        return b""

    pcm_frames = bytes(pcm_frames) if isinstance(pcm_frames, bytearray) else pcm_frames

    if sample_width != 2:
        pcm_frames = audioop.lin2lin(pcm_frames, sample_width, 2)
        sample_width = 2
        pcm_frames = bytes(pcm_frames) if isinstance(pcm_frames, bytearray) else pcm_frames

    if channels == 2:
        pcm_frames = audioop.tomono(pcm_frames, sample_width, 0.5, 0.5)
        pcm_frames = bytes(pcm_frames) if isinstance(pcm_frames, bytearray) else pcm_frames
    elif channels != 1:
        raise ValueError(f"unsupported channel count for mp3 encoding: {channels}")

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate_kbps)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    mp3_data = encoder.encode(pcm_frames)
    mp3_data = bytes(mp3_data) if isinstance(mp3_data, bytearray) else mp3_data
    return mp3_data + encoder.flush()


def extract_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    text = payload.get("input")
    if text is None:
        text = payload.get("text")

    if isinstance(text, list):
        return " ".join(str(item) for item in text if item is not None).strip()
    if text is None:
        return ""
    return str(text).strip()


def find_generated_wav(work_dir: str, prefix: str) -> Path | None:
    candidates = sorted(
        Path(work_dir).glob(f"{prefix}*.wav"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_local_model_path() -> Path | None:
    MODELSCOPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[mlx-server] downloading or reusing model: {MODEL_ID}")
        downloaded_path = snapshot_download(
            model_id=MODEL_ID,
            cache_dir=str(MODELSCOPE_CACHE_DIR),
        )
        local_path = Path(downloaded_path).expanduser().resolve()
        if local_path.exists() and local_path.is_dir():
            print(f"[mlx-server] model available at: {local_path}")
            return local_path
        print("[mlx-server] ModelScope returned an invalid model path")
    except Exception as exc:
        print(f"[mlx-server] failed to download model from ModelScope: {exc}")

    return None


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="mlx-audio OpenAI TTS Compatible Server")
    silent_mp3 = encode_wav_bytes_to_mp3(build_silent_wav())

    model_path = resolve_local_model_path()
    model: Any = None
    generation_lock = threading.Lock()

    if model_path is not None and model_path.exists() and model_path.is_dir():
        try:
            print(f"[mlx-server] loading model from local path: {model_path}")
            model = load_model(model_path)
            print("[mlx-server] model loaded successfully")
        except Exception as exc:
            print(f"[mlx-server] failed to load model: {exc}")
            model = None
    else:
        print("[mlx-server] no local model path available")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/audio/speech")
    async def openai_tts_compatible(request: Request) -> Response:
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        text = extract_text(payload)
        if not text or model is None:
            return Response(content=silent_mp3, media_type="audio/mpeg", status_code=200)

        try:
            with generation_lock:
                with tempfile.TemporaryDirectory(prefix="mlx_tts_") as work_dir:
                    file_prefix = f"tts_{uuid.uuid4().hex}"
                    output_prefix = os.path.join(work_dir, file_prefix)

                    generate_audio(
                        model=model,
                        text=text,
                        ref_audio=args.ref_audio,
                        ref_text=args.ref_text,
                        file_prefix=output_prefix,
                        lang_code=args.lang_code,
                        voice=None,
                    )

                    wav_path = find_generated_wav(work_dir, file_prefix)
                    if wav_path is None or not wav_path.exists():
                        return Response(content=silent_mp3, media_type="audio/mpeg", status_code=200)

                    wav_bytes = wav_path.read_bytes()
                    if not wav_bytes:
                        return Response(content=silent_mp3, media_type="audio/mpeg", status_code=200)

                    mp3_bytes = encode_wav_bytes_to_mp3(wav_bytes, bitrate_kbps=64)
                    if not mp3_bytes:
                        return Response(content=silent_mp3, media_type="audio/mpeg", status_code=200)

                    wav_path.unlink(missing_ok=True)
                    print(f"[mlx-server] sending {len(mp3_bytes)} bytes of mp3.")
                    return Response(content=mp3_bytes, media_type="audio/mpeg", status_code=200)
        except Exception:
            return Response(content=silent_mp3, media_type="audio/mpeg", status_code=200)

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
