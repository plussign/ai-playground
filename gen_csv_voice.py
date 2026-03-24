# coding=utf-8
import csv
import os
from pathlib import Path

import dashscope
from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer


CSV_PATH = Path("voice_id.csv")
OUTPUT_DIR = Path("voice_gen")
MODEL = "cosyvoice-v3.5-plus"
TEXT_COLUMNS = [("文本1", 1), ("文本2", 2)]


def synthesize_one(model: str, voice_id: str, text: str, output_path: Path) -> str:
    synthesizer = SpeechSynthesizer(
        model=model,
        voice=voice_id,
        format=AudioFormat.WAV_22050HZ_MONO_16BIT,
        language_hints=["zh"],
    )
    audio_data = synthesizer.call(text)
    with output_path.open("wb") as f:
        f.write(audio_data)
    return synthesizer.get_last_request_id()


def main() -> None:
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope.api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable not set.")

    dashscope.base_websocket_api_url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            role_id = (row.get("role_id") or "").strip()
            voice_id = (row.get("Voice ID") or "").strip()

            if not role_id or not voice_id:
                print(f"[SKIP] row {row_idx}: missing role_id or Voice ID")
                skipped += 1
                continue

            for text_col, n in TEXT_COLUMNS:
                text = (row.get(text_col) or "").strip()           
                if not text:
                    print(f"[SKIP] row {row_idx}: {text_col} is empty")
                    skipped += 1
                    continue
                else:
                    text = f"<speak><break time=\"600ms\"/></speak>{text}"

                output_path = OUTPUT_DIR / f"{role_id}_{n}.wav"
                try:
                    request_id = synthesize_one(MODEL, voice_id, text, output_path)
                    generated += 1
                    print(
                        f"[OK] row {row_idx} {text_col} -> {output_path.as_posix()} | requestId={request_id}"
                    )
                except Exception as e:
                    skipped += 1
                    print(f"[ERR] row {row_idx} {text_col}: {e}")

    print(f"Done. generated={generated}, skipped={skipped}")


if __name__ == "__main__":
    main()
