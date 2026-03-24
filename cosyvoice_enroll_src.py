import csv
import os
import re
import time
from pathlib import Path

import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, VoiceEnrollmentService


dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope.api_key:
    raise ValueError("DASHSCOPE_API_KEY environment variable not set.")

dashscope.base_websocket_api_url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

TARGET_MODEL = "cosyvoice-v3.5-plus"
BASE_AUDIO_URL = "https://tts-sample.oss-cn-beijing.aliyuncs.com/voice_src"
VOICE_SRC_DIR = Path("voice_src")
VOICE_DEMO_DIR = Path("voice_demo")
SUMMARY_CSV_PATH = Path("voice_demo/cosyvoice_voice_id_summary.csv")
TEXT_TO_SYNTHESIZE = "升级了！以后可以更好地建设浮光岛了！"

MAX_ATTEMPTS = int(os.getenv("COSYVOICE_POLL_MAX_ATTEMPTS", "30"))
POLL_INTERVAL = int(os.getenv("COSYVOICE_POLL_INTERVAL", "10"))


def build_voice_prefix(wav_name: str, used_prefixes: set[str]) -> str:
    stem = Path(wav_name).stem.lower()
    if stem.endswith("yuyin"):
        stem = stem[: -len("yuyin")]

    letters_only = "".join(ch for ch in stem if "a" <= ch <= "z")
    base = letters_only[:10]
    if not base:
        base = "voice"

    # Keep prefixes unique to avoid collision when similar names map to same first 10 letters.
    candidate = base
    suffix_num = 1
    while candidate in used_prefixes:
        suffix = str(suffix_num)
        candidate = f"{base[: 10 - len(suffix)]}{suffix}"
        suffix_num += 1

    used_prefixes.add(candidate)
    return candidate


def poll_voice_ready(service: VoiceEnrollmentService, voice_id: str) -> str:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        voice_info = service.query_voice(voice_id=voice_id)
        status = voice_info.get("status")
        print(f"  Poll {attempt}/{MAX_ATTEMPTS}: status={status}")

        if status == "OK":
            return status
        if status == "UNDEPLOYED":
            return status

        time.sleep(POLL_INTERVAL)

    return "TIMEOUT"


def main() -> None:
    if not VOICE_SRC_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {VOICE_SRC_DIR}")

    VOICE_DEMO_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(p.name for p in VOICE_SRC_DIR.glob("*.wav"))
    if not wav_files:
        raise RuntimeError("No .wav files found under voice_src")

    print(f"Found {len(wav_files)} wav files in {VOICE_SRC_DIR}")

    service = VoiceEnrollmentService()
    used_prefixes: set[str] = set()
    summary_rows: list[dict[str, str]] = []

    for index, wav_name in enumerate(wav_files, start=1):
        voice_prefix = build_voice_prefix(wav_name, used_prefixes)
        audio_url = f"{BASE_AUDIO_URL}/{wav_name}"

        print("\n" + "=" * 72)
        print(f"[{index}/{len(wav_files)}] WAV_NAME={wav_name}")
        print(f"VOICE_PREFIX={voice_prefix}")
        print(f"AUDIO_URL={audio_url}")

        row = {
            "WAV_NAME": wav_name,
            "VOICE_PREFIX": voice_prefix,
            "Generated Voice ID": "",
            "Final Status": "",
            "Demo File": "",
            "Error": "",
        }

        try:
            voice_id = service.create_voice(
                target_model=TARGET_MODEL,
                prefix=voice_prefix,
                url=audio_url,
            )
            create_req_id = service.get_last_request_id()
            print(f"Voice enrollment submitted. Request ID: {create_req_id}")
            print(f"Generated Voice ID: {voice_id}")

            row["Generated Voice ID"] = voice_id

            final_status = poll_voice_ready(service, voice_id)
            row["Final Status"] = final_status

            if final_status != "OK":
                msg = f"Voice is not ready, status={final_status}"
                print(msg)
                row["Error"] = msg
                summary_rows.append(row)
                continue

            synthesizer = SpeechSynthesizer(model=TARGET_MODEL, voice=voice_id)
            audio_data = synthesizer.call(TEXT_TO_SYNTHESIZE)
            synth_req_id = synthesizer.get_last_request_id()
            print(f"Synthesis successful. Request ID: {synth_req_id}")

            demo_file = VOICE_DEMO_DIR / f"{voice_prefix}_demo.mp3"
            with open(demo_file, "wb") as f:
                f.write(audio_data)

            print(f"Saved demo file: {demo_file}")
            row["Demo File"] = str(demo_file)

        except Exception as exc:
            print(f"Error processing {wav_name}: {exc}")
            row["Error"] = str(exc)

        summary_rows.append(row)

    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "WAV_NAME",
                "VOICE_PREFIX",
                "Generated Voice ID",
                "Final Status",
                "Demo File",
                "Error",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 72)
    print(f"Summary CSV saved to: {SUMMARY_CSV_PATH}")
    print(f"Processed {len(summary_rows)} files")


if __name__ == "__main__":
    main()
