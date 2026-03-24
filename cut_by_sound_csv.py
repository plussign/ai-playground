from __future__ import annotations

import csv
import wave
from pathlib import Path


def load_cut_seconds(csv_path: Path) -> dict[str, float]:
    mapping: dict[str, float] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "WAV_NAME" not in reader.fieldnames or "LEN" not in reader.fieldnames:
            raise ValueError("CSV must contain WAV_NAME and LEN columns")

        for row in reader:
            wav_name = (row.get("WAV_NAME") or "").strip()
            len_text = (row.get("LEN") or "").strip()
            if not wav_name or not len_text:
                continue

            try:
                seconds = float(len_text)
            except ValueError:
                print(f"SKIP invalid LEN: {wav_name} -> {len_text}")
                continue

            if seconds <= 0:
                print(f"SKIP non-positive LEN: {wav_name} -> {seconds}")
                continue

            mapping[wav_name] = seconds

    return mapping


def cut_wav(input_wav: Path, output_wav: Path, max_seconds: float) -> None:
    with wave.open(str(input_wav), "rb") as src:
        params = src.getparams()
        frame_rate = src.getframerate()
        total_frames = src.getnframes()

        target_frames = min(int(max_seconds * frame_rate), total_frames)
        frames = src.readframes(target_frames)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir / "voice_src"
    out_dir = base_dir / "voice_src_cut"
    csv_path = src_dir / "sound_cut.csv"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    cut_map = load_cut_seconds(csv_path)
    if not cut_map:
        print("No valid rows found in CSV.")
        return

    wav_files = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if not wav_files:
        print("No wav files found in voice_src.")
        return

    ok = 0
    skipped = 0

    for wav_path in wav_files:
        seconds = cut_map.get(wav_path.name)
        if seconds is None:
            skipped += 1
            continue

        out_path = out_dir / wav_path.name
        try:
            cut_wav(wav_path, out_path, seconds)
            ok += 1
            print(f"OK: {wav_path.name} -> {seconds}s")
        except wave.Error as exc:
            print(f"FAIL wav parse: {wav_path.name} -> {exc}")
        except Exception as exc:
            print(f"FAIL: {wav_path.name} -> {exc}")

    print(f"Done. Cut: {ok}, Skipped(no CSV match): {skipped}")


if __name__ == "__main__":
    main()
