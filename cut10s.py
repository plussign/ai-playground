from pathlib import Path
import re
import subprocess

from pypinyin import lazy_pinyin
import imageio_ffmpeg


def to_pinyin_name(text: str) -> str:
    pinyin = "".join(lazy_pinyin(text)).lower()
    pinyin = re.sub(r"[^a-z0-9_]+", "_", pinyin)
    pinyin = re.sub(r"_+", "_", pinyin).strip("_")
    return pinyin or "audio"


def get_unique_output_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    index = 2

    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def main() -> None:
    cv_dir = Path(__file__).resolve().parent / "cv"
    if not cv_dir.exists():
        raise FileNotFoundError(f"Directory not found: {cv_dir}")

    # Use bundled ffmpeg from imageio-ffmpeg so no system ffmpeg is required.
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    mp3_files = [p for p in cv_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp3"]
    if not mp3_files:
        print(f"No mp3 files found in: {cv_dir}")
        return

    mp3_files.sort(key=lambda p: p.name)

    success_count = 0
    for mp3_path in mp3_files:
        try:
            pinyin_name = to_pinyin_name(mp3_path.stem)
            output_path = get_unique_output_path(cv_dir / f"{pinyin_name}.wav")

            command = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(mp3_path),
                "-t",
                "10",
                "-acodec",
                "pcm_s16le",
                str(output_path),
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "ffmpeg failed")

            success_count += 1
            print(f"OK: {mp3_path.name} -> {output_path.name}")
        except Exception as exc:
            print(f"FAIL: {mp3_path.name} -> {exc}")

    print(f"Done. Success: {success_count}/{len(mp3_files)}")


if __name__ == "__main__":
    main()
