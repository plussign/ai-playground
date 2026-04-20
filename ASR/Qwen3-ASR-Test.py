import torch
import time
import argparse
import ffmpeg
from pathlib import Path
from qwen_asr import Qwen3ASRModel

#https://github.com/QwenLM/Qwen3-ASR

# Download through ModelScope (recommended for users in Mainland China)
# pip install -U modelscope
# modelscope download --model Qwen/Qwen3-ASR-1.7B  --local_dir ./Qwen3-ASR-1.7B
# modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./Qwen3-ASR-0.6B
# modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./Qwen3-ForcedAligner-0.6B

# pip install -U qwen-asr


def format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _strip_trailing_punctuation(text: str) -> str:
    """Remove trailing comma, period, or semicolon from text."""
    trailing_punctuation = set("，。；,.;")
    while text and text[-1] in trailing_punctuation:
        text = text[:-1].rstrip()
    return text


def build_sentence_segments(full_text, time_stamps, comma_split_threshold=20, time_gap_threshold=3.0, space_gap_threshold=0.3):
    # Strong endings always split; commas split only when the segment is long enough.
    # Also split when time gap between characters exceeds threshold.
    # Add space between characters in same segment with moderate time gap.
    strong_sentence_endings = set("。！？；.!?;")
    comma_endings = set("，,")
    punctuation_chars = set("，。！？；：、,.!?;:()（）【】[]《》<>\"'""''—…")

    segments = []
    ts_index = 0
    current_text = []
    current_start = None
    current_end = None
    current_char_count = 0
    prev_ts_end = None
    prev_was_content_char = False

    for ch in full_text:
        if ch.isspace() or ch in punctuation_chars:
            current_text.append(ch)
            prev_was_content_char = False
            should_split = False
            if ch in strong_sentence_endings:
                should_split = True
            elif ch in comma_endings and current_char_count >= comma_split_threshold:
                should_split = True

            if should_split and current_start is not None and current_end is not None:
                text = "".join(current_text).strip()
                text = _strip_trailing_punctuation(text)
                if text:
                    segments.append((current_start, current_end, text))
                current_text = []
                current_start = None
                current_end = None
                current_char_count = 0
                prev_ts_end = None
                prev_was_content_char = False
            continue

        if ts_index >= len(time_stamps):
            current_text.append(ch)
            continue

        ts = time_stamps[ts_index]
        ts_index += 1

        # Check time gap from previous character
        if prev_ts_end is not None and current_start is not None:
            time_gap = ts.start_time - prev_ts_end
            if time_gap >= time_gap_threshold:
                # Split due to large time gap
                text = "".join(current_text).strip()
                text = _strip_trailing_punctuation(text)
                if text:
                    segments.append((current_start, current_end, text))
                current_text = []
                current_start = None
                current_end = None
                current_char_count = 0
                prev_was_content_char = False
            elif prev_was_content_char and time_gap >= space_gap_threshold:
                # Add space for moderate gap between content characters in same segment
                current_text.append(" ")

        current_text.append(ch)
        if current_start is None:
            current_start = ts.start_time
        current_end = ts.end_time
        current_char_count += 1
        prev_ts_end = ts.end_time
        prev_was_content_char = True

    if current_start is not None and current_end is not None:
        text = "".join(current_text).strip()
        text = _strip_trailing_punctuation(text)
        if text:
            segments.append((current_start, current_end, text))

    return segments


def write_srt(segments, output_path: Path):
    lines = []
    for idx, (start, end, text) in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def extend_segment_end_times(segments, min_next_gap=0.2, max_extend=1.0):
    if len(segments) < 2:
        return segments

    adjusted = list(segments)
    for i in range(len(adjusted) - 1):
        start, end, text = adjusted[i]
        next_start, _, _ = adjusted[i + 1]

        original_gap = next_start - end
        if original_gap <= min_next_gap:
            continue

        # Extend current segment as much as possible while preserving required gap.
        extend_by = min(max_extend, original_gap - min_next_gap)
        adjusted[i] = (start, end + extend_by, text)

    return adjusted


def extract_audio_from_video(video_path: Path) -> tuple[Path, Path | None]:
    """Extract audio from video file using ffmpeg-python to temp directory"""
    video_extensions = {'.mp4', '.mkv', '.avi'}

    if video_path.suffix.lower() not in video_extensions:
        return video_path, None

    # Create temp directory: temp_视频文件名 (without extension)
    temp_dir = video_path.parent / f"temp_{video_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Output audio path (extract as wav)
    audio_output = temp_dir / f"{video_path.stem}.wav"

    print(f"Extracting audio from video to: {audio_output.resolve()}")
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(audio_output), vn=None, acodec='pcm_s16le', ar=16000, ac=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg failed to extract audio: {e.stderr.decode()}") from e

    print(f"Audio extracted to: {audio_output.resolve()}")
    return audio_output, temp_dir


def main():
    parser = argparse.ArgumentParser(description="Qwen3 ASR Transcriber")
    parser.add_argument("-d", "--device", choices=["cuda", "xpu", "cpu"], help="Torch device to use (cuda, xpu, cpu)")
    parser.add_argument("input_path", help="Path to audio or video file (supported: .mp4 .mkv .avi wav mp3 etc)")
    args = parser.parse_args()

    input_path = Path(args.input_path)

    # Check if it's a video file, extract audio if needed
    video_extensions = {'.mp4', '.mkv', '.avi'}
    original_input_path = input_path
    temp_dir_to_cleanup = None
    if input_path.suffix.lower() in video_extensions:
        input_path, temp_dir_to_cleanup = extract_audio_from_video(input_path)

    if args.device:
        if args.device == "cuda":
            device = "cuda:0"
        elif args.device == "xpu":
            device = "xpu:0"
        else:
            device = "cpu"
    else:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu:0"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    start_time = time.time()
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map=device,
        # attn_implementation="flash_attention_2",
        max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
        max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map=device,
            # attn_implementation="flash_attention_2",
        ),
    )
    end_time = time.time()
    print(f"from_pretrained time: {end_time - start_time:.2f}s")

    start_time = time.time()
    results = model.transcribe(
        audio=str(input_path),
        language="Chinese", # set "English" to force the language
        return_time_stamps=True,
    )
    end_time = time.time()
    print(f"transcribe time: {end_time - start_time:.2f}s")

    print(results[0].language)
    print(results[0].text)

    #for ts in results[0].time_stamps:
    #    print(f"Start: {ts.start_time:.2f}s, End: {ts.end_time:.2f}s, Text: {ts.text}")

    segments = build_sentence_segments(results[0].text, results[0].time_stamps)
    segments = extend_segment_end_times(segments, min_next_gap=0.2, max_extend=1.0)
    # For video files, output srt next to the original video file
    srt_path = original_input_path.with_suffix(".srt")
    write_srt(segments, srt_path)
    print(f"\nSRT written to: {srt_path.resolve()}")

    for idx, (start, end, text) in enumerate(segments, start=1):
        print(f"[{idx}] {start:.2f}s -> {end:.2f}s | {text}")

    # Clean up temp directory if it exists
    if temp_dir_to_cleanup is not None and temp_dir_to_cleanup.exists():
        import shutil
        try:
            shutil.rmtree(temp_dir_to_cleanup)
            print(f"\nCleaned up temp directory: {temp_dir_to_cleanup.resolve()}")
        except Exception as e:
            print(f"\nWarning: Failed to clean up temp directory {temp_dir_to_cleanup}: {e}")


if __name__ == "__main__":
    main()