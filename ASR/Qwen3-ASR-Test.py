import torch
import time
import argparse
import subprocess
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from pydub import AudioSegment
from qwen_asr import Qwen3ASRModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


@dataclass
class MutableTimeStamp:
    """Mutable timestamp class to work around frozen dataclass issue"""
    start_time: float
    end_time: float
    text: str = ""

#https://github.com/QwenLM/Qwen3-ASR

# Download through ModelScope (recommended for users in Mainland China)
# pip install -U modelscope
# modelscope download --model Qwen/Qwen3-ASR-1.7B  --local_dir ./Qwen3-ASR-1.7B
# modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./Qwen3-ASR-0.6B
# modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./Qwen3-ForcedAligner-0.6B

# pip install -U qwen-asr pydub torch


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


def build_sentence_segments(full_text, time_stamps, comma_split_threshold=20, time_gap_threshold=3.0, space_gap_threshold=0.3, add_spaces=True):
    """
    Simple approach based on actual data structure:
    - time_stamps have content words with timing, but no punctuation
    - full_text has everything with punctuation
    Strategy:
    1. First, split full_text into sentences (by punctuation)
    2. For each sentence, find which time_stamps it corresponds to
    3. Assign the time range from first to last timestamp in that sentence
    """
    if not full_text or not time_stamps:
        return []

    # First, build a list of all content with timing from time_stamps
    content_list = []
    for ts in time_stamps:
        text = getattr(ts, 'text', '') or ''
        if text:
            content_list.append({
                'text': text,
                'start': ts.start_time,
                'end': ts.end_time
            })

    if not content_list:
        return []

    # Now build a "content stream" - concatenated content text with char-level timing
    content_chars = []
    for item in content_list:
        text = item['text']
        duration = item['end'] - item['start']
        char_duration = duration / len(text) if len(text) > 0 else 0
        for i, ch in enumerate(text):
            content_chars.append({
                'char': ch,
                'start': item['start'] + i * char_duration,
                'end': item['start'] + (i + 1) * char_duration
            })

    # Now align full_text with content_chars to get char-level timings for everything
    aligned = []
    cc_idx = 0  # content_chars index

    for ch in full_text:
        # Check if this character is punctuation or space
        is_punc_or_space = ch in "，。！？；：、,.!?;:()（）【】[]《》<>\"'""''—… \t\n"

        if is_punc_or_space:
            # Punctuation/space - use timing from previous or next character
            if aligned:
                start_t = aligned[-1]['end']
                end_t = aligned[-1]['end']
            elif cc_idx < len(content_chars):
                start_t = content_chars[cc_idx]['start']
                end_t = content_chars[cc_idx]['start']
            else:
                start_t = 0
                end_t = 0
            aligned.append({'char': ch, 'start': start_t, 'end': end_t})
        else:
            # Content character - find matching in content_chars
            found = False
            # Look ahead in content_chars to find a match
            temp_idx = cc_idx
            while temp_idx < len(content_chars):
                if content_chars[temp_idx]['char'] == ch:
                    # Found match
                    aligned.append({
                        'char': ch,
                        'start': content_chars[temp_idx]['start'],
                        'end': content_chars[temp_idx]['end']
                    })
                    cc_idx = temp_idx + 1
                    found = True
                    break
                temp_idx += 1
            if not found:
                # No match found - just add with estimated time
                if aligned:
                    start_t = aligned[-1]['end']
                    end_t = aligned[-1]['end']
                else:
                    start_t = 0
                    end_t = 0
                aligned.append({'char': ch, 'start': start_t, 'end': end_t})

    # Now build sentences from aligned characters
    strong_sentence_endings = set("。！？；.!?;")
    comma_endings = set("，,")
    punctuation_chars = set("，。！？；：、,.!?;:()（）【】[]《》<>\"'""''—…")

    segments = []
    current_text = []
    current_start = None
    current_end = None
    current_char_count = 0
    prev_end = None

    for item in aligned:
        ch = item['char']
        start_t = item['start']
        end_t = item['end']

        # Check time gap from previous content character
        time_gap = 0.0
        if (prev_end is not None and current_start is not None and
            not ch.isspace() and ch not in punctuation_chars):
            time_gap = start_t - prev_end

        # Split on large time gap
        if (current_start is not None and prev_end is not None and
            not ch.isspace() and ch not in punctuation_chars and
            time_gap >= time_gap_threshold):
            text = "".join(current_text).strip()
            text = _strip_trailing_punctuation(text)
            if text:
                segments.append((current_start, current_end, text))
            current_text = []
            current_start = None
            current_end = None
            current_char_count = 0

        # Add space if needed (Chinese only)
        if (add_spaces and current_text and current_start is not None and
            not ch.isspace() and ch not in punctuation_chars and
            prev_end is not None and time_gap >= space_gap_threshold):
            last_ch = current_text[-1] if current_text else ""
            if last_ch and not last_ch.isspace() and last_ch not in punctuation_chars:
                current_text.append(" ")

        # Update current segment
        if current_start is None and not ch.isspace() and ch not in punctuation_chars:
            current_start = start_t
        if not ch.isspace() and ch not in punctuation_chars:
            current_end = end_t
            current_char_count += 1
            prev_end = end_t

        current_text.append(ch)

        # Split on sentence endings
        if ch in strong_sentence_endings and current_start is not None:
            text = "".join(current_text).strip()
            text = _strip_trailing_punctuation(text)
            if text:
                segments.append((current_start, current_end, text))
            current_text = []
            current_start = None
            current_end = None
            current_char_count = 0
        elif ch in comma_endings and current_start is not None and current_char_count >= comma_split_threshold:
            text = "".join(current_text).strip()
            text = _strip_trailing_punctuation(text)
            if text:
                segments.append((current_start, current_end, text))
            current_text = []
            current_start = None
            current_end = None
            current_char_count = 0

    # Add last segment
    if current_start is not None:
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


def format_lrc_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    minutes = total_ms // 60000
    secs = (total_ms % 60000) // 1000
    millis = total_ms % 1000
    return f"{minutes:02d}:{secs:02d}.{millis//10:02d}"


def write_lrc(segments, output_path: Path):
    lines = []
    for start, end, text in segments:
        lines.append(f"[{format_lrc_time(start)}]{text}")
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
    """Extract audio from video file using ffmpeg command line to temp directory"""
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
        # Use subprocess to call ffmpeg directly
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            str(audio_output)
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to extract audio: {e.stderr}") from e

    print(f"Audio extracted to: {audio_output.resolve()}")
    return audio_output, temp_dir


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using pydub"""
    audio = AudioSegment.from_file(str(audio_path))
    return len(audio) / 1000.0


def split_audio_with_vad(audio_path: Path, temp_dir: Path, max_chunk_duration: float = 180.0):
    """
    Split audio using Silero VAD into chunks around max_chunk_duration seconds.
    Returns list of (chunk_path, chunk_start_time_seconds)
    """
    # Load Silero VAD model
    print("Loading Silero VAD model...")
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    (get_speech_timestamps, _, read_audio, _, _) = utils

    # Read audio for VAD (16k sampling rate)
    print("Detecting speech segments...")
    wav = read_audio(str(audio_path), sampling_rate=16000)

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        wav, vad_model,
        sampling_rate=16000,
        threshold=0.5,
        min_silence_duration_ms=500
    )

    # Load full audio with pydub
    full_audio = AudioSegment.from_file(str(audio_path))

    # Group speech segments into chunks of approximately max_chunk_duration
    chunks = []
    current_chunk_start = 0  # in milliseconds
    current_chunk_end = 0

    for ts in speech_timestamps:
        # Convert sample indices to milliseconds (16k samples/sec)
        start_ms = ts['start'] / 16
        end_ms = ts['end'] / 16

        # Check if adding this segment would exceed max chunk duration
        potential_end = max(current_chunk_end, end_ms + 200)  # +200ms buffer
        if current_chunk_start > 0 and (potential_end - current_chunk_start) / 1000.0 > max_chunk_duration:
            # Finalize current chunk
            chunk_path = temp_dir / f"chunk_{len(chunks):04d}.wav"
            # Add 200ms buffer at start and end
            chunk_start_ms = max(0, current_chunk_start - 200)
            chunk_end_ms = current_chunk_end + 200
            chunk = full_audio[chunk_start_ms:chunk_end_ms]
            chunk.export(str(chunk_path), format="wav")
            chunks.append((chunk_path, chunk_start_ms / 1000.0))
            # Start new chunk
            current_chunk_start = start_ms
            current_chunk_end = end_ms
        else:
            current_chunk_end = max(current_chunk_end, end_ms)
            if current_chunk_start == 0:
                current_chunk_start = start_ms

    # Add the last chunk
    if current_chunk_start < len(full_audio):
        chunk_path = temp_dir / f"chunk_{len(chunks):04d}.wav"
        chunk_start_ms = max(0, current_chunk_start - 200)
        chunk_end_ms = min(len(full_audio), current_chunk_end + 200)
        chunk = full_audio[chunk_start_ms:chunk_end_ms]
        chunk.export(str(chunk_path), format="wav")
        chunks.append((chunk_path, chunk_start_ms / 1000.0))

    # Clean up VAD model to free VRAM
    print(f"Split audio into {len(chunks)} chunks, releasing VAD model...")
    del vad_model
    del utils
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

    return chunks


def transcribe_single_chunk(model, chunk_path, time_offset, language, lock):
    """Transcribe a single chunk (thread-safe with lock)"""
    print(f"Transcribing chunk: {chunk_path.name} (offset: {time_offset:.2f}s)")

    # Use lock to ensure thread safety for model inference
    with lock:
        results = model.transcribe(
            audio=str(chunk_path),
            language=language,
            return_time_stamps=True,
        )

    chunk_text = ""
    chunk_timestamps = []

    if results and len(results) > 0:
        chunk_text = results[0].text
        # Adjust time stamps by chunk offset - use mutable wrapper
        for ts in results[0].time_stamps:
            mutable_ts = MutableTimeStamp(
                start_time=ts.start_time + time_offset,
                end_time=ts.end_time + time_offset,
                text=getattr(ts, 'text', '')
            )
            chunk_timestamps.append(mutable_ts)

    return time_offset, chunk_text, chunk_timestamps


def transcribe_chunks(model, chunks, language=None, concurrency=2):
    """Transcribe chunks concurrently with thread pool"""
    chunk_results = []
    lock = threading.Lock()

    # Use ThreadPoolExecutor for concurrent transcription
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all tasks
        futures = []
        for chunk_path, time_offset in chunks:
            future = executor.submit(
                transcribe_single_chunk,
                model, chunk_path, time_offset, language, lock
            )
            futures.append(future)

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                time_offset, chunk_text, chunk_timestamps = future.result()
                chunk_results.append((time_offset, chunk_text, chunk_timestamps))
            except Exception as e:
                print(f"Error transcribing chunk: {e}")

    # Sort results by original time offset to maintain order
    chunk_results.sort(key=lambda x: x[0])

    # Combine results in order
    all_text = []
    all_time_stamps = []
    for _, chunk_text, chunk_timestamps in chunk_results:
        all_text.append(chunk_text)
        all_time_stamps.extend(chunk_timestamps)

    full_text = "".join(all_text)
    return full_text, all_time_stamps


def main():
    parser = argparse.ArgumentParser(description="Qwen3 ASR Transcriber")
    parser.add_argument("-d", "--device", choices=["cuda", "xpu", "cpu"], help="Torch device to use (cuda, xpu, cpu)")
    parser.add_argument("-l", "--language", choices=["ch", "en", "jp"], help="Language: ch=Chinese, en=English, jp=Japanese")
    parser.add_argument("-c", "--concurrency", type=int, default=5, help="Number of concurrent transcribe tasks (default: 5)")
    parser.add_argument("input_path", help="Path to audio or video file (supported: .mp4 .mkv .avi wav mp3 etc)")
    args = parser.parse_args()

    # Map language code to full name
    language_map = {
        "ch": "Chinese",
        "en": "English",
        "jp": "Japanese"
    }
    language = language_map.get(args.language)

    input_path = Path(args.input_path)

    # Check if it's a video file, extract audio if needed
    video_extensions = {'.mp4', '.mkv', '.avi'}
    original_input_path = input_path
    temp_dir_to_cleanup = None
    is_video = False
    if input_path.suffix.lower() in video_extensions:
        is_video = True
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

    # Check audio duration
    audio_duration = get_audio_duration(input_path)
    print(f"Audio duration: {audio_duration:.2f}s ({audio_duration/60:.2f} minutes)")

    # Create temp directory for chunks if needed
    chunk_temp_dir = None
    chunks = None

    # Decide whether to split audio first (VAD doesn't need ASR model)
    if audio_duration > 180.0:  # > 3 minutes
        print("Audio exceeds 3 minutes, using VAD-based splitting...")
        # Create temp dir for chunks
        chunk_temp_dir = input_path.parent / f"temp_chunks_{input_path.stem}"
        chunk_temp_dir.mkdir(parents=True, exist_ok=True)

        # Split audio first (VAD model will be released after this)
        chunks = split_audio_with_vad(input_path, chunk_temp_dir, max_chunk_duration=180.0)

    # Now load ASR model (VAD has been released if used)
    start_time = time.time()
    print("Loading Qwen3 ASR model...")
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map=device,
        # attn_implementation="flash_attention_2",
        max_inference_batch_size=-1, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
        max_new_tokens=2048, # Maximum number of tokens to generate. Set a larger value for long audio input.
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

    # Do transcription
    if chunks is not None:
        # Transcribe chunks
        print(f"Using {args.concurrency} concurrent transcribe tasks")
        full_text, all_time_stamps = transcribe_chunks(model, chunks, language, args.concurrency)
    else:
        # Direct transcription for short audio
        print("Audio is short, transcribing directly...")
        results = model.transcribe(
            audio=str(input_path),
            language=language,
            return_time_stamps=True,
        )
        full_text = results[0].text
        # Convert to mutable timestamps for consistency
        all_time_stamps = [
            MutableTimeStamp(
                start_time=ts.start_time,
                end_time=ts.end_time,
                text=getattr(ts, 'text', '')
            )
            for ts in results[0].time_stamps
        ]

    end_time = time.time()
    print(f"transcribe time: {end_time - start_time:.2f}s")

    print(full_text)

    # Debug: print time stamps
    print("\nTime stamps from model:")
    for ts in all_time_stamps:
        print(f"Start: {ts.start_time:.2f}s, End: {ts.end_time:.2f}s, Text: {repr(ts.text)}")

    # Only add spaces for Chinese mode
    add_spaces = True #(language is None) or (language == "Chinese")
    segments = build_sentence_segments(full_text, all_time_stamps, add_spaces=add_spaces)
    segments = extend_segment_end_times(segments, min_next_gap=0.2, max_extend=1.0)

    # For video files, output srt; for audio files, output lrc
    if is_video:
        output_path = original_input_path.with_suffix(".srt")
        write_srt(segments, output_path)
        print(f"\nSRT written to: {output_path.resolve()}")
    else:
        output_path = original_input_path.with_suffix(".lrc")
        write_lrc(segments, output_path)
        print(f"\nLRC written to: {output_path.resolve()}")

    for idx, (start, end, text) in enumerate(segments, start=1):
        print(f"[{idx}] {start:.2f}s -> {end:.2f}s | {text}")

    # Clean up temp directories
    if chunk_temp_dir is not None and chunk_temp_dir.exists():
        import shutil
        try:
            shutil.rmtree(chunk_temp_dir)
            print(f"\nCleaned up chunk temp directory: {chunk_temp_dir.resolve()}")
        except Exception as e:
            print(f"\nWarning: Failed to clean up chunk temp directory {chunk_temp_dir}: {e}")

    if temp_dir_to_cleanup is not None and temp_dir_to_cleanup.exists():
        import shutil
        try:
            shutil.rmtree(temp_dir_to_cleanup)
            print(f"\nCleaned up temp directory: {temp_dir_to_cleanup.resolve()}")
        except Exception as e:
            print(f"\nWarning: Failed to clean up temp directory {temp_dir_to_cleanup}: {e}")


if __name__ == "__main__":
    main()
