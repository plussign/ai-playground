import torch
import time
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


def build_sentence_segments(full_text, time_stamps, comma_split_threshold=20):
    # Strong endings always split; commas split only when the segment is long enough.
    strong_sentence_endings = set("。！？；.!?;")
    comma_endings = set("，,")
    punctuation_chars = set("，。！？；：、,.!?;:()（）【】[]《》<>\"'""''—…")

    segments = []
    ts_index = 0
    current_text = []
    current_start = None
    current_end = None
    current_char_count = 0

    for ch in full_text:
        current_text.append(ch)

        if ch.isspace() or ch in punctuation_chars:
            should_split = False
            if ch in strong_sentence_endings:
                should_split = True
            elif ch in comma_endings and current_char_count >= comma_split_threshold:
                should_split = True

            if should_split and current_start is not None and current_end is not None:
                text = "".join(current_text).strip()
                if text:
                    segments.append((current_start, current_end, text))
                current_text = []
                current_start = None
                current_end = None
                current_char_count = 0
            continue

        if ts_index >= len(time_stamps):
            continue

        ts = time_stamps[ts_index]
        ts_index += 1

        if current_start is None:
            current_start = ts.start_time
        current_end = ts.end_time
        current_char_count += 1

    if current_start is not None and current_end is not None:
        text = "".join(current_text).strip()
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

audio_path = "../cv/胡桃.wav"

start_time = time.time()
results = model.transcribe(
    audio=audio_path,
    language="Chinese", # set "English" to force the language
    return_time_stamps=True,
)
end_time = time.time()
print(f"transcribe time: {end_time - start_time:.2f}s")

print(results[0].language)
print(results[0].text)

for ts in results[0].time_stamps:
    print(f"Start: {ts.start_time:.2f}s, End: {ts.end_time:.2f}s, Text: {ts.text}")

segments = build_sentence_segments(results[0].text, results[0].time_stamps)
segments = extend_segment_end_times(segments, min_next_gap=0.2, max_extend=1.0)
srt_path = Path(audio_path).with_suffix(".srt")
write_srt(segments, srt_path)
print(f"\nSRT written to: {srt_path.resolve()}")

for idx, (start, end, text) in enumerate(segments, start=1):
    print(f"[{idx}] {start:.2f}s -> {end:.2f}s | {text}")