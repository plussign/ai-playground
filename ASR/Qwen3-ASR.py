"""

基于 Qwen3-ASR 的语音识别工具，提供窗口交互界面。

支持拖入或添加音频/视频文件，批量处理为 SRT 字幕或 LRC 歌词格式。

## 功能特性

- **多格式支持**：`.mp4`、`.mkv`、`.avi`、`.wav`、`.mp3` 等
- **长音频处理**：Silero VAD 自动分段
- **多语言支持**：中文、英文、日语
- **并发转录**：可配置并发任务数
- **拖放支持**：文件列表支持拖放添加

## 依赖

```bash
pip install -U qwen-asr pydub torch scipy
```

## 核心函数（复用自 Qwen3-ASR-Test.py）

| 函数 | 功能 |
|------|------|
| `extract_audio_from_video()` | 从视频提取音频 |
| `preprocess_audio()` | 音频预处理 |
| `split_audio_with_vad()` | VAD 分段 |
| `transcribe_chunks()` | 并发转录 |
| `build_sentence_segments()` | 智能分句 |
| `write_srt()` / `write_lrc()` | 输出字幕/歌词 |
"""

import torch
import time
import argparse
import subprocess
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from pydub import AudioSegment, effects
from scipy import signal
from qwen_asr import Qwen3ASRModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import shutil


# ========== 复用 Qwen3-ASR-Test.py 的核心函数 ==========

@dataclass
class MutableTimeStamp:
    start_time: float
    end_time: float
    text: str = ""


def format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _strip_trailing_punctuation(text: str) -> str:
    trailing_punctuation = set("，。；,.;")
    while text and text[-1] in trailing_punctuation:
        text = text[:-1].rstrip()
    return text


def build_sentence_segments(full_text, time_stamps, comma_split_threshold=20, time_gap_threshold=3.0, space_gap_threshold=0.3, add_spaces=True):
    if not full_text or not time_stamps:
        return []

    content_list = []
    for ts in time_stamps:
        text = getattr(ts, 'text', '') or ''
        if text:
            content_list.append({'text': text, 'start': ts.start_time, 'end': ts.end_time})

    if not content_list:
        return []

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

    aligned = []
    cc_idx = 0

    for ch in full_text:
        is_punc_or_space = ch in "，。！？；：、,.!?;:()（）【】[]《》<>\"'""''—… \t\n"

        if is_punc_or_space:
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
            found = False
            temp_idx = cc_idx
            while temp_idx < len(content_chars):
                if content_chars[temp_idx]['char'] == ch:
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
                if aligned:
                    start_t = aligned[-1]['end']
                    end_t = aligned[-1]['end']
                else:
                    start_t = 0
                    end_t = 0
                aligned.append({'char': ch, 'start': start_t, 'end': end_t})

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

        time_gap = 0.0
        if (prev_end is not None and current_start is not None and
            not ch.isspace() and ch not in punctuation_chars):
            time_gap = start_t - prev_end

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

        if (add_spaces and current_text and current_start is not None and
            not ch.isspace() and ch not in punctuation_chars and
            prev_end is not None and time_gap >= space_gap_threshold):
            last_ch = current_text[-1] if current_text else ""
            if last_ch and not last_ch.isspace() and last_ch not in punctuation_chars:
                current_text.append(" ")

        if current_start is None and not ch.isspace() and ch not in punctuation_chars:
            current_start = start_t
        if not ch.isspace() and ch not in punctuation_chars:
            current_end = end_t
            current_char_count += 1
            prev_end = end_t

        current_text.append(ch)

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
        extend_by = min(max_extend, original_gap - min_next_gap)
        adjusted[i] = (start, end + extend_by, text)
    return adjusted


def extract_audio_from_video(video_path: Path) -> tuple[Path, Path | None]:
    video_extensions = {'.mp4', '.mkv', '.avi', '.ts'}
    if video_path.suffix.lower() not in video_extensions:
        return video_path, None

    temp_dir = video_path.parent / f"temp_{video_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    audio_output = temp_dir / f"{video_path.stem}.wav"

    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1', '-y',
        str(audio_output)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return audio_output, temp_dir


def get_audio_duration(audio_path: Path) -> float:
    audio = AudioSegment.from_file(str(audio_path))
    return len(audio) / 1000.0


def apply_highpass_filter(audio: AudioSegment, cutoff_freq: float = 80.0) -> AudioSegment:
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
    filtered_samples = signal.filtfilt(b, a, samples)
    filtered_samples = np.clip(filtered_samples, -32768, 32767).astype(np.int16)
    return AudioSegment(
        filtered_samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )


def preprocess_audio(audio_path: Path, output_path: Path, highpass_cutoff: float = 80.0, use_agc: bool = True) -> Path:
    audio = AudioSegment.from_file(str(audio_path))
    if highpass_cutoff > 0:
        audio = apply_highpass_filter(audio, highpass_cutoff)
    if use_agc:
        audio = effects.normalize(audio)
    audio.export(str(output_path), format="wav")
    return output_path


def split_audio_with_vad(audio_path: Path, temp_dir: Path, max_chunk_duration: float = 10.0, vad_threshold: float = 0.30):
    print("Loading Silero VAD model...")
    vad_model, utils = torch.hub.load('snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, _, read_audio, _, _) = utils

    wav = read_audio(str(audio_path), sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000, threshold=vad_threshold, min_silence_duration_ms=1000)

    full_audio = (
        AudioSegment.from_file(str(audio_path))
        .set_frame_rate(16000).set_channels(1).set_sample_width(2)
    )

    if not speech_timestamps:
        chunk_path = temp_dir / f"chunk_0000_0_{len(full_audio)}.wav"
        full_audio.export(str(chunk_path), format="wav")
        del vad_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        return [(chunk_path, 0.0)]

    chunks = []
    max_chunk_ms = int(max_chunk_duration * 1000)
    speech_ranges_ms = [(int(ts['start'] / 16), int(ts['end'] / 16)) for ts in speech_timestamps]
    chunk_padding_ms = 1000

    chunk_start_ms = max(0, speech_ranges_ms[0][0] - chunk_padding_ms)
    chunk_end_ms = min(len(full_audio), speech_ranges_ms[0][1] + chunk_padding_ms)
    prev_speech_end_ms = speech_ranges_ms[0][1]

    for speech_start_ms, speech_end_ms in speech_ranges_ms[1:]:
        candidate_end_ms = min(len(full_audio), speech_end_ms + chunk_padding_ms)
        would_exceed = (candidate_end_ms - chunk_start_ms) > max_chunk_ms
        has_clear_boundary = (speech_start_ms - prev_speech_end_ms) >= 1000

        if would_exceed and has_clear_boundary:
            chunk_path = temp_dir / f"chunk_{len(chunks):04d}_{int(chunk_start_ms)}_{int(chunk_end_ms)}.wav"
            chunk = full_audio[chunk_start_ms:chunk_end_ms]
            chunk.export(str(chunk_path), format="wav")
            chunks.append((chunk_path, chunk_start_ms / 1000.0))
            chunk_start_ms = max(0, speech_start_ms - chunk_padding_ms)
            chunk_end_ms = min(len(full_audio), speech_end_ms + chunk_padding_ms)
        else:
            chunk_end_ms = max(chunk_end_ms, candidate_end_ms)
        prev_speech_end_ms = speech_end_ms

    chunk_path = temp_dir / f"chunk_{len(chunks):04d}_{int(chunk_start_ms)}_{int(chunk_end_ms)}.wav"
    chunk = full_audio[chunk_start_ms:chunk_end_ms]
    chunk.export(str(chunk_path), format="wav")
    chunks.append((chunk_path, chunk_start_ms / 1000.0))

    del vad_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

    return chunks


def transcribe_single_chunk(model, chunk_path, time_offset, language, lock, debug=False):
    with lock:
        results = model.transcribe(audio=str(chunk_path), language=language, return_time_stamps=True)

    chunk_text = ""
    chunk_timestamps = []

    if results and len(results) > 0:
        chunk_text = results[0].text
        for ts in results[0].time_stamps:
            mutable_ts = MutableTimeStamp(
                start_time=ts.start_time + time_offset,
                end_time=ts.end_time + time_offset,
                text=getattr(ts, 'text', '')
            )
            chunk_timestamps.append(mutable_ts)

    return time_offset, chunk_text, chunk_timestamps


def transcribe_chunks(model, chunks, language=None, concurrency=2, debug=False, progress_callback=None):
    chunk_results = []
    lock = threading.Lock()
    total_chunks = len(chunks)
    completed_chunks = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for chunk_path, time_offset in chunks:
            future = executor.submit(transcribe_single_chunk, model, chunk_path, time_offset, language, lock, debug)
            futures.append(future)

        for future in as_completed(futures):
            try:
                time_offset, chunk_text, chunk_timestamps = future.result()
                chunk_results.append((time_offset, chunk_text, chunk_timestamps))
                completed_chunks += 1
                if progress_callback:
                    progress_callback(completed_chunks, total_chunks)
            except Exception as e:
                print(f"Error transcribing chunk: {e}")
                completed_chunks += 1
                if progress_callback:
                    progress_callback(completed_chunks, total_chunks)

    chunk_results.sort(key=lambda x: x[0])

    all_text = []
    all_time_stamps = []
    for _, chunk_text, chunk_timestamps in chunk_results:
        all_text.append(chunk_text)
        all_time_stamps.extend(chunk_timestamps)

    full_text = "".join(all_text)
    return full_text, all_time_stamps


# ========== GUI 应用类 ==========

class FileItem:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filename = self.filepath.name
        self.status = "未处理"
        self.progress = 0
        self.total_chunks = 0
        self.current_chunk = 0
        self.result_path = None
        self.error = None


class ASRGUI:
    def __init__(self, root, enable_dnd=False):
        self.root = root
        self.root.title("Qwen3-ASR 语音识别工具")
        self.root.geometry("900x700")

        self.files = []
        self.selected_index = None
        self.current_processing = None
        self.is_processing_all = False
        self.model = None
        self.model_device = None

        self._setup_ui()
        if enable_dnd:
            self._setup_drag_drop()

    def _setup_ui(self):
        # 参数设置区
        settings_frame = ttk.LabelFrame(self.root, text="参数设置", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)

        # 第一行
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill="x", pady=2)

        ttk.Label(row1, text="设备:").pack(side="left")
        self.device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
        device_combo = ttk.Combobox(row1, textvariable=self.device_var, values=["cuda", "xpu", "cpu"], state="readonly", width=8)
        device_combo.pack(side="left", padx=5)

        ttk.Label(row1, text="语言:").pack(side="left", padx=(20, 0))
        self.lang_var = tk.StringVar(value="中文")
        self.lang_code_map = {"中文": "ch", "英文": "en", "日语": "jp"}
        self.lang_code_rev = {"ch": "中文", "en": "英文", "jp": "日语"}
        lang_combo = ttk.Combobox(row1, textvariable=self.lang_var, values=["中文", "英文", "日语"], state="readonly", width=8)
        lang_combo.pack(side="left", padx=5)

        ttk.Label(row1, text="并发数:").pack(side="left", padx=(20, 0))
        self.concurrency_var = tk.IntVar(value=2)
        ttk.Spinbox(row1, from_=1, to=8, textvariable=self.concurrency_var, width=5).pack(side="left", padx=5)

        ttk.Label(row1, text="VAD阈值:").pack(side="left", padx=(20, 0))
        self.vad_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(row1, from_=0.1, to=1.0, increment=0.05, textvariable=self.vad_var, width=5).pack(side="left", padx=5)

        # 第二行
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill="x", pady=2)

        ttk.Label(row2, text="最大分段时间(秒):").pack(side="left")
        self.max_chunk_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(row2, from_=5, to=60, increment=5, textvariable=self.max_chunk_var, width=5).pack(side="left", padx=5)

        ttk.Label(row2, text="高通滤波(Hz):").pack(side="left", padx=(20, 0))
        self.highpass_var = tk.DoubleVar(value=80.0)
        ttk.Spinbox(row2, from_=0, to=500, increment=10, textvariable=self.highpass_var, width=5).pack(side="left", padx=5)

        self.use_agc_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="自动增益(AGC)", variable=self.use_agc_var).pack(side="left", padx=(20, 0))

        self.use_preprocess_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="音频预处理", variable=self.use_preprocess_var).pack(side="left", padx=(20, 0))

        self.debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="调试模式", variable=self.debug_var).pack(side="left", padx=(20, 0))

        # 文件列表区
        list_frame = ttk.LabelFrame(self.root, text="文件列表", padding=5)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # 表格
        columns = ("序号", "文件名", "状态")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("序号", text="序号")
        self.tree.heading("文件名", text="文件名")
        self.tree.heading("状态", text="状态")

        self.tree.column("序号", width=60, anchor="center")
        self.tree.column("文件名", width=500)
        self.tree.column("状态", width=280)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._on_double_click)

        # 按钮区
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(btn_frame, text="添加文件", command=self._add_files).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="移除选中", command=self._remove_selected).pack(side="left", padx=5)

        ttk.Button(btn_frame, text="开始处理", command=self._start_selected).pack(side="right", padx=5)
        ttk.Button(btn_frame, text="全部开始", command=self._start_all).pack(side="right", padx=5)

        # 状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief="sunken", anchor="w")
        self.status_bar.pack(fill="x", padx=10, pady=(0, 5))

    def _setup_drag_drop(self):
        from tkinterdnd2 import DND_FILES

        def drop(event):
            for path in self.root.tk.splitlist(event.data):
                path = path.strip('{}')
                if os.path.isfile(path):
                    self._add_file(path)

        self.tree.drop_target_register(DND_FILES)
        self.tree.dnd_bind('<<Drop>>', drop)

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="选择音视频文件",
            filetypes=[
                ("音视频文件", "*.mp4 *.mkv *.avi *.ts *.wav *.mp3 *.m4a *.flac *.ogg"),
                ("所有文件", "*.*")
            ]
        )
        for f in files:
            self._add_file(f)

    def _add_file(self, filepath: str):
        # 检查是否已存在
        for item in self.files:
            if str(item.filepath) == filepath:
                return
        self.files.append(FileItem(filepath))
        self._refresh_list()

    def _remove_selected(self):
        if self.selected_index is not None and self.selected_index < len(self.files):
            del self.files[self.selected_index]
            self.selected_index = None
            self._refresh_list()

    def _on_select(self, event):
        selection = self.tree.selection()
        if selection:
            item_id = selection[0]
            index = int(item_id) - 1
            self.selected_index = index
        else:
            self.selected_index = None

    def _on_double_click(self, event):
        if self.selected_index is not None and self.selected_index < len(self.files):
            item = self.files[self.selected_index]
            directory = item.filepath.parent
            if sys.platform == "win32":
                os.startfile(directory)
            elif sys.platform == "darwin":
                subprocess.run(["open", directory])
            else:
                subprocess.run(["xdg-open", directory])

    def _refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for i, file_item in enumerate(self.files, start=1):
            status = file_item.status
            if file_item.status == "听写中" and file_item.total_chunks > 0:
                pct = int(file_item.current_chunk / file_item.total_chunks * 100)
                status = f"听写中 {pct}%"
            elif file_item.status == "已完成" and file_item.result_path:
                status = f"已完成 ({Path(file_item.result_path).name})"

            self.tree.insert("", "end", iid=str(i), values=(i, file_item.filename, status))

    def _update_status(self, text: str):
        self.status_bar.config(text=text)
        self.root.update_idletasks()

    def _get_device(self):
        device = self.device_var.get()
        if device == "cuda":
            return "cuda:0"
        elif device == "xpu":
            return "xpu:0"
        else:
            return "cpu"

    def _load_model_if_needed(self):
        if self.model is None or self.model_device != self._get_device():
            self._update_status("正在加载 ASR 模型...")
            script_dir = Path(__file__).parent
            device = self._get_device()

            self.model = Qwen3ASRModel.from_pretrained(
                str(script_dir / "Qwen/Qwen3-ASR-1.7B"),
                dtype=torch.bfloat16,
                device_map=device,
                max_inference_batch_size=-1,
                max_new_tokens=2048,
                forced_aligner=str(script_dir / "Qwen/Qwen3-ForcedAligner-0.6B"),
                forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map=device),
            )
            self.model_device = device
            self._update_status("模型加载完成")

    def _start_selected(self):
        if self.selected_index is None or self.selected_index >= len(self.files):
            messagebox.showwarning("提示", "请先选择一个文件")
            return

        if self.files[self.selected_index].status not in ["未处理", "等待开始"]:
            messagebox.showwarning("提示", "该文件已在处理中或已完成")
            return

        threading.Thread(target=self._process_file, args=(self.selected_index,), daemon=True).start()

    def _start_all(self):
        unprocessed = [i for i, f in enumerate(self.files) if f.status == "未处理"]
        if not unprocessed:
            messagebox.showinfo("提示", "没有未处理的文件")
            return

        self.is_processing_all = True
        # 将所有未处理文件标记为等待开始
        for i in unprocessed:
            if self.files[i].status == "未处理":
                self.files[i].status = "等待开始"
        self._refresh_list()

        # 找出下一个未处理的文件开始处理
        for i in unprocessed:
            if self.files[i].status == "等待开始":
                self.current_processing = i
                threading.Thread(target=self._process_file, args=(i,), daemon=True).start()
                break

    def _process_file(self, index: int):
        if index >= len(self.files):
            return

        file_item = self.files[index]
        if file_item.status == "已完成":
            return

        file_item.status = "提取音频中"
        file_item.error = None
        self._refresh_list()

        try:
            self._load_model_if_needed()

            input_path = file_item.filepath
            video_extensions = {'.mp4', '.mkv', '.avi', '.ts'}
            temp_dir_to_cleanup = None
            preprocessed_path = None

            # 提取音频
            if input_path.suffix.lower() in video_extensions:
                self._update_status(f"[{file_item.filename}] 提取音频中...")
                file_item.status = "提取音频中"
                self._refresh_list()

                input_path, temp_dir_to_cleanup = extract_audio_from_video(input_path)

            # 预处理
            if self.use_preprocess_var.get():
                preprocessed_dir = input_path.parent / f"temp_preprocessed_{input_path.stem}"
                preprocessed_dir.mkdir(parents=True, exist_ok=True)
                preprocessed_path = preprocessed_dir / f"preprocessed_{input_path.stem}.wav"

                self._update_status(f"[{file_item.filename}] 预处理音频...")
                preprocess_audio(
                    input_path,
                    preprocessed_path,
                    highpass_cutoff=self.highpass_var.get(),
                    use_agc=self.use_agc_var.get()
                )
                input_path = preprocessed_path
                if temp_dir_to_cleanup is None:
                    temp_dir_to_cleanup = preprocessed_dir

            # VAD 分段
            audio_duration = get_audio_duration(input_path)
            chunks = None
            chunk_temp_dir = None

            if audio_duration > 180.0:
                self._update_status(f"[{file_item.filename}] 分解音频中...")
                file_item.status = "分解音频中"
                self._refresh_list()

                chunk_temp_dir = input_path.parent / f"temp_chunks_{input_path.stem}"
                chunk_temp_dir.mkdir(parents=True, exist_ok=True)
                chunks = split_audio_with_vad(
                    input_path, chunk_temp_dir,
                    max_chunk_duration=self.max_chunk_var.get(),
                    vad_threshold=self.vad_var.get()
                )

            # 转录
            self._update_status(f"[{file_item.filename}] 听写中...")
            file_item.status = "听写中"
            file_item.total_chunks = len(chunks) if chunks else 1
            file_item.current_chunk = 0
            self._refresh_list()

            lang_code = self.lang_code_map.get(self.lang_var.get())
            language = {"ch": "Chinese", "en": "English", "jp": "Japanese"}.get(lang_code)

            if chunks:
                def progress_callback(done, total):
                    file_item.current_chunk = done
                    self._refresh_list()

                full_text, all_time_stamps = transcribe_chunks(
                    self.model, chunks, language,
                    concurrency=self.concurrency_var.get(),
                    debug=self.debug_var.get(),
                    progress_callback=progress_callback
                )
            else:
                file_item.current_chunk = 1
                file_item.total_chunks = 1
                self._refresh_list()

                results = self.model.transcribe(
                    audio=str(input_path),
                    language=language,
                    return_time_stamps=True
                )
                full_text = results[0].text
                all_time_stamps = [
                    MutableTimeStamp(
                        start_time=ts.start_time,
                        end_time=ts.end_time,
                        text=getattr(ts, 'text', '')
                    )
                    for ts in results[0].time_stamps
                ]

            # 生成字幕
            add_spaces = True
            segments = build_sentence_segments(full_text, all_time_stamps, add_spaces=add_spaces)
            segments = extend_segment_end_times(segments, min_next_gap=0.2, max_extend=1.0)

            original_path = file_item.filepath
            is_video = original_path.suffix.lower() in {'.mp4', '.mkv', '.avi', '.ts'}

            if is_video:
                output_path = original_path.with_suffix(".srt")
                write_srt(segments, output_path)
            else:
                output_path = original_path.with_suffix(".lrc")
                write_lrc(segments, output_path)

            file_item.status = "已完成"
            file_item.result_path = str(output_path)
            self._refresh_list()
            self._update_status(f"[{file_item.filename}] 完成: {output_path}")

            # 清理临时文件
            if not self.debug_var.get():
                if chunk_temp_dir and chunk_temp_dir.exists():
                    shutil.rmtree(chunk_temp_dir)
                if temp_dir_to_cleanup and temp_dir_to_cleanup.exists():
                    shutil.rmtree(temp_dir_to_cleanup)
            else:
                if chunk_temp_dir:
                    print(f"调试模式保留: {chunk_temp_dir}")
                if temp_dir_to_cleanup:
                    print(f"调试模式保留: {temp_dir_to_cleanup}")

        except Exception as e:
            file_item.status = f"错误: {str(e)[:30]}"
            file_item.error = str(e)
            self._refresh_list()
            self._update_status(f"[{file_item.filename}] 处理失败: {e}")
            import traceback
            traceback.print_exc()

        # 处理全部模式
        if self.is_processing_all:
            self.is_processing_all = False
            # 找出下一个等待中的文件
            for i, f in enumerate(self.files):
                if f.status == "等待开始":
                    self.is_processing_all = True
                    self.current_processing = i
                    threading.Thread(target=self._process_file, args=(i,), daemon=True).start()
                    break

            # 如果没有更多等待中的文件，检查是否有进行中的
            if not self.is_processing_all:
                for f in self.files:
                    if f.status not in ["已完成", "未处理"] and not str(f.status).startswith("错误"):
                        self.is_processing_all = True
                        break

        self.current_processing = None


def main():
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
        app = ASRGUI(root, enable_dnd=True)
    except ImportError:
        root = tk.Tk()
        app = ASRGUI(root, enable_dnd=False)
        app._update_status("提示: pip install tkinterdnd2 可启用拖放功能")
    root.mainloop()


if __name__ == "__main__":
    main()
