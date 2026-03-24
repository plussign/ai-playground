import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
from pathlib import Path
import time

from modelscope import AutoConfig, AutoModelForCausalLM, AutoTokenizer, snapshot_download

DEFAULT_MODEL_ID = "Tencent-Hunyuan/HY-MT1.5-1.8B-FP8"


def build_prompt(text: str) -> str:
    return (
        "参考下面的翻译：\n猫灵 翻译成 NEKO\n猫娘 翻译成 Neko\n猫蛋 翻译成 Cat Egg."
        "Translate the following segment into English, without additional explanation."
        "请注意保留Html标签为原样，以及注意$Pn模式的标记是待动态填入的占位符。\n\n"
        f"{text}"
    )


def translate_text(model, tokenizer, text: str, max_new_tokens: int) -> str:
    content = (text or "").strip()
    if not content:
        return ""

    messages = [{"role": "user", "content": build_prompt(content)}]
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )

    tokenized_chat = tokenized_chat.to(model.device)
    outputs = model.generate(**tokenized_chat, max_new_tokens=max_new_tokens)
    input_len = tokenized_chat["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return output_text


def ensure_two_columns(row: list[str]) -> list[str]:
    if len(row) == 0:
        return ["", ""]
    if len(row) == 1:
        return [row[0], ""]
    return [row[0], row[1]]


def format_duration(seconds: float) -> str:
    total_millis = int(seconds * 1000)
    hours = total_millis // 3_600_000
    minutes = (total_millis % 3_600_000) // 60_000
    secs = (total_millis % 60_000) // 1000
    millis = total_millis % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate first column in CSV and fill second column.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "src.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV path. Defaults to overwrite input file.",
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="ModelScope model id")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens per row")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent rows to translate")
    parser.add_argument("--start-row", type=int, default=2, help="1-based start row, includes header row in counting")
    parser.add_argument("--end-row", type=int, default=0, help="1-based end row, 0 means to the end")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else input_path

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"CSV is empty: {input_path}")

    model_dir = snapshot_download(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config.tie_word_embeddings = False
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
    )

    # Keep header unchanged and process user-specified data row range.
    start_idx = max(args.start_row - 1, 1)
    end_idx = len(rows) - 1 if args.end_row <= 0 else min(args.end_row - 1, len(rows) - 1)
    concurrency = max(1, args.concurrency)

    jobs: list[tuple[int, str]] = []
    for row_idx in range(start_idx, end_idx + 1):
        row = ensure_two_columns(rows[row_idx])
        source_text = row[0].strip()

        if not source_text:
            row[1] = ""
            rows[row_idx] = row
            continue

        jobs.append((row_idx, source_text))

    total_jobs = len(jobs)
    inference_start_dt = datetime.now()
    inference_start_perf = time.perf_counter()

    if total_jobs > 0:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_job = {
                executor.submit(translate_text, model, tokenizer, source_text, args.max_new_tokens): (row_idx, source_text)
                for row_idx, source_text in jobs
            }

            completed = 0
            for future in as_completed(future_to_job):
                row_idx, source_text = future_to_job[future]
                try:
                    translated = future.result()
                except Exception as exc:
                    translated = ""
                    print(f"[{row_idx + 1}] translate failed: {exc}")

                row = ensure_two_columns(rows[row_idx])
                row[1] = translated
                rows[row_idx] = row
                completed += 1
                print(f"[{completed}/{total_jobs}] row {row_idx + 1}: {source_text} -> {translated}")

    inference_end_dt = datetime.now()
    inference_elapsed = time.perf_counter() - inference_start_perf

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Inference start: {inference_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Inference end:   {inference_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Inference elapsed: {format_duration(inference_elapsed)} ({inference_elapsed:.3f}s)")
    print(f"Done. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
