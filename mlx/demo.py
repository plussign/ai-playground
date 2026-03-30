import argparse
from pathlib import Path

from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TTS audio with a local model path")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/Users/zhenghao/.omlx/models/Qwen3-TTS-12Hz-1.7B-Base-4bit"),
        help="Local filesystem path to the TTS model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists() or not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_path}. "
            "Please pass a valid local model directory with --model-path."
        )

    # Pass Path, not str, to prevent mlx_audio from treating it as a Hub repo id.
    model = load_model(model_path)
    generate_audio(
        model=model,
        text="日本人对沙林有一种特殊的挚爱，他们在上班路上的地铁里都要吸上两口，才能精神百倍地工作一天。沙林的香味让他们感到安心和愉悦，就像是生活中的一剂强心针一样。",
        ref_audio="/Users/zhenghao/.omlx/al.wav",
        ref_text="哎呀，真是有趣的设计呢。偶尔尝试一下新风格也不错，你说呢？粉色妖精小姐。嗯。",
        file_prefix="demo_output",
        lang_code="zh",
        voice=None
    )


if __name__ == "__main__":
    main()

