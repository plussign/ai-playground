#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi"}
CODE_RE = re.compile(r"([A-Za-z\d]{2,6}-\d{3,6})")
STATUS_FILE_NAME = "files_status.json"


def extract_code_from_stem(stem: str) -> str:
    """从文件名中提取编号"""
    stem = stem.upper()
    if stem.find("HEYZO_HD_") != -1:
        stem = stem.replace("HEYZO_HD_", "HEYZO-")[:10]
    match = CODE_RE.search(stem)
    if not match:
        raise ValueError("原始文件名中未找到符合格式的编号（3~6字母-3~6数字）")
    return match.group(1)


def iter_video_files(root: Path):
    """遍历根目录下的所有视频文件"""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def get_status_file_path() -> Path:
    """状态文件路径（脚本所在目录）"""
    return Path(__file__).resolve().parent / STATUS_FILE_NAME


def sync_av_files(root: Path) -> dict[str, str]:
    """同步视频文件，返回编号到相对目录路径的映射"""
    files_map: dict[str, str] = {}

    print(f"开始扫描目录: {root}")

    for video_path in iter_video_files(root):
        try:
            source_code = extract_code_from_stem(video_path.stem)
            normalized_code = source_code.upper()

            if normalized_code in files_map:
                print(f"警告: 编号重复，跳过: {normalized_code}")
                continue

            # 获取相对于根目录的目录路径（不含文件名）
            relative_dir = video_path.parent.relative_to(root)
            files_map[normalized_code] = str(relative_dir)
            print(f"添加: {normalized_code} -> {relative_dir}")

        except Exception as exc:
            print(f"处理文件失败: {video_path}，错误: {exc}")
            continue

    print(f"扫描完成，共找到 {len(files_map)} 个视频文件")
    return files_map


def save_files_status(files_map: dict[str, str]) -> None:
    """保存文件状态到JSON"""
    status_path = get_status_file_path()
    try:
        status_path.write_text(
            json.dumps(files_map, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"状态已保存: {status_path}")
    except Exception as exc:
        print(f"保存状态失败: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同步视频文件列表到files_status.json")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="要遍历的目录路径；省略时使用当前工作目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_root = Path(args.path).expanduser().resolve() if args.path else Path.cwd()

    files_map = sync_av_files(target_root)
    if files_map:
        save_files_status(files_map)
    else:
        print("未找到任何视频文件")


if __name__ == "__main__":
    main()
