from __future__ import annotations

import argparse
import html
import logging
import re
from pathlib import Path
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener, getproxies



VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi"}
INVALID_WINDOWS_CHARS_RE = re.compile(r"[<>:\"/\\|?*\x00-\x1F]")
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
CODE_RE = re.compile(r"([A-Za-z\d]{3,8}-\d{3,8})")
BRACKET_NOTE_RE = re.compile(r"【[^】]*】")
MULTI_DOT_RE = re.compile(r"[.。]{2,}")


OPENER = build_opener(ProxyHandler())
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
    "Range": "bytes=0-1023",
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("fetch_filename")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler("fetch_filename.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def fetch_first_1k_html(name_without_ext: str) -> str:
    url = "https://www.javbus.com/" + quote(name_without_ext)
    request = Request(url, headers=DEFAULT_HEADERS)
    with OPENER.open(request, timeout=10) as response:
        return response.read(1024).decode("utf-8", errors="ignore")


def extract_code_from_stem(stem: str) -> str:
    match = CODE_RE.search(stem)
    if not match:
        raise ValueError("原始文件名中未找到符合格式的编号（3~6字母-3~6数字）")
    return match.group(1)


def build_new_stem(html_head: str, source_code: str, parent_dir_name: str) -> str:
    match = TITLE_RE.search(html_head)
    if not match:
        raise ValueError("未在前1KB网页内容中找到完整 title 标签")

    title_text = html.unescape(match.group(1)).strip().strip("…")
    if len(title_text) <= 9:
        raise ValueError("title 文本长度不足，无法去掉最后9个字符")

    title_text = title_text[:-9]

    title_text = BRACKET_NOTE_RE.sub("", title_text)
    title_text = INVALID_WINDOWS_CHARS_RE.sub("", title_text)
    title_text = title_text.strip()

    if not title_text.upper().startswith(source_code.upper()):
        title_text = f"{source_code} {title_text}".strip()

    if parent_dir_name and title_text.upper().endswith(parent_dir_name.upper()):
        title_text = title_text[: -len(parent_dir_name)].rstrip(" ._-")

    title_text = MULTI_DOT_RE.sub(".", title_text)
    title_text = title_text.rstrip(" .。_-")
    title_text = title_text.strip()

    if title_text.endswith('…'):
        title_text = title_text[: -1]

    title_text = title_text[:230]

    if not title_text:
        raise ValueError("处理后的文件名为空")

    return title_text


def iter_video_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 javbus 页面 title 批量优化视频文件名")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="要遍历的目录路径；省略时使用当前工作目录",
    )
    return parser.parse_args()


def rename_videos_in_current_dir(root: Path) -> None:
    logger = setup_logger()
    processed_codes: set[str] = set()
    logger.info("开始扫描目录: %s", root)
    logger.info("系统代理检测结果: %s", getproxies())

    for video_path in iter_video_files(root):
        try:
            old_stem = video_path.stem
            source_name = extract_code_from_stem(old_stem)

            if source_name in processed_codes:
                logger.info("编号已成功处理过，跳过: %s -> %s", source_name, video_path)
                continue
            
            print(f"提取编号: [{source_name}]")

            html_head = fetch_first_1k_html(source_name)
            parent_dir_name = video_path.parent.name
            new_stem = build_new_stem(html_head, source_name, parent_dir_name)
            
            if new_stem == old_stem:
                logger.info("文件名无需修改，跳过: %s", video_path)
                continue

            new_path = video_path.with_name(new_stem + video_path.suffix)
            try:
                video_path.rename(new_path)
                processed_codes.add(source_name)
                logger.info("重命名成功: %s -> %s", video_path.name, new_path.name)
            except Exception:
                processed_codes.add(source_name)
                logger.exception(
                    "新文件名已生成但重命名失败，编号已加入跳过列表: %s，文件: %s",
                    source_name,
                    video_path,
                )
        except Exception as exc:
            logger.exception("处理失败，已跳过: %s，错误: %s", video_path, exc)

    logger.info("处理完成")


if __name__ == "__main__":
    args = parse_args()
    target_root = Path(args.path).expanduser().resolve() if args.path else Path.cwd()
    rename_videos_in_current_dir(target_root)