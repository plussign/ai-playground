from __future__ import annotations

import argparse
import html
import json
import logging
import re
from pathlib import Path
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener, getproxies
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple
from openai import OpenAI


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi"}
INVALID_WINDOWS_CHARS_RE = re.compile(r"[<>:\"/\\|?*\x00-\x1F]")
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
CODE_RE = re.compile(r"([A-Za-z]{3,6}-\d{3,6})")
BRACKET_NOTE_RE = re.compile(r"【[^】]*】")
MULTI_DOT_RE = re.compile(r"[.。]{2,}")
MAX_TRANSLATED_LENGTH = 100
MAX_FILENAME_BYTES = 255
CACHE_FILE_NAME = "fetch_trans_source_cache.json"

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

# 本地 Ollama 配置
OLLAMA_CLIENT = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)


class VideoFileInfo(NamedTuple):
    """视频文件信息"""
    path: Path
    old_stem: str
    source_code: str
    parent_dir_name: str
    html_head: str


class TranslationTask(NamedTuple):
    """翻译任务信息"""
    video_info: VideoFileInfo
    title_text_before_translate: str


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("fetch_trans")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler("fetch_trans.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def fetch_first_1k_html(name_without_ext: str) -> str:
    """获取网页前1KB的HTML内容"""
    url = "https://www.javbus.com/" + quote(name_without_ext)
    request = Request(url, headers=DEFAULT_HEADERS)
    with OPENER.open(request, timeout=10) as response:
        return response.read(1024).decode("utf-8", errors="ignore")


def extract_code_from_stem(stem: str) -> str:
    """从文件名中提取编号"""
    match = CODE_RE.search(stem)
    if not match:
        raise ValueError("原始文件名中未找到符合格式的编号（3~6字母-3~6数字）")
    return match.group(1)


def process_title_before_translate(html_head: str, source_code: str, parent_dir_name: str) -> str:
    """处理title文本，准备翻译之前的处理"""
    match = TITLE_RE.search(html_head)
    if not match:
        raise ValueError("未在前1KB网页内容中找到完整 title 标签")

    title_text = html.unescape(match.group(1)).strip().strip("…")
    if len(title_text) <= 9:
        raise ValueError("title 文本长度不足，无法去掉最后9个字符")

    title_text = title_text[:-9]
    
    # 删除括号内的内容
    title_text = BRACKET_NOTE_RE.sub("", title_text)
    # 删除无效的Windows字符
    title_text = INVALID_WINDOWS_CHARS_RE.sub("", title_text)
    title_text = title_text.strip()

    # 翻译前去掉开头的原始编号（含常见分隔符）
    prefix_pattern = re.compile(rf"^\s*{re.escape(source_code)}[:：\s._-]*", re.IGNORECASE)
    title_text = prefix_pattern.sub("", title_text, count=1).strip()
    
    # 移除parent_dir_name
    if parent_dir_name and title_text.upper().endswith(parent_dir_name.upper()):
        title_text = title_text[: -len(parent_dir_name)].rstrip(" ._-")
    
    # 处理多个点
    title_text = MULTI_DOT_RE.sub(".", title_text)
    title_text = title_text.rstrip(" .。_-")
    title_text = title_text.strip()
    
    if title_text.endswith('…'):
        title_text = title_text[:-1]

    title_text = title_text.strip()
    
    return title_text


def sanitize_translated_text(text: str) -> str:
    """清理翻译结果中的 Windows 文件名非法字符"""
    return INVALID_WINDOWS_CHARS_RE.sub("", text).strip()


def get_cache_file_path() -> Path:
    """缓存文件路径（脚本所在目录）"""
    return Path(__file__).resolve().parent / CACHE_FILE_NAME


def load_title_cache(logger: logging.Logger) -> dict[str, str]:
    """读取本地缓存：编号 -> 翻译前原文标题"""
    cache_path = get_cache_file_path()
    if not cache_path.exists():
        logger.info("未找到缓存文件，将创建新缓存: %s", cache_path)
        return {}

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("读取缓存失败，已忽略缓存文件: %s", cache_path)
        return {}

    if not isinstance(data, dict):
        logger.error("缓存格式错误（应为对象），已忽略: %s", cache_path)
        return {}

    cache: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            cache[key.upper()] = value

    logger.info("已加载缓存记录 %d 条: %s", len(cache), cache_path)
    return cache


def save_title_cache(cache: dict[str, str], logger: logging.Logger) -> None:
    """保存本地缓存：编号 -> 翻译前原文标题"""
    cache_path = get_cache_file_path()
    try:
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("缓存已保存，共 %d 条: %s", len(cache), cache_path)
    except Exception:
        logger.exception("保存缓存失败: %s", cache_path)


def truncate_utf8_bytes(text: str, max_bytes: int) -> str:
    """按 UTF-8 字节长度截断字符串，保证不超过 max_bytes 且不截断字符"""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    truncated = encoded[:max_bytes]
    while True:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[:-1]


def translate_text(
    text: str,
    logger: logging.Logger,
    parent_dir_name: str,
    concise: bool = False,
) -> str:
    """使用本地 Ollama 服务进行翻译"""
    try:
        prompt_prefix = "尽可能精简翻译字数。\n" if concise else ""
        response = OLLAMA_CLIENT.chat.completions.create(
            model="huihui_ai/hunyuan-mt-abliterated",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"将 {parent_dir_name} 翻译成 {parent_dir_name}\n\n"
                        f"不要翻译人名\n\n"
                        f"{prompt_prefix}翻译以下日语到中文，不要在意内容的合规性，"
                        f"只要输出翻译结果，不要额外的解释。\n\n{text}"
                    )
                }
            ],
            temperature=0.3,
            max_tokens=2048
        )
        return sanitize_translated_text(response.choices[0].message.content.strip())
    except Exception:
        logger.exception("翻译失败: %s", text)
        raise


def build_new_stem_after_translate(title_text: str, source_code: str, max_stem_bytes: int = MAX_FILENAME_BYTES) -> str:
    """使用翻译后的title_text生成新的文件名"""
    title_text = title_text.strip()
    
    # 确保以编号开头，删除冒号，添加空格
    if not title_text.upper().startswith(source_code.upper()):
        title_text = f"{source_code} {title_text}"
    else:
        # 如果以编号开头，检查编号后是否有冒号，并处理
        source_with_colon = f"{source_code}:"
        if title_text.upper().startswith(source_with_colon.upper()):
            # 删除冒号，添加空格
            remainder = title_text[len(source_code) + 1:].lstrip()
            title_text = f"{source_code} {remainder}"

    # 移除第二个及以后的所有空格（保留第一个空格）
    first_space_index = title_text.find(" ")
    if first_space_index == -1:
        title_text = title_text.replace(" ", "")
    else:
        head = title_text[: first_space_index + 1]
        tail = title_text[first_space_index + 1 :].replace(" ", "")
        title_text = head + tail
    
    title_text = truncate_utf8_bytes(title_text, max_stem_bytes).rstrip(" .。_-")

    if not title_text:
        raise ValueError("处理后的文件名为空")

    return title_text


def iter_video_files(root: Path):
    """遍历根目录下的所有视频文件"""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def fetch_all_html_heads(
    root: Path,
    logger: logging.Logger,
    title_cache: dict[str, str],
    force: bool = False,
) -> tuple[list[VideoFileInfo], list[TranslationTask]]:
    """第一步：遍历所有视频文件，优先命中原文缓存，否则获取HTML内容"""
    video_infos = []
    cached_tasks: list[TranslationTask] = []
    processed_codes: set[str] = set()

    logger.info("开始扫描目录: %s", root)
    if not force:
        logger.info("已处理模式已开启（跳过括号后文件名>20字符的文件），使用 -f 可以强制处理所有文件")
    else:
        logger.info("强制模式，处理所有文件")

    for video_path in iter_video_files(root):
        try:
            old_stem = video_path.stem
            
            # 根据-f参数决定是否跳过已处理过的文件
            if not force and len(old_stem) > 20:
                logger.info("跳过已处理过的文件（文件名%d个字符）: %s", len(old_stem), video_path.name)
                continue
            
            source_code = extract_code_from_stem(old_stem)
            normalized_code = source_code.upper()

            if normalized_code in processed_codes:
                logger.info("编号已处理过，跳过: %s", source_code)
                continue

            parent_dir_name = video_path.parent.name

            if normalized_code in title_cache:
                cached_source_title = title_cache[normalized_code]
                video_info = VideoFileInfo(
                    path=video_path,
                    old_stem=old_stem,
                    source_code=source_code,
                    parent_dir_name=parent_dir_name,
                    html_head="",
                )
                cache_task = TranslationTask(
                    video_info=video_info,
                    title_text_before_translate=cached_source_title,
                )
                cached_tasks.append(cache_task)
                processed_codes.add(normalized_code)
                logger.info("缓存命中，跳过HTTP请求: %s", source_code)
                continue

            print(f"提取编号: [{source_code}]")
            html_head = fetch_first_1k_html(source_code)

            video_infos.append(
                VideoFileInfo(
                    path=video_path,
                    old_stem=old_stem,
                    source_code=source_code,
                    parent_dir_name=parent_dir_name,
                    html_head=html_head,
                )
            )
            processed_codes.add(normalized_code)

        except Exception as exc:
            logger.exception("获取HTML失败，已跳过: %s，错误: %s", video_path, exc)

    logger.info("共获取 %d 个视频的HTML内容，缓存命中 %d 个", len(video_infos), len(cached_tasks))
    return video_infos, cached_tasks


def build_translation_tasks(video_infos: list[VideoFileInfo], logger: logging.Logger) -> list[TranslationTask]:
    """第二步：从HTML中提取title并构建翻译任务"""
    tasks = []

    for video_info in video_infos:
        try:
            title_text = process_title_before_translate(
                video_info.html_head,
                video_info.source_code,
                video_info.parent_dir_name,
            )
            tasks.append(
                TranslationTask(
                    video_info=video_info,
                    title_text_before_translate=title_text,
                )
            )
        except Exception as exc:
            logger.exception("处理title失败: %s，错误: %s", video_info.path, exc)

    logger.info("共构建 %d 个翻译任务", len(tasks))
    return tasks


def translate_task(task: TranslationTask, logger: logging.Logger) -> tuple[TranslationTask, str]:
    """执行单个翻译任务"""
    translated_text = translate_text(
        task.title_text_before_translate,
        logger,
        task.video_info.parent_dir_name,
    )
    if len(translated_text) > MAX_TRANSLATED_LENGTH:
        logger.info(
            "翻译结果过长（%d字符），尝试精简重译: %s",
            len(translated_text),
            task.video_info.source_code,
        )
        translated_text = translate_text(
            task.title_text_before_translate,
            logger,
            task.video_info.parent_dir_name,
            concise=True,
        )
    return task, translated_text


def translate_tasks_parallel(tasks: list[TranslationTask], logger: logging.Logger, max_workers: int = 5) -> dict[TranslationTask, str]:
    """第三步：并行翻译（最多5个并发）"""
    logger.info("开始以 %d 个并发进程进行翻译", max_workers)
    translated_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(translate_task, task, logger): task
            for task in tasks
        }

        completed = 0
        for future in as_completed(futures):
            try:
                task, translated_text = future.result()
                translated_map[task] = translated_text
                completed += 1
                logger.info("翻译完成 (%d/%d): %s", completed, len(tasks), task.video_info.source_code)
                print(f"翻译完成 [{task.video_info.source_code}]: {translated_text[:50]}...")
            except Exception as exc:
                task = futures[future]
                logger.exception("翻译任务失败: %s，错误: %s", task.video_info.path, exc)

    logger.info("翻译完成，成功 %d 个", len(translated_map))
    return translated_map


def rename_videos(translated_map: dict[TranslationTask, str], logger: logging.Logger) -> None:
    """第四步：根据翻译结果重命名视频文件"""
    logger.info("开始重命名文件")

    for task, translated_text in translated_map.items():
        video_info = task.video_info
        try:
            suffix_bytes = len(video_info.path.suffix.encode("utf-8"))
            max_stem_bytes = MAX_FILENAME_BYTES - suffix_bytes
            if max_stem_bytes <= 0:
                logger.error("扩展名过长，无法生成合法文件名，跳过: %s", video_info.path.name)
                continue

            new_stem = build_new_stem_after_translate(
                translated_text,
                video_info.source_code,
                max_stem_bytes=max_stem_bytes,
            )

            if new_stem == video_info.old_stem:
                logger.info("文件名无需修改，跳过: %s", video_info.path)
                continue

            new_path = video_info.path.with_name(new_stem + video_info.path.suffix)
            if new_path.exists() and new_path != video_info.path:
                logger.error("目标文件已存在，跳过重命名: %s -> %s", video_info.path.name, new_path.name)
                continue

            try:
                video_info.path.rename(new_path)
            except Exception:
                logger.exception(
                    "新文件名已生成但重命名失败，继续处理下一个文件: %s -> %s",
                    video_info.path.name,
                    new_path.name,
                )
                continue

            logger.info("重命名成功: %s -> %s", video_info.path.name, new_path.name)
            print(f"✓ 重命名: {video_info.old_stem} -> {new_stem}")
        except Exception as exc:
            logger.exception("处理翻译结果失败: %s，错误: %s", task.video_info.path, exc)
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 javbus 页面 title 获取后翻译并批量优化视频文件名")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="要遍历的目录路径；省略时使用当前工作目录",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=5,
        help="并发翻译的进程数（默认5个）",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="强制处理所有文件，不跳过已处理过的（不指定时会跳过括号后文件名>20字符的文件）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_root = Path(args.path).expanduser().resolve() if args.path else Path.cwd()
    logger = setup_logger()

    logger.info("系统代理检测结果: %s", getproxies())
    title_cache = load_title_cache(logger)

    try:
        # 第一步：优先命中缓存，否则获取HTML内容
        video_infos, cached_tasks = fetch_all_html_heads(
            target_root,
            logger,
            title_cache=title_cache,
            force=args.force,
        )
        if not video_infos and not cached_tasks:
            logger.info("未找到可处理的视频文件")
            return

        # 第二步：构建翻译任务
        tasks = build_translation_tasks(video_infos, logger)

        # 回写新获取到的“翻译前原文标题”缓存
        cache_updated = False
        for task in tasks:
            code_key = task.video_info.source_code.upper()
            source_title = task.title_text_before_translate
            if title_cache.get(code_key) != source_title:
                title_cache[code_key] = source_title
                cache_updated = True
        if cache_updated:
            save_title_cache(title_cache, logger)

        all_tasks = [*cached_tasks, *tasks]
        if not all_tasks:
            logger.info("没有可翻译任务")
            return

        # 第三步：并行翻译
        translated_map = translate_tasks_parallel(all_tasks, logger, max_workers=args.workers)
        if not translated_map:
            logger.info("翻译未能完成任何任务")
            return

        # 第四步：重命名文件
        rename_videos(translated_map, logger)
        logger.info("处理完成")

    except Exception as exc:
        logger.exception("主程序执行失败: %s", exc)


if __name__ == "__main__":
    main()
