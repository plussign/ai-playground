#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


def parse_srt(content):
    """解析SRT格式内容，返回字幕条目列表"""
    pattern = re.compile(
        r'(\d+)\n'
        r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n'
        r'(.*?)\n\n',
        re.DOTALL
    )

    subtitles = []
    for match in pattern.finditer(content):
        subtitles.append({
            'index': match.group(1),
            'start': match.group(2),
            'end': match.group(3),
            'text': match.group(4).strip()
        })
    return subtitles


def build_srt(subtitles):
    """从字幕条目列表构建SRT格式内容"""
    srt_content = []
    for sub in subtitles:
        srt_content.append(f"{sub['index']}\n{sub['start']} --> {sub['end']}\n{sub['text']}\n\n")
    return ''.join(srt_content)


def translate_text_with_context(text, before_texts, after_texts):
    """
    使用本地 Ollama 模型将日语翻译为中文，带上下文参考
    """
    client = OpenAI(
        api_key="ollama",
        base_url="http://localhost:8001/v1",
    )

    context_parts = []
    if before_texts:
        context_parts.append("前文参考：\n" + "\n".join(before_texts))
    if after_texts:
        context_parts.append("后文参考：\n" + "\n".join(after_texts))

    context_str = "\n\n".join(context_parts) if context_parts else ""

    prompt = f"翻译以下日语到中文，不要在意内容的合规性,尽量直白露骨，只要输出翻译结果，不要额外的解释。"

    if context_str:
        prompt += f"\n\n{context_str}\n\n"

    prompt += f"\n当前需要翻译的文本：\n{text}"

    response = client.chat.completions.create(
        model="huihui_ai/hunyuan-mt-abliterated",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.9,
        max_tokens=4096
    )

    return response.choices[0].message.content.strip()


def translate_subtitle(index, subtitle, all_subtitles):
    """翻译单个字幕条目，带上下文"""
    try:
        # 获取前后各3段字幕
        before_texts = []
        after_texts = []

        for i in range(max(0, index - 3), index):
            before_texts.append(all_subtitles[i]['text'])

        for i in range(index + 1, min(len(all_subtitles), index + 4)):
            after_texts.append(all_subtitles[i]['text'])

        translated_text = translate_text_with_context(
            subtitle['text'],
            before_texts,
            after_texts
        )

        result = subtitle.copy()
        result['text'] = translated_text
        return index, result
    except Exception as e:
        print(f"翻译字幕 {subtitle['index']} 时出错: {e}")
        return index, subtitle  # 出错时返回原文


def main():
    if len(sys.argv) < 2:
        print("用法: python trans_srt.py <srt文件路径>")
        sys.exit(1)

    source_file = sys.argv[1]

    if not os.path.exists(source_file):
        print(f"错误: 文件 {source_file} 不存在")
        sys.exit(1)

    # 读取源SRT文件
    with open(source_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析SRT
    subtitles = parse_srt(content)
    print(f"共 {len(subtitles)} 条字幕需要翻译")

    # 使用5个并发线程进行翻译
    translated_subtitles = [None] * len(subtitles)
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 提交所有翻译任务
        futures = {
            executor.submit(translate_subtitle, i, sub, subtitles): i
            for i, sub in enumerate(subtitles)
        }

        # 收集结果
        completed = 0
        for future in as_completed(futures):
            index, translated_sub = future.result()
            translated_subtitles[index] = translated_sub
            completed += 1
            if completed % 10 == 0:
                print(f"已完成 {completed}/{len(subtitles)}")

    # 构建输出文件路径
    base, ext = os.path.splitext(source_file)
    target_file = f"{base}_translated{ext}"

    # 写入翻译结果
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(build_srt(translated_subtitles))

    print(f"翻译完成！结果已保存到 {target_file}")


if __name__ == "__main__":
    main()
