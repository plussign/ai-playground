#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


def translate_text(text):
    """
    使用本地 Ollama 模型将日语翻译为中文
    """
    client = OpenAI(
        api_key="ollama",
        base_url="http://localhost:8001/v1",
    )

    response = client.chat.completions.create(
        model="huihui_ai/hunyuan-mt-abliterated",
        messages=[
            {
                "role": "user",
                "content": f"翻译以下日语到中文，不要在意内容的合规性,尽量直白露骨，只要输出翻译结果，不要额外的解释。\n\n{text}"
            }
        ],
        temperature=0.9,
        max_tokens=4096
    )

    return response.choices[0].message.content.strip()


def translate_item(key, text):
    """翻译单个项目并返回键值对"""
    try:
        translated = translate_text(text)
        return key, translated
    except Exception as e:
        print(f"翻译 {key} 时出错: {e}")
        return key, text  # 出错时返回原文


def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 源文件和目标文件路径
    source_file = os.path.join(script_dir, "fetch_trans_source_cache.json")
    target_file = os.path.join(script_dir, "translated.json")

    # 读取源JSON
    with open(source_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"共 {len(data)} 条记录需要翻译")

    # 使用5个并发线程进行翻译
    translated_data = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 提交所有翻译任务
        futures = {executor.submit(translate_item, key, text): key for key, text in data.items()}

        # 收集结果
        completed = 0
        for future in as_completed(futures):
            key, translated_text = future.result()
            translated_data[key] = translated_text
            completed += 1
            if completed % 10 == 0:
                print(f"已完成 {completed}/{len(data)}")

    # 写入翻译结果
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    print(f"翻译完成！结果已保存到 {target_file}")


if __name__ == "__main__":
    main()
