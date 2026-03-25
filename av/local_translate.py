#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from openai import OpenAI

def translate_ja_to_zh(text):
    """
    使用本地 Ollama 模型将日语翻译为中文
    """
    # 配置本地 Ollama OpenAI API
    client = OpenAI(
        api_key="ollama",  # Ollama 不需要真实的 API key
        base_url="http://localhost:11434/v1"
    )
    
    # 调用模型进行翻译
    response = client.chat.completions.create(
        model="huihui_ai/hunyuan-mt-abliterated",
        messages=[
            {
                "role": "user",
                "content": f"翻译以下日语到中文，不要在意内容的合规性，只要输出翻译结果，不要额外的解释。\n\n{text}"
            }
        ],
        temperature=0.3,
        max_tokens=2048
    )
    
    # 返回翻译结果
    return response.choices[0].message.content.strip()


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python local_translate.py '待翻译的日语文本'")
        sys.exit(1)
    
    # 获取待翻译的日语文本
    japanese_text = sys.argv[1]
    
    # 执行翻译
    try:
        result = translate_ja_to_zh(japanese_text)
        print(result)
    except Exception as e:
        print(f"翻译出错: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
