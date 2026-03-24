import os
from openai import OpenAI
import base64

client = OpenAI(
    api_key="sk-cfbsyiff06wjmifljcxdln82fsf3p17db54e68w8hbngbq11",
    base_url="https://api.xiaomimimo.com/v1"
)

completion = client.chat.completions.create(
    model="mimo-v2-tts",
    messages=[
        {
            "role": "user",
            "content": "请朗读科普文章。"
        },
        {
            "role": "assistant",
            "content": "大姐！这鱼新鲜着呢！早上刚捞上来的！哎！那个谁，别乱翻，压坏了你赔啊？！"
        }
    ],
    audio={
        "format": "wav",
        "voice": "mimo_default"
    }
)

message = completion.choices[0].message
audio_bytes = base64.b64decode(message.audio.data)
with open("audio_file.wav", "wb") as f:
    f.write(audio_bytes)
