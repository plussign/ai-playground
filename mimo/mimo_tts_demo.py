import base64
import os
import numpy as np
import soundfile as sf
from openai import OpenAI


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
            "content": "通过 [音频标签] ，你可以对声音进行细粒度控制，精准调节语气、情绪和表达风格——无论是低声耳语、放声大笑，还是带点小情绪的小吐槽，也可以灵活插入呼吸声，停顿，咳嗽等，都能轻松实现。语速同样可以灵活调整，让每句话都有它该有的节奏。"
        }
    ],
    audio={
        "format": "wav",
        "voice": "mimo_default"
    }
)

message = completion.choices[0].message
audio_bytes = base64.b64decode(message.audio.data)
with open("mimo_demo.wav", "wb") as f:
    f.write(audio_bytes)