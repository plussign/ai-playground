# coding=utf-8
# Installation instructions for pyaudio:
# Microsoft Windows
#   python -m pip install pyaudio

import os
from urllib import response
import pyaudio
import dashscope
from dashscope.audio.tts_v2 import *


from http import HTTPStatus
from dashscope import Generation

# 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
# 若没有配置环境变量，请用百炼API Key将下行替换为：dashscope.api_key = "sk-xxx"
dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference
dashscope.base_websocket_api_url='wss://dashscope.aliyuncs.com/api-ws/v1/inference'

# 不同模型版本需要使用对应版本的音色：
# cosyvoice-v3-flash/cosyvoice-v3-plus：使用longanyang等音色。
# cosyvoice-v2：使用longxiaochun_v2等音色。
# 每个音色支持的语言不同，合成日语、韩语等非中文语言时，需选择支持对应语言的音色。详见CosyVoice音色列表。
model = "cosyvoice-v3.5-plus"
voice = "cosyvoice-v3.5-plus-theresa-cd03f9694851482282d20399a89c681b"


class Callback(ResultCallback):
    _player = None
    _stream = None

    def on_open(self):
        print("websocket is open.")
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=44100, output=True
        )

    def on_complete(self):
        print("speech synthesis task complete successfully.")

    def on_error(self, message: str):
        print(f"speech synthesis task failed, {message}")

    def on_close(self):
        print("websocket is closed.")
        # stop player
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()

    def on_event(self, message):
        print(f"recv speech synthsis message {message}")

    def on_data(self, data: bytes) -> None:
        print("audio result length:", len(data))
        self._stream.write(data)


def synthesizer_with_llm():
    callback = Callback()
    synthesizer = SpeechSynthesizer(
        model=model,
        voice=voice,
        format=AudioFormat.PCM_44100HZ_MONO_16BIT,
        callback=callback,
    )
    #synthesizer.streaming_call("哇~[b]这么漂亮的衣服，是给我的吗？<speak effect=\"lolita\">太喜欢了！</speak>")
    #synthesizer.streaming_complete()
    synthesizer.call("哇~[b]我不但夺舍了德莉莎的声音，还得到了这么漂亮的衣服？太开心了！等等，我本名叫啥来着？")
    print('requestId: ', synthesizer.get_last_request_id())


if __name__ == "__main__":
    synthesizer_with_llm()