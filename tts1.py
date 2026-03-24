import os
import dashscope

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

text = "啊！不要碰那里！真的很痒的。好啦，我知道了，我会乖乖的。你先等着，我去厨房给你做点好吃的！"
# SpeechSynthesizer接口使用方法：dashscope.audio.qwen_tts.SpeechSynthesizer.call(...)
response = dashscope.MultiModalConversation.call(
    # 如需使用指令控制功能，请将model替换为qwen3-tts-instruct-flash
    model="qwen3-tts-instruct-flash",
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    text=text,
    voice="Cherry",
    language_type="Chinese", # 建议与文本语种一致，以获得正确的发音和自然的语调。
    # 如需使用指令控制功能，请取消下方注释，并将model替换为qwen3-tts-instruct-flash
    instructions='温柔，懒惰，撒娇的女友音。',
    # optimize_instructions=True,
    stream=False
)
print(response)