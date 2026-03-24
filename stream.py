import os
from dashscope import MultiModalConversation
import dashscope

# 若使用新加坡地域的模型，请取消下列注释
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '梭曼中毒为什么比沙林危险？'}
]

responses = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model='qwen3.5-plus',  # 可按需更换为其它多模态模型，并修改相应的 messages
    messages=messages,
    stream=True,
    incremental_output=True
    )
    
full_content = ""
print("流式输出内容为：")
for response in responses:
    if response.output.choices[0].message.content:
        print(response.output.choices[0].message.content[0]['text'])
        full_content += response.output.choices[0].message.content[0]['text']
print(f"完整内容为：{full_content}")