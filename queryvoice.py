from dashscope.audio.tts_v2 import VoiceEnrollmentService

service = VoiceEnrollmentService()

# 按前缀筛选，或设为None查询所有
voices = service.list_voices(prefix=None, page_index=0, page_size=10)

print(f"Request ID: {service.get_last_request_id()}")
print(f"Found voices: {voices}")