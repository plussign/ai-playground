import os

from modelscope import snapshot_download
from modelscope import AutoConfig, AutoModelForCausalLM, AutoTokenizer

modelscope_model_id = "Tencent-Hunyuan/HY-MT1.5-7B-FP8"

model_dir = snapshot_download(modelscope_model_id)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
config.tie_word_embeddings = False
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    #config=config,
    device_map="auto",
    trust_remote_code=True,
)  # You may want to use bfloat16 and/or move to GPU here
messages = [
    {"role": "user", "content": "参考下面的翻译：\n居民 翻译成 NEKO\n猫娘 翻译成 Neko\n\nTranslate the following segment into English, without additional explanation.请注意保留Html标签为原样，以及注意$Pn模式的标记是待动态填入的占位符。\n\n需要翻译的example。"},
]
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
)

tokenized_chat = tokenized_chat.to(model.device)
outputs = model.generate(**tokenized_chat, max_new_tokens=2048)
input_len = tokenized_chat["input_ids"].shape[1]
new_tokens = outputs[0][input_len:]
output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
print(output_text)