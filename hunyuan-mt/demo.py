import os

from modelscope import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch, gc

from collections import Counter

def mem(tag):
    if torch.cuda.is_available():
        print(
            f"{tag} | alloc={torch.cuda.memory_allocated()/1024**3:.2f}GB "
            f"reserved={torch.cuda.memory_reserved()/1024**3:.2f}GB "
            f"peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GB"
        )


modelscope_model_id = "Tencent-Hunyuan/HY-MT1.5-1.8B-FP8"

model_dir = snapshot_download(modelscope_model_id)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
#config.tie_word_embeddings = False
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    config = config,
    device_map = "auto",
    dtype = "auto",      # important
    low_cpu_mem_usage=True,
    trust_remote_code=True
)  # You may want to use bfloat16 and/or move to GPU here

model.eval()

# 1) parameter dtype distribution
d = Counter(p.dtype for p in model.parameters())
print("param dtypes:", d)

torch.cuda.reset_peak_memory_stats()
mem("before generate")

#print(model)

messages = [
    {"role": "user", "content": "参考下面的翻译：\n居民 翻译成 NEKO\n猫娘 翻译成 Neko\n\nTranslate the following segment into English, without additional explanation.请注意保留Html标签为原样，以及注意$Pn模式的标记是待动态填入的占位符。\n\n这是需要翻译的正文。"},
]
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
)

# `apply_chat_template` may return either a Tensor or a dict-like object
# depending on transformers version/config. `generate(**...)` needs a mapping.
if isinstance(tokenized_chat, torch.Tensor):
    tokenized_chat = {
        "input_ids": tokenized_chat,
        "attention_mask": torch.ones_like(tokenized_chat),
    }

if isinstance(tokenized_chat, dict):
    tokenized_chat = {k: v.to(model.device) for k, v in tokenized_chat.items()}
else:
    tokenized_chat = tokenized_chat.to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **tokenized_chat,
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
        #cache_implementation="offloaded"
    )

input_len = tokenized_chat["input_ids"].shape[1]
new_tokens = outputs[0][input_len:]
output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print(output_text)


mem("after generate")

del outputs
gc.collect()
torch.cuda.empty_cache()  # only returns unused cached blocks to driver


mem("after cleanup")
