# save as evaluation/eval_on_train_sample.py
import json
import torch
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info  # this is already in QWEN3-VL repo

device = "cuda"

base_model_id = "Qwen/Qwen3-VL-4B-Instruct"
adapter_dir = "/output/sft_qwen3_4b_carmel_vad5"
train_json = "/workspace/QWEN3-VL/datasets/CARMEL_VAD/carmel_vad_cat_training_with_system_5cat_mini.json"

with open(train_json, "r") as f:
    data = json.load(f)

sample = data[0]   # take first training example

sys_prompt = sample["system_prompt"]                      # exactly as in JSON
video_path = sample["video"]
question  = sample["conversations"][0]["value"]           # full question text
label     = sample["conversations"][1]["value"]           # category label

print("GT label:", repr(label))
print("Video path:", video_path)
print("Question:", repr(question))

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": sys_prompt}],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text",  "text": question},
        ],
    },
]

processor = AutoProcessor.from_pretrained(base_model_id)

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=prompt,
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt"
).to(device)

gen_kwargs = dict(
    max_new_tokens=8,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    top_k=1,
)

# --- Base model ---
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map={"": device},
)
base_model.eval()

with torch.no_grad():
    base_ids = base_model.generate(**inputs, **gen_kwargs)

base_out = processor.batch_decode(
    base_ids[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("\nBASE OUTPUT:", repr(base_out))

# --- LoRA model ---
lora_model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map={"": device},
)
lora_model = PeftModel.from_pretrained(lora_model, adapter_dir)
lora_model.eval()

with torch.no_grad():
    lora_ids = lora_model.generate(**inputs, **gen_kwargs)

lora_out = processor.batch_decode(
    lora_ids[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("LORA OUTPUT:", repr(lora_out))
