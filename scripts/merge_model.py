import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, AutoConfig
from peft import PeftModel

base_model_id = "Qwen/Qwen3-VL-4B-Instruct"
lora_model_dir = "/output/sft_qwen3_4b_carmel_vad5"
merged_model_dir = "/output/merged_qwen3_4b_carmel_vad5_merged"

# 1) Load base model (FP16/BF16 – avoid 4/8bit for merging)
config = AutoConfig.from_pretrained(base_model_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model_id,
    config=config,
    torch_dtype=torch.bfloat16,  # or torch.float16 / float32 depending on GPU memory
    device_map="auto",
)

# 2) Load LoRA adapter on top of base
model = PeftModel.from_pretrained(model, lora_model_dir)

# 3) Merge LoRA into the base weights
model = model.merge_and_unload()  # folds LoRA into model, removes adapter layers

# 4) Save merged model as a standard Transformers checkpoint
model.save_pretrained(merged_model_dir, safe_serialization=True)

# 5) Save processor/tokenizer (usually just copy from base or LoRA dir)
processor = Qwen3VLProcessor.from_pretrained(base_model_id)
processor.save_pretrained(merged_model_dir)

print("Merged model saved to:", merged_model_dir)
