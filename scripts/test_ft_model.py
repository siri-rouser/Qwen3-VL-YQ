from transformers import AutoConfig, Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model_dir = "/output/sft_qwen3_4b_carmel_vad/pytorch_model.bin"

# config = AutoConfig.from_pretrained("/output/sft_qwen3_4b_av_tau")
# # Load model (handles multimodal reasoning)
# model = Qwen3VLForConditionalGeneration._from_config(config)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir,
    config=AutoConfig.from_pretrained("/output/sft_qwen3_4b_carmel_vad/config.json"),
    torch_dtype=torch.bfloat16,
    device_map="auto")

# Load processor (handles images, videos, text prompts, etc.)
processor = AutoProcessor.from_pretrained("/output/sft_qwen3_4b_carmel_vad")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/data/MedicalDrive-Rangeline-lowres/no7_cat23_sev4_downsize.mp4",
            },
            {"type": "text", "text": "Is there any anomaly in this video? If yes, please describe it."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)