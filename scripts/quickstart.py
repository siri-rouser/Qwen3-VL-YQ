from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import torch

# 1. Paths / IDs
base_model_id = "Qwen/Qwen3-VL-4B-Instruct"
adapter_dir = "/output/sft_qwen3_4b_carmel_vad1"

# 2. Load base model (same as used for training)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# # 3. Load LoRA adapter on top of the base model
# model = PeftModel.from_pretrained(
#     model,
#     adapter_dir,
# )

# model.eval()

# 4. Processor / tokenizer
# You can load from the adapter_dir (it has tokenizer & processor_config),
# or from the base_model_id. Here I use adapter_dir to keep things consistent.
# processor = Qwen3VLProcessor.from_pretrained("/output/sft_qwen3_4b_carmel_vad1")
processor = Qwen3VLProcessor.from_pretrained(base_model_id)
# 5. Your messages
text_cat = """
        #   You are a classification system for identifying anomalous driving behaviors.  
            There are **predefined anomaly categories** listed below.  
            1: "speed_trajectory_irregularities":Abnormal speed choice or unstable movement patterns that raise risk (or clearly deviate from “normal” driving), even if they don’t directly create a near-collision event.
            2: "direction_space_violations":Violations of intended direction of travel or legal use of space, such as entering the oncoming lane, performing illegal turns, or occupying sidewalks.
            3: "conflict_near_collision":Events where interaction with another road user becomes critical: one vehicle clearly cuts off another, a near crash occurs, or emergency maneuvers (hard braking, swerving) are taken to avoid collision.
            4: "stopped_obstruction_right_of_way":Vehicles that are stopped or nearly stopped in the roadway for special reasons—emergency vehicles, breakdowns, letting pedestrians cross—or that act as an obstruction.
            
Could you provide the classification of the anomaly in the video? and explain why you choose this category.
"""
text_cat1 = """
You are a classification system for identifying anomalous driving behaviors. \nThere are 4 predefined anomaly categories listed below. \n1: \"speed_trajectory_irregularities\": Abnormal speed choice or unstable movement patterns that raise risk (or clearly deviate from “normal” driving), even if they don’t directly create a near-collision event. \n2: \"direction_space_violations\": The main problem is where the car is relative to legal lanes/roadway/sidewalk, including wrong-way, illegal turns, and using space not meant for vehicles. \n3: \"conflict_near_collision\": Events where interaction with another road user becomes critical: one vehicle clearly cuts off another, a near crash occurs, or emergency maneuvers (hard braking, swerving) are taken to avoid collision. \n4: \"stopped_obstruction_right_of_way\": Vehicles that are stopped or nearly stopped in the roadway for special reasons—emergency vehicles, breakdowns, letting pedestrians cross—or that act as an obstruction. \n\n

Review the video to identify anomalies, and explain why you choose this category.
"""


text_des = """Do you see any anomalous driving behavior in the video? If yes, please describe it."""

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant that helps people find information."}
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/data/RangelineCityCenterSB-lowres/no2_cat23_sev4_downsize.mp4",
            },
            {"type": "text", "text": text_cat},
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