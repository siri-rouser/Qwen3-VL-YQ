from pathlib import Path
import json
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from QA_pair_database import QA_database
QAB = QA_database()

def inference(messages, model, processor):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

def eval_on_mini(model_dir, data_path):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_dir)

    with open(data_path, "r") as f:
        data = json.load(f)

    refs, preds = [], []
    for conv in data:
        system_prompt = conv["system_prompt"]          # exactly as in training
        video_path = conv["video"]
        question    = conv["conversations"][0]["value"]  # full string, no split()
        ref_ans     = conv["conversations"][1]["value"]

        msgs = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": question},
                ],
            },
        ]

        out = inference(msgs, model, processor)[0]
        refs.append(QAB.four_cat_to_index(ref_ans))
        preds.append(QAB.four_cat_to_index(out))
        print("REF:", ref_ans, "PRED_RAW:", out, "PRED_IDX:", preds[-1])

    # simple accuracy
    correct = sum(int(r == p) for r, p in zip(refs, preds))
    print(f"Train-set accuracy on mini JSON: {correct}/{len(refs)} = {correct/len(refs):.3f}")

if __name__ == "__main__":
    model_dir = "/output/sft_qwen3_4b_carmel_vad2"
    model_dir = "Qwen/Qwen3-VL-4B-Instruct"
    mini_data = "/workspace/QWEN3-VL/datasets/CARMEL_VAD/carmel_vad_cat_training_with_system_5cat_mini.json"
    eval_on_mini(model_dir, mini_data)
