import evaluate
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText
import torch
import argparse
from pathlib import Path
from peft import PeftModel
import json
from QA_pair_database import QA_database
from GPT_Eval import eval_main as g_eval

QAB = QA_database()
GPT_MODEL_ID = "gpt-5.1-2025-11-13"

def text_eval(references, predictions, output_path, category):
    env_score = []
    groudning_score = []
    desc_acc_score = []
    reasoning_score = []
    total_score = []

    with open(output_path / f"evaluation_details_{category}.txt", 'w') as out_file:
        for i in range(len(references)):
            ref = references[i]
            pred = predictions[i]
            results = g_eval(GPT_MODEL_ID, pred, ref)
            print(results)
            out_file.write(f"Sample {i+1}:\n")
            out_file.write(f"Ground Truth: {ref}\n")
            out_file.write(f"Prediction: {pred}\n")
            out_file.write(f"Evaluation Results: {results}\n")
            out_file.write("\n")
            # Process the text JSON result
            if isinstance(results, str):
                try:
                    results = json.loads(results)
                except json.JSONDecodeError:
                # If it's already formatted text or invalid JSON, keep as is
                    raise ValueError(f"Invalid JSON format: {results}")
            
            env_score.append(results["environment_score"])
            groudning_score.append(results["grounding_score"])
            desc_acc_score.append(results["description_accuracy_score"])
            reasoning_score.append(results["reasoning_score"])
            total_score.append(results["total_score"])

        out_file.write(f"\n{'='*40}\n")
        out_file.write(f"Summary for category: {category}\n")
        out_file.write(f"{'='*40}\n")
        summary = {
            "environment_score_avg": sum(env_score) / len(env_score),
            "grounding_score_avg": sum(groudning_score) / len(groudning_score),
            "description_accuracy_score_avg": sum(desc_acc_score) / len(desc_acc_score),
            "reasoning_score_avg": sum(reasoning_score) / len(reasoning_score),
            "total_score_avg": sum(total_score) / len(total_score),
        }
        out_file.write(json.dumps(summary, indent=4))
        out_file.write("\n")
                
def inference(messages, model, processor):
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
    return output_text

def eval_main(model, processor, data_path, evaluation_category, output_path):
    # Load evaluation dataset
    reference = []
    prediction = []

    with open(data_path, 'r') as f:
        data = json.load(f)
        for conversation in data:
            messages = []
            video_path = conversation['video']
            question = conversation['conversations'][0]['value'].split("\n")[-1]
            ref_ans = conversation['conversations'][1]['value']
            question_cat = QAB.question_type_query(question)

            if question_cat != evaluation_category:
                continue
            
            if question_cat == "description":
                question = "Do you see any anomalous driving behavior in the video? If yes, please describe it."

            reference.append(ref_ans)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                        },
                        {"type": "text", "text": question},
                    ],
                })
            
            output_text = inference(messages, model, processor)
            prediction.append(output_text[0])
            print(f"question: {question}")
            print(f"output_text: {output_text}")
        
        # Evaluation
        if evaluation_category == "description":
            results = text_eval(reference, prediction, output_path, evaluation_category)
            with open(output_path / f"evaluation_results_qwen3_8b.txt", 'a') as out_file:
                out_file.write(f"\n{'='*40}\n")
                out_file.write(f"Category: {evaluation_category}\n")
                out_file.write(f"{'='*40}\n")
                out_file.write(f"For text-based evaluation:\n")
                out_file.write(json.dumps(results, indent=4))
                out_file.write("\n\n")

if __name__ == "__main__":
    data_path = "/workspace/QWEN3-VL/datasets/CARMEL_VAD/carmel_vad_test_rewritten.json"
    base_model_id = "Qwen/Qwen3-VL-4B-Instruct"
    adapter_dir = "/output/sft_qwen3_4b_carmel_vad_second"
    parser = argparse.ArgumentParser(description="Evaluate Qwen3VL Model")
    parser.add_argument("--output-path","-o", type=Path, required=True, help="Path to save evaluation results")
    parser.add_argument("--evaluation-category","-e", type=str,
    default="description",
    choices=['description','analysis','severity','category','classification'],
    help="Evaluation category.")

    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
    )

    model.eval()

    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen3-VL-4B-Instruct",
    #     dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # Load processor (handles images, videos, text prompts, etc.)
    processor = AutoProcessor.from_pretrained("/output/sft_qwen3_4b_carmel_vad_second")
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    eval_main(model=model, processor=processor, data_path=data_path, evaluation_category=args.evaluation_category, output_path=args.output_path)