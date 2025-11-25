import evaluate
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText
import torch
import argparse
from pathlib import Path
from peft import PeftModel
import json
from QA_pair_database import QA_database

QAB = QA_database()

def text_eval(references, predictions, output_path, category):
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')

    meteor_scores = meteor.compute(predictions=predictions, references=references)
    bleu_scores = bleu.compute(predictions=predictions, references=[[r] for r in references])
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    results = {
        "BLEU": bleu_scores["bleu"],
        "METEOR": meteor_scores["meteor"],
        "ROUGE-L": rouge_scores["rougeL"]
    }
    return results

def quantitative_eval(references, predictions):
    accuracy = evaluate.load('accuracy')
    predictions = [QAB.four_cat_to_index(pred) for pred in predictions]
    print("References:", references)
    print("Predictions:", predictions)
    results = {"Accuracy": accuracy.compute(predictions=predictions, references=references)["accuracy"]}
    return results

    # accuracy_score = accuracy.compute(predictions=predictions, references=references) # AUC only used for binary classification


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
    generated_ids = model.generate(**inputs, max_new_tokens=8)
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
    count = 0 

    with open(data_path, 'r') as f:
        data = json.load(f)
        for conversation in data:
            
            if count == 50:
                break
            messages = []
            video_path = conversation['video']
            question = conversation['conversations'][0]['value'].split("\n")[-1]
            ref_ans = conversation['conversations'][1]['value']
            question_cat = QAB.question_type_query(question)

            if question_cat != evaluation_category:
                continue
            
            if question_cat == "category":
                count += 1
                ref_ans_bak = ref_ans
                ref_ans = QAB.four_cat_to_index(ref_ans)
                messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": QAB.four_cat_context()}],
                    })  
                if ref_ans is None:
                    print(f"Warning: Unknown category '{ref_ans_bak}' in reference answer.")
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
            print(f"video_path: {video_path}")
            print(f"question: {messages}")
            print(f"output_text: {output_text}")
        
        # Evaluation
        if evaluation_category == "description" or evaluation_category == "analysis":
            results = text_eval(reference, prediction, output_path, evaluation_category)
            with open(output_path / f"evaluation_results_qwen3_4b.txt", 'a') as out_file:
                out_file.write(f"\n{'='*40}\n")
                out_file.write(f"Category: {evaluation_category}\n")
                out_file.write(f"{'='*40}\n")
                out_file.write(f"For text-based evaluation:\n")
                out_file.write(json.dumps(results, indent=4))
                out_file.write("\n\n")

        if evaluation_category == "category":
            results = quantitative_eval(reference, prediction)
            with open(output_path / f"evaluation_results_qwen3_4b.txt", 'a') as out_file:
                out_file.write(f"\n{'='*40}\n")
                out_file.write(f"Category: {evaluation_category}\n")
                out_file.write(f"{'='*40}\n")
                out_file.write(f"For quantitative evaluation:\n")
                out_file.write(json.dumps(results, indent=4))
                out_file.write("\n\n")

if __name__ == "__main__":
    test_data_path = "/workspace/QWEN3-VL/datasets/CARMEL_VAD/carmel_vad_test_rewritten.json"
    training_data_path = "/workspace/QWEN3-VL/datasets/CARMEL_VAD/carmel_vad_cat_training_with_system_5cat.json"
    mini_trainig_data_path = "/workspace/QWEN3-VL/datasets/CARMEL_VAD/carmel_vad_cat_training_with_system_5cat_mini.json"
    base_model_id = "Qwen/Qwen3-VL-4B-Instruct"
    adapter_dir = "/output/sft_qwen3_4b_carmel_vad2"
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
    processor = AutoProcessor.from_pretrained("/output/sft_qwen3_4b_carmel_vad2")
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

    eval_main(model=model, processor=processor, data_path=mini_trainig_data_path, evaluation_category=args.evaluation_category, output_path=args.output_path)