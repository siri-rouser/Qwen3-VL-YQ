import json, os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


# Load dataset from Hugging Face
ds = load_dataset("harryhsing/AV-TAU")

json_list = []

for split in ["train", "test"]:
    print(f"Processing split: {split}, total samples: {len(ds[split])}")
    for item in tqdm(ds[split]):
        # Some samples have QA as list of dicts [{'q': '...', 'a': '...'}]
        qa_pair = item["QA"][0] if isinstance(item["QA"], list) else item["QA"]
        q = qa_pair["q"]
        a = qa_pair["a"]

        # 'video' can be either dict {'path': '...'} or direct string
        video_path = os.path.join(Path("/root/.cache/huggingface/datasets/AV-TAU"), item["video"]["path"] if isinstance(item["video"], dict) else item["video"])

        json_list.append({
            "video": video_path,
            "conversations": [
                {"from": "human", "value": f"<video>\n{q}"},
                {"from": "gpt", "value": a}
            ]
        })

# Save as QwenVL-style JSON
save_path = "/workspace/QWEN3-VL/datasets/AV-TAU/av_tau.json"
with open(save_path, "w") as f:
    json.dump(json_list, f, indent=2)

print(f"\n✅ Saved converted dataset to: {save_path}")
print(f"Total samples: {len(json_list)}")
