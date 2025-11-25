import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Add system prompt to JSON training data")
parser.add_argument("--input", type=Path, default=Path("carmel_vad_cat_training.json"), help="Input JSON file path")
parser.add_argument("--output", type=Path, default=Path("carmel_vad_cat_training_with_system.json"), help="Output JSON file path")
args = parser.parse_args()

INPUT_PATH = args.input
OUTPUT_PATH = args.output

SYSTEM_PROMPT = """You are a classification system for identifying anomalous driving behaviors.
There are 4 predefined anomaly categories listed below.
1: "speed_trajectory_irregularities": Abnormal speed choice or unstable movement patterns that raise risk (or clearly deviate from “normal” driving), even if they don’t directly create a near-collision event.
2: "direction_space_violations": The main problem is where the car is relative to legal lanes/roadway/sidewalk, including wrong-way, illegal turns, and using space not meant for vehicles.
3: "conflict_near_collision": Events where interaction with another road user becomes critical: one vehicle clearly cuts off another, a near crash occurs, or emergency maneuvers (hard braking, swerving) are taken to avoid collision.
4: "stopped_obstruction_right_of_way": Vehicles that are stopped or nearly stopped in the roadway for special reasons—emergency vehicles, breakdowns, letting pedestrians cross—or that act as an obstruction."""

def main():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # data is a list of samples
    for sample in data:
        # Add the system_prompt field (will overwrite if already exists)
        sample["system_prompt"] = SYSTEM_PROMPT

        # (optional) reorder keys so video + system_prompt come first
        if "video" in sample and "conversations" in sample:
            sample_ordered = {
                "video": sample["video"],
                "system_prompt": sample["system_prompt"],
                "conversations": sample["conversations"],
            }
            sample.clear()
            sample.update(sample_ordered)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Written updated file to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
