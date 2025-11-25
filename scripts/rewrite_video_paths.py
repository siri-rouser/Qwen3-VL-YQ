#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List

# PATTERN = re.compile(r'^OTA/([^/]+)/testdata_selected/videos/(.+\.mp4)$')
PATTERN = re.compile(r'^([^_]+)_[^/]+\.mp4$')

def rewrite_path(p: str) -> str:
    m = PATTERN.match(p.strip())
    if not m:
        return p  # leave unchanged if it doesn't match expected pattern
    site =  m.group(1)
    return f"/data/OTA/{site}/{p}"

def main(
    in_path: str = "carmel_vad.json",
    out_path: str = "carmel_vad_rewritten.json",
    key: str = "video",
) -> None:
    src = Path(in_path)
    dst = Path(out_path)

    with src.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array at the root.")

    changed = 0
    for i, item in enumerate(data):
        if isinstance(item, dict) and key in item and isinstance(item[key], str):
            old = item[key]
            new = rewrite_path(old)
            if new != old:
                data[i][key] = new
                changed += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Rewrote {changed} path(s).")
    print(f"Output written to: {dst}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite video paths in JSON file")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="./datasets/CARMEL_VAD/carmel_vad_selected.json",
        help="Input JSON file path (default: ./datasets/CARMEL_VAD/carmel_vad_selected.json)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./datasets/CARMEL_VAD/carmel_vad_selected_rewritten.json",
        help="Output JSON file path (default: ./datasets/CARMEL_VAD/carmel_vad_selected_rewritten.json)"
    )
    parser.add_argument(
        "--key",
        "-k",
        type=str,
        default="video",
        help="JSON key to rewrite (default: video)"
    )
    
    args = parser.parse_args()
    
    main(
        in_path=args.input,
        out_path=args.output,
        key=args.key,
    )
