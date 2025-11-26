for category in category
do
    echo "Running evaluation for task: $category"
    python evaluation/eval_lora.py \
        --output-path /workspace/QWEN3-VL/evaluation/ \
        --evaluation-category $category
done