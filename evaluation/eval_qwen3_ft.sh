for category in category
do
    echo "Running evaluation for task: $category"
    python evaluation/eval.py \
        --output-path /workspace/QWEN3-VL/evaluation/ \
        --evaluation-category $category
done