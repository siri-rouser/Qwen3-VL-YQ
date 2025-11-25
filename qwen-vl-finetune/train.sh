#!/bin/bash

# Run the first script
echo "Starting sft_qwen3_4b.sh..."
bash scripts/sft_qwen3_4b.sh
FIRST_RESULT=$?

if [ $FIRST_RESULT -eq 0 ]; then
    echo "sft_qwen3_4b.sh completed successfully!"
else
    echo "sft_qwen3_4b.sh failed with exit code $FIRST_RESULT"
    exit $FIRST_RESULT
fi

# Run the second script
echo "Starting sft_qwen3_8b.sh..."
bash scripts/sft_qwen3_8b.sh
SECOND_RESULT=$?

if [ $SECOND_RESULT -eq 0 ]; then
    echo "sft_qwen3_8b.sh completed successfully!"
else
    echo "sft_qwen3_8b.sh failed with exit code $SECOND_RESULT"
    exit $SECOND_RESULT
fi

echo "Both scripts completed successfully!"
