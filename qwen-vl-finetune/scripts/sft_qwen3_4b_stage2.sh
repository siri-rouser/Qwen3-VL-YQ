#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# DeepSpeed configuration
deepspeed=./scripts/zero3_gpt.json

# Model configuration
llm=Qwen/Qwen3-VL-4B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-4 # the suggested learning rate is from 1e-6 to 2e-7
batch_size=4
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets="carmel_vad_des"

# Output configuration
run_name="qwen3vl"
second_stage_llm=/output/sft_qwen3_4b_carmel_vad_context

# Training hyperparameters
lr=1e-4 # the suggested learning rate is from 1e-6 to 2e-7
batch_size=4
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets="carmel_vad_des"

# Output configuration
run_name="qwen3vl"
output_dir1=/output/sft_qwen3_4b_carmel_vad_second

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --resume_from_checkpoint "${second_stage_llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir1} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 384 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --run_name ${run_name} \
    --report_to wandb \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 512 \
    --lora_dropout 0.05"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}