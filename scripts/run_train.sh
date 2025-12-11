#!/bin/bash
set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tiser

# Train the model
python -u src/trainer.py \
    --train_file data/TISER_train.json \
    --model_name Qwen/Qwen2.5-7B \
    --output_dir models/tiser_qwen \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 2 \
    --lr 2e-5 \
    --max_length 1024 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05