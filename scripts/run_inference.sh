#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tiser

MODEL_DIR="models/tiser_qwen"
TEST_FILE="data/TISER_test.json"

echo "[INFO] Running batch inference..."

python src/inference.py \
    --model_dir $MODEL_DIR \
    --file $TEST_FILE \
    --max_new_tokens 256

echo "[INFO] Predictions saved next to test file (test_pred.jsonl)."
