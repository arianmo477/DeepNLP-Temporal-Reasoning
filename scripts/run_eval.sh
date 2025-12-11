#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tiser

PRED_FILE="data/test_pred.jsonl"

echo "[INFO] Running evaluation..."

python src/evaluator.py \
    --pred_file $PRED_FILE \
    --output eval_results.txt

echo "[INFO] Evaluation complete. Results in eval_results.txt"
