#!/usr/bin/env bash
set -e

export PYTHONPATH="$(pwd):$PYTHONPATH"

MODEL=$1        # qwen
LANG=$2         # it
MAX_SAMPLES=${3:-}  # 1000

if [[ -z "$MODEL" || -z "$LANG" ]]; then
  echo "Usage: bash multilingual_tiser/training/run_train.sh qwen it 1000"
  exit 1
fi

MAX_SAMPLES_ARG=""
if [[ -n "$MAX_SAMPLES" ]]; then
  MAX_SAMPLES_ARG="--max_train_samples $MAX_SAMPLES"
fi


MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE="data/splits/train/TISER_train_${LANG}.json"
OUT_DIR="experiments/${MODEL}/${LANG}_${MAX_SAMPLES:-full}_8GB_safe"
mkdir -p "$OUT_DIR"

# --- 8GB VRAM "EMERGENCY" SETTINGS ---
# We reduced settings to ensure it fits on RTX 3070 Ti

EPOCHS=3
LR=2e-4

# REDUCED: 1024 -> 512 (Prevents OOM on backward pass)
MAX_LEN=1024

# REDUCED: 64 -> 16 (Saves memory on gradients)
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

PER_DEVICE_BS=1
# INCREASED: 16 -> 32 (To compensate for lower batch/seq len)
GRAD_ACCUM=32

echo "=============================="
echo " STARTING 8GB SAFE TRAINING"
echo "Model         : $MODEL_NAME"
echo "Language      : $LANG"
echo "Max Length    : $MAX_LEN (Reduced)"
echo "LoRA Rank     : $LORA_R (Reduced)"
echo "=============================="

python multilingual_tiser/training/train_qlora.py \
  --model_name "$MODEL_NAME" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --max_length "$MAX_LEN" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --per_device_batch_size "$PER_DEVICE_BS" \
  --grad_accum "$GRAD_ACCUM" \
  --save_steps 200 \
  --logging_steps 5 \
  --dataloader_num_workers 0 \
  --gradient_checkpointing 1 \
  $MAX_SAMPLES_ARG \
  --only_passed \