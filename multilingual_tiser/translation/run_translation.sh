#!/usr/bin/env bash
set -e

export PYTHONPATH="$(pwd):$PYTHONPATH"

# Usage:
# bash multilingual_tiser/translation/run_translation.sh [it|fa|de] [train|test] [max_samples(optional)]

LANGUAGE=$1
CATEGORY=$2
MAX_SAMPLES=${3:-0}

if [[ -z "$LANGUAGE" || -z "$CATEGORY" ]]; then
  echo "Usage: bash multilingual_tiser/translation/run_translation.sh [it|fa|de] [train|test] [max_samples]"
  exit 1
fi

INPUT_FILE="data/splits/${CATEGORY}/TISER_${CATEGORY}_en.json"
OUTPUT_DIR="data/splits/${CATEGORY}"
BATCH_SIZE=32

python multilingual_tiser/translation/translate_dataset.py \
  --input "$INPUT_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --category "$CATEGORY" \
  --language "$LANGUAGE" \
  --batch_size "$BATCH_SIZE" \
  --max_samples "$MAX_SAMPLES" \


echo " DONE"
