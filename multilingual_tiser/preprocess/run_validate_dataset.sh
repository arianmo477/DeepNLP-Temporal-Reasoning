#!/bin/bash
set -e

export PYTHONPATH="$(pwd):$PYTHONPATH"

category="train"

INPUT="data/TISER_${category}.json"
OUTPUT="data/splits/${category}/TISER_${category}_en.json"
INVALID_OUT="data/invalid_samples_${category}.json"

echo " Validating TISER dataset..."
echo " Input : $INPUT"
echo " Invalid samples (if any): $INVALID_OUT"
echo ""

python multilingual_tiser/preprocess/validate_tiser_dataset.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --save_invalid "$INVALID_OUT" \
  --split "$category" \
  --prompt_file "data/prompts/prompt_en.txt"

echo ""
echo "Dataset validation completed."
