#!/usr/bin/env bash
set -euo pipefail

PYTHON=python3
SCRIPT="multilingual_tiser/preprocess/translate_validate.py"

# ---- Args ----
path=${1:?Usage: bash run_translate_validate.sh <TISER_lang.json> [lang] [category]}
lang=${2:-it}
category=${3:-train}

EN_FILE="data/splits/${category}/TISER_${category}_en.json"

echo "▶ Validating translation against English"
echo "  EN   : $EN_FILE"
echo "  LANG : $lang"
echo "  FILE : $path"
echo ""

$PYTHON "$SCRIPT" \
  --en "$EN_FILE" \
  --path "$path" \
  --lang "$lang" \
  --category "$category"

echo ""
echo "✔ Validation completed successfully."
