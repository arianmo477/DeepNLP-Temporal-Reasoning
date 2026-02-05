#!/usr/bin/env bash
set -e

# ===============================
# USAGE
# ===============================
# bash scripts/data/run_build_mixed_dataset.sh train 4000
# bash scripts/data/run_build_mixed_dataset.sh test 2000

CATEGORY=$1          # train | test
SAMPLES_PER_LANG=$2  # e.g. 4000

if [[ -z "$CATEGORY" || -z "$SAMPLES_PER_LANG" ]]; then
  echo "Usage: bash scripts/data/run_build_mixed_dataset.sh [train|test] [samples_per_lang]"
  exit 1
fi

# ===============================
# PATHS
# ===============================
EN_FILE="data/splits/${CATEGORY}/TISER_${CATEGORY}_en.json"
IT_FILE="data/splits/${CATEGORY}/TISER_${CATEGORY}_it.json"
FA_FILE="data/splits/${CATEGORY}/TISER_${CATEGORY}_fa.json"

OUT_FILE="data/splits/${CATEGORY}/TISER_${CATEGORY}_mixed.json"

SCRIPT="scripts/data/build_mixed_dataset.py"

# ===============================
# SANITY CHECKS
# ===============================
for f in "$EN_FILE" "$IT_FILE" "$FA_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing file: $f"
    exit 1
  fi
done

if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing script: $SCRIPT"
  exit 1
fi

mkdir -p "$(dirname "$OUT_FILE")"

# ===============================
# INFO
# ===============================
echo "Building mixed multilingual dataset"
echo "Category         : $CATEGORY"
echo "EN file          : $EN_FILE"
echo "IT file          : $IT_FILE"
echo "FA file          : $FA_FILE"
echo "Samples per lang : $SAMPLES_PER_LANG"
echo "Output           : $OUT_FILE"
echo "=============================="

# ===============================
# RUN
# ===============================
python "$SCRIPT" \
  --en "$EN_FILE" \
  --it "$IT_FILE" \
  --fa "$FA_FILE" \
  --samples_per_lang "$SAMPLES_PER_LANG" \
  --out "$OUT_FILE"

# ===============================
# DONE
# ===============================
echo ""
echo "✅ Mixed dataset created successfully!"
