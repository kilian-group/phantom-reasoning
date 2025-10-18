#!/usr/bin/env bash
set -euo pipefail

# Settings
DATASETS_CSV="aiw, family_relationships, quantum_lock"
SIZE="100"
SEED="42"
TRAIN_FRAC="0.8"
OUT_DIR_BASE="./data/reasoning_gym"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

echo "DATASETS_CSV=${DATASETS_CSV}"
echo "SIZE=${SIZE}  SEED=${SEED}  TRAIN_FRAC=${TRAIN_FRAC}"
echo "OUT_DIR_BASE=${OUT_DIR_BASE}"

IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"

for DATASET in "${DATASETS[@]}"; do
  DATASET_TRIMMED="$(echo "$DATASET" | xargs)"
  [[ -z "$DATASET_TRIMMED" ]] && continue

  if [[ "$OUT_DIR_BASE" == *"{dataset}"* ]]; then
    OUT_DIR="${OUT_DIR_BASE//\{dataset\}/$DATASET_TRIMMED}"
  else
    OUT_DIR="${OUT_DIR_BASE%/}/$DATASET_TRIMMED"
  fi

  echo "==> Generating '${DATASET_TRIMMED}' -> ${OUT_DIR}"
  "$PY" "${SCRIPT_DIR}/generate_dataset.py" \
    --dataset "${DATASET_TRIMMED}" \
    --size "${SIZE}" \
    --seed "${SEED}" \
    --train-frac "${TRAIN_FRAC}" \
    --out-dir "${OUT_DIR}"
done

echo "Done."
