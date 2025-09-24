#!/bin/bash

module load conda
./scripts/anvil/load_modules_cuda.sh
conda activate phantom-reasoning

# --- Paths / constants ---
PRED_DIR="/anvil/projects/x-nairr250102/phantom-reasoning/runs__pw10k/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-1.7B/grpo/x-anmolkab/0913__curr=random__training_seed=1/out-msq500/preds/cot"

# The fixed prefix/suffix around the checkpoint number in filenames
FNAME_PREFIX="dataset=msq500__split=minidev__model_name=share--runs__pw10k--data--wiki-v1-easy-depth_20_size_25--Qwen--Qwen3-1.7B--grpo--x-anmolkab--0913__curr=random__training_seed=1--checkpoint-"
FNAME_SUFFIX="__bs=500__bn=001__seed=1.json"

# Checkpoints you listed
CHECKPOINTS=(500 1000 1500 2000 2500 3000 3500 4000 4500)
CHECKPOINTS=(1000 2000 3000 4000)

# --- Run evaluations sequentially on the same allocation ---
for CKPT in "${CHECKPOINTS[@]}"; do
  FILENAME="${FNAME_PREFIX}${CKPT}${FNAME_SUFFIX}"
  PRED_FILE="${PRED_DIR}/${FILENAME}"
  OUT_FILE="msq_eval_results_06B_naive_ckpt_${CKPT}.json"

  if [[ ! -f "$PRED_FILE" ]]; then
    echo "[WARN] Missing predictions file for ckpt ${CKPT}: $PRED_FILE"
    echo "       Skipping."
    continue
  fi

  echo "------------------------------------------------------------"
  echo "Running evaluator for checkpoint ${CKPT}"
  echo "Predictions: $PRED_FILE"
  echo "Output:      $OUT_FILE"
  echo "------------------------------------------------------------"

  srun python gpt_evaluator.py \
    --use-musique \
    --predictions-file "$PRED_FILE" \
    --api-key "$API_KEY" \
    --output-file "$OUT_FILE"
done

echo "All done."
