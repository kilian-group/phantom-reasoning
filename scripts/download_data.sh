#!/usr/bin/env bash
#
# Download all Phantom Reasoning datasets from the HuggingFace Hub into a target directory.
# Each dataset is a "<name>.zip" that unzips to "<data_dir>/<name>/".
#
# Usage: ./scripts/download_data.sh <data_dir>
#   ./scripts/download_data.sh data        # populate the data/ used by the recipes
#   ./scripts/download_data.sh hf_data      # download elsewhere (e.g. to smoke-test without touching data/)

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <data_dir>"
    echo "  e.g. $0 data"
    exit 1
fi

HF_REPO="kilian-group/phantom-reasoning"
DATA_DIR="$1"

mkdir -p "$DATA_DIR"

echo "==> Downloading all datasets from $HF_REPO into $DATA_DIR/ ..."
hf download "$HF_REPO" --repo-type dataset --local-dir "$DATA_DIR"

echo "==> Unzipping ..."
for zip in "$DATA_DIR"/*.zip; do
    unzip -q -o "$zip" -d "$DATA_DIR"
    rm -f "$zip"
done

echo "==> Done. Datasets ready in $DATA_DIR/"
