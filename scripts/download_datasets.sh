#!/usr/bin/env bash
# scripts/download_datasets.sh
# Downloads all available datasets
set -e
DATASET=${1:-nsl_kdd}
echo "Downloading dataset: $DATASET"
python3 -m src.data.download --dataset "$DATASET"