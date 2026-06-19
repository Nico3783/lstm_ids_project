#!/usr/bin/env bash
# scripts/run_training.sh
set -e
DATASET=${1:-nsl_kdd}
echo "Training on dataset: $DATASET"
python3 train.py --dataset "$DATASET"