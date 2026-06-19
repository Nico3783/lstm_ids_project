#!/usr/bin/env bash
# scripts/run_evaluation.sh
set -e
DATASET=${1:-nsl_kdd}
echo "Evaluating on dataset: $DATASET"
python3 evaluate.py --dataset "$DATASET"
python3 compare_models.py --dataset "$DATASET"