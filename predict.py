#!/usr/bin/env python3
# predict.py — Prediction on new data
# Usage:
#   python predict.py --input data/sample/sample_input.csv
#   python predict.py --input new_traffic.csv --output preds.csv

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate predictions on new network traffic data."
    )
    p.add_argument("--input",   required=True,
                   help="Path to input CSV file.")
    p.add_argument("--output",  default=None,
                   help="Path to save predictions CSV.")
    p.add_argument("--dataset", default="nsl_kdd",
                   choices=["nsl_kdd", "cicids2017", "unsw_nb15"])
    return p.parse_args()


def main():
    args = parse_args()

    from src.utils.helpers import print_banner
    from src.utils.logger import get_pipeline_logger
    from src.deployment.inference_pipeline import run_inference
    from src.config import get_config

    logger = get_pipeline_logger()
    print_banner("LSTM IDS — Prediction")

    results = run_inference(
        input_path=args.input,
        output_path=args.output,
        dataset=args.dataset,
    )

    logger.info("Predicted %d sequences.", len(results))
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()