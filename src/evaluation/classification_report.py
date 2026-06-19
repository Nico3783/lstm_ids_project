
# src/evaluation/classification_report.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Generate, format, and save the full scikit-learn
#          classification report as a text file and structured
#          CSV table for Chapter 4 documentation.

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report  # type: ignore

from src.utils.constants import NSL_KDD_CLASS_NAMES
from src.utils.logger import get_logger
from src.utils.paths import METRICS_DIR, TABLES_DIR, ensure_dir

logger = get_logger(__name__)


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    dataset: str = "nsl_kdd",
    model_name: str = "model",
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Generate a full classification report and save it as both
    a plain-text file and a structured CSV.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    class_names : list of str, optional
    dataset : str
    model_name : str
    output_dir : Path, optional

    Returns
    -------
    dict
        ``{text_path, csv_path, report_dict}``
    """
    names = class_names or (
        NSL_KDD_CLASS_NAMES if dataset == "nsl_kdd" else None
    )
    labels = sorted(np.unique(
        np.concatenate([y_true, y_pred])
    ).tolist())

    target_names = (
        [names[i] for i in labels if i < len(names)]
        if names else [str(i) for i in labels]
    )

    # ---- Text report ----
    report_str = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )

    header = (
        f"Classification Report\n"
        f"Model   : {model_name}\n"
        f"Dataset : {dataset.upper()}\n"
        f"Samples : {len(y_true):,}\n"
        f"{'=' * 60}\n"
    )
    full_text = header + report_str

    # ---- Save text ----
    out_dir = output_dir or METRICS_DIR
    ensure_dir(out_dir / "placeholder")
    txt_path = out_dir / "classification_report.txt"
    txt_path.write_text(full_text, encoding="utf-8")
    logger.info("Classification report (txt) saved: %s", txt_path)

    # ---- Structured dict ----
    report_dict = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    # ---- Save CSV ----
    rows = []
    for cls_name in target_names:
        if cls_name in report_dict:
            r = report_dict[cls_name]
            rows.append({
                "Class":     cls_name,
                "Precision": round(r["precision"], 4),
                "Recall":    round(r["recall"],    4),
                "F1-Score":  round(r["f1-score"],  4),
                "Support":   int(r["support"]),
            })
    for avg in ["macro avg", "weighted avg", "accuracy"]:
        if avg in report_dict:
            r = report_dict[avg]
            if isinstance(r, dict):
                rows.append({
                    "Class":     avg,
                    "Precision": round(r.get("precision", 0), 4),
                    "Recall":    round(r.get("recall",    0), 4),
                    "F1-Score":  round(r.get("f1-score",  0), 4),
                    "Support":   int(r.get("support",     0)),
                })
            else:
                rows.append({
                    "Class": avg, "Precision": "",
                    "Recall": "", "F1-Score": round(float(r), 4),
                    "Support": "",
                })

    ensure_dir(TABLES_DIR / "placeholder")
    csv_path = TABLES_DIR / "final_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Classification report (csv) saved: %s", csv_path)

    # Print to log for screenshot purposes
    logger.info("\n%s", full_text)

    return {
        "text_path":   str(txt_path),
        "csv_path":    str(csv_path),
        "report_dict": report_dict,
        "report_text": full_text,
    }