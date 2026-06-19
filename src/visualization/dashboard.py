# src/visualization/dashboard.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Single-call function that generates ALL Chapter 4
#          figures and exports a ZIP archive ready for the
#          project report.

import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from src.utils.logger import get_logger, log_section_header
from src.utils.paths import (
    FIGURES_DIR, TABLES_DIR, EXPORTED_DIR, ensure_dir
)

logger = get_logger(__name__)


def generate_all_report_figures(
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred_lstm: np.ndarray,
    y_prob_lstm: np.ndarray,
    all_metrics: Dict[str, Dict],
    dataset: str = "nsl_kdd",
    class_names: Optional[List[str]] = None,
    input_shape: tuple = (10, 41),
    n_classes: int = 5,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Generate every figure required for Chapter 4 in a single
    call and return a ``{name: path}`` dictionary.

    Figures generated
    -----------------
    - Training accuracy curve
    - Training loss curve
    - Confusion matrix (LSTM)
    - ROC curves (LSTM)
    - Precision-Recall curves
    - Model comparison bar chart
    - LSTM architecture diagram
    - Preprocessing pipeline diagram
    - System architecture diagram
    - Data flow diagram
    - Feature importance (if available in all_metrics)

    Parameters
    ----------
    history : dict
        Keras training history.
    y_true : np.ndarray
        Ground-truth test labels.
    y_pred_lstm : np.ndarray
        LSTM predicted labels.
    y_prob_lstm : np.ndarray
        LSTM probability matrix.
    all_metrics : dict
        ``{model_name: metrics_dict}``
    dataset : str
    class_names : list of str, optional
    input_shape : tuple
    n_classes : int
    output_dir : Path, optional

    Returns
    -------
    dict
        ``{figure_name: saved_path}``
    """
    log_section_header(logger, "GENERATING ALL REPORT FIGURES")

    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    # 1. Training curves
    from src.visualization.training_curves import plot_training_curves
    curves = plot_training_curves(
        history, dataset=dataset, output_dir=out_dir
    )
    saved.update(curves)

    # 2. Confusion matrix
    from src.evaluation.confusion_matrix import plot_confusion_matrix
    saved["confusion_matrix"] = plot_confusion_matrix(
        y_true, y_pred_lstm,
        class_names=class_names,
        dataset=dataset, model_name="LSTM",
        output_path=out_dir / "confusion_matrix.png",
    )

    # 3. ROC curves
    from src.evaluation.roc_analysis import plot_roc_curves
    saved["roc_curve"] = plot_roc_curves(
        y_true, y_prob_lstm,
        class_names=class_names,
        dataset=dataset, model_name="LSTM",
        output_path=out_dir / "roc_curve.png",
    )

    # 4. Precision-Recall curves
    from src.visualization.plots import plot_precision_recall_curves
    saved["precision_recall"] = plot_precision_recall_curves(
        y_true, y_prob_lstm,
        class_names=class_names,
        dataset=dataset,
        output_path=out_dir / f"precision_recall_curve_{dataset}.png",
    )

    # 5. Model comparison
    from src.evaluation.comparison import plot_model_comparison
    saved["model_comparison"] = plot_model_comparison(
        all_metrics, dataset=dataset,
        output_path=out_dir / "model_comparison_chart.png",
    )

    # 6. LSTM architecture diagram
    from src.models.model_utils import plot_lstm_architecture
    saved["lstm_architecture"] = plot_lstm_architecture(
        input_shape=input_shape,
        n_classes=n_classes,
        output_path=out_dir / "lstm_architecture.png",
    )

    # 7. Preprocessing pipeline
    from src.visualization.plots import plot_preprocessing_pipeline
    saved["preprocessing_pipeline"] = plot_preprocessing_pipeline(
        output_path=out_dir / "preprocessing_pipeline.png"
    )

    # 8. System architecture
    from src.visualization.architecture_diagrams import (
        plot_system_architecture, plot_data_flow_diagram
    )
    saved["system_architecture"] = plot_system_architecture(
        output_path=out_dir / "system_architecture.png"
    )

    # 9. Data flow
    saved["data_flow"] = plot_data_flow_diagram(
        output_path=out_dir / "data_flow_diagram.png"
    )

    logger.info(
        "All report figures generated: %d figures.", len(saved)
    )
    for name, path in saved.items():
        logger.info("  %-30s → %s", name, path)

    return saved


def export_chapter4_zip(
    figures_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Package all Chapter 4 figures and tables into a ZIP
    archive for easy submission and download.

    Parameters
    ----------
    figures_dir : Path, optional
    tables_dir : Path, optional
    output_dir : Path, optional

    Returns
    -------
    Path
        Path to the generated ZIP file.
    """
    figs  = figures_dir or FIGURES_DIR
    tabs  = tables_dir  or TABLES_DIR
    out   = output_dir  or EXPORTED_DIR
    out.mkdir(parents=True, exist_ok=True)

    zip_path = out / "chapter4_results.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Figures
        for img in figs.glob("*.png"):
            zf.write(img, arcname=f"figures/{img.name}")

        # Tables
        for csv in tabs.glob("*.csv"):
            zf.write(csv, arcname=f"tables/{csv.name}")

    logger.info(
        "Chapter 4 ZIP exported: %s (%d files)",
        zip_path,
        len(list(figs.glob("*.png"))) + len(list(tabs.glob("*.csv"))),
    )
    return zip_path