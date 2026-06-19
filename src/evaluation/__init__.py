
# src/evaluation/__init__.py

from src.evaluation.metrics import (
    compute_metrics,
    predict_lstm,
    predict_baseline_model,
)
from src.evaluation.classification_report import (
    generate_classification_report,
)
from src.evaluation.confusion_matrix import (
    plot_confusion_matrix,
    plot_confusion_matrix_comparison,
)
from src.evaluation.roc_analysis import (
    compute_roc_curves,
    plot_roc_curves,
    save_roc_scores,
)
from src.evaluation.comparison import (
    build_comparison_table,
    plot_model_comparison,
    save_evaluation_results,
)

__all__ = [
    "compute_metrics", "predict_lstm", "predict_baseline_model",
    "generate_classification_report",
    "plot_confusion_matrix", "plot_confusion_matrix_comparison",
    "compute_roc_curves", "plot_roc_curves", "save_roc_scores",
    "build_comparison_table", "plot_model_comparison",
    "save_evaluation_results",
]