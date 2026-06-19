
# src/deployment/__init__.py

from src.deployment.predictor import IDSPredictor
from src.deployment.inference_pipeline import (
    run_inference,
    run_test_set_inference,
)

__all__ = [
    "IDSPredictor",
    "run_inference",
    "run_test_set_inference",
]