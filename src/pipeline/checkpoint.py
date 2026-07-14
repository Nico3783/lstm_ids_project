"""Stage completion markers and artifact discovery for auto-resume."""

import json
import time
from typing import Optional, Dict, Any, List
from pathlib import Path


class CheckpointManager:
    """Manages .done markers and artifact discovery for pipeline stages.

    Stages match the pipeline execution order in ``run_pipeline.py``.
    Each stage produces artifacts under ``outputs/<dataset>/``; the
    manager records a ``.done`` marker in ``.checkpoints/`` upon
    completion and validates expected artifacts on resume.
    """

    STAGES: List[str] = [
        "preprocessing",
        "split_save",
        "scale_sequences",
        "tuning",
        "baselines",
        "lstm_train",
        "evaluation",
        "visualization",
        "export",
    ]

    def __init__(self, dataset: str, output_base: str = "outputs"):
        self.dataset = dataset
        self.output_base = Path(output_base)
        self.dataset_dir = self.output_base / dataset
        self.done_dir = self.dataset_dir / ".checkpoints"
        self.done_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Marker operations
    # ------------------------------------------------------------------

    def stage_done_file(self, stage: str) -> Path:
        return self.done_dir / f"{stage}.done"

    def stage_done(self, stage: str) -> bool:
        return self.stage_done_file(stage).exists()

    def mark_done(self, stage: str, metadata: Optional[Dict[str, Any]] = None):
        done_file = self.stage_done_file(stage)
        data: Dict[str, Any] = {
            "stage": stage,
            "dataset": self.dataset,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": time.time(),
        }
        if metadata:
            data["metadata"] = metadata
        done_file.write_text(json.dumps(data, indent=2))

    def get_done_metadata(self, stage: str) -> Optional[Dict[str, Any]]:
        done_file = self.stage_done_file(stage)
        if not done_file.exists():
            return None
        try:
            return json.loads(done_file.read_text())
        except (json.JSONDecodeError, IOError):
            return None

    def clear_stage(self, stage: str):
        done_file = self.stage_done_file(stage)
        if done_file.exists():
            done_file.unlink()

    def clear_all(self):
        for f in self.done_dir.glob("*.done"):
            f.unlink()

    def completed_stages(self) -> List[str]:
        return [s for s in self.STAGES if self.stage_done(s)]

    def pending_stages(self) -> List[str]:
        return [s for s in self.STAGES if not self.stage_done(s)]

    # ------------------------------------------------------------------
    # Artifact discovery
    # ------------------------------------------------------------------

    def find_artifact(self, stage: str, filename: str) -> Optional[Path]:
        """Search for *filename* across directories associated with *stage*."""
        for d in self._artifact_dirs(stage):
            p = d / filename
            if p.exists():
                return p
        return None

    def validate_stage_artifacts(self, stage: str) -> Dict[str, bool]:
        """Check that every expected artifact for *stage* exists.

        Returns ``{artifact_name: exists}`` where *artifact_name* is
        the path relative to ``dataset_dir``.
        """
        expected = self._expected_artifacts(stage)
        results: Dict[str, bool] = {}
        for rel_path in expected:
            p = self.dataset_dir / rel_path
            results[rel_path] = p.exists()
        return results

    def list_stage_artifacts(self, stage: str) -> List[Path]:
        """Return list of existing artifact paths for *stage*."""
        expected = self._expected_artifacts(stage)
        found: List[Path] = []
        for rel_path in expected:
            p = self.dataset_dir / rel_path
            if p.exists():
                found.append(p)
        return found

    # ------------------------------------------------------------------
    # Stage-to-directory mapping
    # ------------------------------------------------------------------

    def _artifact_dirs(self, stage: str) -> List[Path]:
        """Directories to search for artifacts of a given stage."""
        mapping: Dict[str, List[str]] = {
            "preprocessing":    ["preprocessed"],
            "split_save":       ["preprocessed"],
            "scale_sequences":  ["preprocessed"],
            "tuning":           ["config"],
            "baselines":        ["models/baselines"],
            "lstm_train":       ["models/final"],
            "evaluation":       ["tables", "metrics"],
            "visualization":    ["figures"],
            "export":           ["exported"],
        }
        dirs = [self.dataset_dir]
        for sub in mapping.get(stage, []):
            dirs.append(self.dataset_dir / sub)
        return dirs

    def _expected_artifacts(self, stage: str) -> List[str]:
        """Relative paths (from dataset_dir) expected after each stage.

        These match the actual file paths produced by ``run_pipeline.py``.
        Dataset isolation is handled by the output directory structure
        (``outputs/<dataset>/``), not by filename prefixes.
        """
        artifacts: Dict[str, List[str]] = {
            "preprocessing": [
                "preprocessed/X_2d.npy",
                "preprocessed/y_labels.npy",
                "preprocessed/label_encoder.pkl",
                "preprocessed/feature_names.pkl",
            ],
            "split_save": [
                "preprocessed/X_train.npy",
                "preprocessed/X_val.npy",
                "preprocessed/X_test.npy",
                "preprocessed/y_train.npy",
                "preprocessed/y_val.npy",
                "preprocessed/y_test.npy",
            ],
            "scale_sequences": [
                "preprocessed/X_train.npy",
                "preprocessed/X_val.npy",
                "preprocessed/X_test.npy",
                "preprocessed/y_train.npy",
                "preprocessed/y_val.npy",
                "preprocessed/y_test.npy",
                "preprocessed/scaler.pkl",
                "preprocessed/label_encoder.pkl",
            ],
            "tuning": [
                "config/best_config.json",
            ],
            "baselines": [
                "models/baselines/baseline_results.json",
                "models/baselines/svm.pkl",
                "models/baselines/random_forest.pkl",
                "models/baselines/logistic_regression.pkl",
            ],
            "lstm_train": [
                "models/final/lstm_ids_model.keras",
                "models/final/model_metadata.json",
            ],
            "evaluation": [
                "tables/classification_report.csv",
                "tables/classification_report.txt",
                "tables/confusion_matrix.csv",
                "tables/confusion_matrix.png",
            ],
            "visualization": [
                "figures/training_accuracy_curve.png",
                "figures/training_loss_curve.png",
                "figures/roc_curve.png",
                "figures/confusion_matrix.png",
                "figures/precision_recall_curve.png",
                "figures/model_comparison_chart.png",
            ],
            "export": [
                "exported/lstm_ids_model.keras",
                "exported/scaler.pkl",
                "exported/label_encoder.pkl",
                "exported/feature_names.pkl",
                "exported/metadata.json",
            ],
        }
        return artifacts.get(stage, [])
