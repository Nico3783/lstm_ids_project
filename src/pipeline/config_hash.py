"""Configuration fingerprinting for detecting config drift between runs."""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any


class ConfigFingerprint:
    """Computes and validates configuration fingerprints."""

    def __init__(self, dataset: str, output_base: str = "outputs"):
        self.dataset = dataset
        self.output_base = Path(output_base)
        self.dataset_dir = self.output_base / dataset
        self.fingerprint_file = self.dataset_dir / ".checkpoints" / "config_fingerprint.json"
        self.fingerprint_file.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_fingerprint(
        config_path: str,
        dataset: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        hasher = hashlib.sha256()
        with open(config_path, "rb") as f:
            hasher.update(f.read())
        hasher.update(dataset.encode("utf-8"))
        if extra:
            hasher.update(json.dumps(extra, sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()

    def save_fingerprint(
        self,
        config_path: str,
        extra: Optional[Dict[str, Any]] = None,
        stage: str = "start",
    ):
        fp = self.compute_fingerprint(config_path, self.dataset, extra)
        data = {
            "fingerprint": fp,
            "dataset": self.dataset,
            "config_path": config_path,
            "extra": extra or {},
            "stage": stage,
        }
        self.fingerprint_file.write_text(json.dumps(data, indent=2))

    def load_fingerprint(self) -> Optional[str]:
        if not self.fingerprint_file.exists():
            return None
        try:
            data = json.loads(self.fingerprint_file.read_text())
            return data.get("fingerprint")
        except (json.JSONDecodeError, IOError):
            return None

    def validate_fingerprint(
        self,
        config_path: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        old = self.load_fingerprint()
        if old is None:
            return True
        new = self.compute_fingerprint(config_path, self.dataset, extra)
        return old == new

    def clear_fingerprint(self):
        if self.fingerprint_file.exists():
            self.fingerprint_file.unlink()
