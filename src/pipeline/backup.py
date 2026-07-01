"""Extended backup system for Colab results to Google Drive."""

import os
import shutil
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class BackupManager:
    """Manages backup of pipeline results to persistent storage."""

    def __init__(self, dataset: str, drive_base: str = "/content/drive/MyDrive/lstm_ids_results"):
        self.dataset = dataset
        self.drive_base = Path(drive_base)
        self.drive_dir = self.drive_base / dataset
        self.local_base = Path("outputs") / dataset

    def backup_all(self) -> Dict[str, Any]:
        self.drive_dir.mkdir(parents=True, exist_ok=True)
        results = {"dataset": self.dataset, "files": [], "errors": []}

        artifact_dirs = [
            ("data/processed", ["*.npz", "*.npy", "*.pkl", "*.json"]),
            ("models/baselines", ["*.pkl", "*.json"]),
            ("models/final", ["*.keras", "*.h5", "*.pkl", "*.json"]),
            ("models/checkpoints", ["*.keras", "*.json"]),
            ("reports/metrics", ["*.json", "*.csv"]),
            ("reports/tables", ["*.csv", "*.txt"]),
            ("reports/figures", ["*.png", "*.svg"]),
            ("reports/tuning", ["*.json", "*.csv"]),
            ("logs", ["*.log", "*.csv"]),
        ]

        for subdir, patterns in artifact_dirs:
            local_dir = self.local_base / subdir
            if not local_dir.exists():
                continue
            for pattern in patterns:
                for f in local_dir.glob(pattern):
                    try:
                        rel = f.relative_to(self.local_base)
                        dest = self.drive_dir / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(f), str(dest))
                        results["files"].append(str(rel))
                    except Exception as e:
                        results["errors"].append(f"{f}: {e}")

        manifest = {
            "dataset": self.dataset,
            "backed_up_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_count": len(results["files"]),
            "error_count": len(results["errors"]),
            "files": results["files"],
        }
        manifest_path = self.drive_dir / "backup_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        results["manifest"] = str(manifest_path)

        logger.info(
            f"Backup complete: {len(results['files'])} files, "
            f"{len(results['errors'])} errors"
        )
        return results

    def verify_backup(self) -> Dict[str, Any]:
        if not self.drive_dir.exists():
            return {"ok": False, "message": "Backup directory does not exist"}

        manifest_path = self.drive_dir / "backup_manifest.json"
        if not manifest_path.exists():
            return {"ok": False, "message": "No backup manifest found"}

        manifest = json.loads(manifest_path.read_text())
        missing = []
        for f in manifest.get("files", []):
            if not (self.drive_dir / f).exists():
                missing.append(f)

        return {
            "ok": len(missing) == 0,
            "total_files": len(manifest.get("files", [])),
            "missing_files": missing,
            "backed_up_at": manifest.get("backed_up_at"),
        }

    def restore_from_backup(self, target_dir: Optional[str] = None) -> Dict[str, Any]:
        target = Path(target_dir) if target_dir else self.local_base
        if not self.drive_dir.exists():
            return {"ok": False, "message": "No backup found"}

        restored = []
        for f in self.drive_dir.rglob("*"):
            if f.is_file() and f.name != "backup_manifest.json":
                rel = f.relative_to(self.drive_dir)
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(f), str(dest))
                restored.append(str(rel))

        return {"ok": True, "restored": len(restored), "files": restored}
