"""Progress reporting with elapsed time, ETA, memory, and GPU usage."""

import os
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks and reports pipeline progress.

    Usage::

        pm = ProgressTracker("nsl_kdd")
        pm.start_stage("preprocessing")
        ...
        pm.end_stage("preprocessing")
        ...
        pm.print_summary()
    """

    def __init__(self, dataset: str, total_stages: int = 9):
        self.dataset = dataset
        self.total_stages = total_stages
        self.stage_start_time: Optional[float] = None
        self.current_stage: Optional[str] = None
        self.completed_stages: int = 0
        self.stage_timings: Dict[str, float] = {}
        self.pipeline_start_time: float = time.time()

    def start_stage(self, stage: str):
        """Mark the beginning of a pipeline stage."""
        self.current_stage = stage
        self.stage_start_time = time.time()
        mem = self._get_memory_info()
        gpu = self._get_gpu_info()
        disk = self.get_disk_usage()
        logger.info(
            "[PROGRESS] Stage %d/%d: %s | RAM: %s/%s GB | GPU: %s MB | Disk free: %s GB",
            self.completed_stages + 1,
            self.total_stages,
            stage,
            mem.get("used_gb", "?"),
            mem.get("total_gb", "?"),
            gpu.get("used_mb", "?"),
            disk.get("free_gb", "?"),
        )

    def end_stage(self, stage: Optional[str] = None):
        """Mark the end of a pipeline stage and log timing."""
        if self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            name = stage or self.current_stage or "unknown"
            self.stage_timings[name] = elapsed
            self.completed_stages += 1

            total_elapsed = sum(self.stage_timings.values())
            remaining = self.total_stages - self.completed_stages
            avg = total_elapsed / self.completed_stages if self.completed_stages else 0
            eta = avg * remaining

            logger.info(
                "[PROGRESS] Completed %s in %s | Total: %s | ETA: %s",
                name,
                self._fmt(elapsed),
                self._fmt(total_elapsed),
                self._fmt(eta),
            )
            self.stage_start_time = None
            self.current_stage = None

    def summary(self) -> str:
        """Return a formatted summary string."""
        total = sum(self.stage_timings.values())
        lines = [
            f"Dataset: {self.dataset}",
            f"Total time: {self._fmt(total)}",
            "Stage breakdown:",
        ]
        for stage, elapsed in self.stage_timings.items():
            lines.append(f"  {stage}: {self._fmt(elapsed)}")
        return "\n".join(lines)

    def print_summary(self):
        """Print a formatted summary to stdout and logger."""
        total_pipeline = time.time() - self.pipeline_start_time
        total_stages = sum(self.stage_timings.values())

        lines = [
            "",
            "=" * 60,
            f"  Pipeline Summary — {self.dataset}",
            "=" * 60,
            f"  Total pipeline time: {self._fmt(total_pipeline)}",
            f"  Total stage time:    {self._fmt(total_stages)}",
            "-" * 60,
        ]

        if self.stage_timings:
            max_name = max(len(s) for s in self.stage_timings)
            for stage, elapsed in self.stage_timings.items():
                pct = (elapsed / total_stages * 100) if total_stages else 0
                lines.append(
                    f"  {stage:<{max_name}}  {self._fmt(elapsed):>10}  ({pct:5.1f}%)"
                )
        else:
            lines.append("  (no stages recorded)")

        lines.append("=" * 60)

        summary = "\n".join(lines)
        print(summary)
        logger.info(summary)

    # ── Formatting helpers ──────────────────────────────────────────

    @staticmethod
    def _fmt(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h{m:02d}m{s:02d}s"
        return f"{m}m{s:02d}s"

    @staticmethod
    def _get_memory_info() -> Dict[str, Any]:
        try:
            import psutil
            vm = psutil.virtual_memory()
            return {
                "used_gb": f"{vm.used / (1024**3):.1f}",
                "total_gb": f"{vm.total / (1024**3):.1f}",
            }
        except ImportError:
            return {"used_gb": "?", "total_gb": "?"}

    @staticmethod
    def _get_gpu_info() -> Dict[str, Any]:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return {"used_mb": "N/A", "total_mb": "N/A"}
            from tensorflow.python.client import device_lib
            devices = device_lib.list_local_devices()
            for d in devices:
                if d.device_type == "GPU":
                    mem = d.memory_limit / (1024**2)
                    return {"used_mb": f"{mem:.0f}", "total_mb": f"{mem:.0f}"}
            return {"used_mb": "?", "total_mb": "?"}
        except Exception:  # noqa: BLE001
            return {"used_mb": "N/A", "total_mb": "N/A"}

    @staticmethod
    def get_disk_usage(path: str = ".") -> Dict[str, Any]:
        try:
            usage = os.statvfs(path)
            free = (usage.f_bavail * usage.f_frsize) / (1024**3)
            total = (usage.f_blocks * usage.f_frsize) / (1024**3)
            return {"free_gb": f"{free:.1f}", "total_gb": f"{total:.1f}"}
        except OSError:
            return {"free_gb": "?", "total_gb": "?"}
