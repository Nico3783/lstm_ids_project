"""Colab environment reliability checks."""

import os
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ColabEnvironmentValidator:
    """Validates Colab environment before running pipeline."""

    MIN_DISK_GB = 5.0
    MIN_RAM_GB = 8.0
    EXPECTED_GPU_TYPES = ["Tesla T4", "Tesla P100", "A100", "V100", "L4"]

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def validate_all(self) -> Tuple[bool, Dict[str, Any]]:
        checks = {
            "gpu": self.check_gpu(),
            "ram": self.check_ram(),
            "disk": self.check_disk(),
            "drive_mounted": self.check_drive_mounted(),
            "tensorflow": self.check_tensorflow(),
            "cuda_available": self.check_cuda(),
        }
        self.results = checks
        all_ok = all(c.get("ok", False) for c in checks.values())
        return all_ok, checks

    def check_gpu(self) -> Dict[str, Any]:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return {"ok": False, "message": "No GPU found"}
            gpu_name = "unknown"
            try:
                from tensorflow.python.client import device_lib
                for d in device_lib.list_local_devices():
                    if d.device_type == "GPU":
                        gpu_name = d.physical_device_desc
                        break
            except Exception:
                gpu_name = gpus[0].name
            return {"ok": True, "gpu": gpu_name, "count": len(gpus)}
        except ImportError:
            return {"ok": False, "message": "TensorFlow not installed"}

    def check_ram(self) -> Dict[str, Any]:
        try:
            import psutil
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024**3)
            ok = total_gb >= self.MIN_RAM_GB
            return {
                "ok": ok,
                "total_gb": round(total_gb, 1),
                "available_gb": round(vm.available / (1024**3), 1),
                "message": f"{'OK' if ok else 'Low RAM'}: {total_gb:.1f} GB",
            }
        except ImportError:
            return {"ok": True, "message": "psutil not available, skipping RAM check"}

    def check_disk(self) -> Dict[str, Any]:
        try:
            usage = os.statvfs(".")
            free_gb = (usage.f_bavail * usage.f_frsize) / (1024**3)
            ok = free_gb >= self.MIN_DISK_GB
            return {
                "ok": ok,
                "free_gb": round(free_gb, 1),
                "message": f"{'OK' if ok else 'Low disk'}: {free_gb:.1f} GB free",
            }
        except OSError:
            return {"ok": True, "message": "Cannot check disk, skipping"}

    def check_drive_mounted(self) -> Dict[str, Any]:
        drive_path = "/content/drive/MyDrive"
        if os.path.exists(drive_path):
            return {"ok": True, "path": drive_path}
        return {"ok": False, "message": "Google Drive not mounted at /content/drive/MyDrive"}

    def check_tensorflow(self) -> Dict[str, Any]:
        try:
            import tensorflow as tf
            return {"ok": True, "version": tf.__version__}
        except ImportError:
            return {"ok": False, "message": "TensorFlow not installed"}

    def check_cuda(self) -> Dict[str, Any]:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            return {"ok": len(gpus) > 0, "gpu_count": len(gpus)}
        except Exception:
            return {"ok": False, "message": "Cannot check CUDA"}

    def print_report(self):
        ok, checks = self.validate_all()
        print("=" * 60)
        print("COLAB ENVIRONMENT CHECK")
        print("=" * 60)
        for name, result in checks.items():
            status = "PASS" if result.get("ok") else "FAIL"
            msg = result.get("message", "")
            extra = {k: v for k, v in result.items() if k not in ("ok", "message")}
            extra_str = f" ({extra})" if extra else ""
            print(f"  [{status}] {name}{extra_str} {msg}")
        print("=" * 60)
        print(f"Overall: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
        print("=" * 60)
        return ok
