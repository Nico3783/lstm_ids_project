# Kaggle Notebook Cells

Copy these into separate Kaggle notebook cells. Each `---` separator = one cell.
Run cells in order. Each dataset gets its own cell so you can run one at a time.

---

## Cell 1 — Install & Download

```python
!pip install -q gdown scikit-learn pandas pyyaml joblib

import gdown, os, zipfile, subprocess

PROJECT_ZIP_ID = "1NJ3gkaHHQDYwR6QkKzaXMYp6lOcmhGtu"
RAW_DATA_ID = "1C6Hc4yYKe-rwQo8yZD5S2wCSu2rBAjHD"

os.chdir("/kaggle/working")

# Download project code
if not os.path.exists("src"):
    gdown.download(id=PROJECT_ZIP_ID, output="lstm_project.zip", quiet=False)
    with zipfile.ZipFile("lstm_project.zip", "r") as z:
        z.extractall(".")
    print("Project extracted.")

# Download raw datasets
if not os.path.exists("data/raw"):
    gdown.download(id=RAW_DATA_ID, output="lstm_raw.zip", quiet=False)
    with zipfile.ZipFile("lstm_raw.zip", "r") as z:
        z.extractall(".")
    print("Raw data extracted.")
```

---

## Cell 2 — GPU Check & Memory Monitor

```python
import subprocess, os

# Check GPU
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout)
else:
    print("WARNING: No GPU detected! Go to Kernel → Change runtime → GPU.")
    print(result.stderr)

# Memory monitor helper
def mem_usage():
    """Print current GPU and system memory usage."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            from tensorflow.python.client import device_lib
            devices = device_lib.list_local_devices()
            for d in devices:
                if d.device_type == "GPU":
                    print(f"[GPU] {d.name}: {d.memory_limit / 1e9:.1f} GB allocated")
    except Exception:
        pass
    # System RAM
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                avail = int(line.split()[1]) / 1e6
                print(f"[RAM] Available: {avail:.1f} GB")
                break

mem_usage()
```

---

## Cell 3 — Run NSL-KDD

```python
import sys, os, traceback, json, datetime
os.chdir("/kaggle/working")
sys.path.insert(0, ".")

LOG_FILE = "log_nsl_kdd.txt"
results = {}

try:
    # Run pipeline — logs go to file AND stdout
    with open(LOG_FILE, "w") as log_f:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = log_f
        sys.stderr = log_f
        try:
            from run_pipeline import parse_args, main
            sys.argv = ["run_pipeline.py", "--dataset", "nsl_kdd"]
            main()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # Print last 80 lines of log (not tail -60 which hides errors)
    print("=" * 60)
    print("NSL-KDD LOG (last 80 lines):")
    print("=" * 60)
    with open(LOG_FILE) as f:
        lines = f.readlines()
        for line in lines[-80:]:
            print(line, end="")

    results["nsl_kdd"] = "SUCCESS"

except Exception as e:
    print(f"NSL-KDD FAILED: {e}")
    traceback.print_exc()
    # Print full log for debugging
    if os.path.exists(LOG_FILE):
        print("\n--- FULL LOG ---")
        with open(LOG_FILE) as f:
            print(f.read())
    results["nsl_kdd"] = f"FAILED: {e}"

print("\n" + "=" * 60)
print("NSL-KDD RESULT:", results.get("nsl_kdd", "unknown"))
print("=" * 60)
```

---

## Cell 4 — Run UNSW-NB15 (subsampled)

```python
import sys, os, traceback
os.chdir("/kaggle/working")
sys.path.insert(0, ".")

LOG_FILE = "log_unsw_nb15.txt"

try:
    with open(LOG_FILE, "w") as log_f:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = log_f
        sys.stderr = log_f
        try:
            from run_pipeline import parse_args, main
            sys.argv = ["run_pipeline.py", "--dataset", "unsw_nb15", "--subsample", "0.3"]
            main()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    print("=" * 60)
    print("UNSW-NB15 LOG (last 80 lines):")
    print("=" * 60)
    with open(LOG_FILE) as f:
        lines = f.readlines()
        for line in lines[-80:]:
            print(line, end="")

except Exception as e:
    print(f"UNSW-NB15 FAILED: {e}")
    traceback.print_exc()
    if os.path.exists(LOG_FILE):
        print("\n--- FULL LOG ---")
        with open(LOG_FILE) as f:
            print(f.read())
```

---

## Cell 5 — Run CICIDS2017

```python
import sys, os, traceback
os.chdir("/kaggle/working")
sys.path.insert(0, ".")

LOG_FILE = "log_cicids2017.txt"

try:
    with open(LOG_FILE, "w") as log_f:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = log_f
        sys.stderr = log_f
        try:
            from run_pipeline import parse_args, main
            sys.argv = ["run_pipeline.py", "--dataset", "cicids2017"]
            main()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    print("=" * 60)
    print("CICIDS2017 LOG (last 80 lines):")
    print("=" * 60)
    with open(LOG_FILE) as f:
        lines = f.readlines()
        for line in lines[-80:]:
            print(line, end="")

except Exception as e:
    print(f"CICIDS2017 FAILED: {e}")
    traceback.print_exc()
    if os.path.exists(LOG_FILE):
        print("\n--- FULL LOG ---")
        with open(LOG_FILE) as f:
            print(f.read())
```

---

## Cell 6 — Save Results to Google Drive

```python
import os, zipfile, subprocess
from google.colab import drive  # Kaggle doesn't have this; skip if on Kaggle

# On Kaggle, download results manually or use this cell to zip them
os.chdir("/kaggle/working")

# Create a results archive
result_files = []
for root, dirs, files in os.walk("."):
    for f in files:
        if any(f.endswith(ext) for ext in [".csv", ".json", ".npy", ".pkl", ".png", ".txt", ".h5"]):
            result_files.append(os.path.join(root, f))

if result_files:
    with zipfile.ZipFile("kaggle_results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in result_files:
            zf.write(fp)
    print(f"Results archived: kaggle_results.zip ({len(result_files)} files)")
    print("Download from: /kaggle/working/kaggle_results.zip")
else:
    print("No result files found.")
```

---

## Notes

- **UNSW-NB15 uses `--subsample 0.3`** — trains on 30% of the data to avoid OOM.
  Increase to `0.5` if Kaggle has headroom.
- **CICIDS2017 runs full** — borderline on RAM but should fit with the new
  disk-backed chunked builder.
- Each dataset cell saves its own log file for debugging.
- All logs show the last 80 lines plus the full log on failure.
- GPU memory growth is now enabled in `run_pipeline.py` automatically.
