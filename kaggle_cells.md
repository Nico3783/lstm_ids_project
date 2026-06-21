# Kaggle Notebook — Copy each cell below into a new Kaggle notebook

## Setup
1. Go to kaggle.com → Code → New Notebook
2. Settings → Accelerator → GPU P100
3. Copy each `### Cell X` below into a separate Kaggle cell
4. Run all cells top to bottom

---

### Cell 1 — Check GPU

```python
import subprocess, os

result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout)
else:
    print("WARNING: No GPU detected. Go to Settings → Accelerator → GPU P100")
    print("Without GPU, training will be very slow.")
```

---

### Cell 2 — Download from Google Drive

```python
import shutil, os

WORKING = "/kaggle/working"
INPUT = "/kaggle/input"

# Your Google Drive sharing links
PROJECT_DRIVE_ID = "1NJ3gkaHHQDYwR6QkKzaXMYp6lOcmhGtu"  # lstm_project.zip
DATA_DRIVE_ID    = "1C6Hc4yYKe-rwQo8yZD5S2wCSu2rBAjHD"  # lstm_raw.zip

project_zip = None
data_zip = None

# --- Method 1: Download from Google Drive ---
if PROJECT_DRIVE_ID or DATA_DRIVE_ID:
    !pip install -q gdown
    import gdown

    if PROJECT_DRIVE_ID and not os.path.exists(f"{WORKING}/lstm_project.zip"):
        print(f"Downloading project zip from Google Drive ({PROJECT_DRIVE_ID})...")
        gdown.download(id=PROJECT_DRIVE_ID, output=f"{WORKING}/lstm_project.zip", quiet=False)
    project_zip = f"{WORKING}/lstm_project.zip"

    if DATA_DRIVE_ID and not os.path.exists(f"{WORKING}/lstm_raw.zip"):
        print(f"Downloading data zip from Google Drive ({DATA_DRIVE_ID})...")
        gdown.download(id=DATA_DRIVE_ID, output=f"{WORKING}/lstm_raw.zip", quiet=False)
    data_zip = f"{WORKING}/lstm_raw.zip"

# --- Method 2: Find in Kaggle input (added via "Add Data") ---
if not project_zip or not data_zip:
    for root, dirs, files in os.walk(INPUT):
        for f in files:
            path = os.path.join(root, f)
            if f == "lstm_project.zip" and not project_zip:
                project_zip = path
            elif f == "lstm_raw.zip" and not data_zip:
                data_zip = path

print(f"Project zip: {project_zip}")
print(f"Data zip:    {data_zip}")

# Extract project code
if project_zip:
    print("\nExtracting project code...")
    !unzip -q -o "{project_zip}" -d "{WORKING}"
    print("Done.")
else:
    print("ERROR: lstm_project.zip not found.")
    print("Paste your Google Drive ID above, or add the dataset to this notebook.")

# Extract raw data
if data_zip:
    raw_dir = f"{WORKING}/data/raw"
    if not os.path.exists(f"{raw_dir}/nsl_kdd"):
        print("\nExtracting raw data...")
        os.makedirs(raw_dir, exist_ok=True)
        !unzip -q -o "{data_zip}" -d "{raw_dir}"
        print("Done.")
    else:
        print("\nRaw data already extracted.")
else:
    print("ERROR: lstm_raw.zip not found.")
    print("Paste your Google Drive ID above, or add the dataset to this notebook.")

print("\nDataset directories:")
!ls {WORKING}/data/raw/
```

---

### Cell 3 — Rename tensorflow shim + set backend

```python
os.chdir("/kaggle/working")

# Rename the local tensorflow shim so real TF (GPU) is used
if os.path.exists("tensorflow"):
    !mv tensorflow tensorflow_shim_local
    print("Renamed tensorflow/ shim to tensorflow_shim_local/")
else:
    print("No tensorflow/ shim found (already renamed or not present).")

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
```

---

### Cell 4 — Install dependencies

```python
!pip install -q -r requirements_colab.txt
!pip install -q click
print("Dependencies installed.")
```

---

### Cell 5 — Verify environment

```python
import sys, numpy as np, keras

print(f"Python:     {sys.version}")
print(f"NumPy:      {np.__version__}")
print(f"Keras:      {keras.__version__}")
print(f"Backend:    {keras.backend.backend()}")

gpus = tf.config.list_physical_devices("GPU")
print(f"GPU:        {gpus[0].name if gpus else 'NONE'}")

print(f"\nProject files:")
!ls run_pipeline.py requirements_colab.txt 2>&1
```

---

### Cell 6 — Run Pipeline: NSL-KDD

```python
import shutil

# Clean previous run artifacts
for d in ["reports", "models"]:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"Cleaned {d}/")
# Remove stale flat processed files (old bug artifact)
if os.path.exists("data/processed"):
    for item in os.listdir("data/processed"):
        p = f"data/processed/{item}"
        if os.path.isfile(p):
            os.remove(p)
            print(f"Removed stale file: {p}")

os.environ["KERAS_BACKEND"] = "tensorflow"

!python run_pipeline.py --dataset nsl_kdd --skip-eda 2>&1 | tail -60
```

---

### Cell 7 — Run Pipeline: UNSW-NB15

```python
# Clean previous run artifacts
for d in ["reports", "models"]:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"Cleaned {d}/")

os.environ["KERAS_BACKEND"] = "tensorflow"

!python run_pipeline.py --dataset unsw_nb15 --skip-eda 2>&1 | tail -60
```

---

### Cell 8 — Run Pipeline: CICIDS2017 (largest, ~3-5 hrs on P100)

```python
# Clean previous run artifacts
for d in ["reports", "models"]:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"Cleaned {d}/")

os.environ["KERAS_BACKEND"] = "tensorflow"

!python run_pipeline.py --dataset cicids2017 --skip-eda 2>&1 | tail -60
```

---

### Cell 9 — Package all results for download

```python
# Copy final results to a single output folder
import shutil, json
from pathlib import Path

OUT = "/kaggle/working/final_results"
os.makedirs(OUT, exist_ok=True)

for ds in ["nsl_kdd", "unsw_nb15", "cicids2017"]:
    for src in ["reports", "models"]:
        if os.path.exists(src):
            dst = f"{OUT}/{ds}/{src}"
            shutil.copytree(src, dst, dirs_exist_ok=True)

# Create download zip
!cd /kaggle/working && zip -r final_results.zip final_results/
print("\nDownload 'final_results.zip' from the Kaggle output panel.")
print("(Right sidebar → Output → final_results.zip → Download)")
```

---

### Cell 10 — Summary of all runs

```python
print("=" * 60)
print("RESULTS SUMMARY — All Datasets")
print("=" * 60)

for ds in ["nsl_kdd", "unsw_nb15", "cicids2017"]:
    for subdir in ["reports/metrics", "reports", "metrics"]:
        result_file = Path(f"{OUT}/{ds}/{subdir}/evaluation_results.json")
        if result_file.exists():
            break
    else:
        # Also check working dir
        for subdir in ["reports/metrics", "reports"]:
            result_file = Path(f"/kaggle/working/{subdir}/evaluation_results.json")
            if result_file.exists():
                break
        else:
            result_file = None

    print(f"\n--- {ds.upper()} ---")
    if result_file and result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        for model_name, metrics in data.items():
            acc = metrics.get("accuracy", "N/A")
            f1 = metrics.get("f1_macro", "N/A")
            roc = metrics.get("roc_auc_macro", "N/A")
            if isinstance(acc, float):
                print(f"  {model_name:25s}  Acc={acc:.4f}  F1={f1:.4f}  ROC={roc:.4f}")
            else:
                print(f"  {model_name:25s}  Acc={acc}  F1={f1}  ROC={roc}")
    else:
        print("  No results found.")
```
