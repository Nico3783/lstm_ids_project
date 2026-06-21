#!/usr/bin/env bash
# Package the project for Google Colab upload (excludes data, venv, git, caches)
set -euo pipefail

OUT="lstm_project.zip"
rm -f "$OUT"

zip -r "$OUT" . \
  -x 'data/*' \
  -x 'venv/*' \
  -x '.venv/*' \
  -x '.git/*' \
  -x '*.pyc' \
  -x '__pycache__/*' \
  -x '*.egg-info/*' \
  -x 'node_modules/*' \
  -x '.ipynb_checkpoints/*' \
  -x 'zSamples/*' \
  -x 'lstm_raw.zip' \
  -x 'lstm_project.zip' \
  -x '.matplotlib_cache/*' \
  -x 'journals/*' \
  -x 'outputs/*'

echo ""
echo "Created: $OUT ($(du -h "$OUT" | cut -f1))"
echo ""
echo "Upload both zips to Google Drive:"
echo "  1. $OUT  →  MyDrive/"
echo "  2. lstm_raw.zip  →  MyDrive/"
echo ""
echo "Then open colab_setup.ipynb in Google Colab and run all cells."
