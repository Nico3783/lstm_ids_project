#!/usr/bin/env bash
# setup.sh — One-command environment setup
# Usage: chmod +x setup.sh && ./setup.sh
set -e

echo "============================================================"
echo " Deep Learning IDS Using LSTM — Environment Setup"
echo "============================================================"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create project directory tree
python3 -c "
from src.utils.paths import create_project_directories
create_project_directories()
print('All project directories created.')
"

echo ""
echo "Setup complete. Activate with:"
echo "  source venv/bin/activate"
echo ""
echo "Then download the NSL-KDD dataset:"
echo "  python -m src.data.download --dataset nsl_kdd"
echo ""
echo "Then run the full pipeline:"
echo "  python run_pipeline.py --dataset nsl_kdd"