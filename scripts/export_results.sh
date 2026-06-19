#!/usr/bin/env bash
# scripts/export_results.sh
set -e
echo "Exporting Chapter 4 results..."
python3 -c "
from src.visualization.dashboard import export_chapter4_zip
path = export_chapter4_zip()
print(f'Exported: {path}')
"