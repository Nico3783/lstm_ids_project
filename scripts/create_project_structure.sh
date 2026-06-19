#!/usr/bin/env bash
# scripts/create_project_structure.sh
# Creates all project directories
set -e
python3 -c "
from src.utils.paths import create_project_directories
create_project_directories()
print('Project structure created.')
"