
# tests/__init__.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Marks the tests/ directory as a Python package,
#          enabling pytest discovery and cross-test imports.
#
# Test Suite Overview:
#   test_data_loading.py     -- NSL-KDD loader unit tests
#   test_preprocessing.py   -- Preprocessing pipeline unit tests
#   test_sequence_builder.py -- Sliding window sequence tests
#   test_models.py           -- LSTM + baseline model unit tests
#   test_training.py         -- Callback + class weight tests
#   test_pipeline.py         -- End-to-end integration tests
#
# Run all tests:
#   pytest tests/ -v
#
# Run specific module:
#   pytest tests/test_pipeline.py -v
#
# Run with coverage:
#   pytest tests/ --cov=src --cov-report=term-missing
