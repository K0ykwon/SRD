#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-artifacts/parameter_scaling_long}"
PYTHONPATH=src python3 scripts/run_block_refresh_large_suite.py \
  --config configs/experiment/parameter_scaling_long.json \
  --output-dir "${OUTPUT_DIR}"
