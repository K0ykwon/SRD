#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-artifacts/length_scaling}"
PYTHONPATH=src python3 scripts/run_block_refresh_large_suite.py \
  --config configs/experiment/length_scaling.json \
  --output-dir "${OUTPUT_DIR}"
