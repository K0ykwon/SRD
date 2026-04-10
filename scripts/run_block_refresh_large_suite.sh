#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-artifacts/block_refresh_large_suite}"
export PYTHONPATH="${PYTHONPATH:-src}"

python3 scripts/run_block_refresh_large_suite.py \
  --config configs/experiment/block_refresh_large_suite.json \
  --output-dir "${OUTPUT_DIR}"
