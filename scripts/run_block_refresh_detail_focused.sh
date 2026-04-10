#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-artifacts/block_refresh_detail_focused}"
export PYTHONPATH="${PYTHONPATH:-src}"

python3 scripts/run_block_refresh_large_suite.py \
  --config configs/experiment/block_refresh_detail_focused.json \
  --output-dir "${OUTPUT_DIR}"
