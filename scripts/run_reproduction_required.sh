#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-outputs/reproduction/required_synthetic}"
export PYTHONPATH="${PYTHONPATH:-src}"

python3 scripts/run_block_refresh_large_suite.py \
  --config configs/experiment/reproduction_required.json \
  --output-dir "${OUTPUT_DIR}"
