#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-artifacts/synthetic_suite}"
export PYTHONPATH="${PYTHONPATH:-src}"

python3 -m srd.eval.ablation_runner \
  --config configs/experiment/synthetic_suite.json \
  --output-dir "${OUTPUT_DIR}"
