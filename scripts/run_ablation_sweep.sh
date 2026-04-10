#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-artifacts/ablation_sweep}"
export PYTHONPATH="${PYTHONPATH:-src}"

python3 -m srd.eval.ablation_runner \
  --config configs/experiment/ablation_sweep.json \
  --output-dir "${OUTPUT_DIR}"
