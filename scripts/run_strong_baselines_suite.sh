#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-artifacts/strong_baselines}"
export PYTHONPATH="${PYTHONPATH:-src}"

python3 -m srd.eval.ablation_runner \
  --config configs/experiment/strong_baselines.json \
  --output-dir "${OUTPUT_DIR}"
