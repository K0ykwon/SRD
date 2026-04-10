#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-artifacts/needle}"
export PYTHONPATH="${PYTHONPATH:-src}"

python3 -m srd.eval.benchmark_runner \
  --config configs/experiment/needle_retrieval.json \
  --variant srd_with_sufficiency \
  --train-steps 160 \
  --eval-batches 12 \
  --batch-size 32 \
  --learning-rate 0.002 \
  --answer-loss-weight 1 \
  --lm-loss-weight 0.25 \
  --output-dir "${OUTPUT_DIR}"
