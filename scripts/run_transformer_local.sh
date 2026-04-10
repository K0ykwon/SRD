#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_CONFIG="${1:-configs/experiment/delayed_kv_easy.json}"
OUTPUT_DIR="${2:-artifacts/transformer_local}"
export PYTHONPATH="${PYTHONPATH:-src}"

python3 -m srd.eval.benchmark_runner \
  --config "${BENCHMARK_CONFIG}" \
  --variant transformer_local \
  --model-config configs/model/transformer_local_matched.json \
  --train-steps 160 \
  --eval-batches 12 \
  --batch-size 32 \
  --learning-rate 0.002 \
  --answer-loss-weight 1 \
  --lm-loss-weight 0.25 \
  --output-dir "${OUTPUT_DIR}"
