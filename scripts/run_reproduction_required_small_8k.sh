#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SUITE_PATH="${SUITE_PATH:-configs/experiment/set_a/suite_reproduction_required_small_8k.json}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-configs/train/set_a/reproduction_required_small_8k_16gb.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/reproduction/required_small_8k}"
MAX_RUNS="${MAX_RUNS:-}"

CMD=(
  python3 -m srd.eval.benchmark_runner
  --suite "$SUITE_PATH"
  --train-config "$TRAIN_CONFIG_PATH"
  --output-dir "$OUTPUT_DIR"
  --include-ablations
)

if [[ -n "$MAX_RUNS" ]]; then
  CMD+=(--max-runs "$MAX_RUNS")
fi

PYTHONPATH=src "${CMD[@]}"
