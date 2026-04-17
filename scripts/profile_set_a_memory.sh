#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/outputs/set_a/memory_profile}"
SUITE_PATH="${2:-$ROOT_DIR/configs/experiment/set_a/suite_reasoning_context_sweep.json}"
TRAIN_CONFIG_PATH="${3:-$ROOT_DIR/configs/train/set_a/pilot_ultracompact.json}"

PYTHONPATH="$ROOT_DIR/src" python3 -m srd.eval.profile_set_a_memory \
  --suite "$SUITE_PATH" \
  --train-config "$TRAIN_CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR"
