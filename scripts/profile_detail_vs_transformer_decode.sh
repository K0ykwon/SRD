#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TASK="${TASK:-delayed_kv}"
MODEL_SIZE="${MODEL_SIZE:-compact}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
SEED="${SEED:-11}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DECODE_STEPS="${DECODE_STEPS:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/profiling/detail_vs_transformer_${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_${TASK}}"

PYTHONPATH=src python3 -m srd.eval.profile_decode_detail \
  --task "$TASK" \
  --model-size "$MODEL_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --decode-steps "$DECODE_STEPS" \
  --output-dir "$OUTPUT_DIR"
