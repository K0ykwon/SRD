#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/artifacts/detail_150m_pilot}"
LOG_PATH="${OUTPUT_DIR}/run.log"

mkdir -p "${OUTPUT_DIR}"

cd "${ROOT_DIR}"
PYTHONUNBUFFERED=1 PYTHONPATH=src python3 scripts/run_block_refresh_large_suite.py \
  --config configs/experiment/detail_150m_eval.json \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"
