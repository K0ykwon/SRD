#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

for dir in outputs/reproduction/required_longctx_test outputs/reproduction/required_small_8k_test; do
  if [[ -d "$dir" ]]; then
    PYTHONPATH=src python3 -m srd.eval.set_a_aggregate --input-dir "$dir" --output-dir "$dir"
  fi
done

echo "snapshots updated"
