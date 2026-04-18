#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-outputs/reproduction/logs}"
mkdir -p "$LOG_DIR"

LONGCTX_OUTPUT_DIR="${LONGCTX_OUTPUT_DIR:-outputs/reproduction/required_longctx_test}"
SMALL8K_OUTPUT_DIR="${SMALL8K_OUTPUT_DIR:-outputs/reproduction/required_small_8k_test}"

systemctl --user stop required_longctx_resumable.service >/dev/null 2>&1 || true
systemctl --user stop required_small_8k_resumable.service >/dev/null 2>&1 || true

for unit in required_longctx_resumable required_small_8k_resumable; do
  for _ in {1..20}; do
    load_state="$(systemctl --user show "${unit}.service" -p LoadState --value 2>/dev/null || true)"
    if [[ "$load_state" == "not-found" || -z "$load_state" ]]; then
      break
    fi
    sleep 0.2
  done
done

systemd-run --user --unit=required_longctx_resumable \
  bash -lc "cd '$ROOT_DIR' && export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src && python3 -m srd.eval.benchmark_runner --suite configs/experiment/set_a/suite_reproduction_required_longctx.json --train-config configs/train/set_a/reproduction_required_longctx_16gb.json --output-dir '$LONGCTX_OUTPUT_DIR' --include-ablations --skip-existing >> '$LOG_DIR/required_longctx_resumable.log' 2>&1"

systemd-run --user --unit=required_small_8k_resumable \
  bash -lc "cd '$ROOT_DIR' && export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src && python3 -m srd.eval.benchmark_runner --suite configs/experiment/set_a/suite_reproduction_required_small_8k.json --train-config configs/train/set_a/reproduction_required_small_8k_16gb.json --output-dir '$SMALL8K_OUTPUT_DIR' --include-ablations --skip-existing >> '$LOG_DIR/required_small_8k_resumable.log' 2>&1"

echo "required_longctx_resumable unit=required_longctx_resumable.service"
echo "required_small_8k_resumable unit=required_small_8k_resumable.service"
