#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m srd.eval.benchmark_runner \
  --suite configs/experiment/set_a/suite_full.json \
  --train-config configs/train/set_a/main.json \
  --output-dir outputs/set_a/runs
