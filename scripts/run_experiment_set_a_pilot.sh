#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m srd.eval.benchmark_runner \
  --suite configs/experiment/set_a/suite_pilot.json \
  --train-config configs/train/set_a/pilot.json \
  --output-dir outputs/set_a/runs
