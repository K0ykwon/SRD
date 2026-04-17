#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m srd.eval.set_a_aggregate \
  --input-dir outputs/set_a/runs \
  --output-dir outputs/set_a/aggregates
