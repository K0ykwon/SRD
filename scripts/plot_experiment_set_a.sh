#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m srd.eval.set_a_plot \
  --input-csv outputs/set_a/aggregates/aggregate_grouped.csv \
  --output-dir outputs/set_a/plots
