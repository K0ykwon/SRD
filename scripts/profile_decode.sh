#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
python3 -m srd.eval.benchmark_runner --preset srd_suf_tiny --decode-steps 16
