#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-outputs/reproduction/audit}"
export PYTHONPATH="${PYTHONPATH:-src}"

python3 -m srd.eval.reproduction_audit --output-dir "${OUTPUT_DIR}"
