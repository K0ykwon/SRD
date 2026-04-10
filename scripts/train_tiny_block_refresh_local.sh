#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
python3 -m srd.training.train --preset block_refresh_local_tiny
