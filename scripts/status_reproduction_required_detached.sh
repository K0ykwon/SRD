#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-outputs/reproduction/logs}"

for name in required_longctx_resumable required_small_8k_resumable; do
  echo "[$name]"
  systemctl --user status "${name}.service" --no-pager || true
  log_file="$LOG_DIR/${name}.log"
  if [[ -f "$log_file" ]]; then
    tail -n 5 "$log_file" || true
  fi
  echo
done
