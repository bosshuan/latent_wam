#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/sfs_turbo/fyy/checkpoints/Wan2.2-TI2V-5B}"
OUT_DIR="${OUT_DIR:-reports/wan2_2}"

python scripts/inspect_wan_checkpoint.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --out-dir "${OUT_DIR}"
