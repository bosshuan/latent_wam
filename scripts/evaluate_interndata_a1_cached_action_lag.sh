#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/sfs_turbo/fyy/latent_wam}"
CONFIG="${CONFIG:-configs/data/interndata_a1_dual_arm_unified_cached_wan_real_fsdp_pilot.yaml}"
OUTPUT_JSON="${OUTPUT_JSON:-reports/interndata_a1/cached_action_lag.json}"
OUTPUT_MD="${OUTPUT_MD:-reports/interndata_a1/cached_action_lag.md}"
DEVICE="${DEVICE:-cuda}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python scripts/evaluate_cached_action_lag.py \
  --config "${CONFIG}" \
  --output-json "${OUTPUT_JSON}" \
  --output-md "${OUTPUT_MD}" \
  --device "${DEVICE}"
