#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/sfs_turbo/fyy/latent_wam}"
CONFIG="${CONFIG:-configs/data/interndata_a1_dual_arm_vj_rae_train_broad.yaml}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python scripts/cache_robot_latents.py \
  --config "${CONFIG}" \
  --plan-only
