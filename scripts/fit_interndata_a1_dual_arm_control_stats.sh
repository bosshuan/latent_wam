#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/sfs_turbo/fyy/latent_wam}"
CONFIG="${CONFIG:-configs/data/interndata_a1_dual_arm_control_stats.yaml}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python scripts/fit_cached_robot_control_stats.py --config "${CONFIG}"
