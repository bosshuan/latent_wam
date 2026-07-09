#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/sfs_turbo/fyy/latent_wam}"
export CONFIG="${CONFIG:-configs/data/interndata_a1_dual_arm_unified_cached_wan_real_fsdp_action_recovery.yaml}"
export RESUME_FROM="${RESUME_FROM:-checkpoints/interndata_a1/wan_fsdp_pilot/step_000064}"
export TOTAL_STEPS="${TOTAL_STEPS:-128}"
export METRICS_PATH="${METRICS_PATH:-reports/interndata_a1/wan_fsdp_action_recovery_128.jsonl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"

bash "${PROJECT_ROOT}/scripts/train_interndata_a1_dual_arm_wan_fsdp_pilot.sh"
