#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/sfs_turbo/fyy/latent_wam}"
export CONFIG="${CONFIG:-configs/data/interndata_a1_dual_arm_unified_cached_wan_real_fsdp_pilot_broad.yaml}"
export CHECKPOINT="${CHECKPOINT:-checkpoints/interndata_a1/wan_fsdp_pilot_broad/step_000256}"
export OUTPUT="${OUTPUT:-reports/interndata_a1/wan_fsdp_pilot_broad_step256_eval_192.json}"
export MAX_BATCHES="${MAX_BATCHES:-32}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"

bash "${PROJECT_ROOT}/scripts/evaluate_interndata_a1_wan_fsdp_checkpoint.sh"
