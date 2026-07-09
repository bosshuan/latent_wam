#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/sfs_turbo/fyy/latent_wam}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG="${CONFIG:-configs/data/interndata_a1_dual_arm_unified_cached_wan_real_fsdp_backward_smoke.yaml}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/smoke_wan_real_fsdp_backward.py --config "${CONFIG}"
