#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/sfs_turbo/fyy/latent_wam}"
CONFIG="${CONFIG:-configs/data/interndata_a1_dual_arm_unified_cached_wan_real_fsdp_pilot.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#GPU_IDS[@]}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

ARGS=(--config "${CONFIG}")
if [[ -n "${RESUME_FROM:-}" ]]; then
  ARGS+=(--resume "${RESUME_FROM}")
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/train_wan_real_fsdp_pilot.py "${ARGS[@]}"
