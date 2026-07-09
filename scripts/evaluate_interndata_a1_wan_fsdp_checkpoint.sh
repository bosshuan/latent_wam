#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/sfs_turbo/fyy/latent_wam}"
CONFIG="${CONFIG:-configs/data/interndata_a1_dual_arm_unified_cached_wan_real_fsdp_pilot.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/interndata_a1/wan_fsdp_pilot/step_000064}"
OUTPUT="${OUTPUT:-reports/interndata_a1/wan_fsdp_step64_aggregate_eval.json}"
MAX_BATCHES="${MAX_BATCHES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#GPU_IDS[@]}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/evaluate_wan_real_fsdp_checkpoint.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  --max-batches "${MAX_BATCHES}"
