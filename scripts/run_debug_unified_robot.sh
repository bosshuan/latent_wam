#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG="${CONFIG:-configs/debug/unified_robot_synthetic.yaml}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m train.train_unified_flow \
  --config "${CONFIG}" \
  --synthetic
