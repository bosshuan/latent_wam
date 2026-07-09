"""Save and restore a full real-Wan FSDP model+Adam DCP checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from flow.losses import UnifiedLossWeights
from flow.schedulers import TimestepScheduler
from scripts.smoke_wan_real_fsdp_backward import (
    _all_reduce_scalar,
    _rank_aware_model,
    _wrap_fsdp,
)
from scripts.train_cached_unified_flow import _build_loader, _cfg_get, _to_step_inputs
from scripts.train_wan_real_fsdp_short import _LOSS_KEYS, _evaluate, _optimizer
from train.dist_utils import barrier, cleanup, init_distributed, set_seed
from train.fsdp_checkpoint import load_fsdp_checkpoint, save_fsdp_checkpoint
from train.train_unified_flow import unified_train_step


def _train_one(model, optimizer, inp, scheduler, weights, max_grad_norm: float) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss, _metrics = unified_train_step(
        model,
        inp,
        scheduler,
        weights,
        compute_monitors=False,
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("checkpoint smoke encountered non-finite loss")
    loss.backward()
    grad_norm = float(model.clip_grad_norm_(max_grad_norm))
    optimizer.step()
    return grad_norm


def _optimizer_step_range(optimizer, device: torch.device) -> tuple[float, float]:
    local_steps = []
    for state in optimizer.state.values():
        step = state.get("step")
        if step is not None:
            local_steps.append(float(step))
    local_min = min(local_steps) if local_steps else float("inf")
    local_max = max(local_steps) if local_steps else float("-inf")
    global_min = _all_reduce_scalar(local_min, device, op=dist.ReduceOp.MIN)
    global_max = _all_reduce_scalar(local_max, device, op=dist.ReduceOp.MAX)
    if global_min == float("inf") or global_max == float("-inf"):
        raise RuntimeError("Adam optimizer has no initialized step state")
    return global_min, global_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    dcfg = cfg["distributed"]
    if str(dcfg.get("sharding_strategy", "full_shard")).lower() != "full_shard":
        raise ValueError("checkpoint round-trip smoke currently requires FULL_SHARD")
    ctx = init_distributed(backend=str(dcfg.get("backend", "nccl")))
    try:
        min_world_size = int(dcfg.get("min_world_size", 2))
        if not ctx.distributed or ctx.world_size < min_world_size:
            raise RuntimeError(
                f"launch with torchrun world_size>={min_world_size}; "
                f"got world_size={ctx.world_size}"
            )
        set_seed(int(cfg.get("seed", 0)), ctx)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", True))

        batch = next(iter(_build_loader(cfg, "train")))
        inp = _to_step_inputs(batch, cfg, ctx.device)
        raw_model = _rank_aware_model(cfg, ctx.rank)
        model = _wrap_fsdp(raw_model, cfg, ctx.device)
        model.train()
        barrier(ctx)

        weight_fields = {field.name for field in dataclasses.fields(UnifiedLossWeights)}
        weights = UnifiedLossWeights(
            **{key: value for key, value in cfg["loss"].items() if key in weight_fields}
        )
        scheduler = TimestepScheduler(
            coupled=bool(_cfg_get(cfg, "timestep", "coupled", default=True))
        )
        optimizer = _optimizer(model, cfg)
        max_grad_norm = float(cfg["optimizer"].get("max_grad_norm", 1.0))
        scfg = cfg["checkpoint_smoke"]
        checkpoint_backend = str(scfg.get("backend", "legacy_fsdp1"))
        eval_seed = int(scfg.get("eval_seed", 1901))

        first_grad = _train_one(
            model, optimizer, inp, scheduler, weights, max_grad_norm
        )
        reference = _evaluate(
            model, inp, scheduler, weights, ctx, eval_seed, monitors=False
        )
        checkpoint_dir = save_fsdp_checkpoint(
            model,
            optimizer,
            cfg["checkpoint_dir"],
            step=1,
            metadata={
                "config": args.config,
                "world_size": ctx.world_size,
                "sharding_strategy": "full_shard",
            },
            backend=checkpoint_backend,
        )
        saved_step_range = _optimizer_step_range(optimizer, ctx.device)

        second_grad = _train_one(
            model, optimizer, inp, scheduler, weights, max_grad_norm
        )
        perturbed = _evaluate(
            model, inp, scheduler, weights, ctx, eval_seed, monitors=False
        )
        loaded_step = load_fsdp_checkpoint(
            model,
            optimizer,
            checkpoint_dir,
            backend=checkpoint_backend,
        )
        restored_step_range = _optimizer_step_range(optimizer, ctx.device)
        restored = _evaluate(
            model, inp, scheduler, weights, ctx, eval_seed, monitors=False
        )

        restore_diff = max(
            abs(restored[key] - reference[key]) for key in _LOSS_KEYS
        )
        perturb_diff = max(
            abs(perturbed[key] - reference[key]) for key in _LOSS_KEYS
        )
        tolerance = float(scfg.get("restore_tolerance", 1.0e-5))
        step_ok = (
            loaded_step == 1
            and saved_step_range == (1.0, 1.0)
            and restored_step_range == (1.0, 1.0)
        )
        restore_ok = restore_diff <= tolerance and step_ok

        if ctx.is_main:
            size_bytes = sum(
                path.stat().st_size
                for path in Path(checkpoint_dir).rglob("*")
                if path.is_file()
            )
            print(
                f"[dcp-smoke] first_grad_norm={first_grad:.6f} "
                f"second_grad_norm={second_grad:.6f}",
                flush=True,
            )
            print(
                f"[dcp-smoke] perturb_diff={perturb_diff:.8f} "
                f"restore_diff={restore_diff:.8f} tolerance={tolerance:.8f}",
                flush=True,
            )
            print(
                f"[dcp-smoke] loaded_step={loaded_step} "
                f"backend={checkpoint_backend} "
                f"saved_optimizer_steps={saved_step_range} "
                f"restored_optimizer_steps={restored_step_range} "
                f"checkpoint_gb={size_bytes / (1024**3):.2f}",
                flush=True,
            )
            print(f"[dcp-smoke] restore_ok={restore_ok}", flush=True)
        if not restore_ok:
            raise RuntimeError("DCP checkpoint round-trip did not restore exact state")
        if ctx.is_main:
            print("[dcp-smoke] ok", flush=True)
        barrier(ctx)
    finally:
        cleanup(ctx)


if __name__ == "__main__":  # pragma: no cover
    main()
