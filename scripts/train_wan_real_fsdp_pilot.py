"""Streaming real-Wan FSDP pilot with validation, JSONL logs, and DCP resume."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from flow.losses import UnifiedLossWeights
from flow.schedulers import TimestepScheduler
from scripts.smoke_wan_real_fsdp_backward import (
    _all_reduce_scalar,
    _max_cuda_memory,
    _rank_aware_model,
    _wrap_fsdp,
)
from scripts.train_cached_unified_flow import _build_loader, _cfg_get, _to_step_inputs
from scripts.train_wan_real_fsdp_short import (
    _LOSS_KEYS,
    _MONITOR_KEYS,
    _evaluate,
    _format_metrics,
    _global_means,
    _optimizer,
)
from train.dist_utils import barrier, cleanup, init_distributed, set_seed
from train.fsdp_checkpoint import load_fsdp_checkpoint, save_fsdp_checkpoint
from train.train_unified_flow import unified_train_step


def _write_jsonl(path: str | Path, record: dict, is_main: bool) -> None:
    if not is_main:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _position_train_iterator(loader, completed_steps: int):
    batches_per_epoch = len(loader)
    if batches_per_epoch <= 0:
        raise RuntimeError("streaming train loader has zero batches")
    epoch, offset = divmod(int(completed_steps), batches_per_epoch)
    sampler = loader.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(offset):
        next(iterator)
    return epoch, iterator


def _next_train_batch(loader, epoch: int, iterator):
    try:
        return epoch, iterator, next(iterator)
    except StopIteration:
        epoch += 1
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        iterator = iter(loader)
        return epoch, iterator, next(iterator)


def _checkpoint_path(root: str | Path, step: int) -> Path:
    return Path(root) / f"step_{int(step):06d}"


def _pilot_acceptance(initial_val: dict, final_val: dict, cfg: dict) -> dict:
    max_ratio = float(cfg.get("max_final_val_total_ratio", 1.10))
    min_sensitivity = float(cfg.get("min_final_action_sensitivity", 0.01))
    min_delta_cond = float(cfg.get("min_final_delta_cond", -1.0e-3))
    val_ratio = final_val["total"] / max(initial_val["total"], 1.0e-12)
    checks = {
        "val_ratio": val_ratio,
        "max_val_ratio": max_ratio,
        "ratio_ok": val_ratio <= max_ratio,
        "cf_conclusive_ok": (
            not bool(cfg.get("require_cf_conclusive", True))
            or not final_val["cf_inconclusive"]
        ),
        "action_sensitivity_ok": final_val["S_a"] >= min_sensitivity,
        "delta_cond_ok": final_val["delta_cond"] >= min_delta_cond,
        "min_action_sensitivity": min_sensitivity,
        "min_delta_cond": min_delta_cond,
    }
    checks["pilot_ok"] = bool(
        checks["ratio_ok"]
        and checks["cf_conclusive_ok"]
        and checks["action_sensitivity_ok"]
        and checks["delta_cond_ok"]
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    dcfg = cfg["distributed"]
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
        torch.cuda.reset_peak_memory_stats(ctx.device)

        train_loader = _build_loader(
            cfg,
            "train",
            distributed_rank=ctx.rank,
            distributed_world_size=ctx.world_size,
        )
        val_loader = _build_loader(
            cfg,
            "val",
            distributed_rank=ctx.rank,
            distributed_world_size=ctx.world_size,
        )
        if hasattr(val_loader.sampler, "set_epoch"):
            val_loader.sampler.set_epoch(0)
        val_inp = _to_step_inputs(next(iter(val_loader)), cfg, ctx.device)

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
        pcfg = cfg["pilot"]
        checkpoint_backend = str(pcfg.get("checkpoint_backend", "legacy_fsdp1"))
        resume_from = args.resume or pcfg.get("resume_from")
        completed_steps = 0
        if resume_from:
            completed_steps = load_fsdp_checkpoint(
                model,
                optimizer,
                resume_from,
                backend=checkpoint_backend,
            )

        total_steps = int(pcfg.get("total_steps", 32))
        if completed_steps >= total_steps:
            raise ValueError(
                f"resume step={completed_steps} is not below total_steps={total_steps}"
            )
        epoch, train_iterator = _position_train_iterator(
            train_loader, completed_steps
        )
        metrics_path = pcfg["metrics_path"]
        eval_seed = int(pcfg.get("eval_seed", 2201))
        val_pre = _evaluate(
            model,
            val_inp,
            scheduler,
            weights,
            ctx,
            eval_seed,
            monitors=True,
        )
        if ctx.is_main:
            print(
                f"[wan-pilot] world_size={ctx.world_size} start_step={completed_steps} "
                f"batches_per_epoch={len(train_loader)}",
                flush=True,
            )
            print(
                f"[wan-pilot][val] step={completed_steps} "
                f"{_format_metrics(val_pre, _LOSS_KEYS + _MONITOR_KEYS)} "
                f"cf_inconclusive={val_pre['cf_inconclusive']} "
                f"collapse={val_pre['collapse']}",
                flush=True,
            )
        _write_jsonl(
            metrics_path,
            {"kind": "val", "step": completed_steps, **val_pre},
            ctx.is_main,
        )

        log_every = int(pcfg.get("log_every", 2))
        val_every = int(pcfg.get("val_every", 8))
        max_grad_norm = float(cfg["optimizer"].get("max_grad_norm", 1.0))
        final_val = val_pre
        while completed_steps < total_steps:
            epoch, train_iterator, batch = _next_train_batch(
                train_loader, epoch, train_iterator
            )
            inp = _to_step_inputs(batch, cfg, ctx.device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = unified_train_step(
                model,
                inp,
                scheduler,
                weights,
                compute_monitors=False,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite pilot loss rank={ctx.rank} step={completed_steps}"
                )
            loss.backward()
            grad_norm = float(model.clip_grad_norm_(max_grad_norm))
            optimizer.step()
            completed_steps += 1

            if completed_steps % log_every == 0:
                means = _global_means(metrics, _LOSS_KEYS, ctx)
                grad_norm = _all_reduce_scalar(
                    grad_norm, ctx.device, op=dist.ReduceOp.MAX
                )
                if ctx.is_main:
                    print(
                        f"[wan-pilot] step={completed_steps} "
                        f"{_format_metrics(means, _LOSS_KEYS)} "
                        f"grad_norm={grad_norm:.6f}",
                        flush=True,
                    )
                _write_jsonl(
                    metrics_path,
                    {
                        "kind": "train",
                        "step": completed_steps,
                        "grad_norm": grad_norm,
                        **means,
                    },
                    ctx.is_main,
                )

            if completed_steps % val_every == 0 or completed_steps == total_steps:
                final_val = _evaluate(
                    model,
                    val_inp,
                    scheduler,
                    weights,
                    ctx,
                    eval_seed,
                    monitors=True,
                )
                if ctx.is_main:
                    print(
                        f"[wan-pilot][val] step={completed_steps} "
                        f"{_format_metrics(final_val, _LOSS_KEYS + _MONITOR_KEYS)} "
                        f"cf_inconclusive={final_val['cf_inconclusive']} "
                        f"collapse={final_val['collapse']}",
                        flush=True,
                    )
                _write_jsonl(
                    metrics_path,
                    {"kind": "val", "step": completed_steps, **final_val},
                    ctx.is_main,
                )

        acceptance = _pilot_acceptance(val_pre, final_val, pcfg)
        checkpoint_dir = _checkpoint_path(
            pcfg["checkpoint_root"], completed_steps
        )
        if checkpoint_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing checkpoint: {checkpoint_dir}"
            )
        save_fsdp_checkpoint(
            model,
            optimizer,
            checkpoint_dir,
            step=completed_steps,
            metadata={
                "config": args.config,
                "world_size": ctx.world_size,
                "sharding_strategy": dcfg.get("sharding_strategy", "full_shard"),
                "val_total_ratio": acceptance["val_ratio"],
                "strict_final_collapse": final_val["collapse"],
                "pilot_ok": acceptance["pilot_ok"],
            },
            backend=checkpoint_backend,
        )
        alloc_gb, reserved_gb = _max_cuda_memory(ctx.device)
        _write_jsonl(
            metrics_path,
            {
                "kind": "acceptance",
                "step": completed_steps,
                "strict_collapse": final_val["collapse"],
                **acceptance,
            },
            ctx.is_main,
        )
        if ctx.is_main:
            print(
                f"[wan-pilot] val_total_ratio={acceptance['val_ratio']:.6f} "
                f"S_a_ok={acceptance['action_sensitivity_ok']} "
                f"delta_cond_ok={acceptance['delta_cond_ok']} "
                f"strict_collapse={final_val['collapse']} "
                f"checkpoint={checkpoint_dir}",
                flush=True,
            )
            print(
                f"[wan-pilot] peak_cuda_alloc_gb={alloc_gb:.2f} "
                f"peak_cuda_reserved_gb={reserved_gb:.2f}",
                flush=True,
            )
        if not acceptance["pilot_ok"]:
            raise RuntimeError(
                "pilot acceptance failed after checkpoint save: "
                f"{acceptance}"
            )
        if ctx.is_main:
            print("[wan-pilot] ok", flush=True)
        barrier(ctx)
    finally:
        cleanup(ctx)


if __name__ == "__main__":  # pragma: no cover
    main()
