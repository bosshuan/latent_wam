"""Aggregate validation for a resumable real-Wan FSDP checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import yaml

from flow.losses import UnifiedLossWeights
from flow.schedulers import TimestepScheduler
from scripts.smoke_wan_real_fsdp_backward import (
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
    _optimizer,
)
from train.dist_utils import barrier, cleanup, init_distributed, set_seed
from train.fsdp_checkpoint import load_fsdp_checkpoint
from train.train_unified_flow import MIN_ACTION_SENSITIVITY


def _aggregate(metrics: list[dict], weights: UnifiedLossWeights) -> dict:
    keys = _LOSS_KEYS + _MONITOR_KEYS
    result = {
        key: sum(float(item[key]) for item in metrics) / len(metrics)
        for key in keys
    }
    result["batch_alarm_count"] = sum(bool(item["collapse"]) for item in metrics)
    result["cf_inconclusive"] = (
        result["cf_valid_frac"]
        < weights.min_counterfactual_valid_frac_for_alarm
    )
    result["collapse"] = bool(
        not result["cf_inconclusive"]
        and (
            result["S_a"] < MIN_ACTION_SENSITIVITY
            or result["delta_cond"] <= 0.0
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Per-rank validation batches; 0 evaluates the complete loader.",
    )
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
        if args.max_batches < 0:
            raise ValueError("--max-batches must be >= 0")

        set_seed(int(cfg.get("seed", 0)), ctx)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", True))
        torch.cuda.reset_peak_memory_stats(ctx.device)

        val_loader = _build_loader(
            cfg,
            "val",
            distributed_rank=ctx.rank,
            distributed_world_size=ctx.world_size,
        )
        if hasattr(val_loader.sampler, "set_epoch"):
            val_loader.sampler.set_epoch(0)

        raw_model = _rank_aware_model(cfg, ctx.rank)
        model = _wrap_fsdp(raw_model, cfg, ctx.device)
        optimizer = _optimizer(model, cfg)
        checkpoint_backend = str(
            cfg["pilot"].get("checkpoint_backend", "legacy_fsdp1")
        )
        step = load_fsdp_checkpoint(
            model,
            optimizer,
            args.checkpoint,
            backend=checkpoint_backend,
        )
        barrier(ctx)

        weight_fields = {field.name for field in dataclasses.fields(UnifiedLossWeights)}
        weights = UnifiedLossWeights(
            **{key: value for key, value in cfg["loss"].items() if key in weight_fields}
        )
        scheduler = TimestepScheduler(
            coupled=bool(_cfg_get(cfg, "timestep", "coupled", default=True))
        )
        eval_seed = int(cfg["pilot"].get("eval_seed", 2201))
        batch_metrics = []
        for batch_index, batch in enumerate(val_loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            inp = _to_step_inputs(batch, cfg, ctx.device)
            metrics = _evaluate(
                model,
                inp,
                scheduler,
                weights,
                ctx,
                eval_seed + batch_index * 1009,
                monitors=True,
            )
            batch_metrics.append(metrics)
            if ctx.is_main:
                print(
                    f"[wan-eval] batch={batch_index} "
                    f"{_format_metrics(metrics, _LOSS_KEYS + _MONITOR_KEYS)} "
                    f"cf_inconclusive={metrics['cf_inconclusive']} "
                    f"collapse={metrics['collapse']}",
                    flush=True,
                )

        if not batch_metrics:
            raise RuntimeError("validation loader produced zero evaluated batches")
        aggregate = _aggregate(batch_metrics, weights)
        per_rank_batches = len(batch_metrics)
        global_samples = (
            per_rank_batches
            * int(cfg["data"]["batch_size"])
            * ctx.world_size
        )
        result = {
            "checkpoint": str(args.checkpoint),
            "step": step,
            "world_size": ctx.world_size,
            "per_rank_batches": per_rank_batches,
            "global_samples": global_samples,
            **aggregate,
        }
        alloc_gb, reserved_gb = _max_cuda_memory(ctx.device)
        result["peak_cuda_alloc_gb"] = alloc_gb
        result["peak_cuda_reserved_gb"] = reserved_gb

        if ctx.is_main:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(
                f"[wan-eval] step={step} batches_per_rank={per_rank_batches} "
                f"global_samples={global_samples} "
                f"{_format_metrics(aggregate, _LOSS_KEYS + _MONITOR_KEYS)} "
                f"batch_alarm_count={aggregate['batch_alarm_count']} "
                f"cf_inconclusive={aggregate['cf_inconclusive']} "
                f"collapse={aggregate['collapse']}",
                flush=True,
            )
            print(
                f"[wan-eval] peak_cuda_alloc_gb={alloc_gb:.2f} "
                f"peak_cuda_reserved_gb={reserved_gb:.2f}",
                flush=True,
            )
            print(f"[wan-eval] wrote {output}", flush=True)
            print("[wan-eval] ok", flush=True)
    finally:
        cleanup(ctx)


if __name__ == "__main__":
    main()
