"""M5 short real-Wan FSDP fixed-batch overfit with pre/post probes.

This is deliberately between the one-step M4 backward smoke and a resumable
training job. It proves repeated optimizer steps reduce latent and action flow
losses on fixed real robot batches, while an independent validation batch
reports action-conditioning monitors. It does not save a 5B checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import math

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
from scripts.train_cached_unified_flow import (
    _build_loader,
    _cfg_get,
    _to_step_inputs,
)
from train.dist_utils import barrier, cleanup, init_distributed, set_seed
from train.train_unified_flow import MIN_ACTION_SENSITIVITY, unified_train_step


_LOSS_KEYS = ("total", "z_fm", "a_fm", "clean", "cf")
_MONITOR_KEYS = (
    "S_a",
    "S_a_cos",
    "delta_cond",
    "cf_valid_frac",
    "cf_action_delta",
)


def _global_means(metrics: dict, keys: tuple[str, ...], ctx) -> dict[str, float]:
    return {
        key: _all_reduce_scalar(float(metrics[key]), ctx.device) / ctx.world_size
        for key in keys
    }


@torch.no_grad()
def _evaluate(model, inp, scheduler, weights, ctx, seed: int, monitors: bool) -> dict:
    model.eval()
    generator = torch.Generator(device=ctx.device)
    generator.manual_seed(int(seed) + ctx.rank)
    _loss, metrics = unified_train_step(
        model,
        inp,
        scheduler,
        weights,
        generator=generator,
        compute_monitors=monitors,
    )
    keys = _LOSS_KEYS + (_MONITOR_KEYS if monitors else ())
    result = _global_means(metrics, keys, ctx)
    if monitors:
        result["cf_inconclusive"] = (
            result["cf_valid_frac"]
            < weights.min_counterfactual_valid_frac_for_alarm
        )
        result["collapse"] = (
            not result["cf_inconclusive"]
            and (
                result["S_a"] < MIN_ACTION_SENSITIVITY
                or result["delta_cond"] <= 0.0
            )
        )
    model.train()
    return result


def _format_metrics(metrics: dict, keys: tuple[str, ...]) -> str:
    return " ".join(f"{key}={metrics[key]:.6f}" for key in keys)


def _optimizer(model, cfg: dict):
    ocfg = cfg["optimizer"]
    backbone = []
    new_modules = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "backbone." in name:
            backbone.append(parameter)
        else:
            new_modules.append(parameter)
    if not backbone or not new_modules:
        raise RuntimeError(
            f"optimizer grouping failed: backbone={len(backbone)} new={len(new_modules)}"
        )
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(ocfg["backbone_lr"])},
            {"params": new_modules, "lr": float(ocfg["new_module_lr"])},
        ],
        weight_decay=float(ocfg.get("weight_decay", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    dcfg = cfg.get("distributed", {})
    ctx = init_distributed(backend=str(dcfg.get("backend", "nccl")))
    try:
        min_world_size = int(dcfg.get("min_world_size", 2))
        if not ctx.distributed or ctx.world_size < min_world_size:
            raise RuntimeError(
                f"launch with torchrun world_size>={min_world_size}; "
                f"got world_size={ctx.world_size}"
            )
        if ctx.device.type != "cuda":
            raise RuntimeError("real Wan short training requires CUDA")
        set_seed(int(cfg.get("seed", 0)), ctx)
        if bool(cfg.get("allow_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(ctx.device)

        train_batch = next(iter(_build_loader(cfg, "train")))
        val_batch = next(iter(_build_loader(cfg, "val")))
        train_inp = _to_step_inputs(train_batch, cfg, ctx.device)
        val_inp = _to_step_inputs(val_batch, cfg, ctx.device)
        if ctx.is_main:
            print(
                f"[wan-short] world_size={ctx.world_size} "
                f"train={tuple(train_inp.context_latent.shape)} "
                f"val={tuple(val_inp.context_latent.shape)}",
                flush=True,
            )

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
        scfg = cfg["short_train"]
        eval_seed = int(scfg.get("eval_seed", 1701))

        train_pre = _evaluate(
            model, train_inp, scheduler, weights, ctx, eval_seed, monitors=False
        )
        val_pre = _evaluate(
            model, val_inp, scheduler, weights, ctx, eval_seed + 1000, monitors=True
        )
        if ctx.is_main:
            print(f"[wan-short][train-pre] {_format_metrics(train_pre, _LOSS_KEYS)}")
            print(
                f"[wan-short][val-pre] {_format_metrics(val_pre, _LOSS_KEYS + _MONITOR_KEYS)} "
                f"cf_inconclusive={val_pre['cf_inconclusive']} "
                f"collapse={val_pre['collapse']}",
                flush=True,
            )

        steps = int(scfg.get("steps", 8))
        log_every = int(scfg.get("log_every", 1))
        max_grad_norm = float(cfg["optimizer"].get("max_grad_norm", 1.0))
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = unified_train_step(
                model,
                train_inp,
                scheduler,
                weights,
                compute_monitors=False,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite train loss on rank {ctx.rank} step={step}"
                )
            loss.backward()
            grad_norm = float(model.clip_grad_norm_(max_grad_norm))
            optimizer.step()

            if step % log_every == 0:
                means = _global_means(metrics, _LOSS_KEYS, ctx)
                grad_norm = _all_reduce_scalar(
                    grad_norm, ctx.device, op=dist.ReduceOp.MAX
                )
                if ctx.is_main:
                    print(
                        f"[wan-short] step={step} {_format_metrics(means, _LOSS_KEYS)} "
                        f"grad_norm={grad_norm:.6f}",
                        flush=True,
                    )

        train_post = _evaluate(
            model, train_inp, scheduler, weights, ctx, eval_seed, monitors=False
        )
        val_post = _evaluate(
            model, val_inp, scheduler, weights, ctx, eval_seed + 1000, monitors=True
        )
        overfit_ok = all(train_post[key] < train_pre[key] for key in ("total", "z_fm", "a_fm"))
        if ctx.is_main:
            print(f"[wan-short][train-post] {_format_metrics(train_post, _LOSS_KEYS)}")
            print(
                f"[wan-short][val-post] {_format_metrics(val_post, _LOSS_KEYS + _MONITOR_KEYS)} "
                f"cf_inconclusive={val_post['cf_inconclusive']} "
                f"collapse={val_post['collapse']}",
                flush=True,
            )
            alloc_gb, reserved_gb = _max_cuda_memory(ctx.device)
            print(
                f"[wan-short] train_overfit_ok={overfit_ok} "
                f"peak_cuda_alloc_gb={alloc_gb:.2f} "
                f"peak_cuda_reserved_gb={reserved_gb:.2f}",
                flush=True,
            )
        else:
            # _max_cuda_memory contains collectives and must run on every rank.
            _max_cuda_memory(ctx.device)
        if not overfit_ok:
            raise RuntimeError(
                "fixed-batch overfit failed: expected total, z_fm, and a_fm to decrease"
            )
        if ctx.is_main:
            print("[wan-short] ok", flush=True)
        barrier(ctx)
    finally:
        cleanup(ctx)


if __name__ == "__main__":  # pragma: no cover
    main()
