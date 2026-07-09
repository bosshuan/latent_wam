"""One real Wan2.2-5B FSDP forward/backward/update over cached robot data.

Rank 0 owns checkpoint IO and constructs the initialized CPU model. Other ranks
construct the same module on the meta device; FSDP materializes and synchronizes
them from rank 0. This avoids eight full CPU copies of the 5B checkpoint.

This script deliberately does not validate, create EMA weights, or save a model.
It proves the M4 distributed training path and exits after one optimizer step.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import math
from functools import partial

import torch
import torch.distributed as dist
import yaml

from flow.losses import UnifiedLossWeights
from flow.schedulers import TimestepScheduler
from models.wan_blocks import WanAttentionBlock
from models.wan_latent_world_action_dit import WanLatentWorldActionDiT
from models.weight_loading import load_wan_backbone
from scripts.train_cached_unified_flow import (
    _build_loader,
    _cfg_get,
    _load_state_dict,
    _to_step_inputs,
    _wan_checkpoint_path,
    _wan_config_from_model_cfg,
)
from train.dist_utils import barrier, cleanup, init_distributed, set_seed
from train.train_unified_flow import unified_train_step


_GRAD_GROUPS = {
    "backbone": ("backbone.",),
    "latent_adapter": ("latent_adapter.",),
    "latent_head": ("latent_head.",),
    "action_encoder": ("action_encoder.",),
    "action_to_latent": ("action_to_latent.",),
    "action_head": ("action_head.",),
    "state_adapter": ("state_adapter.",),
}
_UPDATE_PROBES = (
    "latent_head.proj.bias",
    "action_head.decoder.layer2.b",
)


def _new_model(cfg: dict) -> WanLatentWorldActionDiT:
    mcfg = cfg["model"]
    if str(mcfg.get("type")) != "wan":
        raise ValueError("real FSDP smoke requires model.type=wan")
    return WanLatentWorldActionDiT(
        cfg=_wan_config_from_model_cfg(mcfg),
        latent_dim=int(mcfg.get("latent_dim", 384)),
        action_dim=int(mcfg.get("action_dim", 14)),
        num_embodiments=int(mcfg.get("num_embodiments", 2)),
        grid_hw=tuple(mcfg.get("grid_hw", [12, 12])),
        max_chunks=int(mcfg.get("max_chunks", 16)),
        max_actions=int(mcfg.get("max_actions", 12)),
        state_dim=int(mcfg.get("state_dim", 14)),
        text_seq_len=int(mcfg.get("text_seq_len", 4)),
        action_token_scale=float(mcfg.get("action_token_scale", 1.0)),
        action_latent_bridge_scale=float(mcfg.get("action_latent_bridge_scale", 0.0)),
    )


def _rank_aware_model(cfg: dict, rank: int) -> WanLatentWorldActionDiT:
    if rank == 0:
        model = _new_model(cfg)
        checkpoint = _wan_checkpoint_path(cfg)
        if checkpoint is None:
            raise ValueError("weights.wan_checkpoint is required")
        state = _load_state_dict(checkpoint)
        load_wan_backbone(model, state, verbose=True)
        del state
        gc.collect()
        return model

    with torch.device("meta"):
        return _new_model(cfg)


def _materialize_meta(module: torch.nn.Module, device: torch.device) -> None:
    if any(p.is_meta for p in module.parameters(recurse=False)) or any(
        b.is_meta for b in module.buffers(recurse=False)
    ):
        module.to_empty(device=device, recurse=False)


def _wrap_fsdp(model, cfg: dict, device: torch.device):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    dcfg = cfg["distributed"]
    precision = str(dcfg.get("precision", "bf16"))
    if precision != "bf16":
        raise ValueError(f"FSDP smoke only supports bf16, got {precision!r}")
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    auto_wrap = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={WanAttentionBlock},
    )
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        param_init_fn=partial(_materialize_meta, device=device),
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,
    )

    if bool(dcfg.get("activation_checkpointing", True)):
        wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=wrapper,
            check_fn=lambda module: isinstance(module, WanAttentionBlock),
        )
    return fsdp_model


def _all_reduce_scalar(value: float, device: torch.device, op=dist.ReduceOp.SUM) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=op)
    return float(tensor)


def _matches(name: str, fragments: tuple[str, ...]) -> bool:
    return any(fragment in name for fragment in fragments)


def _gradient_report(model, device: torch.device) -> dict[str, float]:
    local = {group: 0.0 for group in _GRAD_GROUPS}
    local_nonfinite = 0.0
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        grad_f = grad.detach().float()
        if not bool(torch.isfinite(grad_f).all()):
            local_nonfinite = 1.0
        for group, fragments in _GRAD_GROUPS.items():
            if _matches(name, fragments):
                local[group] += float(grad_f.pow(2).sum())
                break

    if _all_reduce_scalar(local_nonfinite, device, op=dist.ReduceOp.MAX) != 0.0:
        raise FloatingPointError("non-finite gradients detected")
    return {
        group: math.sqrt(_all_reduce_scalar(sq, device))
        for group, sq in local.items()
    }


def _capture_update_probes(model) -> dict[str, torch.Tensor]:
    probes = {}
    for name, parameter in model.named_parameters():
        if _matches(name, _UPDATE_PROBES):
            probes[name] = parameter.detach().float().clone()
    return probes


def _max_probe_update(
    model,
    before: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    local_max = 0.0
    matched = 0.0
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        matched = 1.0
        if parameter.numel() > 0:
            delta = (parameter.detach().float() - before[name]).abs().max()
            local_max = max(local_max, float(delta))
    if _all_reduce_scalar(matched, device) == 0.0:
        raise RuntimeError("optimizer update probes did not match any FSDP parameter")
    return _all_reduce_scalar(local_max, device, op=dist.ReduceOp.MAX)


def _max_cuda_memory(device: torch.device) -> tuple[float, float]:
    allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    return (
        _all_reduce_scalar(allocated, device, op=dist.ReduceOp.MAX),
        _all_reduce_scalar(reserved, device, op=dist.ReduceOp.MAX),
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
            raise RuntimeError("real Wan FSDP smoke requires CUDA")

        set_seed(int(cfg.get("seed", 0)), ctx)
        if bool(cfg.get("allow_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(ctx.device)
        if ctx.is_main:
            print(
                f"[wan-fsdp] world_size={ctx.world_size} precision=bf16 "
                "sharding=FULL_SHARD activation_checkpointing="
                f"{bool(dcfg.get('activation_checkpointing', True))}",
                flush=True,
            )

        # Every rank uses the exact train modulo split. Rank-offset seeds make
        # shuffle choose different batches without shifting start_item (shifting
        # before modulo filtering would change the original train/val remainder).
        train_loader = _build_loader(cfg, "train")
        batch = next(iter(train_loader))
        inp = _to_step_inputs(batch, cfg, ctx.device)
        action_data_rms = math.sqrt(
            _all_reduce_scalar(float(inp.actions.float().pow(2).mean()), ctx.device)
            / ctx.world_size
        )
        if ctx.is_main:
            print(
                "[wan-fsdp] batch "
                f"context={tuple(inp.context_latent.shape)} "
                f"target={tuple(inp.r1.shape)} actions={tuple(inp.actions.shape)} "
                f"action_data_rms={action_data_rms:.6f}",
                flush=True,
            )

        raw_model = _rank_aware_model(cfg, ctx.rank)
        model = _wrap_fsdp(raw_model, cfg, ctx.device)
        model.train()
        barrier(ctx)

        weight_fields = {field.name for field in dataclasses.fields(UnifiedLossWeights)}
        weights = UnifiedLossWeights(
            **{k: v for k, v in cfg.get("loss", {}).items() if k in weight_fields}
        )
        scheduler = TimestepScheduler(
            coupled=bool(_cfg_get(cfg, "timestep", "coupled", default=True))
        )
        ocfg = cfg.get("optimizer", {})
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(ocfg.get("lr", 1.0e-5)),
            weight_decay=float(ocfg.get("weight_decay", 0.0)),
        )
        update_before = _capture_update_probes(model)

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = unified_train_step(
            model,
            inp,
            scheduler,
            weights,
            compute_monitors=False,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss on rank {ctx.rank}: {float(loss)}")
        loss.backward()

        grad_norms = _gradient_report(model, ctx.device)
        required_grad_groups = (
            "backbone",
            "latent_adapter",
            "latent_head",
            "action_encoder",
            "action_to_latent",
            "action_head",
        )
        missing_grad = [name for name in required_grad_groups if grad_norms[name] <= 0.0]
        if missing_grad:
            raise RuntimeError(f"zero gradient in required module(s): {missing_grad}")

        max_grad_norm = float(ocfg.get("max_grad_norm", 1.0))
        total_grad_norm = float(model.clip_grad_norm_(max_grad_norm))
        optimizer.step()
        update_max = _max_probe_update(model, update_before, ctx.device)
        if update_max <= 0.0:
            raise RuntimeError("optimizer step did not update latent/action head probes")

        loss_mean = _all_reduce_scalar(float(loss.detach()), ctx.device) / ctx.world_size
        metric_means = {
            name: _all_reduce_scalar(float(metrics[name]), ctx.device) / ctx.world_size
            for name in ("z_fm", "a_fm", "cf")
        }
        max_action_fm = float(
            _cfg_get(cfg, "smoke", "max_initial_action_fm", default=2.0)
        )
        if metric_means["a_fm"] > max_action_fm:
            raise RuntimeError(
                f"initial weighted action FM loss {metric_means['a_fm']:.6f} exceeds "
                f"smoke.max_initial_action_fm={max_action_fm:.6f}; check action "
                "normalization and action-head initialization"
            )
        alloc_gb, reserved_gb = _max_cuda_memory(ctx.device)
        if ctx.is_main:
            print(
                f"[wan-fsdp] loss_mean={loss_mean:.6f} "
                f"z_fm={metric_means['z_fm']:.6f} "
                f"a_fm={metric_means['a_fm']:.6f} "
                f"cf={metric_means['cf']:.6f}",
                flush=True,
            )
            print(
                "[wan-fsdp] grad_norms "
                + " ".join(f"{name}={value:.6f}" for name, value in grad_norms.items()),
                flush=True,
            )
            print(
                f"[wan-fsdp] total_grad_norm={total_grad_norm:.6f} "
                f"optimizer_update_max={update_max:.8f}",
                flush=True,
            )
            print(
                f"[wan-fsdp] peak_cuda_alloc_gb={alloc_gb:.2f} "
                f"peak_cuda_reserved_gb={reserved_gb:.2f}",
                flush=True,
            )
            print("[wan-fsdp] ok", flush=True)
        barrier(ctx)
    finally:
        cleanup(ctx)


if __name__ == "__main__":  # pragma: no cover
    main()
