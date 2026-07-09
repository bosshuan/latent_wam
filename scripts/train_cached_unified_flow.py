"""Unified flow smoke over cached real VJ-RAE latents.

This is the first Stage-A unified predictor smoke that consumes real cached
VJ-RAE latents plus real robot actions/proprio. By default it uses the tiny DiT
backbone, but it can also instantiate a Wan-style tiny trunk to validate the
M4 architecture wiring before loading the real 5B checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.cached_latent_robot_dataset import (
    CachedLatentRobotDataset,
    collate_cached_latent_robot,
)
from flow.losses import UnifiedLossWeights
from flow.schedulers import TimestepScheduler
from models.latent_world_action_dit import LatentWorldActionDiT
from models.wan_config import WanConfig
from models.wan_latent_world_action_dit import WanLatentWorldActionDiT
from models.weight_loading import load_wan_backbone
from train.train_unified_flow import EmaModel, StepInputs, unified_train_step, validate


def _cfg_get(cfg: dict, *path, default=None):
    cur = cfg
    for key in path:
        if cur is None:
            return default
        if hasattr(cur, "get"):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def _build_loader(
    cfg: dict,
    split: str,
    *,
    distributed_rank: int | None = None,
    distributed_world_size: int | None = None,
) -> DataLoader:
    dcfg = cfg.get("data", {})
    history_chunks = int(_cfg_get(cfg, "temporal", "history_chunks", default=4))
    future_chunks = int(_cfg_get(cfg, "temporal", "future_chunks", default=4))
    dataset = CachedLatentRobotDataset(
        schema_report=cfg["schema_report"],
        manifest_path=cfg["manifest_path"],
        history_chunks=history_chunks,
        future_chunks=future_chunks,
        tubelet=int(_cfg_get(cfg, "temporal", "tubelet", default=2)),
        start_item=int(dcfg.get(f"{split}_start_item", dcfg.get("start_item", 0))),
        index_modulus=dcfg.get(f"{split}_index_modulus", dcfg.get("index_modulus")),
        index_remainders=dcfg.get(f"{split}_index_remainders", dcfg.get("index_remainders")),
        max_items=int(dcfg.get(f"{split}_max_items", dcfg.get("max_items", 16))),
        control_stats_path=cfg.get("control_stats_path"),
    )
    sampler = None
    shuffle = bool(dcfg.get(f"{split}_shuffle", dcfg.get("shuffle", split == "train")))
    if distributed_rank is not None or distributed_world_size is not None:
        if distributed_rank is None or distributed_world_size is None:
            raise ValueError("distributed_rank and distributed_world_size must be set together")
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            num_replicas=int(distributed_world_size),
            rank=int(distributed_rank),
            shuffle=shuffle,
            seed=int(cfg.get("seed", 0)) + (10000 if split == "val" else 0),
            drop_last=bool(dcfg.get(f"{split}_sampler_drop_last", True)),
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=int(dcfg.get("batch_size", 2)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(dcfg.get("num_workers", 0)),
        collate_fn=collate_cached_latent_robot,
        drop_last=bool(
            dcfg.get(f"{split}_drop_last", dcfg.get("drop_last", True))
        ),
    )


def _to_step_inputs(batch: dict, cfg: dict, device: torch.device) -> StepInputs:
    latent = batch["latent"].to(device=device, dtype=torch.float32)
    history_chunks = int(_cfg_get(cfg, "temporal", "history_chunks", default=4))
    future_chunks = int(_cfg_get(cfg, "temporal", "future_chunks", default=4))
    if latent.shape[1] < history_chunks + future_chunks:
        raise ValueError(
            f"latent T={latent.shape[1]} < history+future={history_chunks + future_chunks}"
        )
    context = latent[:, :history_chunks]
    r1 = latent[:, history_chunks : history_chunks + future_chunks]
    actions = batch["actions"].to(device=device, dtype=torch.float32)
    a_mask = batch["action_mask"].to(device=device)
    proprio = None
    if batch["proprio"] is not None:
        proprio = batch["proprio"].to(device=device, dtype=torch.float32)

    return StepInputs(
        context_latent=context,
        r1=r1,
        actions=actions,
        a_mask=a_mask,
        proprio=proprio,
        action_valid=batch["action_valid"].to(device),
        embodiment_id=batch["embodiment_id"].to(device),
        action_schema_id=batch["action_schema_id"].to(device),
        text=list(batch["text"]),
    )


def _load_checkpoint_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - depends on server env
            raise ImportError(
                f"{path} is a safetensors checkpoint, but safetensors is not installed"
            ) from exc
        return load_file(str(path), device="cpu")

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch<2.0 compatibility on some debug images
        obj = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - checkpoint-format dependent
        print(f"[checkpoint] weights_only load failed for {path}: {exc}; retrying full load")
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module"):
            nested = obj.get(key)
            if isinstance(nested, dict):
                obj = nested
                break
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint {path} did not contain a state_dict-like mapping")
    return {k: v for k, v in obj.items() if torch.is_tensor(v)}


def _strip_prefix_if_all(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if sd and all(k.startswith(prefix) for k in sd):
        return {k[len(prefix) :]: v for k, v in sd.items()}
    return sd


def _load_state_dict(path_like: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path_like)
    if path.is_dir():
        files = sorted(path.glob("diffusion_pytorch_model*.safetensors"))
        if not files:
            files = sorted(path.glob("*.safetensors"))
        if not files:
            files = sorted(path.glob("*.pt")) + sorted(path.glob("*.pth")) + sorted(path.glob("*.bin"))
        if not files:
            raise FileNotFoundError(f"no checkpoint shards found under {path}")
    else:
        files = [path]

    merged: dict[str, torch.Tensor] = {}
    for file in files:
        part = _load_checkpoint_file(file)
        overlap = set(merged).intersection(part)
        if overlap:
            raise ValueError(f"duplicate checkpoint keys while merging {file}: {sorted(overlap)[:5]}")
        merged.update(part)

    for prefix in (
        "module.model.diffusion_model.",
        "model.diffusion_model.",
        "diffusion_model.",
        "module.model.",
        "module.",
        "model.",
    ):
        merged = _strip_prefix_if_all(merged, prefix)
    print(f"[checkpoint] loaded {len(merged)} tensor(s) from {path}")
    return merged


def _wan_config_from_model_cfg(mcfg: dict) -> WanConfig:
    if "wan" in mcfg:
        return WanConfig.from_dict(mcfg["wan"])
    if str(mcfg.get("type", "tiny")) == "wan":
        path = mcfg.get("wan_config_path", "configs/model/latent_wam_dit.yaml")
        key_path = tuple(mcfg.get("wan_key_path", ["wan"]))
        return WanConfig.from_yaml(path, key_path=key_path)
    return WanConfig(
        dim=192,
        num_layers=4,
        num_heads=4,
        ffn_dim=512,
        freq_dim=64,
        text_dim=64,
    )


def _wan_checkpoint_path(cfg: dict) -> str | None:
    candidates = (
        _cfg_get(cfg, "weights", "wan_checkpoint"),
        _cfg_get(cfg, "model", "weights", "wan_checkpoint"),
        _cfg_get(cfg, "model", "wan_checkpoint"),
    )
    for path in candidates:
        if path:
            return str(path)
    return None


def _build_model(cfg: dict) -> nn.Module:
    mcfg = cfg.get("model", {})
    model_type = str(mcfg.get("type", "tiny"))
    if model_type in {"wan_tiny", "wan"}:
        wan_cfg = _wan_config_from_model_cfg(mcfg)
        model = WanLatentWorldActionDiT(
            cfg=wan_cfg,
            latent_dim=int(mcfg.get("latent_dim", 384)),
            action_dim=int(mcfg.get("action_dim", 14)),
            num_embodiments=int(mcfg.get("num_embodiments", 2)),
            grid_hw=tuple(mcfg.get("grid_hw", [12, 12])),
            max_chunks=int(mcfg.get("max_chunks", 16)),
            max_actions=int(mcfg.get("max_actions", 12)),
            state_dim=int(mcfg.get("state_dim", 14)),
            text_seq_len=int(mcfg.get("text_seq_len", 8)),
            action_token_scale=float(mcfg.get("action_token_scale", 1.0)),
            action_latent_bridge_scale=float(mcfg.get("action_latent_bridge_scale", 0.0)),
        )
        ckpt_path = _wan_checkpoint_path(cfg)
        if ckpt_path:
            load_wan_backbone(model, _load_state_dict(ckpt_path), verbose=True)
        else:
            print(f"[wan-load] no Wan checkpoint provided for model.type={model_type}; random trunk")
        return model

    if model_type != "tiny":
        raise ValueError(f"unknown model.type={model_type!r}; expected tiny, wan_tiny, or wan")
    return LatentWorldActionDiT(
        latent_dim=int(mcfg.get("latent_dim", 384)),
        action_dim=int(mcfg.get("action_dim", 14)),
        hidden_dim=int(mcfg.get("hidden_dim", 128)),
        depth=int(mcfg.get("depth", 2)),
        heads=int(mcfg.get("heads", 4)),
        num_embodiments=int(mcfg.get("num_embodiments", 2)),
        grid_n=int(mcfg.get("grid_n", 144)),
        max_chunks=int(mcfg.get("max_chunks", 16)),
        max_actions=int(mcfg.get("max_actions", 12)),
        state_dim=int(mcfg.get("state_dim", 14)),
        text_dim=0,
        adaln_gate_init=float(mcfg.get("adaln_gate_init", 0.05)),
        action_token_scale=float(mcfg.get("action_token_scale", 1.0)),
        action_latent_bridge_scale=float(mcfg.get("action_latent_bridge_scale", 0.0)),
    )


def _save_checkpoint(model, ema: EmaModel, cfg: dict) -> Path:
    out_path = Path(cfg.get("out_path", "checkpoints/unified_cached_smoke.pt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "ema": {k: v.detach().cpu() for k, v in ema.shadow.items()},
            "config": cfg,
        },
        out_path,
    )
    return out_path


def _module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum())
    return total ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    if bool(cfg.get("allow_tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    train_loader = _build_loader(cfg, "train")
    val_loader = _build_loader(cfg, "val")
    val_batches = [
        _to_step_inputs(batch, cfg, device)
        for _, batch in zip(range(int(cfg.get("num_val_batches", 1))), val_loader)
    ]

    model = _build_model(cfg).to(device)
    ema = EmaModel(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    weight_fields = {f.name for f in dataclasses.fields(UnifiedLossWeights)}
    weights = UnifiedLossWeights(
        **{k: v for k, v in cfg.get("loss", {}).items() if k in weight_fields}
    )
    scheduler = TimestepScheduler(coupled=bool(_cfg_get(cfg, "timestep", "coupled", default=True)))

    max_steps = int(cfg.get("total_steps", 4))
    log_every = int(cfg.get("log_every", 1))
    val_every = int(cfg.get("val_every", 4))
    grad_probe = bool(_cfg_get(cfg, "debug", "grad_probe", default=False))
    model.train()
    batch_iter = cycle(train_loader)
    for step in range(max_steps):
        inp = _to_step_inputs(next(batch_iter), cfg, device)
        loss, metrics = unified_train_step(model, inp, scheduler, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_probe and step % log_every == 0:
            metrics["grad_action_encoder"] = _module_grad_norm(model.action_encoder)
            metrics["grad_action_to_latent"] = _module_grad_norm(model.action_to_latent)
            metrics["grad_latent_head"] = _module_grad_norm(model.latent_head)
        optimizer.step()
        ema.update(model)

        if step % log_every == 0:
            print(
                f"[unified-cached] step={step} "
                + " ".join(
                    f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in sorted(metrics.items())
                ),
                flush=True,
            )
        if val_batches and step % val_every == 0:
            vm = validate(model, val_batches, scheduler, weights)
            print(
                f"[unified-cached][val] total={vm['total']:.6f} "
                f"S_a={vm['S_a']:.6f} S_a_cos={vm['S_a_cos']:.6f} "
                f"delta_cond={vm['delta_cond']:.6f} cf_valid_frac={vm['cf_valid_frac']:.6f} "
                f"cf_action_delta={vm['cf_action_delta']:.6f} "
                f"cf_inconclusive={vm['cf_alarm_inconclusive']} "
                f"batch_alarm_count={vm['batch_alarm_count']} "
                f"collapse={vm['collapse']}",
                flush=True,
            )
            model.train()

    out_path = _save_checkpoint(model, ema, cfg)
    print(f"[unified-cached] saved {out_path}")
    print("[unified-cached] ok", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
