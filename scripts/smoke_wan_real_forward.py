"""Single-forward smoke for real Wan2.2-TI2V-5B weights.

This script is intentionally narrower than training:
  * load the official diffusion safetensor shards into our Wan-Latent-WAM model;
  * drop Wan VAE patch embed / pixel head via ``load_wan_backbone``;
  * move the model to one GPU in bf16/fp32;
  * run one no-grad forward over a cached VJ-RAE robot batch.

It is the bridge between metadata inspection and FSDP training. No optimizer,
EMA, checkpoint save, validation, or backward pass is created here.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
import yaml

from flow.interpolation import make_noisy
from scripts.train_cached_unified_flow import (
    _build_loader,
    _cfg_get,
    _load_state_dict,
    _to_step_inputs,
    _wan_config_from_model_cfg,
    _wan_checkpoint_path,
)
from models.wan_latent_world_action_dit import WanLatentWorldActionDiT
from models.weight_loading import load_wan_backbone


def _dtype_from_cfg(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unknown dtype {name!r}; expected bf16, fp16, or fp32")


def _build_wan_model(cfg: dict) -> WanLatentWorldActionDiT:
    mcfg = cfg.get("model", {})
    if str(mcfg.get("type", "wan")) != "wan":
        raise ValueError("real forward smoke requires model.type: wan")
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
    if not ckpt_path:
        raise ValueError("weights.wan_checkpoint is required for real Wan forward smoke")
    state = _load_state_dict(ckpt_path)
    load_wan_backbone(model, state, verbose=True)
    del state
    gc.collect()
    return model


def _memory(prefix: str, device: torch.device) -> None:
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"[wan-real-forward] {prefix} cuda_alloc_gb={alloc:.2f} cuda_reserved_gb={reserved:.2f}")


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
    dtype = _dtype_from_cfg(cfg.get("dtype", "bf16"))
    if device.type != "cuda" and dtype != torch.float32:
        print(f"[wan-real-forward][warning] dtype={dtype} requested on {device}; using fp32")
        dtype = torch.float32

    loader = _build_loader(cfg, "train")
    batch = next(iter(loader))
    inp = _to_step_inputs(batch, cfg, device)

    print(
        "[wan-real-forward] batch "
        f"context={tuple(inp.context_latent.shape)} "
        f"target={tuple(inp.r1.shape)} "
        f"actions={tuple(inp.actions.shape) if inp.actions is not None else None} "
        f"proprio={tuple(inp.proprio.shape) if inp.proprio is not None else None}",
        flush=True,
    )

    model = _build_wan_model(cfg)
    model.requires_grad_(False)
    model.eval()
    target_dtype = dtype if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=target_dtype)
    _memory("after_model_to_device", device)

    t = float(_cfg_get(cfg, "smoke", "timestep", default=0.5))
    t_z = torch.full((inp.r1.shape[0],), t, device=device, dtype=torch.float32)
    t_a = torch.full((inp.actions.shape[0],), t, device=device, dtype=torch.float32)
    noisy_latent, _, _ = make_noisy(inp.r1, t_z)
    noisy_action, _, _ = make_noisy(inp.actions, t_a)

    autocast_enabled = device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            out = model(
                context_latent=inp.context_latent,
                noisy_latent=noisy_latent,
                latent_timestep=t_z,
                action_timestep=t_a,
                noisy_action=noisy_action,
                action_valid=inp.action_valid,
                embodiment_id=inp.embodiment_id,
                proprio=inp.proprio,
                text=inp.text,
            )

    _memory("after_forward", device)
    action_shape = None if out.action_velocity is None else tuple(out.action_velocity.shape)
    action_abs = None
    if out.action_velocity is not None:
        action_abs = float(out.action_velocity.detach().float().abs().mean().cpu())
    print(
        "[wan-real-forward] output "
        f"latent_velocity={tuple(out.latent_velocity.shape)} "
        f"action_velocity={action_shape} "
        f"latent_abs_mean={float(out.latent_velocity.detach().float().abs().mean().cpu()):.6f} "
        f"action_abs_mean={action_abs if action_abs is None else round(action_abs, 6)}",
        flush=True,
    )
    print("[wan-real-forward] ok", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
