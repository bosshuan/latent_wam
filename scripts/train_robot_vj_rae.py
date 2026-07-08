"""Tiny real-data VJ-RAE training smoke.

This trains a small number of VJ-RAE steps on real robot-action paired clips
using a frozen local V-JEPA checkpoint. It is meant to validate the real-data
M2 path before launching larger VJ-RAE pretraining:

    real robot frames/actions -> frozen V-JEPA features -> normalizer fit
    -> VJ-RAE reconstruction + action-discriminability probe -> checkpoint
"""

from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path

import torch
import yaml

from flow.losses import CodecLossWeights
from models.vj_rae import ActionDiscriminabilityProbe
from scripts.cache_robot_latents import (
    _build_encoder,
    _build_loader,
    _cfg_get,
    _preprocess_pixels,
)
from train.train_vj_rae import build_codec_from_encoder, codec_train_step


def _actions_to_transitions(
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    t_tok: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool control actions to one target per latent transition.

    actions: [B, T_action, A], where T_action = T_tok * actions_per_chunk.
    Returns:
        action_target [B, T_tok-1, A]
        transition_mask [B, T_tok-1]
    """
    if actions.ndim != 3:
        raise ValueError(f"expected actions [B,T_action,A], got {tuple(actions.shape)}")
    b, steps, a_dim = actions.shape
    if steps < t_tok:
        raise ValueError(f"actions steps={steps} < t_tok={t_tok}")
    steps_per_token = steps // t_tok
    usable = steps_per_token * t_tok
    if usable != steps:
        raise ValueError(
            f"actions steps={steps} not divisible by t_tok={t_tok}; "
            "check temporal alignment"
        )

    chunks = actions[:, :usable].reshape(b, t_tok, steps_per_token, a_dim)
    if action_mask is None:
        mask = torch.ones(b, t_tok, steps_per_token, device=actions.device, dtype=actions.dtype)
    else:
        mask = action_mask[:, :usable].to(device=actions.device, dtype=actions.dtype)
        mask = mask.reshape(b, t_tok, steps_per_token)

    # Transition k predicts z_{k+1}-z_k, so use action chunk k+1 as the target.
    trans_chunks = chunks[:, 1:]
    trans_mask = mask[:, 1:]
    denom = trans_mask.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
    pooled = (trans_chunks * trans_mask.unsqueeze(-1)).sum(dim=2) / denom
    valid = trans_mask.any(dim=2)
    return pooled, valid


def _cpu_state_dict(module: torch.nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _cpu_stats(stats: dict) -> dict:
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in stats.items()}


@torch.no_grad()
def _fit_normalizer(codec, encoder, loader, cfg: dict, device: torch.device) -> int:
    max_batches = int(_cfg_get(cfg, "normalizer", "max_batches", default=4))
    seen = 0
    codec.normalizer.to(device)
    codec.normalizer.train(False)
    for batch in loader:
        pixels = _preprocess_pixels(batch.pixels, cfg, device)
        feats = encoder(pixels)
        codec.normalizer.update(feats.features)
        seen += 1
        print(
            f"[vj-rae-real] normalizer batch={seen} "
            f"features={tuple(feats.features.shape)}",
            flush=True,
        )
        if seen >= max_batches:
            break
    if seen <= 0:
        raise RuntimeError("normalizer saw zero batches")
    codec.normalizer.finalize()
    print(
        "[vj-rae-real] normalizer finalized "
        f"mean_abs={float(codec.normalizer.mean.abs().mean()):.6f} "
        f"var_mean={float(codec.normalizer.var.mean()):.6f}",
        flush=True,
    )
    return seen


def _save_checkpoint(codec, probe, cfg: dict) -> Path:
    out_path = Path(_cfg_get(cfg, "vj_rae", "out_path", default="checkpoints/vj_rae_real_smoke.pt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    codec.freeze()
    torch.save(
        {
            "vj_rae": _cpu_state_dict(codec),
            "codec": _cpu_state_dict(codec),  # backward-compatible key
            "probe": _cpu_state_dict(probe),
            "normalizer_stats": _cpu_stats(codec.normalizer.stats_state()),
            "config": cfg,
        },
        out_path,
    )
    return out_path


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

    device_name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name)
    loader = _build_loader(cfg)
    encoder = _build_encoder(cfg, device)
    vcfg = cfg.get("vj_rae", {})
    codec = build_codec_from_encoder(
        encoder,
        hidden_dim=int(vcfg.get("hidden_dim", 1024)),
        latent_dim=int(vcfg.get("latent_dim", 384)),
        pool=int(vcfg.get("pool", 2)),
        norm_version=str(_cfg_get(cfg, "feature_cache", "norm_version", default="v0")),
    ).to(device)

    _fit_normalizer(codec, encoder, loader, cfg, device)

    probe = ActionDiscriminabilityProbe(
        int(vcfg.get("latent_dim", 384)),
        int(vcfg.get("probe_action_dim", 14)),
        hidden_dim=int(vcfg.get("probe_hidden_dim", 256)),
    ).to(device)
    params = list(codec.parameters()) + list(probe.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=float(vcfg.get("lr", 1e-4)),
        weight_decay=float(vcfg.get("weight_decay", 0.0)),
    )
    weights = CodecLossWeights(**vcfg.get("loss", {}))

    max_steps = int(vcfg.get("max_steps", 8))
    log_every = int(vcfg.get("log_every", 1))
    batch_iter = cycle(loader)
    codec.train()
    probe.train()
    for step in range(max_steps):
        batch = next(batch_iter)
        pixels = _preprocess_pixels(batch.pixels, cfg, device)
        feats = encoder(pixels)
        if batch.actions is None:
            raise RuntimeError("VJ-RAE robot training requires action labels")
        actions = batch.actions.to(device=device, dtype=torch.float32, non_blocking=True)
        action_mask = None if batch.action_pad_mask is None else batch.action_pad_mask.to(device)
        action_pt, trans_valid = _actions_to_transitions(actions, action_mask, feats.token_grid[0])
        m_a = batch.action_valid.to(device=device).unsqueeze(1) & trans_valid

        logs = codec_train_step(
            codec,
            feats,
            optimizer,
            weights,
            probe=probe,
            action_per_transition=action_pt,
            m_a=m_a,
        )
        if step % log_every == 0:
            print(
                f"[vj-rae-real] step={step} "
                + " ".join(f"{k}={v:.6f}" for k, v in sorted(logs.items())),
                flush=True,
            )

    out_path = _save_checkpoint(codec, probe, cfg)
    print(f"[vj-rae-real] saved {out_path}")
    print("[vj-rae-real] ok", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
