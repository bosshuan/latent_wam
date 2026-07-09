"""Evaluate VJ-RAE action-discriminability on real robot clips.

The Stage-A VJ-RAE should compress pooled V-JEPA features without erasing the
action signal in temporal differences. This diagnostic trains two temporary
inverse-dynamics probes with identical settings:

* pooled V-JEPA feature delta -> action
* VJ-RAE latent delta -> action

Then it evaluates both on held-out windows and reports the relative MSE increase
of the compressed latent probe. The probes are diagnostic only; they are not
saved into the Stage-A model.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from models.vj_rae import ActionDiscriminabilityProbe, pooled_latent_delta
from scripts.cache_robot_latents import (
    _build_encoder,
    _build_loader,
    _cfg_get,
    _load_vj_rae,
    _preprocess_pixels,
)
from scripts.train_robot_vj_rae import _actions_to_transitions
from train.train_vj_rae import build_codec_from_encoder


def _safe_torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


def _split_cfg(cfg: dict, split: str) -> dict:
    out = copy.deepcopy(cfg)
    pcfg = cfg.get("probe_eval", {})
    data = out.setdefault("data", {})
    for src, dst in (
        (f"{split}_start_index", "start_index"),
        (f"{split}_samples_per_dataset", "max_samples_per_dataset"),
        (f"{split}_max_episodes_per_dataset", "max_episodes_per_dataset"),
        (f"{split}_episode_modulus", "episode_modulus"),
        (f"{split}_episode_remainders", "episode_remainders"),
        (f"{split}_batch_size", "batch_size"),
        (f"{split}_num_workers", "num_workers"),
    ):
        if src in pcfg:
            data[dst] = pcfg[src]
    if "batch_size" in pcfg and f"{split}_batch_size" not in pcfg:
        data["batch_size"] = pcfg["batch_size"]
    if "num_workers" in pcfg and f"{split}_num_workers" not in pcfg:
        data["num_workers"] = pcfg["num_workers"]
    out["log_prefix"] = str(pcfg.get(f"{split}_log_prefix", f"probe-{split}"))
    return out


def _build_codec(cfg: dict, encoder, device: torch.device):
    vcfg = cfg.get("vj_rae", {})
    codec = build_codec_from_encoder(
        encoder,
        hidden_dim=int(vcfg.get("hidden_dim", 1024)),
        latent_dim=int(vcfg.get("latent_dim", 384)),
        pool=int(vcfg.get("pool", 2)),
        norm_version=str(_cfg_get(cfg, "feature_cache", "norm_version", default="v0")),
    )
    _load_vj_rae(codec, cfg, device)
    codec.to(device)
    codec.freeze()
    return codec


@torch.no_grad()
def _extract_probe_batch(batch, cfg: dict, encoder, codec, device: torch.device) -> dict:
    pixels = _preprocess_pixels(batch.pixels, cfg, device)
    feats = encoder(pixels)
    latent = codec.encode(feats).latent
    latent_delta = pooled_latent_delta(latent)

    pooled_features = codec.pooled_target(feats).mean(dim=2)
    pooled_features = pooled_features.flatten(start_dim=-2)
    pooled_feature_delta = pooled_features[:, 1:] - pooled_features[:, :-1]

    if batch.actions is None:
        raise RuntimeError("VJ-RAE action probe requires action labels")
    actions = batch.actions.to(device=device, dtype=torch.float32, non_blocking=True)
    action_mask = None if batch.action_pad_mask is None else batch.action_pad_mask.to(device)
    action_target, transition_valid = _actions_to_transitions(
        actions,
        action_mask,
        feats.token_grid[0],
    )
    action_valid = batch.action_valid.to(device=device).unsqueeze(1) & transition_valid
    return {
        "latent_delta": latent_delta.detach(),
        "pooled_feature_delta": pooled_feature_delta.detach(),
        "action_target": action_target.detach(),
        "valid": action_valid.detach(),
    }


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    per = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
    mask = valid.to(dtype=per.dtype)
    return (per * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def _collect_probe_batches(
    cfg: dict,
    loader,
    encoder,
    codec,
    device: torch.device,
    max_batches: int,
) -> list[dict]:
    batches = []
    for step, batch_raw in enumerate(loader):
        if step >= max_batches:
            break
        batches.append(_extract_probe_batch(batch_raw, cfg, encoder, codec, device))
    if not batches:
        raise RuntimeError("probe collection saw zero batches")
    return batches


@torch.no_grad()
def _action_stats(
    batches: list[dict],
    eps: float = 1e-6,
    std_floor: float = 1e-3,
) -> dict[str, torch.Tensor]:
    action_dim = batches[0]["action_target"].shape[-1]
    device = batches[0]["action_target"].device
    total = torch.zeros(action_dim, device=device)
    total_sq = torch.zeros(action_dim, device=device)
    count = torch.zeros((), device=device)
    for batch in batches:
        target = batch["action_target"]
        mask = batch["valid"].to(dtype=target.dtype).unsqueeze(-1)
        total += (target * mask).sum(dim=(0, 1))
        total_sq += (target.square() * mask).sum(dim=(0, 1))
        count += mask.sum()
    if float(count.item()) <= 0.0:
        raise RuntimeError("probe action stats saw no valid transitions")
    mean = total / count.clamp_min(1.0)
    var = total_sq / count.clamp_min(1.0) - mean.square()
    std = torch.sqrt(var.clamp_min(eps)).clamp_min(float(std_floor))
    return {"mean": mean, "std": std, "count": count}


def _standardize_batches(batches: list[dict], stats: dict[str, torch.Tensor]) -> list[dict]:
    out = []
    mean = stats["mean"].view(1, 1, -1)
    std = stats["std"].view(1, 1, -1)
    for batch in batches:
        b = dict(batch)
        b["action_target_raw"] = batch["action_target"]
        b["action_target"] = (batch["action_target"] - mean) / std
        out.append(b)
    return out


def _flatten_probe_batches(batches: list[dict]) -> dict[str, torch.Tensor | int]:
    latent = []
    pooled = []
    action = []
    action_raw = []
    for batch in batches:
        valid = batch["valid"].reshape(-1).to(torch.bool)
        latent.append(batch["latent_delta"].reshape(-1, batch["latent_delta"].shape[-1])[valid])
        pooled.append(batch["pooled_feature_delta"].reshape(-1, batch["pooled_feature_delta"].shape[-1])[valid])
        action.append(batch["action_target"].reshape(-1, batch["action_target"].shape[-1])[valid])
        raw = batch.get("action_target_raw", batch["action_target"])
        action_raw.append(raw.reshape(-1, raw.shape[-1])[valid])
    flat = {
        "latent_delta": torch.cat(latent, dim=0),
        "pooled_feature_delta": torch.cat(pooled, dim=0),
        "action_target": torch.cat(action, dim=0),
        "action_target_raw": torch.cat(action_raw, dim=0),
    }
    flat["valid_transition_count"] = int(flat["action_target"].shape[0])
    if flat["valid_transition_count"] <= 0:
        raise RuntimeError("probe flatten saw no valid transitions")
    return flat


def _train_temporary_probes(
    cfg: dict,
    train_batches: list[dict],
    codec,
    device: torch.device,
) -> tuple[ActionDiscriminabilityProbe, ActionDiscriminabilityProbe]:
    pcfg = cfg.get("probe_eval", {})
    vcfg = cfg.get("vj_rae", {})
    latent_dim = int(vcfg.get("latent_dim", 384))
    pooled_dim = int(codec.num_layers * codec.embed_dim)
    action_dim = int(vcfg.get("probe_action_dim", 14))
    hidden_dim = int(pcfg.get("probe_hidden_dim", vcfg.get("probe_hidden_dim", 256)))

    latent_probe = ActionDiscriminabilityProbe(latent_dim, action_dim, hidden_dim).to(device)
    pooled_probe = ActionDiscriminabilityProbe(pooled_dim, action_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(latent_probe.parameters()) + list(pooled_probe.parameters()),
        lr=float(pcfg.get("lr", 1e-3)),
        weight_decay=float(pcfg.get("weight_decay", 0.0)),
    )
    max_steps = int(pcfg.get("train_steps", 200))
    log_every = int(pcfg.get("log_every", 25))
    probe_batch_size = int(pcfg.get("probe_batch_size", 256))
    train_flat = _flatten_probe_batches(train_batches)
    n = int(train_flat["valid_transition_count"])
    latent_probe.train()
    pooled_probe.train()

    for step in range(max_steps):
        idx = torch.randint(n, (min(probe_batch_size, n),), device=device)
        action_target = train_flat["action_target"][idx]
        latent_pred = latent_probe(train_flat["latent_delta"][idx])
        pooled_pred = pooled_probe(train_flat["pooled_feature_delta"][idx])
        latent_loss = F.mse_loss(latent_pred, action_target)
        pooled_loss = F.mse_loss(pooled_pred, action_target)
        loss = latent_loss + pooled_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % log_every == 0:
            print(
                "[vj-rae-probe] "
                f"step={step} latent_mse={float(latent_loss.detach()):.6f} "
                f"pooled_vjepa_mse={float(pooled_loss.detach()):.6f}",
                flush=True,
            )
    return latent_probe, pooled_probe


def _checkpoint_probe(cfg: dict, device: torch.device) -> ActionDiscriminabilityProbe | None:
    ckpt_path = _cfg_get(cfg, "vj_rae", "checkpoint_path", default=None)
    if not ckpt_path:
        return None
    ckpt = _safe_torch_load(ckpt_path)
    state = ckpt.get("probe")
    if not state:
        return None
    vcfg = cfg.get("vj_rae", {})
    probe = ActionDiscriminabilityProbe(
        int(vcfg.get("latent_dim", 384)),
        int(vcfg.get("probe_action_dim", 14)),
        hidden_dim=int(vcfg.get("probe_hidden_dim", 256)),
    ).to(device)
    result = probe.load_state_dict(state, strict=False)
    print(
        "[vj-rae-probe] loaded checkpoint probe "
        f"missing={list(result.missing_keys)} unexpected={list(result.unexpected_keys)}",
        flush=True,
    )
    probe.eval()
    return probe


@torch.no_grad()
def _eval_probes(
    cfg: dict,
    batches: list[dict],
    latent_probe,
    pooled_probe,
    checkpoint_probe,
    device: torch.device,
    split: str,
) -> dict[str, float | int | bool | None]:
    pcfg = cfg.get("probe_eval", {})
    latent_probe.eval()
    pooled_probe.eval()
    if checkpoint_probe is not None:
        checkpoint_probe.eval()

    flat = _flatten_probe_batches(batches)
    valid_count = int(flat["valid_transition_count"])
    has_checkpoint = checkpoint_probe is not None

    target = flat["action_target"]
    latent_mse = float(F.mse_loss(latent_probe(flat["latent_delta"]), target).item())
    pooled_mse = float(F.mse_loss(pooled_probe(flat["pooled_feature_delta"]), target).item())
    # Targets are standardized by train stats, so the train-mean baseline is
    # exactly zero for every split.
    baseline_mse = float(F.mse_loss(torch.zeros_like(target), target).item())
    checkpoint_raw_mse = None
    if has_checkpoint:
        checkpoint_raw_mse = float(
            F.mse_loss(
                checkpoint_probe(flat["latent_delta"]),
                flat["action_target_raw"],
            ).item()
        )

    rel_increase = latent_mse / max(pooled_mse, 1e-12) - 1.0
    max_rel_increase = float(pcfg.get("max_relative_mse_increase", 0.15))
    latent_r2 = 1.0 - latent_mse / max(baseline_mse, 1e-12)
    pooled_r2 = 1.0 - pooled_mse / max(baseline_mse, 1e-12)
    return {
        "split": split,
        "eval_batches": len(batches),
        "valid_transition_count": valid_count,
        "latent_mse": latent_mse,
        "pooled_vjepa_mse": pooled_mse,
        "mean_baseline_mse": baseline_mse,
        "checkpoint_probe_raw_mse": checkpoint_raw_mse,
        "latent_r2": latent_r2,
        "pooled_vjepa_r2": pooled_r2,
        "relative_mse_increase": rel_increase,
        "max_relative_mse_increase": max_rel_increase,
        "drop_ok": rel_increase <= max_rel_increase,
    }


def _write_reports(metrics: dict, cfg: dict) -> None:
    pcfg = cfg.get("probe_eval", {})
    json_path = Path(pcfg.get("out_json", "reports/vj_rae_action_probe.json"))
    md_path = Path(pcfg.get("out_md", json_path.with_suffix(".md")))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    status = metrics.get("status", "CHECK")
    md = [
        "# VJ-RAE Action-Discriminability Probe",
        "",
        f"Status: **{status}**",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in (
        "retention_ok",
        "predictive_ok",
        "strong_pass",
        "min_latent_r2",
        "max_relative_mse_increase",
        "action_std_mean",
        "action_std_min",
    ):
        value = metrics.get(key)
        if isinstance(value, float):
            md.append(f"| `{key}` | {value:.6f} |")
        else:
            md.append(f"| `{key}` | {value} |")
    md.append("")
    for split in ("train", "val"):
        md.extend([
            f"## {split.title()} Metrics",
            "",
            "| metric | value |",
            "| --- | ---: |",
        ])
        split_metrics = metrics.get(split, {})
        for key in (
            "latent_mse",
            "pooled_vjepa_mse",
            "mean_baseline_mse",
            "checkpoint_probe_raw_mse",
            "latent_r2",
            "pooled_vjepa_r2",
            "relative_mse_increase",
            "eval_batches",
            "valid_transition_count",
        ):
            value = split_metrics.get(key)
            if isinstance(value, float):
                md.append(f"| `{key}` | {value:.6f} |")
            else:
                md.append(f"| `{key}` | {value} |")
        md.append("")
    md.extend([
        "`retention_ok` checks whether VJ-RAE latent deltas are no worse than pooled "
        "V-JEPA feature deltas by the configured relative MSE margin.",
        "",
        "`predictive_ok` checks whether the standardized latent probe beats the "
        "train-mean action baseline on held-out data.",
    ])
    md_path.write_text("\n".join(md) + "\n")
    print(f"[vj-rae-probe] wrote {json_path}")
    print(f"[vj-rae-probe] wrote {md_path}")


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
    encoder = _build_encoder(cfg, device)
    codec = _build_codec(cfg, encoder, device)
    train_loader = _build_loader(_split_cfg(cfg, "train"))
    val_loader = _build_loader(_split_cfg(cfg, "val"))
    pcfg = cfg.get("probe_eval", {})
    train_batches_raw = _collect_probe_batches(
        cfg,
        train_loader,
        encoder,
        codec,
        device,
        max_batches=int(pcfg.get("train_eval_batches", pcfg.get("train_samples_per_dataset", 64))),
    )
    val_batches_raw = _collect_probe_batches(
        cfg,
        val_loader,
        encoder,
        codec,
        device,
        max_batches=int(pcfg.get("eval_batches", 32)),
    )
    stats = _action_stats(
        train_batches_raw,
        std_floor=float(pcfg.get("action_std_floor", 1e-3)),
    )
    train_batches = _standardize_batches(train_batches_raw, stats)
    val_batches = _standardize_batches(val_batches_raw, stats)
    print(
        "[vj-rae-probe] action_stats "
        f"count={int(stats['count'].item())} "
        f"std_mean={float(stats['std'].mean()):.6f} "
        f"std_min={float(stats['std'].min()):.6f}",
        flush=True,
    )

    latent_probe, pooled_probe = _train_temporary_probes(
        cfg,
        train_batches,
        codec,
        device,
    )
    ckpt_probe = _checkpoint_probe(cfg, device)
    train_metrics = _eval_probes(
        cfg,
        train_batches,
        latent_probe,
        pooled_probe,
        ckpt_probe,
        device,
        split="train",
    )
    val_metrics = _eval_probes(
        cfg,
        val_batches,
        latent_probe,
        pooled_probe,
        ckpt_probe,
        device,
        split="val",
    )
    max_rel = float(pcfg.get("max_relative_mse_increase", 0.15))
    min_r2 = float(pcfg.get("min_latent_r2", 0.0))
    retention_ok = bool(val_metrics["relative_mse_increase"] <= max_rel)
    predictive_ok = bool(val_metrics["latent_r2"] >= min_r2)
    metrics = {
        "status": "PASS" if retention_ok and predictive_ok else "CHECK",
        "retention_ok": retention_ok,
        "predictive_ok": predictive_ok,
        "strong_pass": bool(retention_ok and predictive_ok),
        "max_relative_mse_increase": max_rel,
        "min_latent_r2": min_r2,
        "action_std_mean": float(stats["std"].mean().item()),
        "action_std_min": float(stats["std"].min().item()),
        "train": train_metrics,
        "val": val_metrics,
    }
    print(
        "[vj-rae-probe] "
        f"status={metrics['status']} "
        f"retention_ok={retention_ok} predictive_ok={predictive_ok} "
        f"val_latent_r2={val_metrics['latent_r2']:.6f} "
        f"val_pooled_vjepa_r2={val_metrics['pooled_vjepa_r2']:.6f} "
        f"val_relative_mse_increase={val_metrics['relative_mse_increase']:.6f}",
        flush=True,
    )
    _write_reports(metrics, cfg)
    print("[vj-rae-probe] ok", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
