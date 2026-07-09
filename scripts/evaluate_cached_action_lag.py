"""Sweep action-chunk offsets against cached VJ-RAE latent transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.train_cached_unified_flow import _build_loader


def _aligned_pairs(
    latent_delta: torch.Tensor,
    action_chunks: torch.Tensor,
    action_valid: torch.Tensor,
    lag: int,
    action_origin: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align delta[j] with full-window action[action_origin + j + lag]."""
    if latent_delta.ndim != 3 or action_chunks.ndim != 3:
        raise ValueError("latent_delta and action_chunks must be [B,T,C]")
    if latent_delta.shape[0] != action_chunks.shape[0]:
        raise ValueError("latent and action batch axes must match")
    if action_valid.shape != action_chunks.shape[:2]:
        raise ValueError("action_valid must match the full action window")

    action_start = int(action_origin) + int(lag)
    action_stop = action_start + latent_delta.shape[1]
    if action_start < 0 or action_stop > action_chunks.shape[1]:
        raise ValueError(
            f"lag={lag} selects action chunks [{action_start}:{action_stop}] "
            f"outside window T={action_chunks.shape[1]}"
        )

    actions = action_chunks[:, action_start:action_stop]
    valid = action_valid[:, action_start:action_stop]
    return latent_delta[valid], actions[valid]


def _collect(
    loader,
    history_chunks: int,
    future_chunks: int,
    lags: list[int],
) -> dict[int, dict]:
    collected: dict[int, dict[str, list[torch.Tensor]]] = {}
    for lag in lags:
        collected[lag] = {"latent_delta": [], "action": []}

    for batch in loader:
        latent = batch["latent"].float().mean(dim=2)
        future = latent[
            :, history_chunks : history_chunks + future_chunks
        ]
        previous = torch.cat(
            [latent[:, history_chunks - 1 : history_chunks], future[:, :-1]],
            dim=1,
        )
        latent_delta = future - previous

        actions = batch["action_window"].float()
        action_mask = batch["action_window_mask"].bool()
        denom = action_mask.sum(dim=2, keepdim=True).clamp_min(1)
        action_chunks = (
            actions * action_mask.unsqueeze(-1)
        ).sum(dim=2) / denom
        chunk_valid = action_mask.any(dim=2) & batch["action_valid"].unsqueeze(1)

        for lag in lags:
            x, y = _aligned_pairs(
                latent_delta,
                action_chunks,
                chunk_valid,
                lag,
                action_origin=history_chunks,
            )
            collected[lag]["latent_delta"].append(x.cpu())
            collected[lag]["action"].append(y.cpu())

    result = {}
    for lag, fields in collected.items():
        result[lag] = {
            key: torch.cat(parts, dim=0) for key, parts in fields.items()
        }
    return result


def _fit_ridge(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    ridge: float,
    device: torch.device,
) -> dict[str, float]:
    eps = 1.0e-6
    x_mean = train_x.mean(dim=0)
    x_std = train_x.std(dim=0, unbiased=False).clamp_min(eps)
    y_mean = train_y.mean(dim=0)
    y_std = train_y.std(dim=0, unbiased=False).clamp_min(eps)

    x_train = ((train_x - x_mean) / x_std).to(device)
    y_train = ((train_y - y_mean) / y_std).to(device)
    x_val = ((val_x - x_mean) / x_std).to(device)
    y_val = ((val_y - y_mean) / y_std).to(device)

    n = x_train.shape[0]
    covariance = x_train.T @ x_train / n
    rhs = x_train.T @ y_train / n
    identity = torch.eye(covariance.shape[0], device=device)
    weights = torch.linalg.solve(covariance + ridge * identity, rhs)

    def metrics(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        prediction = x @ weights
        mse = float((prediction - y).square().mean().item())
        baseline = float(y.square().mean().item())
        return mse, 1.0 - mse / max(baseline, 1.0e-12)

    train_mse, train_r2 = metrics(x_train, y_train)
    val_mse, val_r2 = metrics(x_val, y_val)
    return {
        "train_mse": train_mse,
        "train_r2": train_r2,
        "val_mse": val_mse,
        "val_r2": val_r2,
    }


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Cached Action-Lag Sweep",
        "",
        f"Current alignment: `lag=0`",
        f"Best held-out lag: `{report['best_lag']}`",
        "",
        "| lag | train pairs | val pairs | train R2 | val R2 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["results"]:
        lines.append(
            f"| {record['lag']} | {record['train_pairs']} | "
            f"{record['val_pairs']} | {record['train_r2']:.6f} | "
            f"{record['val_r2']:.6f} |"
        )
    lines.extend(
        [
            "",
            "`lag=0` pairs each future latent transition with the action chunk "
            "currently used by the unified trainer. Negative lag uses an earlier "
            "action chunk; positive lag uses a later action chunk.",
            "",
            "Use the relative lag ranking as an alignment diagnostic. The manifest "
            "contains overlapping windows, so these R2 values are not a standalone "
            "generalization benchmark.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    parser.add_argument("--lags", type=int, nargs="+", default=[-2, -1, 0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.ridge <= 0:
        raise ValueError("--ridge must be positive")
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    torch.manual_seed(int(cfg.get("seed", 0)))
    history_chunks = int(cfg["temporal"]["history_chunks"])
    future_chunks = int(cfg["temporal"]["future_chunks"])
    for lag in args.lags:
        if lag > 0:
            raise ValueError(
                f"lag={lag} needs action chunks beyond the cached target window; "
                "equal-support causal sweep requires lag <= 0"
            )
        if history_chunks + lag < 0:
            raise ValueError(
                f"lag={lag} exceeds available history_chunks={history_chunks}"
            )

    train = _collect(
        _build_loader(cfg, "train"),
        history_chunks,
        future_chunks,
        args.lags,
    )
    val = _collect(
        _build_loader(cfg, "val"),
        history_chunks,
        future_chunks,
        args.lags,
    )
    device = torch.device(args.device)
    results = []
    for lag in args.lags:
        train_pairs = train[lag]
        val_pairs = val[lag]
        metrics = _fit_ridge(
            train_pairs["latent_delta"],
            train_pairs["action"],
            val_pairs["latent_delta"],
            val_pairs["action"],
            args.ridge,
            device,
        )
        record = {
            "lag": lag,
            "train_pairs": int(train_pairs["action"].shape[0]),
            "val_pairs": int(val_pairs["action"].shape[0]),
            **metrics,
        }
        results.append(record)
        print(
            f"[action-lag] lag={lag:+d} "
            f"train_pairs={record['train_pairs']} "
            f"val_pairs={record['val_pairs']} "
            f"train_r2={record['train_r2']:.6f} "
            f"val_r2={record['val_r2']:.6f}",
            flush=True,
        )

    best = max(results, key=lambda record: record["val_r2"])
    report = {
        "config": args.config,
        "ridge": args.ridge,
        "current_lag": 0,
        "best_lag": int(best["lag"]),
        "results": results,
    }
    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(md_path, report)
    print(
        f"[action-lag] current_lag=0 best_lag={best['lag']:+d} "
        f"best_val_r2={best['val_r2']:.6f}",
        flush=True,
    )
    print(f"[action-lag] wrote {json_path}", flush=True)
    print(f"[action-lag] wrote {md_path}", flush=True)
    print("[action-lag] ok", flush=True)


if __name__ == "__main__":
    main()
