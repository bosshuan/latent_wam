"""Sweep action-chunk offsets against cached VJ-RAE latent transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data.cached_latent_robot_dataset import (
    CachedLatentRobotDataset,
    collate_cached_latent_robot,
)


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


def _episode_holdout_indices(
    entries: list[dict],
    dataset_id: str | None = None,
    holdout_episode: int | None = None,
) -> tuple[list[int], list[int], dict]:
    groups: dict[str, dict[int, list[int]]] = {}
    for index, entry in enumerate(entries):
        ds = str(entry["dataset_id"])
        episode = int(entry["episode"])
        groups.setdefault(ds, {}).setdefault(episode, []).append(index)

    eligible = {
        ds: episodes for ds, episodes in groups.items() if len(episodes) >= 2
    }
    if dataset_id is None:
        if not eligible:
            raise ValueError("lag sweep needs one dataset with at least two episodes")
        dataset_id = max(
            eligible,
            key=lambda ds: sum(len(indices) for indices in eligible[ds].values()),
        )
    if dataset_id not in eligible:
        available = {ds: sorted(episodes) for ds, episodes in groups.items()}
        raise ValueError(
            f"dataset_id={dataset_id!r} lacks two episodes; available={available}"
        )

    episodes = eligible[dataset_id]
    if holdout_episode is None:
        holdout_episode = max(episodes)
    if holdout_episode not in episodes:
        raise ValueError(
            f"holdout_episode={holdout_episode} absent from "
            f"dataset_id={dataset_id!r}; episodes={sorted(episodes)}"
        )

    val_indices = list(episodes[holdout_episode])
    train_indices = [
        index
        for episode, indices in episodes.items()
        if episode != holdout_episode
        for index in indices
    ]
    if not train_indices or not val_indices:
        raise ValueError("episode holdout produced an empty train or val split")
    metadata = {
        "mode": "same_dataset_episode_holdout",
        "dataset_id": dataset_id,
        "train_episodes": sorted(
            episode for episode in episodes if episode != holdout_episode
        ),
        "holdout_episode": int(holdout_episode),
        "train_windows": len(train_indices),
        "val_windows": len(val_indices),
    }
    return train_indices, val_indices, metadata


def _build_episode_holdout_loaders(
    cfg: dict,
    dataset_id: str | None,
    holdout_episode: int | None,
):
    temporal = cfg["temporal"]
    dataset = CachedLatentRobotDataset(
        schema_report=cfg["schema_report"],
        manifest_path=cfg["manifest_path"],
        history_chunks=int(temporal["history_chunks"]),
        future_chunks=int(temporal["future_chunks"]),
        tubelet=int(temporal.get("tubelet", 2)),
        control_stats_path=cfg.get("control_stats_path"),
    )
    train_indices, val_indices, metadata = _episode_holdout_indices(
        dataset.entries,
        dataset_id=dataset_id,
        holdout_episode=holdout_episode,
    )
    batch_size = int(cfg.get("data", {}).get("batch_size", 2))

    def loader(indices: list[int]) -> DataLoader:
        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
            collate_fn=collate_cached_latent_robot,
            drop_last=False,
        )

    return loader(train_indices), loader(val_indices), metadata


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
    x_std = train_x.std(dim=0, unbiased=False)
    nonconstant = x_std[x_std > eps]
    if nonconstant.numel() == 0:
        raise ValueError("all latent-delta probe features are constant")
    x_scale_floor = nonconstant.median() * 0.05
    x_std = x_std.clamp_min(max(float(x_scale_floor), eps))
    y_mean = train_y.mean(dim=0)

    x_train = ((train_x - x_mean) / x_std).to(device)
    # Cached controls are already normalized by the global train-set control
    # statistics. Re-standardizing on one episode makes near-constant action
    # dimensions explode on the held-out episode.
    y_train = (train_y - y_mean).to(device)
    x_val = ((val_x - x_mean) / x_std).to(device)
    y_val = (val_y - y_mean).to(device)

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
    train_action_std = train_y.std(dim=0, unbiased=False)
    val_action_std = val_y.std(dim=0, unbiased=False)
    return {
        "train_mse": train_mse,
        "train_r2": train_r2,
        "val_mse": val_mse,
        "val_r2": val_r2,
        "train_action_std_mean": float(train_action_std.mean()),
        "train_action_std_min": float(train_action_std.min()),
        "val_action_std_mean": float(val_action_std.mean()),
        "val_action_std_min": float(val_action_std.min()),
        "action_mean_shift_rms": float(
            (val_y.mean(dim=0) - y_mean).square().mean().sqrt()
        ),
    }


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Cached Action-Lag Sweep",
        "",
        f"Current alignment: `lag=0`",
        f"Best held-out lag: `{report['best_lag']}`",
        f"Split: `{report['split']['mode']}`",
        f"Dataset: `{report['split']['dataset_id']}`",
        f"Train episodes: `{report['split']['train_episodes']}`",
        f"Holdout episode: `{report['split']['holdout_episode']}`",
        "",
        "| lag | train pairs | val pairs | train R2 | val R2 | action mean shift |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["results"]:
        lines.append(
            f"| {record['lag']} | {record['train_pairs']} | "
            f"{record['val_pairs']} | {record['train_r2']:.6f} | "
            f"{record['val_r2']:.6f} | "
            f"{record['action_mean_shift_rms']:.6f} |"
        )
    lines.extend(
        [
            "",
            "`lag=0` pairs each future latent transition with the action chunk "
            "currently used by the unified trainer. Negative lag uses an earlier "
            "action chunk.",
            "",
            "Train and validation windows come from different episodes of the same "
            "dataset, so no overlapping video window crosses the split. Use the "
            "relative held-out lag ranking as the alignment diagnostic.",
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
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--holdout-episode", type=int, default=None)
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

    train_loader, val_loader, split_metadata = _build_episode_holdout_loaders(
        cfg,
        dataset_id=args.dataset_id,
        holdout_episode=args.holdout_episode,
    )
    print(
        f"[action-lag] split={split_metadata['mode']} "
        f"dataset={split_metadata['dataset_id']} "
        f"train_episodes={split_metadata['train_episodes']} "
        f"holdout_episode={split_metadata['holdout_episode']} "
        f"train_windows={split_metadata['train_windows']} "
        f"val_windows={split_metadata['val_windows']}",
        flush=True,
    )
    train = _collect(
        train_loader,
        history_chunks,
        future_chunks,
        args.lags,
    )
    val = _collect(
        val_loader,
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
            f"val_r2={record['val_r2']:.6f} "
            f"action_mean_shift={record['action_mean_shift_rms']:.6f}",
            flush=True,
        )

    best = max(results, key=lambda record: record["val_r2"])
    report = {
        "config": args.config,
        "ridge": args.ridge,
        "current_lag": 0,
        "best_lag": int(best["lag"]),
        "split": split_metadata,
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
