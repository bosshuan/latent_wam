"""Fit fixed action/state statistics from the cached robot train split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from data.control_normalizer import RunningMoments
from scripts.train_cached_unified_flow import _build_loader


def _finalize(groups: dict[int, RunningMoments], std_floor: float) -> dict[str, dict]:
    return {
        str(group_id): groups[group_id].finalize(std_floor)
        for group_id in sorted(groups)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("control_stats_path"):
        raise ValueError("stats fitting config must not normalize its own input")

    torch.manual_seed(int(cfg.get("seed", 0)))
    loader = _build_loader(cfg, "train")
    action_moments: dict[int, RunningMoments] = defaultdict(RunningMoments)
    state_moments: dict[int, RunningMoments] = defaultdict(RunningMoments)

    batches = 0
    samples = 0
    for batch in loader:
        batches += 1
        samples += int(batch["actions"].shape[0])
        schema_ids = batch["action_schema_id"]
        for schema_id in schema_ids.unique().tolist():
            rows = schema_ids == int(schema_id)
            actions = batch["actions"][rows]
            mask = batch["action_mask"][rows]
            action_moments[int(schema_id)].update(actions[mask])

        if batch["proprio"] is not None:
            embodiment_ids = batch["embodiment_id"]
            for embodiment_id in embodiment_ids.unique().tolist():
                rows = embodiment_ids == int(embodiment_id)
                state_moments[int(embodiment_id)].update(batch["proprio"][rows])

    if not action_moments:
        raise RuntimeError("control stats loader produced no actions")
    std_floor = float(cfg.get("stats", {}).get("std_floor", 1.0e-3))
    payload = {
        "format_version": 1,
        "source_manifest": str(cfg["manifest_path"]),
        "split": {
            "episode_modulus": cfg["data"].get("train_episode_modulus"),
            "episode_remainders": cfg["data"].get(
                "train_episode_remainders"
            ),
            "index_modulus": cfg["data"].get("train_index_modulus"),
            "index_remainders": cfg["data"].get("train_index_remainders"),
            "max_items": cfg["data"].get("train_max_items"),
        },
        "std_floor": std_floor,
        "num_batches": batches,
        "num_samples": samples,
        "actions": _finalize(action_moments, std_floor),
        "states": _finalize(state_moments, std_floor) if state_moments else {},
    }

    out_path = Path(cfg["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[control-stats] batches={batches} samples={samples} "
        f"action_schemas={sorted(payload['actions'])} "
        f"embodiments={sorted(payload['states'])}",
        flush=True,
    )
    for schema_id, record in payload["actions"].items():
        mean_abs = sum(abs(x) for x in record["mean"]) / record["dim"]
        std_mean = sum(record["std"]) / record["dim"]
        print(
            f"[control-stats] action_schema={schema_id} count={record['count']} "
            f"dim={record['dim']} mean_abs={mean_abs:.6f} std_mean={std_mean:.6f}",
            flush=True,
        )
    for embodiment_id, record in payload["states"].items():
        mean_abs = sum(abs(x) for x in record["mean"]) / record["dim"]
        std_mean = sum(record["std"]) / record["dim"]
        print(
            f"[control-stats] embodiment={embodiment_id} count={record['count']} "
            f"dim={record['dim']} mean_abs={mean_abs:.6f} std_mean={std_mean:.6f}",
            flush=True,
        )
    print(f"[control-stats] wrote {out_path}")
    print("[control-stats] ok", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
