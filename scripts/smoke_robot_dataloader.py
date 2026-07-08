"""Real LeRobot dataloader smoke test.

This script intentionally stops before V-JEPA. It verifies that a scanned schema
report can be turned into robot-only ``TrajectoryBatch`` objects with real pixels,
concatenated action vectors, and concatenated proprio/state vectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data.collate import collate_trajectory
from data.robot_dataset import RobotSchemaBinding, RobotTrajectoryDataset


def _load_records(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _filter_records(records: list[dict], cfg: dict) -> list[dict]:
    fcfg = cfg.get("filter", {})
    embodiments = set(fcfg.get("embodiments", []))
    schema_ids = set(int(x) for x in fcfg.get("action_schema_ids", []))
    out = []
    for rec in records:
        if embodiments and rec["embodiment"] not in embodiments:
            continue
        if schema_ids and int(rec["action_schema_id"]) not in schema_ids:
            continue
        out.append(rec)
    return out


def _subset(ds, start_index: int, max_samples: int):
    n = len(ds)
    if n <= 0:
        raise ValueError("dataset has length 0")
    start = min(max(start_index, 0), n - 1)
    end = min(n, start + max_samples)
    return Subset(ds, list(range(start, end)))


def _build_dataset(rec: dict, cfg: dict):
    temporal = cfg.get("temporal", {})
    binding_cfg = cfg.get("binding", {})
    image_key = binding_cfg.get("image_key", "images.rgb.head")
    if image_key not in rec["camera_keys"]:
        raise ValueError(
            f"{rec['dataset_id']}: requested image_key={image_key!r} not in "
            f"camera_keys={rec['camera_keys']}"
        )

    binding = RobotSchemaBinding(
        image_key=image_key,
        state_keys=tuple(rec["state_keys"]),
        action_keys=tuple(rec["action_keys"]),
        embodiment_id=int(rec["embodiment_id"]),
        action_schema_id=int(rec["action_schema_id"]),
        fps=float(rec["fps"]),
        dataset_id=str(rec["dataset_id"]),
    )
    return RobotTrajectoryDataset(
        repo_id=rec["repo_or_path"],
        binding=binding,
        token_grid=tuple(int(x) for x in temporal.get("token_grid", [8, 24, 24])),
        history_chunks=int(temporal.get("history_chunks", 4)),
        future_chunks=int(temporal.get("future_chunks", 4)),
        tubelet=int(temporal.get("tubelet", 2)),
        backend=str(cfg.get("reader_backend", "auto")),
    )


def _shape(x):
    return None if x is None else tuple(x.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    records = _filter_records(_load_records(cfg["schema_report"]), cfg)
    if not records:
        raise SystemExit(f"no schema records selected by {args.config}")

    dcfg = cfg.get("data", {})
    start_index = int(dcfg.get("start_index", 0))
    max_samples = int(dcfg.get("max_samples_per_dataset", 16))
    datasets = []
    print(f"[smoke] selected {len(records)} dataset(s)")
    for rec in records:
        ds = _build_dataset(rec, cfg)
        sub = _subset(ds, start_index=start_index, max_samples=max_samples)
        datasets.append(sub)
        print(
            "[smoke] dataset "
            f"id={rec['dataset_id']} len={len(ds)} subset_len={len(sub)} "
            f"image={cfg.get('binding', {}).get('image_key', 'images.rgb.head')} "
            f"state_dim={rec['selected_state_dim']} action_dim={rec['selected_action_dim']} "
            f"embodiment={rec['embodiment_id']} schema={rec['action_schema_id']}",
            flush=True,
        )

    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=int(dcfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(dcfg.get("num_workers", 0)),
        collate_fn=collate_trajectory,
    )

    max_batches = int(dcfg.get("max_batches", 2))
    for step, batch in enumerate(loader):
        print(
            f"[smoke] batch={step} "
            f"pixels={_shape(batch.pixels)} "
            f"actions={_shape(batch.actions)} "
            f"proprio={_shape(batch.proprio)} "
            f"action_mask={_shape(batch.action_pad_mask)} "
            f"embodiment_ids={batch.embodiment_id.tolist()} "
            f"schema_ids={batch.action_schema_id.tolist()} "
            f"fps={batch.fps_meta.tolist()}",
            flush=True,
        )
        if batch.pixels is None or batch.actions is None or batch.proprio is None:
            raise RuntimeError("real robot smoke expected pixels/actions/proprio")
        if batch.actions.shape[-1] != 14:
            raise RuntimeError(f"dual-arm smoke expected action_dim=14, got {batch.actions.shape[-1]}")
        if batch.proprio.shape[-1] != 14:
            raise RuntimeError(f"dual-arm smoke expected state_dim=14, got {batch.proprio.shape[-1]}")
        if step + 1 >= max_batches:
            break

    print("[smoke] ok", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
