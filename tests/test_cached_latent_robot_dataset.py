from __future__ import annotations

import json

import torch

from data.cached_latent_robot_dataset import (
    CachedLatentRobotDataset,
    CachedLatentRobotSample,
    collate_cached_latent_robot,
    split_future_actions,
)
from scripts.train_cached_unified_flow import _build_loader


def test_split_future_actions_drops_history_chunks():
    actions = torch.arange(8 * 12 * 2, dtype=torch.float32).reshape(8 * 12, 2)
    future, mask = split_future_actions(
        actions,
        t_tok=8,
        history_chunks=4,
        future_chunks=4,
    )
    assert future.shape == (4, 12, 2)
    assert mask.shape == (4, 12)
    assert torch.equal(future[0], actions.reshape(8, 12, 2)[4])
    assert mask.all()


def _sample(i: int) -> CachedLatentRobotSample:
    return CachedLatentRobotSample(
        latent=torch.ones(8, 144, 384) * i,
        actions=torch.ones(4, 12, 14) * i,
        action_mask=torch.ones(4, 12, dtype=torch.bool),
        proprio=torch.ones(14) * i,
        action_valid=True,
        embodiment_id=0,
        action_schema_id=0,
        dataset_id="ds",
        episode_index=0,
        frame_start=80 + i,
        frame_end=96 + i,
        sample_index=80 + i,
        text="task",
    )


def test_collate_cached_latent_robot_shapes():
    batch = collate_cached_latent_robot([_sample(0), _sample(1)])
    assert batch["latent"].shape == (2, 8, 144, 384)
    assert batch["actions"].shape == (2, 4, 12, 14)
    assert batch["action_mask"].shape == (2, 4, 12)
    assert batch["proprio"].shape == (2, 14)
    assert batch["action_valid"].tolist() == [True, True]
    assert batch["action_schema_id"].tolist() == [0, 0]


def test_cached_latent_dataset_slices_manifest_before_max_items(tmp_path):
    schema_path = tmp_path / "schema.json"
    manifest_path = tmp_path / "manifest.jsonl"
    schema_path.write_text(json.dumps([]))
    rows = [
        {
            "dataset_id": "ds",
            "sample_index": i,
            "cache_path": str(tmp_path / f"{i}.pt"),
        }
        for i in range(5)
    ]
    manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    ds = CachedLatentRobotDataset(
        schema_report=schema_path,
        manifest_path=manifest_path,
        start_item=2,
        max_items=2,
    )

    assert len(ds) == 2
    assert [entry["sample_index"] for entry in ds.entries] == [2, 3]


def test_cached_latent_dataset_supports_modulo_split(tmp_path):
    schema_path = tmp_path / "schema.json"
    manifest_path = tmp_path / "manifest.jsonl"
    schema_path.write_text(json.dumps([]))
    rows = [
        {
            "dataset_id": "ds",
            "sample_index": i,
            "cache_path": str(tmp_path / f"{i}.pt"),
        }
        for i in range(8)
    ]
    manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    ds = CachedLatentRobotDataset(
        schema_report=schema_path,
        manifest_path=manifest_path,
        index_modulus=4,
        index_remainders=[1, 3],
    )

    assert [entry["sample_index"] for entry in ds.entries] == [1, 3, 5, 7]


def test_cached_loader_distributed_sampler_partitions_without_overlap(tmp_path):
    schema_path = tmp_path / "schema.json"
    manifest_path = tmp_path / "manifest.jsonl"
    schema_path.write_text(json.dumps([]))
    rows = [
        {
            "dataset_id": "ds",
            "sample_index": i,
            "cache_path": str(tmp_path / f"{i}.pt"),
        }
        for i in range(8)
    ]
    manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    cfg = {
        "seed": 5,
        "schema_report": str(schema_path),
        "manifest_path": str(manifest_path),
        "data": {
            "batch_size": 2,
            "train_max_items": 8,
            "train_shuffle": False,
            "train_sampler_drop_last": True,
        },
    }

    rank0 = _build_loader(
        cfg, "train", distributed_rank=0, distributed_world_size=2
    )
    rank1 = _build_loader(
        cfg, "train", distributed_rank=1, distributed_world_size=2
    )
    indices0 = set(iter(rank0.sampler))
    indices1 = set(iter(rank1.sampler))
    assert indices0.isdisjoint(indices1)
    assert indices0 | indices1 == set(range(8))
