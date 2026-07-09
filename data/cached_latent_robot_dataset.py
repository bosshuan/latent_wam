"""Cached VJ-RAE latent + real robot action dataset.

This is the bridge between M2 cache generation and M5 unified flow training:
latents are read from the VJ-RAE cache manifest, while actions/proprio are read
again from the original local LeRobot-v2.1-format dataset. Keeping labels out of
the latent cache avoids duplicating supervision and keeps cache files
representation-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from data.control_normalizer import FixedControlNormalizer
from data.lerobot_v21_direct import DirectLeRobotV21Dataset
from data.robot_dataset import _concat_current_features, _concat_time_features
from data.temporal_alignment import build_delta_timestamps


@dataclass
class CachedLatentRobotSample:
    latent: torch.Tensor  # [T_tok, N, C]
    actions: torch.Tensor  # [T_fut, n_act, A]
    action_mask: torch.Tensor  # [T_fut, n_act]
    proprio: torch.Tensor | None  # [S]
    action_valid: bool
    embodiment_id: int
    action_schema_id: int
    dataset_id: str
    episode_index: int
    frame_start: int
    frame_end: int
    sample_index: int
    text: str


def _load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _safe_torch_load(path: str | Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


def split_future_actions(
    actions: torch.Tensor,
    t_tok: int,
    history_chunks: int,
    future_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """[T_tok*n_act,A] -> future [T_fut,n_act,A] + mask.

    The cache window covers history+future V-JEPA time tokens. Stage-A flow
    trains on the future latent tokens, so the action branch must receive the
    future action chunks only.
    """
    if actions.ndim != 2:
        raise ValueError(f"expected actions [T_action,A], got {tuple(actions.shape)}")
    if t_tok != history_chunks + future_chunks:
        raise ValueError(
            f"t_tok={t_tok} != history+future={history_chunks + future_chunks}"
        )
    steps = actions.shape[0]
    if steps % t_tok:
        raise ValueError(f"action steps={steps} not divisible by t_tok={t_tok}")
    n_act = steps // t_tok
    chunks = actions.reshape(t_tok, n_act, actions.shape[-1])
    future = chunks[history_chunks : history_chunks + future_chunks]
    mask = torch.ones(future.shape[:2], dtype=torch.bool)
    return future, mask


class CachedLatentRobotDataset(Dataset):
    def __init__(
        self,
        schema_report: str | Path,
        manifest_path: str | Path,
        history_chunks: int = 4,
        future_chunks: int = 4,
        tubelet: int = 2,
        start_item: int = 0,
        index_modulus: int | None = None,
        index_remainders: Sequence[int] | None = None,
        max_items: int | None = None,
        control_stats_path: str | Path | None = None,
    ) -> None:
        self.schema_records = {
            str(rec["dataset_id"]): rec for rec in _load_json(schema_report)
        }
        entries = _load_jsonl(manifest_path)
        start = max(int(start_item), 0)
        if start:
            entries = entries[start:]
        if index_modulus is not None:
            modulus = int(index_modulus)
            if modulus <= 0:
                raise ValueError(f"index_modulus must be positive, got {modulus}")
            remainders = {int(r) % modulus for r in (index_remainders or [0])}
            entries = [
                entry
                for local_idx, entry in enumerate(entries)
                if local_idx % modulus in remainders
            ]
        if max_items is not None:
            entries = entries[: int(max_items)]
        if not entries:
            raise ValueError(f"manifest has no entries: {manifest_path}")
        self.entries = entries
        self.history_chunks = int(history_chunks)
        self.future_chunks = int(future_chunks)
        self.tubelet = int(tubelet)
        self.control_normalizer = (
            None
            if control_stats_path is None
            else FixedControlNormalizer.from_json(control_stats_path)
        )
        self._backends: dict[str, DirectLeRobotV21Dataset] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> CachedLatentRobotSample:
        entry = self.entries[idx]
        dataset_id = str(entry["dataset_id"])
        rec = self.schema_records[dataset_id]
        backend = self._backend_for(rec)
        item = backend[int(entry["sample_index"])]

        latent = _safe_torch_load(entry["cache_path"]).float()
        if latent.ndim != 3:
            raise ValueError(
                f"cached latent must be [T_tok,N,C], got {tuple(latent.shape)}"
            )
        t_tok = int(latent.shape[0])
        actions = _concat_time_features(item, tuple(rec["action_keys"])).float()
        future_actions, future_mask = split_future_actions(
            actions,
            t_tok=t_tok,
            history_chunks=self.history_chunks,
            future_chunks=self.future_chunks,
        )
        if self.control_normalizer is not None:
            future_actions = self.control_normalizer.normalize_action(
                future_actions, int(entry["action_schema_id"])
            )
        proprio = None
        if rec.get("state_keys"):
            proprio = _concat_current_features(item, tuple(rec["state_keys"])).float()
            if self.control_normalizer is not None:
                proprio = self.control_normalizer.normalize_state(
                    proprio, int(entry["embodiment_id"])
                )

        return CachedLatentRobotSample(
            latent=latent,
            actions=future_actions,
            action_mask=future_mask,
            proprio=proprio,
            action_valid=True,
            embodiment_id=int(entry["embodiment_id"]),
            action_schema_id=int(entry["action_schema_id"]),
            dataset_id=dataset_id,
            episode_index=int(entry["episode"]),
            frame_start=int(entry["frame_start"]),
            frame_end=int(entry["frame_end"]),
            sample_index=int(entry["sample_index"]),
            text=str(entry.get("text", "")),
        )

    def _backend_for(self, rec: dict) -> DirectLeRobotV21Dataset:
        dataset_id = str(rec["dataset_id"])
        backend = self._backends.get(dataset_id)
        if backend is not None:
            return backend
        delta = build_delta_timestamps(
            dataset_fps=float(rec["fps"]),
            history_chunks=self.history_chunks,
            future_chunks=self.future_chunks,
            tubelet=self.tubelet,
        )
        delta_timestamps = {
            key: delta.flat_action_offsets() for key in rec["action_keys"]
        }
        backend = DirectLeRobotV21Dataset(
            root=rec["repo_or_path"],
            delta_timestamps=delta_timestamps,
            current_keys=tuple(rec.get("state_keys", [])),
        )
        self._backends[dataset_id] = backend
        return backend


def collate_cached_latent_robot(samples: Sequence[CachedLatentRobotSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty cached latent batch")
    token_shape = tuple(samples[0].latent.shape)
    action_shape = tuple(samples[0].actions.shape)
    schema = samples[0].action_schema_id
    for sample in samples:
        if tuple(sample.latent.shape) != token_shape:
            raise ValueError("cached latent batch mixes latent shapes")
        if tuple(sample.actions.shape) != action_shape:
            raise ValueError("cached latent batch mixes action shapes")
        if sample.action_schema_id != schema:
            raise ValueError("cached latent batch mixes action schemas")

    proprio = None
    if samples[0].proprio is not None:
        proprio = torch.stack([s.proprio for s in samples], dim=0)

    return {
        "latent": torch.stack([s.latent for s in samples], dim=0),
        "actions": torch.stack([s.actions for s in samples], dim=0),
        "action_mask": torch.stack([s.action_mask for s in samples], dim=0),
        "proprio": proprio,
        "action_valid": torch.tensor([s.action_valid for s in samples], dtype=torch.bool),
        "embodiment_id": torch.tensor([s.embodiment_id for s in samples], dtype=torch.long),
        "action_schema_id": torch.tensor([s.action_schema_id for s in samples], dtype=torch.long),
        "dataset_id": [s.dataset_id for s in samples],
        "episode_index": torch.tensor([s.episode_index for s in samples], dtype=torch.long),
        "frame_start": torch.tensor([s.frame_start for s in samples], dtype=torch.long),
        "frame_end": torch.tensor([s.frame_end for s in samples], dtype=torch.long),
        "sample_index": torch.tensor([s.sample_index for s in samples], dtype=torch.long),
        "text": [s.text for s in samples],
    }
