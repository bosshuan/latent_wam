"""Robot trajectory dataset over LeRobot v2.1 robot data.

We prefer the official ``lerobot.LeRobotDataset`` when it is installed and pull
history + future observations + the future action chunk via ``delta_timestamps``
(CLAUDE.md §4). CUDA debug servers may not be able to install that package
without upgrading torch, so local v2.1 datasets can fall back to a narrow direct
reader that parses parquet/video files.

The ``lerobot`` import is lazy so this module imports on CPU without the package
installed; ``assert_codebase_version`` is a free function so the version gate is
unit-testable with a mock metadata object.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from data.collate import TrajectorySample
from data.lerobot_v21_direct import DirectLeRobotV21Dataset
from data.schema_scanner import EXPECTED_CODEBASE_VERSION
from data.temporal_alignment import (
    DeltaTimestamps,
    build_delta_timestamps,
)


def assert_codebase_version(codebase_version: str) -> None:
    """Fail loud unless the dataset is LeRobot v2.1.

    # TODO(v3.0): when datasets migrate to v3.0, branch here on version and
    # route to a v3.0 adapter; do not silently accept a different layout.
    """
    if str(codebase_version) != EXPECTED_CODEBASE_VERSION:
        raise ValueError(
            f"LeRobotDataset codebase_version={codebase_version!r} != "
            f"{EXPECTED_CODEBASE_VERSION!r}; Stage A robot data must be v2.1."
        )


@dataclass
class RobotSchemaBinding:
    """How a dataset's raw LeRobot fields map into a ``TrajectorySample``."""

    image_key: str
    # Backward-compatible single-key fields. New InternData-style configs should
    # prefer state_keys/action_keys below.
    state_key: str = ""
    action_key: str = ""
    embodiment_id: int = 0
    action_schema_id: int = 0
    fps: float = 30.0
    view_id: int = 0
    dataset_id: str = ""
    state_keys: tuple[str, ...] = field(default_factory=tuple)
    action_keys: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.state_keys = tuple(self.state_keys)
        self.action_keys = tuple(self.action_keys)
        if not self.state_keys and self.state_key:
            self.state_keys = (self.state_key,)
        if not self.action_keys and self.action_key:
            self.action_keys = (self.action_key,)
        if not self.action_keys:
            raise ValueError("RobotSchemaBinding requires at least one action key")


class RobotTrajectoryDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        binding: RobotSchemaBinding,
        token_grid: tuple[int, int, int],
        history_chunks: int = 4,
        future_chunks: int = 4,
        tubelet: int = 2,
        backend: str = "auto",
        lerobot_dataset: Optional[Any] = None,
    ) -> None:
        """
        Args:
            repo_id: LeRobot repo id / local path.
            binding: field map + ids (typically derived from a SchemaRecord).
            token_grid: (T_tok, grid_h, grid_w) target grid.
            backend: ``auto`` (official then direct), ``lerobot``, or ``direct``.
            lerobot_dataset: pre-built dataset (injectable for tests); when None
                the real ``LeRobotDataset`` is constructed lazily.
        """
        self.repo_id = repo_id
        self.binding = binding
        self.token_grid = token_grid
        self.backend = str(backend)
        self.delta: DeltaTimestamps = build_delta_timestamps(
            dataset_fps=binding.fps,
            history_chunks=history_chunks,
            future_chunks=future_chunks,
            tubelet=tubelet,
        )

        if lerobot_dataset is None:
            lerobot_dataset = self._build_lerobot(repo_id)
        self._ds = lerobot_dataset
        assert_codebase_version(self._read_codebase_version(self._ds))

    # --- lazy backend ---------------------------------------------------
    def _build_lerobot(self, repo_id: str) -> Any:
        delta_timestamps = {
            self.binding.image_key: self.delta.flat_observation_offsets(),
        }
        for key in self.binding.action_keys:
            delta_timestamps[key] = self.delta.flat_action_offsets()

        if self.backend == "direct":
            return self._build_direct_lerobot(repo_id, delta_timestamps)
        if self.backend not in {"auto", "lerobot"}:
            raise ValueError(
                f"unknown robot dataset backend={self.backend!r}; "
                "expected 'auto', 'lerobot', or 'direct'"
            )

        try:
            LeRobotDataset = _import_lerobot_dataset()
            return _instantiate_lerobot_dataset(
                LeRobotDataset,
                repo_or_path=repo_id,
                delta_timestamps=delta_timestamps,
            )
        except (ModuleNotFoundError, ImportError):
            if self.backend == "lerobot" or not Path(repo_id).exists():
                raise
            return self._build_direct_lerobot(repo_id, delta_timestamps)

    def _build_direct_lerobot(self, repo_id: str, delta_timestamps: dict) -> Any:
        return DirectLeRobotV21Dataset(
            root=repo_id,
            delta_timestamps=delta_timestamps,
            current_keys=self.binding.state_keys,
        )

    @staticmethod
    def _read_codebase_version(ds: Any) -> str:
        # v2.1 exposes meta.info["codebase_version"]; tolerate a few shapes.
        meta = getattr(ds, "meta", None)
        if meta is not None:
            info = getattr(meta, "info", None)
            if isinstance(info, dict) and "codebase_version" in info:
                return str(info["codebase_version"])
        info = getattr(ds, "info", None)
        if isinstance(info, dict) and "codebase_version" in info:
            return str(info["codebase_version"])
        raise AttributeError("cannot locate codebase_version on LeRobot dataset")

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> TrajectorySample:
        item = self._ds[idx]
        b = self.binding

        pixels = _normalize_pixels(_read_item(item, b.image_key))
        actions = _concat_time_features(item, b.action_keys)  # [T_chunk, A]
        proprio = _concat_current_features(item, b.state_keys) if b.state_keys else None
        text = self._extract_text(item)
        meta = _read_sample_meta(item)

        if actions.ndim != 2:
            raise ValueError(
                f"action tensor must be [T_chunk, A]; got {tuple(actions.shape)}"
            )

        return TrajectorySample(
            pixels=pixels,
            token_grid=self.token_grid,
            action_valid=True,
            embodiment_id=b.embodiment_id,
            action_schema_id=b.action_schema_id,
            view_id=b.view_id,
            fps=b.fps,
            text=text,
            dataset_id=b.dataset_id,
            episode_index=int(meta.get("episode_index", -1)),
            frame_start=int(meta.get("frame_start", -1)),
            frame_end=int(meta.get("frame_end", -1)),
            sample_index=int(meta.get("global_index", idx)),
            actions=actions,
            action_step_mask=None,  # all real here; collate fills True
            proprio=proprio,
        )

    @staticmethod
    def _extract_text(item: Any) -> str:
        for key in ("task", "language_instruction", "instruction"):
            if isinstance(item, dict) and key in item and item[key]:
                return str(item[key])
        return ""


def _read_item(item: Any, key: str):
    if hasattr(item, "get"):
        value = item.get(key)
        if value is not None:
            return value
    return item[key]


def _read_sample_meta(item: Any) -> dict:
    if isinstance(item, dict):
        meta = item.get("__meta__", {})
        if isinstance(meta, dict):
            return meta
    return {}


def _import_lerobot_dataset():
    """Import ``LeRobotDataset`` across LeRobot package-layout changes."""

    candidates = (
        # LeRobot releases used by v2.1-era datasets.
        "lerobot.common.datasets.lerobot_dataset",
        # Current Hugging Face LeRobot layout.
        "lerobot.datasets.lerobot_dataset",
    )
    errors = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        cls = getattr(module, "LeRobotDataset", None)
        if cls is not None:
            return cls
        errors.append(f"{module_name}: module has no LeRobotDataset")
    msg = "\n".join(errors)
    raise ModuleNotFoundError(
        "Could not import LeRobotDataset. Install Hugging Face LeRobot, or use a "
        "LeRobot release compatible with v2.1 datasets.\nTried:\n"
        f"{msg}"
    )


def _instantiate_lerobot_dataset(LeRobotDataset, repo_or_path: str, delta_timestamps: dict):
    """Instantiate old/new LeRobotDataset APIs with local-path awareness."""

    repo_path = Path(repo_or_path)
    kwargs = {"delta_timestamps": delta_timestamps}
    params = inspect.signature(LeRobotDataset).parameters

    if repo_path.exists() and "root" in params:
        # Newer LeRobot separates the hub repo id from the local dataset root.
        # Our schema report stores local subset roots in repo_or_path.
        return LeRobotDataset(repo_id=repo_path.name, root=repo_path, **kwargs)
    return LeRobotDataset(repo_or_path, **kwargs)


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        if not value.flags.writeable:
            value = value.copy()
        return torch.from_numpy(value)
    return torch.as_tensor(value)


def _flatten_time_feature(value) -> torch.Tensor:
    """Feature with a time axis -> [T, D]."""
    x = _as_tensor(value)
    if x.ndim == 0:
        raise ValueError("time feature must have a leading time dimension")
    if x.ndim == 1:
        return x.unsqueeze(-1)
    return x.reshape(x.shape[0], -1)


def _flatten_current_feature(value) -> torch.Tensor:
    """Current-step feature -> [D]."""
    return _as_tensor(value).reshape(-1)


def _concat_time_features(item: Any, keys: tuple[str, ...]) -> torch.Tensor:
    parts = [_flatten_time_feature(_read_item(item, key)) for key in keys]
    if not parts:
        raise ValueError("cannot concatenate zero time features")
    steps = {p.shape[0] for p in parts}
    if len(steps) != 1:
        shapes = {key: tuple(part.shape) for key, part in zip(keys, parts)}
        raise ValueError(f"action/state time keys have inconsistent lengths: {shapes}")
    return torch.cat(parts, dim=-1)


def _concat_current_features(item: Any, keys: tuple[str, ...]) -> torch.Tensor:
    parts = [_flatten_current_feature(_read_item(item, key)) for key in keys]
    if not parts:
        raise ValueError("cannot concatenate zero current features")
    return torch.cat(parts, dim=-1)


def _normalize_pixels(value) -> torch.Tensor:
    """Return pixels as float [T, 3, H, W].

    LeRobot converted datasets may expose images as [T, H, W, 3] or [T, 3, H, W].
    Keep resizing/normalization policy out of this smoke dataset; V-JEPA-specific
    transforms can be added in the feature extraction step.
    """
    x = _as_tensor(value)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"expected image tensor [T,3,H,W] or [T,H,W,3]; got {tuple(x.shape)}")
    if x.shape[1] == 3:
        out = x
    elif x.shape[-1] == 3:
        out = x.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"cannot infer RGB channel axis from image shape {tuple(x.shape)}")
    out = out.float()
    if out.numel() and float(out.max()) > 2.0:
        out = out / 255.0
    return out
