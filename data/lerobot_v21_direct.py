"""Lightweight direct reader for local LeRobot v2.1 datasets.

This is a fallback for CUDA servers where installing the official ``lerobot``
package would force an incompatible torch upgrade. It intentionally supports the
small surface needed by ``RobotTrajectoryDataset``:

* read tabular state/action columns from per-episode parquet files;
* decode local video files for image keys;
* return a dict shaped like a LeRobotDataset sample after ``delta_timestamps``.

The official LeRobotDataset remains the preferred backend when it is installed.
"""

from __future__ import annotations

import bisect
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _EpisodeInfo:
    index: int
    parquet_path: Path
    length: int


class DirectLeRobotV21Dataset(Dataset):
    """Local LeRobot v2.1 reader with a LeRobotDataset-like item contract."""

    def __init__(
        self,
        root: str | Path,
        delta_timestamps: Mapping[str, Sequence[float]],
        current_keys: Sequence[str] = (),
    ) -> None:
        self.root = Path(root)
        self.delta_timestamps = {
            str(k): [float(x) for x in v] for k, v in delta_timestamps.items()
        }
        self.current_keys = tuple(str(k) for k in current_keys)

        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"LeRobot v2.1 info.json not found: {info_path}")
        with info_path.open() as f:
            self.info = json.load(f)
        self.meta = SimpleNamespace(info=self.info)
        self.fps = float(self.info.get("fps", 0.0))
        if self.fps <= 0:
            raise ValueError(f"{info_path}: missing/invalid fps={self.fps!r}")
        self.features = dict(self.info.get("features", {}))

        self._tasks = _read_tasks(self.root / "meta" / "tasks.jsonl")
        self._episodes = self._discover_episodes()
        if not self._episodes:
            raise FileNotFoundError(f"no parquet episodes found under {self.root / 'data'}")
        total = 0
        self._cum_lengths: list[int] = []
        for ep in self._episodes:
            total += ep.length
            self._cum_lengths.append(total)

        self._table_cache: dict[Path, Any] = {}
        self._video_path_cache: dict[tuple[int, str], Path] = {}
        self._video_reader_cache: dict[Path, Any] = {}

    def __len__(self) -> int:
        return self._cum_lengths[-1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep, local_idx = self._locate(idx)
        table = self._load_table(ep.parquet_path)
        out: dict[str, Any] = {}
        frame_window: list[int] | None = None

        for key, offsets in self.delta_timestamps.items():
            row_indices = _clamped_offset_indices(local_idx, offsets, self.fps, ep.length)
            if self._should_decode_video(key, table):
                if frame_window is None:
                    frame_window = list(row_indices)
                out[key] = self._read_video_frames(ep, key, row_indices)
            else:
                out[key] = _stack_rows(table, key, row_indices)

        for key in self.current_keys:
            if key not in out:
                out[key] = _read_row_value(table, key, local_idx)

        task = self._read_task(table, local_idx)
        if task:
            out["task"] = task
        if frame_window is None:
            frame_window = [local_idx]
        out["__meta__"] = {
            "episode_index": ep.index,
            "local_index": local_idx,
            "global_index": int(idx),
            "frame_start": min(frame_window),
            "frame_end": max(frame_window) + 1,
            "parquet_path": str(ep.parquet_path),
        }
        return out

    def _discover_episodes(self) -> list[_EpisodeInfo]:
        paths = sorted((self.root / "data").glob("**/*.parquet"))
        episodes: list[_EpisodeInfo] = []
        for path in paths:
            match = re.search(r"episode_(\d+)", path.stem)
            index = int(match.group(1)) if match else len(episodes)
            episodes.append(
                _EpisodeInfo(
                    index=index,
                    parquet_path=path,
                    length=_parquet_num_rows(path),
                )
            )
        return sorted(episodes, key=lambda e: (e.index, str(e.parquet_path)))

    def _locate(self, idx: int) -> tuple[_EpisodeInfo, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        ep_i = bisect.bisect_right(self._cum_lengths, idx)
        prev = 0 if ep_i == 0 else self._cum_lengths[ep_i - 1]
        return self._episodes[ep_i], idx - prev

    def _load_table(self, path: Path):
        table = self._table_cache.get(path)
        if table is None:
            try:
                import pandas as pd
            except ImportError as exc:  # pragma: no cover - environment specific
                raise ImportError(
                    "Direct LeRobot reader requires pandas + pyarrow to read parquet."
                ) from exc
            table = pd.read_parquet(path)
            self._table_cache[path] = table
        return table

    def _should_decode_video(self, key: str, table) -> bool:
        feat = self.features.get(key, {})
        dtype = str(feat.get("dtype", "")).lower()
        if dtype in {"image", "video"}:
            if key not in table.columns:
                return True
            try:
                value = _read_row_value(table, key, 0)
                tensor = _value_to_tensor(value)
                return tensor.ndim < 3
            except Exception:
                return True
        return key not in table.columns

    def _read_video_frames(
        self, ep: _EpisodeInfo, key: str, row_indices: Sequence[int]
    ) -> torch.Tensor:
        path = self._video_path(ep.index, key)
        indices = [int(i) for i in row_indices]
        try:
            return _read_video_frames_decord(path, indices, self._video_reader_cache)
        except Exception as decord_error:
            try:
                return _read_video_frames_av(path, indices)
            except Exception as av_error:
                raise RuntimeError(
                    f"failed to decode {path} for key={key!r}; "
                    f"decord error={decord_error}; av error={av_error}"
                ) from av_error

    def _video_path(self, episode_index: int, key: str) -> Path:
        cache_key = (episode_index, key)
        cached = self._video_path_cache.get(cache_key)
        if cached is not None:
            return cached

        stem = f"episode_{episode_index:06d}"
        patterns = [
            self.root / "videos" / "**" / key / f"{stem}.mp4",
            self.root / "videos" / "**" / key.replace(".", "/") / f"{stem}.mp4",
        ]
        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(Path(p) for p in glob.glob(str(pattern), recursive=True))

        if not matches:
            all_episode_videos = sorted((self.root / "videos").glob(f"**/{stem}.mp4"))
            key_as_path = key.replace(".", "/")
            matches = [
                p
                for p in all_episode_videos
                if key in str(p) or key_as_path in str(p)
            ]
            if not matches and len(all_episode_videos) == 1:
                matches = all_episode_videos

        if not matches:
            raise FileNotFoundError(
                f"video for key={key!r}, episode={episode_index} not found under "
                f"{self.root / 'videos'}"
            )
        path = sorted(matches)[0]
        self._video_path_cache[cache_key] = path
        return path

    def _read_task(self, table, local_idx: int) -> str:
        if "task" in table.columns:
            value = table.iloc[local_idx]["task"]
            if value:
                return str(value)
        if "task_index" in table.columns:
            task_index = int(table.iloc[local_idx]["task_index"])
            return self._tasks.get(task_index, "")
        return ""


def _read_tasks(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    tasks: dict[int, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "task_index" in row and "task" in row:
                tasks[int(row["task_index"])] = str(row["task"])
    return tasks


def _parquet_num_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment specific
            raise ImportError(
                "Direct LeRobot reader requires pyarrow or pandas to inspect parquet."
            ) from exc
        return int(len(pd.read_parquet(path)))


def _clamped_offset_indices(
    local_idx: int, offsets: Sequence[float], fps: float, episode_len: int
) -> list[int]:
    out = []
    for dt in offsets:
        frame = int(round(local_idx + float(dt) * fps))
        out.append(max(0, min(episode_len - 1, frame)))
    return out


def _read_row_value(table, key: str, row_idx: int):
    if key not in table.columns:
        raise KeyError(f"column {key!r} not found in parquet table")
    return table.iloc[int(row_idx)][key]


def _stack_rows(table, key: str, row_indices: Sequence[int]) -> torch.Tensor:
    values = [_value_to_tensor(_read_row_value(table, key, i)) for i in row_indices]
    try:
        return torch.stack(values, dim=0)
    except RuntimeError:
        shapes = [tuple(v.shape) for v in values]
        raise ValueError(f"cannot stack column {key!r} values with shapes={shapes}")


def _value_to_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.ndarray):
        if not value.flags.writeable:
            value = value.copy()
        return torch.from_numpy(value)
    return torch.as_tensor(value)


def _read_video_frames_decord(
    path: Path, indices: Sequence[int], cache: dict[Path, Any]
) -> torch.Tensor:
    from decord import VideoReader, cpu

    reader = cache.get(path)
    if reader is None:
        reader = VideoReader(str(path), ctx=cpu(0))
        cache[path] = reader
    max_idx = len(reader) - 1
    safe = [max(0, min(max_idx, int(i))) for i in indices]
    frames = reader.get_batch(safe).asnumpy()
    return torch.as_tensor(frames)


def _read_video_frames_av(path: Path, indices: Sequence[int]) -> torch.Tensor:
    import av

    wanted = [int(i) for i in indices]
    wanted_set = set(wanted)
    max_wanted = max(wanted_set) if wanted_set else -1
    frames_by_index = {}
    last_frame = None
    with av.open(str(path)) as container:
        for frame_i, frame in enumerate(container.decode(video=0)):
            last_frame = torch.as_tensor(frame.to_ndarray(format="rgb24"))
            if frame_i in wanted_set:
                frames_by_index[frame_i] = last_frame
            if frame_i >= max_wanted:
                break
    if not frames_by_index and last_frame is not None:
        frames_by_index = {i: last_frame for i in wanted}
    if not frames_by_index:
        raise RuntimeError(f"no frames decoded from {path}")
    last = frames_by_index[max(frames_by_index)]
    frames = [frames_by_index.get(i, last) for i in wanted]
    return torch.stack(frames, dim=0)
