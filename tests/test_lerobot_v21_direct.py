from __future__ import annotations

import numpy as np
import torch

from data.lerobot_v21_direct import (
    DirectLeRobotV21Dataset,
    _EpisodeInfo,
    _clamped_offset_indices,
    _read_tasks,
    _value_to_tensor,
)


def test_clamped_offset_indices_convert_seconds_to_rows():
    assert _clamped_offset_indices(
        local_idx=80,
        offsets=[-1.6, -0.4, 0.0, 0.4],
        fps=30.0,
        episode_len=100,
    ) == [32, 68, 80, 92]


def test_clamped_offset_indices_stay_inside_episode():
    assert _clamped_offset_indices(
        local_idx=2,
        offsets=[-1.6, 0.0, 10.0],
        fps=30.0,
        episode_len=50,
    ) == [0, 2, 49]


def test_read_tasks_jsonl(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"task_index": 0, "task": "open drawer"}\n'
        '{"task_index": 2, "task": "close drawer"}\n'
    )
    assert _read_tasks(path) == {0: "open drawer", 2: "close drawer"}


def test_value_to_tensor_copies_readonly_numpy_array():
    arr = np.arange(4, dtype=np.float32)
    arr.flags.writeable = False
    tensor = _value_to_tensor(arr)
    assert torch.equal(tensor, torch.arange(4, dtype=torch.float32))
    tensor[0] = 99.0
    assert float(arr[0]) == 0.0


def test_episode_ranges_are_global_half_open_intervals(tmp_path):
    dataset = DirectLeRobotV21Dataset.__new__(DirectLeRobotV21Dataset)
    dataset._episodes = [
        _EpisodeInfo(3, tmp_path / "episode_3.parquet", 100),
        _EpisodeInfo(7, tmp_path / "episode_7.parquet", 60),
    ]
    assert dataset.episode_ranges() == [
        {"episode": 3, "start": 0, "stop": 100, "length": 100},
        {"episode": 7, "start": 100, "stop": 160, "length": 60},
    ]
