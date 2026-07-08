"""M1 temporal-alignment arithmetic tests (CLAUDE.md §3)."""

from __future__ import annotations

import torch

from data.temporal_alignment import (
    actions_per_chunk,
    build_delta_timestamps,
    pad_actions,
    resample_actions,
)


def test_actions_per_chunk_values():
    assert actions_per_chunk(30) == 12        # 30 * 0.4
    assert actions_per_chunk(50) == 12        # clamped to 30 -> 12
    assert actions_per_chunk(10) == 4         # 10 * 0.4
    assert actions_per_chunk(15) == 6


def test_delta_timestamps_num_chunks():
    dt = build_delta_timestamps(dataset_fps=30, history_chunks=4, future_chunks=4)
    assert dt.num_chunks == 8                  # history + future
    assert len(dt.observation_chunks) == 8
    assert len(dt.action_chunks) == 8
    # each obs chunk = tubelet frames; each action chunk = actions_per_chunk
    assert all(len(c) == 2 for c in dt.observation_chunks)
    assert dt.actions_per_chunk == 12
    assert all(len(c) == 12 for c in dt.action_chunks)
    # t=0 boundary: last history chunk starts at -0.4, first future chunk at 0.0
    assert dt.observation_chunks[3][0] == -0.4
    assert dt.observation_chunks[4][0] == 0.0


def test_resample_downsamples_above_target():
    actions = torch.arange(60).float().unsqueeze(1)  # 60 steps @ 60Hz
    out = resample_actions(actions, src_fps=60, target_fps=30)
    assert out.shape[0] == 30                  # stride 2
    # at or below target: untouched
    same = resample_actions(actions, src_fps=20, target_fps=30)
    assert same.shape[0] == 60


def test_pad_actions_mask():
    actions = torch.randn(5, 7)
    padded, mask = pad_actions(actions, max_steps=12)
    assert padded.shape == (12, 7)
    assert mask.sum().item() == 5
    assert mask[:5].all() and not mask[5:].any()
