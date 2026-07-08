"""M1 missing-action tests (CLAUDE.md §2.3).

Actionless video must never carry a fabricated action tensor — at the sample
level, at collate, and at the batch container.
"""

from __future__ import annotations

import torch

from data.collate import TrajectorySample, collate_trajectory
from data.schemas import TrajectoryBatch


def _video_sample(grid=(2, 24, 24)):
    t = grid[0] * 2
    return TrajectorySample(
        pixels=torch.randn(t, 3, 384, 384),
        token_grid=grid,
        action_valid=False,
        embodiment_id=0,
        action_schema_id=-1,
        fps=5.0,
        text="",
    )


def _robot_sample(grid=(2, 24, 24), a_dim=7, steps=12):
    t = grid[0] * 2
    return TrajectorySample(
        pixels=torch.randn(t, 3, 384, 384),
        token_grid=grid,
        action_valid=True,
        embodiment_id=1,
        action_schema_id=3,
        fps=30.0,
        text="pick up the cube",
        dataset_id="robot_ds",
        episode_index=5,
        frame_start=80,
        frame_end=96,
        sample_index=123,
        actions=torch.randn(steps, a_dim),
        proprio=torch.randn(9),
    )


def test_video_sample_rejects_action_tensor():
    try:
        TrajectorySample(
            pixels=torch.randn(4, 3, 384, 384),
            token_grid=(2, 24, 24),
            action_valid=False,
            embodiment_id=0,
            action_schema_id=-1,
            actions=torch.randn(12, 7),  # illegal
        )
    except ValueError:
        return
    raise AssertionError("actionless sample must reject an action tensor")


def test_video_only_batch_has_no_action_tensor():
    batch = collate_trajectory([_video_sample(), _video_sample()])
    assert batch.actions is None
    assert batch.action_pad_mask is None
    assert batch.proprio is None
    assert batch.has_any_action() is False
    assert batch.action_valid.dtype == torch.bool
    assert not batch.action_valid.any()


def test_mixed_batch_masks_video_rows_inert():
    batch = collate_trajectory([_robot_sample(), _video_sample()])
    assert batch.actions is not None  # robot present
    # video row (index 1) is zero and masked-out, not a pseudo action
    assert batch.action_valid.tolist() == [True, False]
    assert batch.action_pad_mask[1].any().item() is False
    assert torch.count_nonzero(batch.actions[1]).item() == 0
    assert batch.dataset_id[0] == "robot_ds"
    assert batch.episode_index.tolist() == [5, -1]
    assert batch.frame_start.tolist() == [80, -1]
    assert batch.frame_end.tolist() == [96, -1]


def test_batch_container_rejects_action_without_validity():
    # all-actionless validity but an actions tensor => container raises
    try:
        TrajectoryBatch(
            pixels=torch.randn(2, 4, 3, 384, 384),
            token_grid=(2, 24, 24),
            actions=torch.randn(2, 12, 7),
            action_pad_mask=torch.ones(2, 12, dtype=torch.bool),
            action_valid=torch.zeros(2, dtype=torch.bool),
            embodiment_id=torch.zeros(2, dtype=torch.long),
            action_schema_id=torch.zeros(2, dtype=torch.long),
        )
    except ValueError:
        return
    raise AssertionError("batch must reject actions under all-zero action_valid")
