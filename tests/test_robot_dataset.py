from __future__ import annotations

import torch

from data.robot_dataset import (
    RobotSchemaBinding,
    RobotTrajectoryDataset,
    _instantiate_lerobot_dataset,
)


class _Meta:
    info = {"codebase_version": "v2.1"}


class _MockLeRobot:
    meta = _Meta()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0
        return {
            "images.rgb.head": torch.randint(0, 255, (4, 32, 32, 3), dtype=torch.uint8),
            "states.left_joint.position": torch.arange(6, dtype=torch.float32),
            "states.left_gripper.position": torch.ones(1),
            "states.right_joint.position": torch.arange(6, dtype=torch.float32) + 10,
            "states.right_gripper.position": torch.ones(1) * 2,
            "actions.left_joint.position": torch.ones(12, 6),
            "actions.left_gripper.position": torch.ones(12, 1) * 2,
            "actions.right_joint.position": torch.ones(12, 6) * 3,
            "actions.right_gripper.position": torch.ones(12, 1) * 4,
            "task": "heat the food",
            "__meta__": {
                "episode_index": 3,
                "frame_start": 80,
                "frame_end": 96,
                "global_index": 123,
            },
        }


def test_robot_dataset_concats_intern_data_style_keys():
    binding = RobotSchemaBinding(
        image_key="images.rgb.head",
        state_keys=(
            "states.left_joint.position",
            "states.left_gripper.position",
            "states.right_joint.position",
            "states.right_gripper.position",
        ),
        action_keys=(
            "actions.left_joint.position",
            "actions.left_gripper.position",
            "actions.right_joint.position",
            "actions.right_gripper.position",
        ),
        embodiment_id=0,
        action_schema_id=0,
        fps=30.0,
        dataset_id="interndata_mock",
    )
    ds = RobotTrajectoryDataset(
        repo_id="mock",
        binding=binding,
        token_grid=(2, 24, 24),
        lerobot_dataset=_MockLeRobot(),
    )
    sample = ds[0]
    assert sample.pixels.shape == (4, 3, 32, 32)
    assert sample.pixels.dtype == torch.float32
    assert sample.actions.shape == (12, 14)
    assert sample.proprio.shape == (14,)
    assert sample.embodiment_id == 0
    assert sample.action_schema_id == 0
    assert sample.text == "heat the food"
    assert sample.dataset_id == "interndata_mock"
    assert sample.episode_index == 3
    assert sample.frame_start == 80
    assert sample.frame_end == 96
    assert sample.sample_index == 123


def test_lerobot_instantiation_uses_root_for_local_paths(tmp_path):
    calls = {}

    class _FakeLeRobotDataset:
        def __init__(self, repo_id, root=None, delta_timestamps=None):
            calls["repo_id"] = repo_id
            calls["root"] = root
            calls["delta_timestamps"] = delta_timestamps

    ds = _instantiate_lerobot_dataset(
        _FakeLeRobotDataset,
        repo_or_path=str(tmp_path),
        delta_timestamps={"images.rgb.head": [0.0]},
    )
    assert isinstance(ds, _FakeLeRobotDataset)
    assert calls["repo_id"] == tmp_path.name
    assert calls["root"] == tmp_path
    assert calls["delta_timestamps"] == {"images.rgb.head": [0.0]}
