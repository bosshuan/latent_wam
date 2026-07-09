from __future__ import annotations

import torch

from train.fsdp_checkpoint import TrainingProgress


def test_training_progress_state_roundtrip():
    source = TrainingProgress(step=37)
    target = TrainingProgress()
    target.load_state_dict(source.state_dict())
    assert target.step == 37
    assert source.state_dict()["step"].dtype == torch.int64
