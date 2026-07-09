from __future__ import annotations

import torch

from data.control_normalizer import FixedControlNormalizer, RunningMoments


def test_running_moments_matches_direct_statistics():
    values = torch.tensor([[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]])
    moments = RunningMoments()
    moments.update(values[:1])
    moments.update(values[1:])
    record = moments.finalize(std_floor=1.0e-6)

    assert record["count"] == 3
    assert torch.allclose(torch.tensor(record["mean"]), values.mean(dim=0))
    assert torch.allclose(
        torch.tensor(record["std"]),
        values.var(dim=0, unbiased=False).sqrt(),
    )


def test_control_normalizer_routes_action_and_state_stats():
    normalizer = FixedControlNormalizer(
        {
            "format_version": 1,
            "actions": {
                "3": {"count": 2, "dim": 2, "mean": [1.0, 2.0], "std": [2.0, 4.0]}
            },
            "states": {
                "7": {"count": 2, "dim": 2, "mean": [5.0, 6.0], "std": [5.0, 3.0]}
            },
        }
    )

    action = normalizer.normalize_action(torch.tensor([[3.0, 6.0]]), 3)
    state = normalizer.normalize_state(torch.tensor([10.0, 3.0]), 7)
    assert torch.equal(action, torch.tensor([[1.0, 1.0]]))
    assert torch.equal(state, torch.tensor([1.0, -1.0]))
