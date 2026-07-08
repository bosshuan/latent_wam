from __future__ import annotations

import torch

from scripts.train_robot_vj_rae import _actions_to_transitions


def test_actions_to_transitions_uses_next_chunk_mean():
    # B=1, T_tok=4, 3 action steps/chunk, A=1.
    actions = torch.tensor(
        [[[0.0], [1.0], [2.0], [10.0], [11.0], [12.0],
          [20.0], [21.0], [22.0], [30.0], [31.0], [32.0]]]
    )
    pooled, valid = _actions_to_transitions(actions, None, t_tok=4)
    assert pooled.shape == (1, 3, 1)
    assert torch.equal(pooled[0, :, 0], torch.tensor([11.0, 21.0, 31.0]))
    assert valid.tolist() == [[True, True, True]]


def test_actions_to_transitions_respects_step_mask():
    actions = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
    mask = torch.ones(1, 12, dtype=torch.bool)
    mask[:, 3:6] = torch.tensor([[True, False, False]])
    pooled, valid = _actions_to_transitions(actions, mask, t_tok=4)
    assert float(pooled[0, 0, 0]) == 3.0
    assert valid.tolist() == [[True, True, True]]
