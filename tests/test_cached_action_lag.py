import torch

from scripts.evaluate_cached_action_lag import _aligned_pairs


def _marked_inputs():
    delta = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
    action = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    valid = torch.ones(1, 8, dtype=torch.bool)
    return delta, action, valid


def test_zero_lag_pairs_same_chunk():
    delta, action, valid = _marked_inputs()
    x, y = _aligned_pairs(delta, action, valid, lag=0, action_origin=4)
    assert x[:, 0].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert y[:, 0].tolist() == [4.0, 5.0, 6.0, 7.0]


def test_negative_lag_pairs_transition_with_earlier_action():
    delta, action, valid = _marked_inputs()
    x, y = _aligned_pairs(delta, action, valid, lag=-1, action_origin=4)
    assert x[:, 0].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert y[:, 0].tolist() == [3.0, 4.0, 5.0, 6.0]


def test_positive_lag_uses_same_transitions_and_honors_mask():
    delta, action, valid = _marked_inputs()
    valid[:, 6] = False
    x, y = _aligned_pairs(delta, action, valid, lag=1, action_origin=3)
    assert x[:, 0].tolist() == [10.0, 20.0, 40.0]
    assert y[:, 0].tolist() == [4.0, 5.0, 7.0]
