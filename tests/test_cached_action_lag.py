import torch

from scripts.evaluate_cached_action_lag import _aligned_pairs


def _marked_inputs():
    delta = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
    action = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    return delta, action, valid


def test_zero_lag_pairs_same_chunk():
    delta, action, valid = _marked_inputs()
    x, y = _aligned_pairs(delta, action, valid, lag=0)
    assert x[:, 0].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert y[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_negative_lag_pairs_transition_with_earlier_action():
    delta, action, valid = _marked_inputs()
    x, y = _aligned_pairs(delta, action, valid, lag=-1)
    assert x[:, 0].tolist() == [20.0, 30.0, 40.0]
    assert y[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_positive_lag_pairs_transition_with_later_action_and_mask():
    delta, action, valid = _marked_inputs()
    valid[:, 2] = False
    x, y = _aligned_pairs(delta, action, valid, lag=1)
    assert x[:, 0].tolist() == [10.0, 30.0]
    assert y[:, 0].tolist() == [2.0, 4.0]
