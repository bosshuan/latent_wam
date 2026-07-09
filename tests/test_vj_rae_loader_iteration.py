from itertools import islice

import pytest

from scripts.train_robot_vj_rae import _repeat_loader


def test_repeat_loader_restarts_without_cycle_object():
    values = list(islice(_repeat_loader([1, 2, 3]), 8))
    assert values == [1, 2, 3, 1, 2, 3, 1, 2]


def test_repeat_loader_rejects_empty_input():
    with pytest.raises(RuntimeError, match="empty"):
        next(_repeat_loader([]))
