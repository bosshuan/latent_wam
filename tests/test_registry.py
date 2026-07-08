"""M1 registry tests: id 0 is a real embodiment, NEW_EMBODIMENT is trailing.

Guards the correctness fix: the untrained RoboTwin slot must never collide with
the default/uninitialized ``embodiment_id=0`` that real trained data uses, or
robot data could silently route through untrained CategorySpecific weights.
"""

from __future__ import annotations

import torch

from data.registry import (
    INVALID_EMBODIMENT_ID,
    NEW_EMBODIMENT,
    EmbodimentRegistry,
)


def test_real_embodiments_start_at_zero():
    reg = EmbodimentRegistry()
    assert reg.embodiment_id("panda") == 0
    assert reg.embodiment_id("widowx") == 1
    assert reg.embodiment_id("panda") == 0  # stable


def test_new_embodiment_not_id_zero():
    reg = EmbodimentRegistry()
    a = reg.embodiment_id("panda")        # 0
    b = reg.embodiment_id("ur5")          # 1
    nid = reg.new_embodiment_id()         # trailing
    assert nid not in (a, b)
    assert nid != 0
    assert reg.reserved_new_embodiment_id == nid


def test_embodiment_id_refuses_new_embodiment_name():
    reg = EmbodimentRegistry()
    try:
        reg.embodiment_id(NEW_EMBODIMENT)
    except ValueError:
        return
    raise AssertionError("embodiment_id() must refuse the NEW_EMBODIMENT name")


def test_invalid_sentinel_is_negative():
    assert INVALID_EMBODIMENT_ID == -1


def test_assert_trainable_batch():
    reg = EmbodimentRegistry()
    reg.embodiment_id("panda")            # 0
    reg.embodiment_id("ur5")              # 1
    nid = reg.new_embodiment_id()

    reg.assert_trainable_batch(torch.tensor([0, 1, 0]))  # ok

    for bad in ([INVALID_EMBODIMENT_ID, 0], [0, nid]):
        try:
            reg.assert_trainable_batch(torch.tensor(bad))
        except ValueError:
            continue
        raise AssertionError(f"assert_trainable_batch must reject {bad}")
