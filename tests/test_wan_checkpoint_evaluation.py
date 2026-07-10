import pytest

from flow.losses import UnifiedLossWeights
from scripts.evaluate_wan_real_fsdp_checkpoint import _aggregate


def _metrics(*, sensitivity: float, delta_cond: float, collapse: bool) -> dict:
    return {
        "total": 2.0,
        "z_fm": 1.0,
        "a_fm": 0.2,
        "clean": 0.7,
        "cf": 0.1,
        "S_a": sensitivity,
        "S_a_cos": 0.01,
        "delta_cond": delta_cond,
        "cf_valid_frac": 1.0,
        "cf_action_delta": 1.0,
        "cf_inconclusive": False,
        "collapse": collapse,
    }


def test_aggregate_uses_split_mean_for_collapse_decision():
    result = _aggregate(
        [
            _metrics(sensitivity=0.005, delta_cond=-0.1, collapse=True),
            _metrics(sensitivity=0.035, delta_cond=0.3, collapse=False),
        ],
        UnifiedLossWeights(),
    )

    assert result["S_a"] == pytest.approx(0.02)
    assert result["delta_cond"] == pytest.approx(0.1)
    assert result["delta_cond_sem"] == pytest.approx(0.2)
    assert result["delta_cond_ci95_low"] == pytest.approx(-0.292)
    assert result["delta_cond_ci95_high"] == pytest.approx(0.492)
    assert result["delta_cond_confident_positive"] is False
    assert result["batch_alarm_count"] == 1
    assert result["collapse"] is False


def test_aggregate_detects_split_level_action_collapse():
    result = _aggregate(
        [
            _metrics(sensitivity=0.004, delta_cond=0.1, collapse=True),
            _metrics(sensitivity=0.006, delta_cond=0.2, collapse=True),
        ],
        UnifiedLossWeights(),
    )

    assert result["batch_alarm_count"] == 2
    assert result["S_a_confident"] is False
    assert result["collapse"] is True
