from __future__ import annotations

from scripts.train_wan_real_fsdp_pilot import _pilot_acceptance


def _val(total: float, sensitivity: float, delta_cond: float) -> dict:
    return {
        "total": total,
        "S_a": sensitivity,
        "delta_cond": delta_cond,
        "cf_inconclusive": False,
    }


def test_pilot_accepts_tiny_negative_delta_without_hiding_strict_alarm():
    initial = _val(total=4.05, sensitivity=0.008, delta_cond=-0.002)
    final = _val(total=3.54, sensitivity=0.0142, delta_cond=-0.0004)
    result = _pilot_acceptance(
        initial,
        final,
        {
            "max_final_val_total_ratio": 1.10,
            "min_final_action_sensitivity": 0.01,
            "min_final_delta_cond": -0.001,
            "require_cf_conclusive": True,
        },
    )
    assert result["pilot_ok"]
    assert result["action_sensitivity_ok"]
    assert result["delta_cond_ok"]


def test_pilot_rejects_real_action_sensitivity_collapse():
    initial = _val(total=4.0, sensitivity=0.02, delta_cond=0.0)
    final = _val(total=3.5, sensitivity=0.005, delta_cond=0.001)
    result = _pilot_acceptance(initial, final, {})
    assert not result["pilot_ok"]
    assert not result["action_sensitivity_ok"]
