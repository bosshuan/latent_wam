from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    path = ROOT / "configs" / "data" / name
    return yaml.safe_load(path.read_text())


def test_broad_pilot_is_episode_disjoint_and_medium_independent():
    cfg = _load(
        "interndata_a1_dual_arm_unified_cached_wan_real_fsdp_pilot_broad.yaml"
    )
    train_remainders = set(cfg["data"]["train_episode_remainders"])
    val_remainders = set(cfg["data"]["val_episode_remainders"])

    assert train_remainders.isdisjoint(val_remainders)
    assert cfg["data"]["train_episode_modulus"] == 5
    assert cfg["data"]["val_episode_modulus"] == 5
    assert "broad" in cfg["manifest_path"]
    assert "broad" in cfg["control_stats_path"]
    assert "broad" in cfg["pilot"]["checkpoint_root"]
    assert cfg["pilot"]["resume_from"] is None
    assert cfg["pilot"]["total_steps"] == 64


def test_broad_stats_uses_only_pilot_train_episode_remainders():
    pilot = _load(
        "interndata_a1_dual_arm_unified_cached_wan_real_fsdp_pilot_broad.yaml"
    )
    stats = _load("interndata_a1_dual_arm_control_stats_broad.yaml")
    assert stats["data"]["train_episode_modulus"] == 5
    assert stats["data"]["train_episode_remainders"] == pilot["data"][
        "train_episode_remainders"
    ]
    assert stats["manifest_path"] == pilot["manifest_path"]
