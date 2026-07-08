"""M1 schema-scanner tests: scan, report, early-fail on mismatch / wrong version."""

from __future__ import annotations

import json

from data.registry import EmbodimentRegistry, ExpectedSchema
from data.schema_scanner import (
    scan_dataset,
    write_schema_report,
)


def _write_info(tmp_path, dataset_id, version="v2.1", fps=30, action_dim=7, state_dim=9):
    d = tmp_path / dataset_id / "meta"
    d.mkdir(parents=True)
    info = {
        "codebase_version": version,
        "fps": fps,
        "robot_type": "panda",
        "features": {
            "observation.images.top": {"dtype": "video", "shape": [3, 224, 224]},
            "observation.state": {"dtype": "float32", "shape": [state_dim]},
            "action": {"dtype": "float32", "shape": [action_dim]},
        },
    }
    p = d / "info.json"
    p.write_text(json.dumps(info))
    return p


def test_scan_produces_record(tmp_path):
    p = _write_info(tmp_path, "agibot_x", action_dim=14, state_dim=16)
    rec = scan_dataset(p, semantics="ee_pose+gripper")
    assert rec.dataset_id == "agibot_x"
    assert rec.codebase_version == "v2.1"
    assert rec.fps == 30
    assert rec.selected_action_dim == 14
    assert rec.selected_state_dim == 16
    assert rec.camera_keys == ["observation.images.top"]
    assert rec.action_keys == ["action"]
    assert rec.embodiment_id >= 0
    assert rec.action_schema_id >= 0


def test_write_report(tmp_path):
    recs = [scan_dataset(_write_info(tmp_path, "ds_a"))]
    out = write_schema_report(recs, tmp_path / "schema_report.md")
    assert out.exists()
    text = out.read_text()
    assert text.startswith("# Schema Report")
    assert "ds_a" in text
    assert "action_schema_id" in text


def test_wrong_codebase_version_raises(tmp_path):
    p = _write_info(tmp_path, "old_ds", version="v3.0")
    try:
        scan_dataset(p)
    except ValueError:
        return
    raise AssertionError("scanner must reject non-v2.1 datasets")


def test_oxe_expected_mismatch_raises(tmp_path):
    p = _write_info(tmp_path, "oxe_bridge", action_dim=7, state_dim=9)
    reg = EmbodimentRegistry()
    reg.register_expected(
        ExpectedSchema("oxe_bridge", "widowx", state_dim=9, action_dim=99)
    )
    try:
        scan_dataset(p, registry=reg)
    except ValueError:
        return
    raise AssertionError("scanner must fail early on OXE dim mismatch")
