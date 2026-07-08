"""Schema scanner for LeRobot v2.1 datasets.

Reads ``meta/info.json`` (the v2.1 metadata file) for each dataset, extracts the
camera / state / action layout, allocates ids via the registry, asserts known
(OXE) dims, and writes a human-readable ``schema_report.md``. A schema mismatch
fails **early** rather than silently mis-padding action vectors (CLAUDE.md §4).

This does not depend on the ``lerobot`` package — it parses the JSON directly so
it runs on CPU in the unit tests with a synthetic fixture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from data.registry import (
    NEW_EMBODIMENT,
    EmbodimentRegistry,
    default_registry,
)

EXPECTED_CODEBASE_VERSION = "v2.1"


@dataclass
class SchemaRecord:
    """One row of ``schema_report.md`` (fields per doc §4.2)."""

    dataset_id: str
    repo_or_path: str
    codebase_version: str
    fps: float
    feature_keys: list[str]
    visual_candidate_keys: list[str]
    camera_keys: list[str]
    state_keys: list[str]
    action_keys: list[str]
    raw_state_dim: int
    raw_action_dim: int
    selected_state_dim: int
    selected_action_dim: int
    semantics: str
    embodiment: str
    embodiment_id: int
    action_schema_id: int
    norm_stats_path: str = ""
    warnings: list[str] = field(default_factory=list)


def _feature_dim(feat: dict) -> int:
    """Dim of a LeRobot feature = product of its ``shape`` (default 1)."""
    shape = feat.get("shape", [1])
    dim = 1
    for s in shape:
        dim *= int(s)
    return dim


def _keys_by_dtype_prefix(features: dict, prefixes: tuple[str, ...]) -> list[str]:
    return [k for k in features if any(k.startswith(p) for p in prefixes)]


def _visual_candidate_keys(features: dict) -> list[str]:
    """Heuristic image/video feature candidates for datasets with nonstandard keys."""
    tokens = ("image", "images", "camera", "cam", "rgb", "video", "videos")
    out: list[str] = []
    for key, feat in features.items():
        dtype = str(feat.get("dtype", "")).lower()
        low = key.lower()
        if dtype in {"image", "video"} or any(tok in low for tok in tokens):
            out.append(key)
    return out


def scan_dataset(
    info_json_path: str | Path,
    dataset_id: Optional[str] = None,
    embodiment: Optional[str] = None,
    semantics: str = "",
    registry: Optional[EmbodimentRegistry] = None,
    norm_stats_path: str = "",
) -> SchemaRecord:
    """Scan one dataset's ``info.json`` into a ``SchemaRecord``.

    Raises if ``codebase_version`` is not v2.1 or if dims contradict a
    registered OXE expectation.
    """
    info_json_path = Path(info_json_path)
    registry = registry or default_registry()
    with info_json_path.open() as f:
        info = json.load(f)

    codebase_version = str(info.get("codebase_version", ""))
    if codebase_version != EXPECTED_CODEBASE_VERSION:
        raise ValueError(
            f"{info_json_path}: codebase_version={codebase_version!r} != "
            f"{EXPECTED_CODEBASE_VERSION!r} (Stage A assumes LeRobot v2.1)."
        )

    fps = float(info.get("fps", 0.0))
    if fps <= 0:
        raise ValueError(f"{info_json_path}: missing/invalid fps {fps!r}")

    features: dict = info.get("features", {})
    feature_keys = sorted(features.keys())
    visual_candidate_keys = _visual_candidate_keys(features)
    camera_keys = _keys_by_dtype_prefix(
        features, ("observation.images", "observation.image")
    )
    if not camera_keys:
        # InternData-style converted sets may not follow observation.images.*
        # exactly; keep the broader candidates visible and usable for inspection.
        camera_keys = list(visual_candidate_keys)
    state_keys = _keys_by_dtype_prefix(
        features, ("observation.state", "states", "state")
    )
    action_keys = _keys_by_dtype_prefix(features, ("actions", "action"))

    raw_state_dim = sum(_feature_dim(features[k]) for k in state_keys)
    raw_action_dim = sum(_feature_dim(features[k]) for k in action_keys)
    # M1: selected == raw; key-subset selection is a per-dataset config refinement.
    selected_state_dim = raw_state_dim
    selected_action_dim = raw_action_dim

    dataset_id = dataset_id or info_json_path.parent.parent.name
    embodiment = embodiment or str(info.get("robot_type", dataset_id))

    # OXE expectation assertion (no-op for scanned-only datasets).
    registry.assert_matches_expected(
        dataset_id, selected_state_dim, selected_action_dim
    )

    if embodiment == NEW_EMBODIMENT:
        # RoboTwin benchmark slot: trailing reserved id, never trained.
        embodiment_id = registry.new_embodiment_id()
    else:
        embodiment_id = registry.embodiment_id(embodiment)
    action_schema_id = registry.action_schema_id(
        selected_action_dim, selected_state_dim, semantics
    )

    warnings: list[str] = []
    if not features:
        warnings.append("features dict is empty")
    if not camera_keys:
        warnings.append("no camera/image features found")
    if not action_keys:
        warnings.append("no action features found (video-only dataset?)")

    return SchemaRecord(
        dataset_id=dataset_id,
        repo_or_path=str(info_json_path.parent.parent),
        codebase_version=codebase_version,
        fps=fps,
        feature_keys=feature_keys,
        visual_candidate_keys=visual_candidate_keys,
        camera_keys=camera_keys,
        state_keys=state_keys,
        action_keys=action_keys,
        raw_state_dim=raw_state_dim,
        raw_action_dim=raw_action_dim,
        selected_state_dim=selected_state_dim,
        selected_action_dim=selected_action_dim,
        semantics=semantics,
        embodiment=embodiment,
        embodiment_id=embodiment_id,
        action_schema_id=action_schema_id,
        norm_stats_path=norm_stats_path,
        warnings=warnings,
    )


_REPORT_COLUMNS = [
    "dataset_id",
    "embodiment",
    "embodiment_id",
    "action_schema_id",
    "codebase_version",
    "fps",
    "feature_keys",
    "visual_candidate_keys",
    "camera_keys",
    "state_keys",
    "action_keys",
    "raw_state_dim",
    "raw_action_dim",
    "selected_state_dim",
    "selected_action_dim",
    "semantics",
    "norm_stats_path",
    "warnings",
]


def _cell(value) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "-"
    s = str(value)
    return s if s else "-"


def write_schema_report(
    records: list[SchemaRecord], out_path: str | Path
) -> Path:
    """Write a Markdown table of scanned schemas; returns the written path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(_REPORT_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _REPORT_COLUMNS) + " |"
    lines = ["# Schema Report", "", header, sep]
    for rec in records:
        row = [_cell(getattr(rec, col)) for col in _REPORT_COLUMNS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    out_path.write_text("\n".join(lines))
    return out_path
