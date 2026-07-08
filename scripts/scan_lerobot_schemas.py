"""Scan LeRobot v2.1 dataset schemas from a YAML config.

This is the first real-data smoke step: discover ``meta/info.json`` files under
the server dataset mount, parse image/state/action keys, allocate embodiment and
action-schema ids, and write a report for manual inspection before wiring actual
training dataloaders.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import asdict
from pathlib import Path

import yaml

from data.registry import default_registry
from data.schema_scanner import SchemaRecord, scan_dataset, write_schema_report


def _sanitize_id(text: str) -> str:
    text = text.strip().replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _glob_paths(root: Path, pattern: str) -> list[Path]:
    pat = Path(pattern)
    full_pattern = str(pat if pat.is_absolute() else root / pattern)
    return [Path(p) for p in sorted(glob.glob(full_pattern, recursive=True))]


def _dataset_id(root: Path, info_path: Path, family: str, group_name: str) -> str:
    # info_path = <dataset_root>/meta/info.json for LeRobot v2.1.
    ds_root = info_path.parent.parent
    try:
        rel = ds_root.relative_to(root)
    except ValueError:
        rel = ds_root.name
    return _sanitize_id(f"{family}__{group_name}__{rel}")


def discover_records(cfg: dict) -> list[SchemaRecord]:
    root = Path(cfg["root"])
    family = str(cfg.get("dataset_family", root.name))
    registry = default_registry()
    records: list[SchemaRecord] = []
    seen: set[Path] = set()

    for group in cfg.get("datasets", []):
        group_name = str(group["name"])
        semantics = str(group.get("semantics", group_name))
        embodiment = group.get("embodiment")
        norm_stats_path = str(group.get("norm_stats_path", ""))
        for pattern in group.get("info_globs", []):
            for info_path in _glob_paths(root, str(pattern)):
                info_path = info_path.resolve()
                if info_path in seen:
                    continue
                seen.add(info_path)
                records.append(
                    scan_dataset(
                        info_path,
                        dataset_id=_dataset_id(root.resolve(), info_path, family, group_name),
                        embodiment=embodiment,
                        semantics=semantics,
                        registry=registry,
                        norm_stats_path=norm_stats_path,
                    )
                )
    return records


def _write_json(records: list[SchemaRecord], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(r) for r in records], indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="override markdown report path")
    parser.add_argument("--out-json", default=None, help="override json report path")
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="write reports but do not fail when scanner warnings are present",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    records = discover_records(cfg)
    scan_cfg = cfg.get("scan", {})
    expected_min = int(scan_cfg.get("expected_min_count", 1))
    if len(records) < expected_min:
        raise SystemExit(
            f"discovered {len(records)} LeRobot dataset(s), expected at least "
            f"{expected_min}; check root/info_globs in {args.config}"
        )

    out_md = args.out or scan_cfg.get("output_markdown", "reports/schema_report.md")
    out_json = args.out_json or scan_cfg.get("output_json", "")
    md_path = write_schema_report(records, out_md)
    json_path = _write_json(records, out_json) if out_json else None

    print(f"[schema] discovered {len(records)} dataset(s)")
    print(f"[schema] wrote {md_path}")
    if json_path is not None:
        print(f"[schema] wrote {json_path}")

    warnings = [(r.dataset_id, r.warnings) for r in records if r.warnings]
    fail_on_warnings = bool(scan_cfg.get("fail_on_warnings", True))
    if warnings:
        for dataset_id, msgs in warnings:
            print(f"[schema][warning] {dataset_id}: {'; '.join(msgs)}")
        if fail_on_warnings and not args.allow_warnings:
            raise SystemExit(
                "schema warnings present; fix the dataset/config or rerun with "
                "--allow-warnings for inspection only"
            )


if __name__ == "__main__":  # pragma: no cover
    main()
