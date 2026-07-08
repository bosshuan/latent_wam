"""Inspect a downloaded Wan2.2-TI2V-5B checkpoint without loading tensors.

The official Hugging Face/ModelScope directory contains diffusion safetensor
shards plus VAE/T5 files. This script reads safetensor metadata only, maps the
diffusion keys through our Latent-WAM loader rules, and compares them against a
meta-device WanLatentWorldActionDiT so we can validate key/shape compatibility
before attempting a real 5B load or FSDP run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from models.wan_config import WAN22_TI2V_5B, WanConfig
from models.wan_latent_world_action_dit import WanLatentWorldActionDiT
from models.weight_loading import remap_wan_key


DEFAULT_CKPT_DIR = "/mnt/sfs_turbo/fyy/checkpoints/Wan2.2-TI2V-5B"
DEFAULT_REPORT_DIR = "reports/wan2_2"


def _as_list(x: tuple[int, ...] | list[int]) -> list[int]:
    return [int(v) for v in x]


def _find_diffusion_shards(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(path.glob(pattern))
    if not files:
        files = sorted(path.glob("diffusion_pytorch_model*.safetensors"))
    if not files:
        files = sorted(path.glob("*.safetensors"))
    return [p for p in files if p.suffix == ".safetensors"]


def _read_safetensor_metadata(files: list[Path]) -> dict[str, dict[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - depends on server env
        raise ImportError("safetensors is required to inspect Wan checkpoint metadata") from exc

    meta: dict[str, dict[str, Any]] = {}
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor_slice = handle.get_slice(key)
                if key in meta:
                    raise ValueError(f"duplicate tensor key across shards: {key}")
                meta[key] = {
                    "file": file.name,
                    "shape": _as_list(tensor_slice.get_shape()),
                    "dtype": str(tensor_slice.get_dtype()),
                }
    return meta


def _strip_prefix_if_all(
    meta: dict[str, dict[str, Any]], prefix: str
) -> tuple[dict[str, dict[str, Any]], str | None]:
    if meta and all(k.startswith(prefix) for k in meta):
        return {k[len(prefix) :]: v for k, v in meta.items()}, prefix
    return meta, None


def _normalize_key_prefixes(meta: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], str | None]:
    stripped = None
    for prefix in (
        "module.model.diffusion_model.",
        "model.diffusion_model.",
        "diffusion_model.",
        "module.model.",
        "module.",
        "model.",
    ):
        meta, matched = _strip_prefix_if_all(meta, prefix)
        if matched is not None:
            stripped = matched
    return meta, stripped


def _build_meta_model_state(cfg: WanConfig, args: argparse.Namespace) -> dict[str, list[int]]:
    with torch.device("meta"):
        model = WanLatentWorldActionDiT(
            cfg=cfg,
            latent_dim=args.latent_dim,
            action_dim=args.action_dim,
            num_embodiments=args.num_embodiments,
            grid_hw=tuple(args.grid_hw),
            max_chunks=args.max_chunks,
            max_actions=args.max_actions,
            state_dim=args.state_dim,
            text_seq_len=args.text_seq_len,
            action_token_scale=1.0,
            action_latent_bridge_scale=0.0,
        )
    return {k: _as_list(v.shape) for k, v in model.state_dict().items()}


def _companion_files(path: Path) -> dict[str, bool]:
    if path.is_file():
        root = path.parent
    else:
        root = path
    names = {
        "config_json": "config.json",
        "configuration_json": "configuration.json",
        "safetensors_index": "diffusion_pytorch_model.safetensors.index.json",
        "vae": "Wan2.2_VAE.pth",
        "umt5": "models_t5_umt5-xxl-enc-bf16.pth",
    }
    return {k: (root / v).exists() for k, v in names.items()}


def _config_report(cfg: WanConfig) -> dict[str, Any]:
    expected = WAN22_TI2V_5B
    fields = ("dim", "num_layers", "num_heads", "ffn_dim", "freq_dim", "text_dim", "eps")
    values = {f: getattr(cfg, f) for f in fields}
    expected_values = {f: getattr(expected, f) for f in fields}
    return {
        "values": values,
        "expected": expected_values,
        "matches_known_ti2v_5b": values == expected_values,
        "head_dim": cfg.head_dim,
    }


def _inspect(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    files = _find_diffusion_shards(path, args.pattern)
    if not files:
        raise FileNotFoundError(f"no safetensor diffusion shards found under {path}")
    tensor_meta = _read_safetensor_metadata(files)
    tensor_meta, stripped_prefix = _normalize_key_prefixes(tensor_meta)
    cfg = WanConfig.from_yaml(args.wan_config, key_path=tuple(args.wan_key_path))
    model_state = _build_meta_model_state(cfg, args)

    dropped: dict[str, list[str]] = defaultdict(list)
    remapped_shapes: dict[str, list[int]] = {}
    raw_to_mapped: dict[str, str] = {}
    for raw_key, meta in tensor_meta.items():
        mapped_key, drop_cat = remap_wan_key(raw_key)
        if drop_cat is not None:
            dropped[drop_cat].append(raw_key)
            continue
        remapped_shapes[mapped_key] = meta["shape"]
        raw_to_mapped[raw_key] = mapped_key

    model_keys = set(model_state)
    remapped_keys = set(remapped_shapes)
    missing = sorted(model_keys - remapped_keys)
    unexpected = sorted(remapped_keys - model_keys)
    backbone_missing = sorted(k for k in missing if k.startswith("backbone."))
    shape_mismatches = []
    for key in sorted(model_keys.intersection(remapped_keys)):
        if model_state[key] != remapped_shapes[key]:
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint": remapped_shapes[key],
                    "model": model_state[key],
                }
            )

    dtype_counts = Counter(meta["dtype"] for meta in tensor_meta.values())
    file_key_counts = Counter(meta["file"] for meta in tensor_meta.values())
    expected_missing_prefixes = (
        "latent_adapter",
        "latent_head",
        "action_encoder",
        "action_to_latent",
        "action_head",
        "state_adapter",
        "tokenizer",
        "value_",
        "value_head",
        "text_encoder",
    )
    unexpected_missing = [
        k for k in missing if not k.startswith(expected_missing_prefixes) and not k.startswith("backbone.")
    ]
    status = "PASS"
    if unexpected or backbone_missing or shape_mismatches:
        status = "FAIL"
    elif unexpected_missing:
        status = "CHECK"

    return {
        "status": status,
        "checkpoint_path": str(path),
        "shards": [str(p) for p in files],
        "num_shards": len(files),
        "num_checkpoint_tensors": len(tensor_meta),
        "num_loadable_tensors": len(remapped_shapes),
        "num_model_tensors": len(model_state),
        "dtype_counts": dict(dtype_counts),
        "file_key_counts": dict(file_key_counts),
        "stripped_prefix": stripped_prefix,
        "config": _config_report(cfg),
        "companion_files": _companion_files(path),
        "dropped_counts": {k: len(v) for k, v in sorted(dropped.items())},
        "dropped_examples": {k: v[:8] for k, v in sorted(dropped.items())},
        "missing_count": len(missing),
        "missing_examples": missing[:40],
        "backbone_missing_count": len(backbone_missing),
        "backbone_missing_examples": backbone_missing[:40],
        "unexpected_count": len(unexpected),
        "unexpected_examples": unexpected[:40],
        "shape_mismatch_count": len(shape_mismatches),
        "shape_mismatch_examples": shape_mismatches[:40],
        "unexpected_missing_count": len(unexpected_missing),
        "unexpected_missing_examples": unexpected_missing[:40],
        "raw_to_mapped_examples": list(raw_to_mapped.items())[:16],
    }


def _write_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "wan2_2_ti2v_5b_checkpoint_report.json"
    md_path = out_dir / "wan2_2_ti2v_5b_checkpoint_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Wan2.2-TI2V-5B Checkpoint Report",
        "",
        f"- status: `{report['status']}`",
        f"- checkpoint_path: `{report['checkpoint_path']}`",
        f"- shards: `{report['num_shards']}`",
        f"- checkpoint tensors: `{report['num_checkpoint_tensors']}`",
        f"- loadable tensors: `{report['num_loadable_tensors']}`",
        f"- model tensors: `{report['num_model_tensors']}`",
        f"- config matches known TI2V-5B: `{report['config']['matches_known_ti2v_5b']}`",
        f"- stripped prefix: `{report['stripped_prefix']}`",
        f"- dropped counts: `{report['dropped_counts']}`",
        f"- missing count: `{report['missing_count']}`",
        f"- backbone missing count: `{report['backbone_missing_count']}`",
        f"- unexpected count: `{report['unexpected_count']}`",
        f"- shape mismatch count: `{report['shape_mismatch_count']}`",
        f"- companion files: `{report['companion_files']}`",
        "",
        "## Examples",
        "",
        "### Missing",
        "",
    ]
    lines.extend(f"- `{k}`" for k in report["missing_examples"][:20])
    lines.extend(["", "### Unexpected", ""])
    lines.extend(f"- `{k}`" for k in report["unexpected_examples"][:20])
    lines.extend(["", "### Shape Mismatches", ""])
    if report["shape_mismatch_examples"]:
        for item in report["shape_mismatch_examples"][:20]:
            lines.append(f"- `{item['key']}` ckpt={item['checkpoint']} model={item['model']}")
    else:
        lines.append("- none")
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--pattern", default="diffusion_pytorch_model*.safetensors")
    parser.add_argument("--wan-config", default="configs/model/latent_wam_dit.yaml")
    parser.add_argument("--wan-key-path", nargs="*", default=["wan"])
    parser.add_argument("--out-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--grid-hw", type=int, nargs=2, default=[12, 12])
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--max-chunks", type=int, default=16)
    parser.add_argument("--num-embodiments", type=int, default=2)
    parser.add_argument("--text-seq-len", type=int, default=8)
    args = parser.parse_args()

    report = _inspect(Path(args.checkpoint_dir), args)
    json_path, md_path = _write_reports(report, Path(args.out_dir))
    print(
        "[wan-inspect] "
        f"status={report['status']} shards={report['num_shards']} "
        f"loadable={report['num_loadable_tensors']} dropped={report['dropped_counts']} "
        f"backbone_missing={report['backbone_missing_count']} "
        f"unexpected={report['unexpected_count']} "
        f"shape_mismatch={report['shape_mismatch_count']}"
    )
    print(f"[wan-inspect] wrote {json_path}")
    print(f"[wan-inspect] wrote {md_path}")
    if report["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":  # pragma: no cover
    main()
