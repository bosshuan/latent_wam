"""Generate a tiny V-JEPA/VJ-RAE latent cache from real robot data.

This is a smoke/integration step, not full training. It verifies:

* real LeRobot-v2.1-format robot clips can be resized to V-JEPA's native 384;
* frozen V-JEPA features flow into VJ-RAE;
* 384-d 12x12 latents are written through ``FeatureCache``;
* a JSONL manifest records dataset/episode/window provenance for later readers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data.collate import collate_trajectory
from data.feature_cache import CacheKeyFields, FeatureCache
from data.robot_dataset import RobotSchemaBinding, RobotTrajectoryDataset
from data.schemas import MultiLevelFeatures
from models.vjepa_encoder import FrozenVJEPAEncoder
from train.train_vj_rae import build_codec_from_encoder


def _cfg_get(cfg: dict, *path, default=None):
    cur: Any = cfg
    for key in path:
        if cur is None:
            return default
        if hasattr(cur, "get"):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def _load_records(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _filter_records(records: list[dict], cfg: dict) -> list[dict]:
    fcfg = cfg.get("filter", {})
    embodiments = set(fcfg.get("embodiments", []))
    schema_ids = set(int(x) for x in fcfg.get("action_schema_ids", []))
    out = []
    for rec in records:
        if embodiments and rec["embodiment"] not in embodiments:
            continue
        if schema_ids and int(rec["action_schema_id"]) not in schema_ids:
            continue
        out.append(rec)
    return out


def _subset(ds, start_index: int, max_samples: int):
    n = len(ds)
    if n <= 0:
        raise ValueError("dataset has length 0")
    start = min(max(start_index, 0), n - 1)
    end = min(n, start + max_samples)
    return Subset(ds, list(range(start, end)))


def _build_dataset(rec: dict, cfg: dict):
    temporal = cfg.get("temporal", {})
    binding_cfg = cfg.get("binding", {})
    image_key = binding_cfg.get("image_key", "images.rgb.head")
    if image_key not in rec["camera_keys"]:
        raise ValueError(
            f"{rec['dataset_id']}: requested image_key={image_key!r} not in "
            f"camera_keys={rec['camera_keys']}"
        )

    binding = RobotSchemaBinding(
        image_key=image_key,
        state_keys=tuple(rec["state_keys"]),
        action_keys=tuple(rec["action_keys"]),
        embodiment_id=int(rec["embodiment_id"]),
        action_schema_id=int(rec["action_schema_id"]),
        fps=float(rec["fps"]),
        dataset_id=str(rec["dataset_id"]),
    )
    return RobotTrajectoryDataset(
        repo_id=rec["repo_or_path"],
        binding=binding,
        token_grid=tuple(int(x) for x in temporal.get("token_grid", [8, 24, 24])),
        history_chunks=int(temporal.get("history_chunks", 4)),
        future_chunks=int(temporal.get("future_chunks", 4)),
        tubelet=int(temporal.get("tubelet", 2)),
        backend=str(cfg.get("reader_backend", "auto")),
    )


def _build_loader(cfg: dict) -> DataLoader:
    records = _filter_records(_load_records(cfg["schema_report"]), cfg)
    if not records:
        raise SystemExit(f"no schema records selected by {cfg['schema_report']}")

    dcfg = cfg.get("data", {})
    start_index = int(dcfg.get("start_index", 0))
    max_samples = int(dcfg.get("max_samples_per_dataset", 8))
    datasets = []
    log_prefix = str(cfg.get("log_prefix", "cache"))
    print(f"[{log_prefix}] selected {len(records)} dataset(s)")
    for rec in records:
        ds = _build_dataset(rec, cfg)
        sub = _subset(ds, start_index=start_index, max_samples=max_samples)
        datasets.append(sub)
        print(
            f"[{log_prefix}] dataset "
            f"id={rec['dataset_id']} len={len(ds)} subset_len={len(sub)} "
            f"action_dim={rec['selected_action_dim']} "
            f"state_dim={rec['selected_state_dim']}",
            flush=True,
        )

    return DataLoader(
        ConcatDataset(datasets),
        batch_size=int(dcfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(dcfg.get("num_workers", 0)),
        collate_fn=collate_trajectory,
        pin_memory=bool(dcfg.get("pin_memory", True)),
    )


def _preprocess_pixels(pixels: torch.Tensor, cfg: dict, device: torch.device) -> torch.Tensor:
    pcfg = cfg.get("preprocess", {})
    img_size = int(pcfg.get("img_size", 384))
    resize_mode = str(pcfg.get("resize_mode", "bilinear"))
    normalize = bool(pcfg.get("normalize", True))

    if pixels.ndim != 5:
        raise ValueError(f"expected pixels [B,T,3,H,W], got {tuple(pixels.shape)}")
    b, t, c, h, w = pixels.shape
    if c != 3:
        raise ValueError(f"expected RGB pixels, got c={c}")

    x = pixels.to(device=device, dtype=torch.float32, non_blocking=True)
    x = x.reshape(b * t, c, h, w)
    if h != img_size or w != img_size:
        align_corners = False if resize_mode in {"linear", "bilinear", "bicubic", "trilinear"} else None
        x = F.interpolate(
            x,
            size=(img_size, img_size),
            mode=resize_mode,
            align_corners=align_corners,
        )
    if normalize:
        mean = torch.tensor(
            pcfg.get("mean", [0.485, 0.456, 0.406]),
            device=device,
            dtype=x.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            pcfg.get("std", [0.229, 0.224, 0.225]),
            device=device,
            dtype=x.dtype,
        ).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x.reshape(b, t, c, img_size, img_size)


def _build_encoder(cfg: dict, device: torch.device) -> FrozenVJEPAEncoder:
    ecfg = cfg.get("encoder", {})
    encoder = FrozenVJEPAEncoder(
        hub_name=str(ecfg.get("hub_name", "vjepa2_1_vit_gigantic_384")),
        hub_repo=str(ecfg.get("hub_repo", "facebookresearch/vjepa2")),
        hub_source=str(ecfg.get("hub_source", "github")),
        extract_layers=tuple(int(x) for x in ecfg.get("extract_layers", [11, 23, 37, 47])),
        pretrained=bool(ecfg.get("pretrained", True)),
        checkpoint_path=ecfg.get("checkpoint_path"),
        checkpoint_key=str(ecfg.get("checkpoint_key", "target_encoder")),
        checkpoint_strict=bool(ecfg.get("checkpoint_strict", True)),
        assert_gigantic=bool(ecfg.get("assert_gigantic", True)),
    )
    encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def _load_vj_rae(codec: torch.nn.Module, cfg: dict, device: torch.device) -> bool:
    vcfg = cfg.get("vj_rae", {})
    ckpt_path = vcfg.get("checkpoint_path")
    allow_random = bool(vcfg.get("allow_random_vj_rae", False))
    if not ckpt_path:
        if not allow_random:
            raise ValueError(
                "vj_rae.checkpoint_path is required unless "
                "vj_rae.allow_random_vj_rae=true for a smoke-only cache."
            )
        print("[cache][warning] using randomly initialized VJ-RAE; smoke only", flush=True)
        return False

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("vj_rae") or ckpt.get("codec") or ckpt
    state = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    result = codec.load_state_dict(state, strict=bool(vcfg.get("strict", True)))
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    print(f"[cache] loaded VJ-RAE checkpoint {ckpt_path}")
    print(f"[cache] missing_keys ({len(missing)}): {missing}")
    print(f"[cache] unexpected_keys ({len(unexpected)}): {unexpected}")
    return True


def _build_codec(encoder: FrozenVJEPAEncoder, cfg: dict, device: torch.device):
    vcfg = cfg.get("vj_rae", {})
    codec = build_codec_from_encoder(
        encoder,
        hidden_dim=int(vcfg.get("hidden_dim", 1024)),
        latent_dim=int(vcfg.get("latent_dim", 384)),
        pool=int(vcfg.get("pool", 2)),
        norm_version=str(_cfg_get(cfg, "feature_cache", "norm_version", default="v0")),
    )
    loaded = _load_vj_rae(codec, cfg, device)
    codec.to(device)
    codec.freeze()
    return codec, loaded


def _build_cache(cfg: dict) -> FeatureCache:
    ccfg = cfg.get("feature_cache", {})
    vjepa_version = str(ccfg.get("vjepa_version", "vjepa2_1_vitG_384"))
    vj_rae_version = str(ccfg.get("vj_rae_version", ""))
    allow_random = bool(_cfg_get(cfg, "vj_rae", "allow_random_vj_rae", default=False))
    encoder_pretrained = bool(_cfg_get(cfg, "encoder", "pretrained", default=True))
    encoder_checkpoint = _cfg_get(cfg, "encoder", "checkpoint_path", default=None)
    encoder_random = (not encoder_pretrained) and not encoder_checkpoint
    if encoder_random and "random" not in vjepa_version and "smoke" not in vjepa_version:
        raise ValueError(
            "random/untrained V-JEPA caches must use a feature_cache.vjepa_version "
            "containing 'random' or 'smoke' to avoid polluting real cache keys."
        )
    if allow_random and "random" not in vj_rae_version and "smoke" not in vj_rae_version:
        raise ValueError(
            "random VJ-RAE caches must use a feature_cache.vj_rae_version "
            "containing 'random' or 'smoke' to avoid polluting real cache keys."
        )
    return FeatureCache(
        cache_dir=ccfg.get("cache_dir", ".feature_cache/interndata_a1_smoke"),
        vjepa_version=vjepa_version,
        extract_layers=tuple(int(x) for x in _cfg_get(cfg, "encoder", "extract_layers", default=[11, 23, 37, 47])),
        norm_version=str(ccfg.get("norm_version", "v0")),
        max_bytes=ccfg.get("max_bytes"),
        vj_rae_version=vj_rae_version,
    )


def _require_cache_metadata(batch) -> None:
    if not batch.dataset_id:
        raise ValueError("batch is missing dataset_id metadata")
    for name in ("episode_index", "frame_start", "frame_end"):
        value = getattr(batch, name)
        if value is None or bool((value < 0).any().item()):
            raise ValueError(f"batch is missing valid {name} metadata: {value}")


def _write_manifest_record(f, record: dict) -> None:
    f.write(json.dumps(record, sort_keys=True) + "\n")
    f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.set_grad_enabled(False)
    if bool(cfg.get("allow_tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device_name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name)
    loader = _build_loader(cfg)
    encoder = _build_encoder(cfg, device)
    codec, loaded_vj_rae = _build_codec(encoder, cfg, device)
    cache = _build_cache(cfg)
    vjepa_loaded = bool(
        _cfg_get(cfg, "encoder", "pretrained", default=True)
        or _cfg_get(cfg, "encoder", "checkpoint_path", default=None)
    )

    ccfg = cfg.get("feature_cache", {})
    manifest_path = Path(ccfg.get("manifest_path", Path(cache.cache_dir) / "manifest.jsonl"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    max_batches = int(_cfg_get(cfg, "data", "max_batches", default=1))
    verify_readback = bool(ccfg.get("verify_readback", True))

    wrote = 0
    with manifest_path.open("w") as manifest:
        for step, batch in enumerate(loader):
            if step >= max_batches:
                break
            _require_cache_metadata(batch)
            pixels = _preprocess_pixels(batch.pixels, cfg, device)
            feats: MultiLevelFeatures = encoder(pixels)
            latent = codec.encode(feats).latent.detach().cpu()

            print(
                f"[cache] batch={step} pixels={tuple(pixels.shape)} "
                f"features={tuple(feats.features.shape)} latent={tuple(latent.shape)}",
                flush=True,
            )

            for i in range(latent.shape[0]):
                fields = CacheKeyFields(
                    dataset_id=batch.dataset_id[i],
                    episode=int(batch.episode_index[i].item()),
                    frame_start=int(batch.frame_start[i].item()),
                    frame_end=int(batch.frame_end[i].item()),
                )
                cache.put(fields, latent[i])
                if verify_readback:
                    got = cache.get(fields)
                    if got is None or tuple(got.shape) != tuple(latent[i].shape):
                        raise RuntimeError(
                            f"cache readback failed for {fields}: "
                            f"got={None if got is None else tuple(got.shape)}"
                        )
                record = {
                    "dataset_id": fields.dataset_id,
                    "episode": fields.episode,
                    "frame_start": fields.frame_start,
                    "frame_end": fields.frame_end,
                    "sample_index": int(batch.sample_index[i].item()),
                    "cache_key": cache.key_hash(fields),
                    "cache_path": str(cache.path_for(fields)),
                    "latent_shape": list(latent[i].shape),
                    "latent_dtype": "float16",
                    "vjepa_loaded": vjepa_loaded,
                    "vj_rae_loaded": loaded_vj_rae,
                    "embodiment_id": int(batch.embodiment_id[i].item()),
                    "action_schema_id": int(batch.action_schema_id[i].item()),
                    "action_shape": None if batch.actions is None else list(batch.actions[i].shape),
                    "proprio_shape": None if batch.proprio is None else list(batch.proprio[i].shape),
                    "fps": None if batch.fps_meta is None else float(batch.fps_meta[i].item()),
                    "text": batch.text[i] if batch.text else "",
                }
                _write_manifest_record(manifest, record)
                wrote += 1

    print(f"[cache] wrote {wrote} latent item(s)")
    print(f"[cache] manifest {manifest_path}")
    print(f"[cache] cache_dir {cache.cache_dir}")
    print("[cache] ok", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
