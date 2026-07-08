"""On-disk feature cache for frozen V-JEPA features.

V-JEPA encoding is expensive, so Stage A caches features and reuses them across
epochs. The cache key folds in every version knob so stale entries can never be
served after a config change (CLAUDE.md §4 cache-key rule):
    (dataset_id, episode, frame_range, vjepa_version, extract_layers, norm_version)

Capacity guard (user note 2): the M1/M2 payload is the *raw* multi-level feature
``[B, 576, 6656]`` ≈ 7.6 MB per time-token in fp16 — at video scale this blows
up disk. So the cache supports:
  * a **subset allowlist** (only cache an explicit set of (dataset, episode)
    pairs — e.g. a fixed training subset), and
  * an **LRU eviction + byte cap** that drops the least-recently-used entries.
All payloads are stored fp16.

M5 switches the payload to the 384-d VJ-RAE latent; the key already versions the
norm/VJ-RAE so that switch invalidates the raw-feature entries automatically.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass(frozen=True)
class CacheKeyFields:
    dataset_id: str
    episode: int
    frame_start: int
    frame_end: int

    def as_pair(self) -> tuple[str, int]:
        return (self.dataset_id, self.episode)


class FeatureCache:
    def __init__(
        self,
        cache_dir: str | Path,
        vjepa_version: str,
        extract_layers: tuple[int, ...],
        norm_version: str,
        max_bytes: Optional[int] = None,
        allowed_pairs: Optional[set[tuple[str, int]]] = None,
        vj_rae_version: Optional[str] = None,
        codec_version: Optional[str] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Version components folded into every key (changing any invalidates all).
        self.vjepa_version = str(vjepa_version)
        self.extract_layers = tuple(int(x) for x in extract_layers)
        self.norm_version = str(norm_version)
        # M5 cache switch: once the VJ-RAE is frozen, Stage A caches the **384-d
        # VJ-RAE latent** instead of the raw multi-level feature. ``vj_rae_version``
        # (checkpoint + latent_dim id) is folded into the key so VJ-RAE-latent
        # entries can NEVER be served for a raw-feature key (and vice versa).
        # ``codec_version`` remains as a backward-compatible alias.
        if vj_rae_version is not None and codec_version is not None:
            raise ValueError("pass only one of vj_rae_version or codec_version")
        payload_version = vj_rae_version if vj_rae_version is not None else codec_version
        self.vj_rae_version = None if payload_version is None else str(payload_version)
        self.codec_version = self.vj_rae_version
        self.max_bytes = max_bytes
        # None => cache everything; a set => only these (dataset_id, episode).
        self.allowed_pairs = allowed_pairs

        # LRU index: key_hash -> (path, nbytes); ordered by recency (oldest first).
        self._index: "OrderedDict[str, tuple[Path, int]]" = OrderedDict()
        self._total_bytes = 0

    # --- key construction ----------------------------------------------
    def _key_str(self, fields: CacheKeyFields) -> str:
        layers = "-".join(str(x) for x in self.extract_layers)
        payload = "raw" if self.vj_rae_version is None else f"vj_rae={self.vj_rae_version}"
        return (
            f"{fields.dataset_id}|ep{fields.episode}|"
            f"f{fields.frame_start}:{fields.frame_end}|"
            f"vj={self.vjepa_version}|layers={layers}|norm={self.norm_version}|{payload}"
        )

    def _key_hash(self, fields: CacheKeyFields) -> str:
        return hashlib.sha256(self._key_str(fields).encode()).hexdigest()[:32]

    def _path(self, key_hash: str) -> Path:
        return self.cache_dir / f"{key_hash}.pt"

    def key_hash(self, fields: CacheKeyFields) -> str:
        """Public stable hash for manifests and debugging."""
        return self._key_hash(fields)

    def path_for(self, fields: CacheKeyFields) -> Path:
        """Path where ``fields`` is or would be stored."""
        return self._path(self._key_hash(fields))

    # --- policy ---------------------------------------------------------
    def should_cache(self, fields: CacheKeyFields) -> bool:
        if self.allowed_pairs is None:
            return True
        return fields.as_pair() in self.allowed_pairs

    # --- public API -----------------------------------------------------
    def get(self, fields: CacheKeyFields) -> Optional[torch.Tensor]:
        key = self._key_hash(fields)
        entry = self._index.get(key)
        path = entry[0] if entry else self._path(key)
        if not path.exists():
            self._index.pop(key, None)
            return None
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        # touch for LRU
        if key in self._index:
            self._index.move_to_end(key)
        else:
            nbytes = path.stat().st_size
            self._index[key] = (path, nbytes)
            self._total_bytes += nbytes
        return tensor

    def put(self, fields: CacheKeyFields, tensor: torch.Tensor) -> bool:
        """Store ``tensor`` (cast fp16). Returns False if skipped by allowlist."""
        if not self.should_cache(fields):
            return False
        key = self._key_hash(fields)
        path = self._path(key)
        payload = tensor.detach().to(torch.float16).contiguous()
        torch.save(payload, path)
        nbytes = path.stat().st_size

        if key in self._index:
            self._total_bytes -= self._index[key][1]
        self._index[key] = (path, nbytes)
        self._index.move_to_end(key)
        self._total_bytes += nbytes

        self._evict_if_needed()
        return True

    def _evict_if_needed(self) -> None:
        if self.max_bytes is None:
            return
        while self._total_bytes > self.max_bytes and len(self._index) > 1:
            old_key, (old_path, old_bytes) = self._index.popitem(last=False)
            self._total_bytes -= old_bytes
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass

    # --- introspection (tests / monitoring) ----------------------------
    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._index)

    def contains(self, fields: CacheKeyFields) -> bool:
        return self._path(self._key_hash(fields)).exists()
