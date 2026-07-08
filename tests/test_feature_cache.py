"""M1 feature-cache tests: round-trip, version keying, allowlist, eviction."""

from __future__ import annotations

import torch

from data.feature_cache import CacheKeyFields, FeatureCache


def _cache(tmp_path, **kw):
    defaults = dict(
        cache_dir=tmp_path,
        vjepa_version="vitG",
        extract_layers=(11, 23, 37, 47),
        norm_version="v0",
    )
    defaults.update(kw)
    return FeatureCache(**defaults)


def _fields(ep=0):
    return CacheKeyFields(dataset_id="ds", episode=ep, frame_start=0, frame_end=16)


def test_put_get_roundtrip_identical(tmp_path):
    cache = _cache(tmp_path)
    x = torch.randn(2, 576, 64)
    assert cache.put(_fields(), x) is True
    got = cache.get(_fields())
    assert got is not None
    assert torch.equal(got, x.half())  # stored fp16


def test_key_changes_with_version(tmp_path):
    c0 = _cache(tmp_path, norm_version="v0")
    c0.put(_fields(), torch.randn(1, 4, 8))
    # different norm_version => different key => miss
    c1 = _cache(tmp_path, norm_version="v1")
    assert c1.get(_fields()) is None
    # different extract_layers => miss
    c2 = _cache(tmp_path, extract_layers=(47,))
    assert c2.get(_fields()) is None
    # same config => hit
    c0b = _cache(tmp_path, norm_version="v0")
    assert c0b.get(_fields()) is not None


def test_public_manifest_path_matches_cache_file(tmp_path):
    cache = _cache(tmp_path)
    fields = _fields()
    cache.put(fields, torch.randn(1, 4, 8))
    assert cache.path_for(fields).exists()
    assert cache.path_for(fields).name == f"{cache.key_hash(fields)}.pt"


def test_allowlist_skips_disallowed(tmp_path):
    cache = _cache(tmp_path, allowed_pairs={("ds", 0)})
    assert cache.put(_fields(ep=0), torch.randn(1, 4, 8)) is True
    assert cache.put(_fields(ep=1), torch.randn(1, 4, 8)) is False  # not allowed
    assert cache.get(_fields(ep=1)) is None


def test_codec_version_separates_raw_and_codec_latent(tmp_path):
    """M5 cache switch: a codec-latent cache (codec_version set) must never serve
    a raw-feature entry for the same (dataset, episode, frames), and vice-versa."""
    raw = _cache(tmp_path)  # codec_version=None -> raw-feature cache
    raw.put(_fields(), torch.randn(2, 576, 6656))
    # codec-latent cache (384-d) with the same key fields -> miss (different key)
    codec = _cache(tmp_path, codec_version="codecv1")
    assert codec.get(_fields()) is None
    codec.put(_fields(), torch.randn(2, 144, 384))
    assert codec.get(_fields()) is not None
    # raw cache still sees only its own entry; a different codec version misses
    codec2 = _cache(tmp_path, codec_version="codecv2")
    assert codec2.get(_fields()) is None
    assert raw.get(_fields()) is not None


def test_lru_eviction_under_cap(tmp_path):
    x = torch.randn(64, 64)  # ~8KB fp16
    one = None
    # measure one payload size by putting then reading total
    probe = _cache(tmp_path / "probe")
    probe.put(_fields(0), x)
    one = probe.total_bytes
    cap = int(one * 2.5)  # room for ~2 entries

    cache = _cache(tmp_path / "main", max_bytes=cap)
    for ep in range(5):
        cache.put(_fields(ep), x)
    assert cache.total_bytes <= cap
    assert len(cache) <= 3
    # most recent survives
    assert cache.get(_fields(4)) is not None
