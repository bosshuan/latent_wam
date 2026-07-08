"""KV-cache interface placeholder (M5 stub; filled in M8 closed-loop).

Per the plan (§8), the working KV cache + receding-horizon closed-loop is **M8**
(deployment). M5 only fixes the *interface* so the training-time forward carries
the right signature (``forward(..., kv_cache=None)`` -> ``WAMOutput.kv_cache``) and
M8 can fill the mechanism without touching the training path.

Design intent for M8 (recorded here so the shape is fixed now):
  * the chunk-causal mask makes clean ``context`` tokens a pure causal encoder
    (they never attend the noisy future), so their per-layer K/V are **stable** and
    cacheable across receding-horizon steps — exactly the expensive
    frozen-V-JEPA+codec context the plan says to cache;
  * **prefill** runs the context tokens once (context-only mask) storing each
    layer's K/V; **decode** runs the future tokens attending
    ``[cached_context_kv ++ current_future_kv]`` with the future rows of the same
    mask. Because context output is independent of the future, decode is exact
    (a unit test in M8 must assert decode == full-forward on the concatenation).

Training never passes a cache (full teacher-forced forward), so a non-None cache
in Stage A is a programming error and fails loud rather than silently no-op'ing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LayerKV:
    """Cached post-RoPE key/value for one attention layer's context tokens."""

    k: torch.Tensor  # [B, n_ctx, num_heads, head_dim]
    v: torch.Tensor


@dataclass
class KVCache:
    """Per-layer context K/V + bookkeeping for incremental (M8) decoding.

    Stage A leaves this empty; it exists so the forward signature and
    ``WAMOutput.kv_cache`` are stable now.
    """

    layers: list[LayerKV] = field(default_factory=list)
    context_len: int = 0
    start_chunk: int = 0

    def is_empty(self) -> bool:
        return len(self.layers) == 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.layers.append(LayerKV(k=k, v=v))


class KVCacheNotImplemented(NotImplementedError):
    """Raised if Stage-A code is handed a non-empty KV cache (it's an M8 feature)."""
