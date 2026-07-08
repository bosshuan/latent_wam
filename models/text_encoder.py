"""Frozen umT5-XXL text encoder + task cache + CFG (doc §2.4; CLAUDE.md §3).

Wan2.2 ships an **umT5-XXL** text encoder; we use it as a FROZEN module (eval +
``requires_grad_(False)`` + no_grad encode -> zero gradient, covered by
``test_frozen_modules``). Its output is the cross-attention context that the Wan
text cross-attention (weights loaded) consumes, after the backbone's
``text_embedding`` projection.

Responsibilities:
  * pre-encode + **cache** LeRobot task strings (``meta/tasks.jsonl``) — encode
    once, look up at train time;
  * a fixed **null-text** embedding for caption-less video (unified with
    ``action_valid=0``) and for **CFG** (``text_cfg_dropout=0.1``): with prob ``p``
    a sample's text is replaced by null during training.

The real umT5 is loaded on the server (``from_pretrained``); the unit tests use
:meth:`mock`, a deterministic frozen embedding stand-in (real params so the frozen
test has something to check, deterministic so caching is verifiable on CPU).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

NULL_TEXT = ""  # unified with data.schemas.NULL_TEXT (caption-less video)


class FrozenUMT5TextEncoder(nn.Module):
    """Frozen text encoder with a string->embedding cache and CFG null-text.

    ``text_dim`` is the umT5 hidden (cross-attn context dim, e.g. 4096); ``seq_len``
    is the (fixed, padded) token length we expose to cross-attention.
    """

    def __init__(self, text_dim: int, seq_len: int = 8, num_buckets: int = 4096) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.seq_len = seq_len
        self.num_buckets = num_buckets
        # Stand-in for umT5; on the server this is replaced by the real frozen
        # encoder. Kept as real params so the frozen-module test is meaningful.
        self.embed = nn.Embedding(num_buckets, text_dim)
        self._cache: dict[str, torch.Tensor] = {}
        self.freeze()

    # -- freezing (CLAUDE.md §2.2) --------------------------------------
    def freeze(self) -> None:
        self.eval()
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenUMT5TextEncoder":
        # stays in eval no matter what the parent .train() does
        return super().train(False)

    # -- tokenization stand-in ------------------------------------------
    def _bucket_ids(self, text: str) -> torch.Tensor:
        if text == NULL_TEXT:
            return torch.zeros(self.seq_len, dtype=torch.long)
        # deterministic hash of (word, position) -> bucket; pad/truncate to seq_len
        words = text.split()
        ids = []
        for i in range(self.seq_len):
            if i < len(words):
                ids.append((hash((words[i], i)) % (self.num_buckets - 1)) + 1)
            else:
                ids.append(0)  # pad with the null bucket
        return torch.tensor(ids, dtype=torch.long)

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device | str = "cpu") -> torch.Tensor:
        """``list[str]`` length B -> ``[B, seq_len, text_dim]`` (detached, cached)."""
        out = []
        for t in texts:
            if t not in self._cache:
                ids = self._bucket_ids(t).to(self.embed.weight.device)
                self._cache[t] = self.embed(ids).detach()  # [seq_len, text_dim]
            out.append(self._cache[t])
        return torch.stack(out, dim=0).to(device)

    @torch.no_grad()
    def null_embedding(self, batch_size: int, device) -> torch.Tensor:
        return self.encode([NULL_TEXT] * batch_size, device)

    @torch.no_grad()
    def precache(self, tasks: list[str]) -> None:
        """Pre-encode every task string (server: from ``meta/tasks.jsonl``)."""
        self.encode(tasks, device=self.embed.weight.device)

    @torch.no_grad()
    def encode_with_cfg(
        self,
        texts: list[str],
        training: bool,
        p: float = 0.1,
        generator: Optional[torch.Generator] = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Encode with CFG text dropout: in training, each sample's text is
        replaced by null with prob ``p`` (caption-less video is already null)."""
        emb = self.encode(texts, device)
        if not training or p <= 0.0:
            return emb
        b = emb.shape[0]
        drop = torch.rand(b, generator=generator) < p
        if drop.any():
            null = self.null_embedding(b, device)
            emb = torch.where(drop.view(b, 1, 1).to(emb.device), null, emb)
        return emb
