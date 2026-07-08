"""Distributed helpers — centralized so both train scripts share one path.

Decisions (user-confirmed): `torchrun` launches; `train_codec.py` uses DDP,
`train_unified_flow.py` uses FSDP (M5). No accelerate/deepspeed. Everything must
degrade to a single process (`world_size=1`) so the server can smoke-test before
going multi-GPU, and so CPU unit tests can exercise the code with the gloo
backend without launching torchrun.

This module does only process-group setup / seeding / rank helpers / checkpoint
IO. The FSDP-vs-DDP wrapping choice stays in each train script.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

try:  # numpy is optional — only used to seed its RNG if present
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


@dataclass
class DistContext:
    rank: int
    world_size: int
    local_rank: int
    distributed: bool
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(backend: str = "nccl") -> DistContext:
    """Init the process group from torchrun env vars, or run single-process.

    Reads ``RANK``/``WORLD_SIZE``/``LOCAL_RANK``. If they are absent (not under
    torchrun) we return a single-process context without creating a group, so
    the same code path runs on a laptop/CPU. ``backend`` should be "gloo" for CPU.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return DistContext(rank, world_size, local_rank, True, device)

    # single-process degrade
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return DistContext(rank=0, world_size=1, local_rank=0, distributed=False, device=device)


def set_seed(seed: int, ctx: Optional[DistContext] = None) -> None:
    """Seed RNGs; offset by rank so workers draw different noise but stay
    reproducible."""
    offset = ctx.rank if ctx is not None else 0
    s = seed + offset
    random.seed(s)
    if _np is not None:
        _np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def barrier(ctx: DistContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.barrier()


def cleanup(ctx: DistContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module behind a DDP/FSDP wrapper (or itself)."""
    return getattr(model, "module", model)


def save_checkpoint(
    state: dict, path: str | Path, ctx: DistContext
) -> Optional[Path]:
    """Save on rank 0 only. ``state`` is a plain dict of (already-unwrapped)
    state_dicts/scalars. Returns the path on the main process, else None.

    NOTE: this is the simple *full* (rank-0-gathered) checkpoint path used for
    DDP/codec. FSDP full-vs-sharded state-dict handling is added in M5's train
    script where the FSDP wrapper lives.
    """
    if not ctx.is_main:
        barrier(ctx)
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    barrier(ctx)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)
