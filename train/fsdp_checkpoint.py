"""Distributed Checkpoint (DCP) helpers for FSDP model/optimizer state.

Uses PyTorch's canonical distributed state-dict API so checkpoints contain
unwrapped parameter FQNs and may be resharded when the world size changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _dcp_api():
    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            get_state_dict,
            set_state_dict,
        )
        from torch.distributed.checkpoint.stateful import Stateful
    except ImportError as exc:  # pragma: no cover - server PyTorch dependent
        raise ImportError(
            "PyTorch Distributed Checkpoint with get_state_dict/set_state_dict "
            "is required; use the project server's supported PyTorch build"
        ) from exc
    return dcp, get_state_dict, set_state_dict, Stateful


class TrainingProgress:
    """Small Stateful-compatible object for resumable trainer metadata."""

    def __init__(self, step: int = 0) -> None:
        self.step = int(step)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"step": torch.tensor(self.step, dtype=torch.int64)}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.step = int(state_dict["step"].item())


class _AppState:
    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.load_result = None

    def state_dict(self) -> dict[str, Any]:
        _dcp, get_state_dict, _set_state_dict, _stateful = _dcp_api()
        model_state, optim_state = get_state_dict(self.model, self.optimizer)
        return {"model": model_state, "optim": optim_state}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        _dcp, _get_state_dict, set_state_dict, _stateful = _dcp_api()
        self.load_result = set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


def dcp_capabilities() -> dict[str, Any]:
    dcp, get_state_dict, set_state_dict, stateful = _dcp_api()
    return {
        "torch_version": torch.__version__,
        "dcp_save": callable(getattr(dcp, "save", None)),
        "dcp_load": callable(getattr(dcp, "load", None)),
        "get_state_dict": callable(get_state_dict),
        "set_state_dict": callable(set_state_dict),
        "stateful": stateful is not None,
    }


def save_fsdp_checkpoint(
    model,
    optimizer,
    checkpoint_dir: str | Path,
    step: int,
    *,
    metadata: dict[str, Any] | None = None,
    process_group=None,
) -> Path:
    """Collectively save sharded model + optimizer state."""
    dcp, _get_state_dict, _set_state_dict, _stateful = _dcp_api()
    checkpoint_dir = Path(checkpoint_dir)
    progress = TrainingProgress(step)
    state = {
        "app": _AppState(model, optimizer),
        "progress": progress,
    }
    dcp.save(
        state,
        checkpoint_id=str(checkpoint_dir),
        process_group=process_group,
    )
    dist.barrier()
    if dist.get_rank() == 0:
        manifest = {
            "format": "torch.distributed.checkpoint",
            "torch_version": torch.__version__,
            "step": int(step),
            **(metadata or {}),
        }
        (checkpoint_dir / "latent_wam_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    dist.barrier()
    return checkpoint_dir


def load_fsdp_checkpoint(
    model,
    optimizer,
    checkpoint_dir: str | Path,
    *,
    process_group=None,
) -> int:
    """Collectively restore sharded model + optimizer state; return saved step."""
    dcp, _get_state_dict, _set_state_dict, _stateful = _dcp_api()
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"DCP checkpoint does not exist: {checkpoint_dir}")
    progress = TrainingProgress()
    app = _AppState(model, optimizer)
    state = {"app": app, "progress": progress}
    dcp.load(
        state,
        checkpoint_id=str(checkpoint_dir),
        process_group=process_group,
    )
    dist.barrier()
    if app.load_result is not None:
        missing = list(app.load_result.missing_keys)
        unexpected = list(app.load_result.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"DCP state mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
            )
    return progress.step
