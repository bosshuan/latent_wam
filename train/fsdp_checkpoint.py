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
    legacy_optimizer_load = getattr(dcp, "load_sharded_optimizer_state_dict", None)
    if legacy_optimizer_load is None:
        try:
            from torch.distributed.checkpoint.optimizer import (
                load_sharded_optimizer_state_dict,
            )

            legacy_optimizer_load = load_sharded_optimizer_state_dict
        except ImportError:
            legacy_optimizer_load = None
    return {
        "torch_version": torch.__version__,
        "dcp_save": callable(getattr(dcp, "save", None)),
        "dcp_load": callable(getattr(dcp, "load", None)),
        "get_state_dict": callable(get_state_dict),
        "set_state_dict": callable(set_state_dict),
        "stateful": stateful is not None,
        "legacy_optimizer_load": callable(legacy_optimizer_load),
    }


def _legacy_optimizer_loader(dcp):
    loader = getattr(dcp, "load_sharded_optimizer_state_dict", None)
    if loader is None:
        try:
            from torch.distributed.checkpoint.optimizer import (
                load_sharded_optimizer_state_dict,
            )

            loader = load_sharded_optimizer_state_dict
        except ImportError as exc:  # pragma: no cover - server PyTorch dependent
            raise ImportError(
                "legacy FSDP1 checkpoint loading requires "
                "load_sharded_optimizer_state_dict"
            ) from exc
    return loader


def _legacy_state_dict_context(model):
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardedOptimStateDictConfig,
        ShardedStateDictConfig,
        StateDictType,
    )

    return FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True),
    )


def _save_legacy_fsdp1(dcp, model, optimizer, checkpoint_dir, step, process_group):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    with _legacy_state_dict_context(model):
        state = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
            "progress": torch.tensor(int(step), dtype=torch.int64),
        }
        dcp.save(
            state,
            checkpoint_id=str(checkpoint_dir),
            process_group=process_group,
        )


def _load_legacy_fsdp1(dcp, model, optimizer, checkpoint_dir, process_group) -> int:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    with _legacy_state_dict_context(model):
        model_state = model.state_dict()
        state = {
            "model": model_state,
            "progress": torch.tensor(0, dtype=torch.int64),
        }
        dcp.load(
            state,
            checkpoint_id=str(checkpoint_dir),
            process_group=process_group,
        )
        incompatible = model.load_state_dict(state["model"])
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"DCP model mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
            )

        load_optimizer = _legacy_optimizer_loader(dcp)
        optim_state = load_optimizer(
            model_state,
            optimizer_key="optimizer",
            storage_reader=dcp.FileSystemReader(str(checkpoint_dir)),
        )
        flattened = FSDP.optim_state_dict_to_load(
            model,
            optimizer,
            optim_state["optimizer"],
        )
        optimizer.load_state_dict(flattened)
    return int(state["progress"].item())


def save_fsdp_checkpoint(
    model,
    optimizer,
    checkpoint_dir: str | Path,
    step: int,
    *,
    metadata: dict[str, Any] | None = None,
    process_group=None,
    backend: str = "modern",
) -> Path:
    """Collectively save sharded model + optimizer state."""
    dcp, _get_state_dict, _set_state_dict, _stateful = _dcp_api()
    checkpoint_dir = Path(checkpoint_dir)
    if backend == "legacy_fsdp1":
        _save_legacy_fsdp1(
            dcp, model, optimizer, checkpoint_dir, step, process_group
        )
    elif backend == "modern":
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
    else:
        raise ValueError(
            f"unknown DCP backend={backend!r}; expected 'legacy_fsdp1' or 'modern'"
        )
    dist.barrier()
    if dist.get_rank() == 0:
        manifest = {
            "format": "torch.distributed.checkpoint",
            "torch_version": torch.__version__,
            "step": int(step),
            "backend": backend,
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
    backend: str = "modern",
) -> int:
    """Collectively restore sharded model + optimizer state; return saved step."""
    dcp, _get_state_dict, _set_state_dict, _stateful = _dcp_api()
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"DCP checkpoint does not exist: {checkpoint_dir}")
    if backend == "legacy_fsdp1":
        step = _load_legacy_fsdp1(
            dcp, model, optimizer, checkpoint_dir, process_group
        )
    elif backend == "modern":
        progress = TrainingProgress()
        app = _AppState(model, optimizer)
        state = {"app": app, "progress": progress}
        dcp.load(
            state,
            checkpoint_id=str(checkpoint_dir),
            process_group=process_group,
        )
        if app.load_result is not None:
            missing = list(app.load_result.missing_keys)
            unexpected = list(app.load_result.unexpected_keys)
            if missing or unexpected:
                raise RuntimeError(
                    f"DCP state mismatch: missing={missing[:8]} "
                    f"unexpected={unexpected[:8]}"
                )
        step = progress.step
    else:
        raise ValueError(
            f"unknown DCP backend={backend!r}; expected 'legacy_fsdp1' or 'modern'"
        )
    dist.barrier()
    return step
