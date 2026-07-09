"""Fixed action/state normalization for Stage-A robot training.

Action statistics are keyed by action_schema_id; proprio statistics are keyed
by embodiment_id. Statistics are fit offline on the robot train split and then
treated as immutable data metadata during unified-flow training.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


class RunningMoments:
    """Float64 per-channel moments for an arbitrary stream of ``[N, D]`` rows."""

    def __init__(self) -> None:
        self.count = 0
        self.sum: torch.Tensor | None = None
        self.sum_sq: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1, values.shape[-1]).to("cpu", torch.float64)
        if values.numel() == 0:
            return
        if self.sum is None:
            self.sum = torch.zeros(values.shape[-1], dtype=torch.float64)
            self.sum_sq = torch.zeros_like(self.sum)
        if values.shape[-1] != self.sum.shape[0]:
            raise ValueError(
                f"moment dimension changed from {self.sum.shape[0]} to {values.shape[-1]}"
            )
        self.count += int(values.shape[0])
        self.sum += values.sum(dim=0)
        self.sum_sq += values.square().sum(dim=0)

    def finalize(self, std_floor: float) -> dict:
        if self.count <= 0 or self.sum is None or self.sum_sq is None:
            raise RuntimeError("cannot finalize empty control statistics")
        mean = self.sum / self.count
        var = (self.sum_sq / self.count - mean.square()).clamp_min(0.0)
        std = var.sqrt().clamp_min(float(std_floor))
        return {
            "count": self.count,
            "dim": int(mean.numel()),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }


class FixedControlNormalizer:
    """Apply precomputed schema/embodiment-specific standardization."""

    def __init__(self, payload: dict) -> None:
        if int(payload.get("format_version", -1)) != 1:
            raise ValueError(
                f"unsupported control stats format_version={payload.get('format_version')}"
            )
        self.payload = payload
        self.action_stats = payload.get("actions", {})
        self.state_stats = payload.get("states", {})

    @classmethod
    def from_json(cls, path: str | Path) -> "FixedControlNormalizer":
        with open(path) as handle:
            return cls(json.load(handle))

    @staticmethod
    def _normalize(values: torch.Tensor, record: dict, label: str) -> torch.Tensor:
        expected_dim = int(record["dim"])
        if values.shape[-1] != expected_dim:
            raise ValueError(
                f"{label} dim={values.shape[-1]} does not match stats dim={expected_dim}"
            )
        mean = torch.as_tensor(record["mean"], device=values.device, dtype=values.dtype)
        std = torch.as_tensor(record["std"], device=values.device, dtype=values.dtype)
        if not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
            raise ValueError(f"{label} stats contain invalid std")
        return (values - mean) / std

    def normalize_action(self, values: torch.Tensor, action_schema_id: int) -> torch.Tensor:
        key = str(int(action_schema_id))
        if key not in self.action_stats:
            raise KeyError(f"no action stats for action_schema_id={key}")
        return self._normalize(values, self.action_stats[key], f"action schema {key}")

    def normalize_state(self, values: torch.Tensor, embodiment_id: int) -> torch.Tensor:
        key = str(int(embodiment_id))
        if key not in self.state_stats:
            raise KeyError(f"no state stats for embodiment_id={key}")
        return self._normalize(values, self.state_stats[key], f"state embodiment {key}")
