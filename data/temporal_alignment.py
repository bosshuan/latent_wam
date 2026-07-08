"""Temporal alignment between V-JEPA vision tubelets and action control rate.

Pure, side-effect-free functions (no torch state, CPU-trivial) so the unit
tests pin the arithmetic the rest of the data layer depends on.

Locked parameters (CLAUDE.md §3):
  * 1 chunk = 1 V-JEPA time token = 1 tubelet = 2 frames.
  * ``vision_fps = 5`` => ``seconds_per_chunk = tubelet / vision_fps = 0.4``.
  * ``history_chunks = 4``, ``future_chunks = 4`` (1.6 s each).
  * ``target_control_fps = 30``; sources >30 Hz are downsampled.
  * ``actions_per_chunk = round(min(dataset_fps, 30) * seconds_per_chunk)``.

Hard rule: **fps comes from ``meta/info.json`` per dataset, never hardcoded.**
These functions take ``dataset_fps`` as an explicit argument for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_TARGET_CONTROL_FPS = 30
DEFAULT_SECONDS_PER_CHUNK = 0.4
DEFAULT_VISION_FPS = 5
DEFAULT_TUBELET = 2
DEFAULT_HISTORY_CHUNKS = 4
DEFAULT_FUTURE_CHUNKS = 4


def actions_per_chunk(
    dataset_fps: float,
    target_control_fps: int = DEFAULT_TARGET_CONTROL_FPS,
    seconds_per_chunk: float = DEFAULT_SECONDS_PER_CHUNK,
) -> int:
    """Number of action steps that fall inside one 0.4 s vision chunk.

    >30 Hz sources are clamped to ``target_control_fps`` first (downsample).
    Examples: fps=30 -> 12, fps=50 -> 12 (clamped), fps=10 -> 4.
    """
    if dataset_fps <= 0:
        raise ValueError(f"dataset_fps must be > 0; got {dataset_fps}")
    effective = min(float(dataset_fps), float(target_control_fps))
    return int(round(effective * seconds_per_chunk))


@dataclass
class DeltaTimestamps:
    """Chunk-grouped relative timestamps (seconds) for LeRobot delta fetch.

    ``observation_chunks`` and ``action_chunks`` each have length
    ``history_chunks + future_chunks``; t=0 is the boundary between the last
    history chunk and the first future chunk. History offsets are negative.

    Each observation chunk holds ``tubelet`` frame offsets; each action chunk
    holds ``actions_per_chunk`` step offsets (future chunks only carry real
    actions — history action offsets are kept for context/teacher-forcing but
    the loss masks them as needed downstream).
    """

    observation_chunks: list[list[float]]
    action_chunks: list[list[float]]
    history_chunks: int
    future_chunks: int
    actions_per_chunk: int
    seconds_per_chunk: float

    @property
    def num_chunks(self) -> int:
        return self.history_chunks + self.future_chunks

    def flat_observation_offsets(self) -> list[float]:
        return [t for chunk in self.observation_chunks for t in chunk]

    def flat_action_offsets(self) -> list[float]:
        return [t for chunk in self.action_chunks for t in chunk]


def build_delta_timestamps(
    dataset_fps: float,
    history_chunks: int = DEFAULT_HISTORY_CHUNKS,
    future_chunks: int = DEFAULT_FUTURE_CHUNKS,
    tubelet: int = DEFAULT_TUBELET,
    vision_fps: int = DEFAULT_VISION_FPS,
    target_control_fps: int = DEFAULT_TARGET_CONTROL_FPS,
    seconds_per_chunk: float = DEFAULT_SECONDS_PER_CHUNK,
) -> DeltaTimestamps:
    """Build chunk-grouped relative timestamps for one training window.

    Returns a ``DeltaTimestamps`` whose ``num_chunks == history + future``.
    """
    n_per_chunk = actions_per_chunk(
        dataset_fps, target_control_fps, seconds_per_chunk
    )
    frame_dt = 1.0 / float(vision_fps)
    action_dt = 1.0 / min(float(dataset_fps), float(target_control_fps))

    obs_chunks: list[list[float]] = []
    act_chunks: list[list[float]] = []
    num_chunks = history_chunks + future_chunks
    for ci in range(num_chunks):
        # chunk index relative to t=0 boundary: history chunks are negative
        chunk_start = (ci - history_chunks) * seconds_per_chunk
        obs_chunks.append(
            [round(chunk_start + f * frame_dt, 6) for f in range(tubelet)]
        )
        act_chunks.append(
            [round(chunk_start + s * action_dt, 6) for s in range(n_per_chunk)]
        )

    return DeltaTimestamps(
        observation_chunks=obs_chunks,
        action_chunks=act_chunks,
        history_chunks=history_chunks,
        future_chunks=future_chunks,
        actions_per_chunk=n_per_chunk,
        seconds_per_chunk=seconds_per_chunk,
    )


def resample_actions(
    actions: torch.Tensor,
    src_fps: float,
    target_fps: int = DEFAULT_TARGET_CONTROL_FPS,
) -> torch.Tensor:
    """Downsample an action stream [T, A] to <= ``target_fps`` (no upsampling).

    Sources at or below target are returned untouched. Above target we stride by
    ``round(src_fps / target_fps)`` — keep it simple/deterministic; smoothing is
    a downstream concern.
    """
    if actions.ndim != 2:
        raise ValueError(f"expected actions [T, A]; got {tuple(actions.shape)}")
    if src_fps <= target_fps:
        return actions
    stride = max(1, int(round(src_fps / target_fps)))
    return actions[::stride]


def pad_actions(
    actions: torch.Tensor, max_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad/truncate an action chunk [T, A] to ``max_steps`` with a step mask.

    Returns ``(padded [max_steps, A], mask [max_steps])`` where mask=1 marks a
    real step. This is the *temporal* (step) pad; the *feature* pad to the
    schema's max action dim is handled by the embodiment adapter, not here.
    """
    if actions.ndim != 2:
        raise ValueError(f"expected actions [T, A]; got {tuple(actions.shape)}")
    t, a = actions.shape
    mask = torch.zeros(max_steps, dtype=torch.bool)
    out = torch.zeros(max_steps, a, dtype=actions.dtype)
    keep = min(t, max_steps)
    out[:keep] = actions[:keep]
    mask[:keep] = True
    return out, mask
