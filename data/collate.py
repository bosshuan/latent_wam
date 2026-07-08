"""Per-sample type + collate into a ``TrajectoryBatch``.

Collate is where the **no-fabricated-action** invariant (CLAUDE.md §2.3) is
enforced concretely:
  * A batch in which *no* sample has a real action produces ``actions=None`` /
    ``proprio=None`` — there is simply no action tensor. This is the common
    case: the sampler prefers homogeneous (all-video / all-robot) batches.
  * A *mixed* batch (some robot, some video) zero-fills the video rows purely so
    the robot rows can live in one dense tensor. Those zero rows are **inert
    storage**, not a representation the model consumes: ``action_valid`` /
    ``action_pad_mask`` are 0 for them.

BINDING M1 -> M3 CONTRACT (see schemas.py): downstream the DiT must
**structurally omit the ``Ak`` action tokens** for ``action_valid=0`` rows
(gather only valid rows), NOT build-then-mask. A zero row must never enter the
action flow loss nor condition latent prediction. The masks here are the
temporal/feature pad for the genuine robot rows — not a license to feed zeros
through the action branch.

Samples are grouped by ``action_schema_id`` upstream (sampler), so within one
batch the real action/state dims are homogeneous; we only pad the *temporal*
(step) axis here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import torch

from data.schemas import NULL_TEXT, TrajectoryBatch


@dataclass
class TrajectorySample:
    """A single pre-collate item (one clip)."""

    pixels: torch.Tensor  # [T, 3, H, W]
    token_grid: tuple[int, int, int]
    action_valid: bool
    embodiment_id: int
    action_schema_id: int
    view_id: int = 0
    fps: float = 0.0
    text: str = NULL_TEXT
    # cache provenance; optional for synthetic/video tests, required for real
    # latent-cache generation.
    dataset_id: str = ""
    episode_index: int = -1
    frame_start: int = -1
    frame_end: int = -1
    sample_index: int = -1
    # robot-only; MUST be None for actionless video
    actions: Optional[torch.Tensor] = None  # [T_chunk, A]
    action_step_mask: Optional[torch.Tensor] = None  # [T_chunk] bool
    proprio: Optional[torch.Tensor] = None  # [S]

    def __post_init__(self) -> None:
        if not self.action_valid and self.actions is not None:
            raise ValueError(
                "actionless sample (action_valid=False) must not carry an "
                "`actions` tensor (CLAUDE.md §2.3)."
            )
        if self.action_valid and self.actions is None:
            raise ValueError("action_valid=True but no `actions` provided.")


def _max_steps(samples: list[TrajectorySample]) -> int:
    steps = [s.actions.shape[0] for s in samples if s.actions is not None]
    return max(steps) if steps else 0


def collate_trajectory(samples: list[TrajectorySample]) -> TrajectoryBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample list")

    b = len(samples)
    pixels = torch.stack([s.pixels for s in samples], dim=0)  # [B,T,3,H,W]
    token_grid = samples[0].token_grid
    for s in samples:
        if s.token_grid != token_grid:
            raise ValueError("mixed token_grid in one batch is not supported")

    action_valid = torch.tensor(
        [bool(s.action_valid) for s in samples], dtype=torch.bool
    )
    embodiment_id = torch.tensor([int(s.embodiment_id) for s in samples], dtype=torch.long)
    action_schema_id = torch.tensor(
        [int(s.action_schema_id) for s in samples], dtype=torch.long
    )
    view_id = torch.tensor([int(s.view_id) for s in samples], dtype=torch.long)
    fps_meta = torch.tensor([float(s.fps) for s in samples], dtype=torch.float32)
    text = [s.text for s in samples]
    dataset_id = [str(s.dataset_id) for s in samples]
    episode_index = torch.tensor([int(s.episode_index) for s in samples], dtype=torch.long)
    frame_start = torch.tensor([int(s.frame_start) for s in samples], dtype=torch.long)
    frame_end = torch.tensor([int(s.frame_end) for s in samples], dtype=torch.long)
    sample_index = torch.tensor([int(s.sample_index) for s in samples], dtype=torch.long)

    if not action_valid.any():
        # video-only batch: no action/proprio tensors at all.
        return TrajectoryBatch(
            pixels=pixels,
            token_grid=token_grid,
            actions=None,
            action_pad_mask=None,
            proprio=None,
            state_pad_mask=None,
            action_valid=action_valid,
            embodiment_id=embodiment_id,
            action_schema_id=action_schema_id,
            view_id=view_id,
            fps_meta=fps_meta,
            text=text,
            dataset_id=dataset_id,
            episode_index=episode_index,
            frame_start=frame_start,
            frame_end=frame_end,
            sample_index=sample_index,
        )

    # mixed / robot batch: build padded tensors; video rows are inert (mask=0).
    max_steps = _max_steps(samples)
    action_dim = next(s.actions.shape[1] for s in samples if s.actions is not None)

    # Proprio consistency guard: never silently drop the whole batch's proprio
    # because one robot row lacks it. Within a schema, proprio presence should be
    # uniform across the action_valid rows.
    robot_proprio_present = [
        s.proprio is not None for s in samples if s.action_valid
    ]
    if any(robot_proprio_present) and not all(robot_proprio_present):
        raise ValueError(
            "inconsistent proprio within a batch: some action_valid samples "
            "have proprio and some do not — this would silently zero-fill the "
            "missing rows. Group by schema / fix the dataset."
        )
    if robot_proprio_present and not any(robot_proprio_present):
        warnings.warn(
            "all action_valid samples in this batch have proprio=None; "
            "proprio will be None for the batch (no state conditioning).",
            stacklevel=2,
        )
    state_dim = next(
        (s.proprio.shape[0] for s in samples if s.proprio is not None), 0
    )

    actions = torch.zeros(b, max_steps, action_dim)
    action_pad_mask = torch.zeros(b, max_steps, dtype=torch.bool)
    proprio = torch.zeros(b, state_dim) if state_dim > 0 else None
    state_pad_mask = (
        torch.zeros(b, state_dim, dtype=torch.bool) if state_dim > 0 else None
    )

    for i, s in enumerate(samples):
        if s.actions is None:
            # video row stays zero with mask=0: inert storage only. M3 omits the
            # Ak tokens for this row entirely (no build-then-mask).
            continue
        if s.actions.shape[1] != action_dim:
            raise ValueError(
                "inconsistent action dim within a batch — samples must be "
                "grouped by action_schema_id before collate."
            )
        t = s.actions.shape[0]
        actions[i, :t] = s.actions
        if s.action_step_mask is not None:
            action_pad_mask[i, :t] = s.action_step_mask[:t]
        else:
            action_pad_mask[i, :t] = True
        if proprio is not None and s.proprio is not None:
            proprio[i] = s.proprio
            state_pad_mask[i] = True

    return TrajectoryBatch(
        pixels=pixels,
        token_grid=token_grid,
        actions=actions,
        action_pad_mask=action_pad_mask,
        proprio=proprio,
        state_pad_mask=state_pad_mask,
        action_valid=action_valid,
        embodiment_id=embodiment_id,
        action_schema_id=action_schema_id,
        view_id=view_id,
        fps_meta=fps_meta,
        text=text,
        dataset_id=dataset_id,
        episode_index=episode_index,
        frame_start=frame_start,
        frame_end=frame_end,
        sample_index=sample_index,
    )
