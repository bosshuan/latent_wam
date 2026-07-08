"""Robot-schema batch sampler + Stage-A curriculum.

Two jobs:
  * **Robot-only homogeneous batches** — Stage A now follows DreamZero/VLA-style
    paired robot video+action data. Every yielded training batch is all robot and
    contains one ``action_schema_id`` so the padded action layout is uniform and
    the counterfactual permutation has within-schema rows.
  * **Curriculum** — the main curriculum ramps action / rollout loss weights and
    switches coupled -> decoupled timesteps. It does **not** mix actionless video.

The legacy ``MixedBatchSampler`` is kept for Stage-A+ ablations with actionless
video, but the default ``CurriculumSchedule`` and the new
``RobotSchemaBatchSampler`` are robot-only.

The sampler yields *lists of dataset indices* (use as a ``batch_sampler``); the
dataset + ``collate_trajectory`` turn those into a ``TrajectoryBatch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch


@dataclass
class CurriculumStage:
    until_frac: float          # active while progress < until_frac
    video_ratio: float = 0.0   # legacy only; Stage-A main keeps this at 0
    action_weight: float = 1.0 # multiplies lambda_a
    rollout_weight: float = 0.0  # multiplies lambda_roll
    coupled: bool = True        # timestep schedule: t_z==t_a (early) -> decoupled


@dataclass
class CurriculumSchedule:
    """Robot-only piecewise schedule over training progress in [0, 1].

    Default: warm latent/action alignment -> full joint flow -> rollout/action
    emphasis. The timestep schedule starts **coupled** (``t_z==t_a``) and switches
    to **decoupled** late in training. ``video_ratio`` stays zero by design.
    """

    stages: list[CurriculumStage] = field(
        default_factory=lambda: [
            CurriculumStage(until_frac=0.15, action_weight=0.3, coupled=True),
            CurriculumStage(until_frac=0.80, action_weight=1.0, coupled=True),
            CurriculumStage(
                until_frac=1.01, action_weight=1.5,
                rollout_weight=1.0, coupled=False,
            ),
        ]
    )

    def at(self, progress: float) -> CurriculumStage:
        for st in self.stages:
            if progress < st.until_frac:
                return st
        return self.stages[-1]


class RobotSchemaBatchSampler:
    """Yields robot-only homogeneous batches grouped by ``action_schema_id``.

    This is the Stage-A main sampler. It never yields actionless-video indices and
    never mixes schemas inside one batch.
    """

    def __init__(
        self,
        robot_indices_by_schema: dict[int, list[int]],
        batch_size: int,
        require_min_schema: int = 2,
        num_batches: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.robot_by_schema = {k: list(v) for k, v in robot_indices_by_schema.items()}
        self.batch_size = int(batch_size)
        self.require_min_schema = int(require_min_schema)
        self.seed = int(seed)
        self._epoch = 0
        self._robot_schemas = [
            s for s, idx in self.robot_by_schema.items()
            if len(idx) >= max(self.require_min_schema, 1)
        ]
        total = sum(len(v) for v in self.robot_by_schema.values())
        self.num_batches = num_batches if num_batches is not None else total // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _gen(self) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        return g

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        if not self._robot_schemas:
            return
        g = self._gen()
        for _ in range(self.num_batches):
            si = int(torch.randint(len(self._robot_schemas), (1,), generator=g))
            schema = self._robot_schemas[si]
            yield _sample_pool(self.robot_by_schema[schema], self.batch_size, g)


class MixedBatchSampler:
    """Legacy Stage-A+ sampler with a per-step actionless-video/robot coin flip.

    Parameters
    ----------
    video_indices : list[int]
        dataset indices of actionless video clips.
    robot_indices_by_schema : dict[int, list[int]]
        robot clip indices grouped by ``action_schema_id`` (uniform pad layout).
    batch_size : int
    video_ratio : float
        probability a given batch is drawn from the video pool (else robot). May be
        overridden per-epoch via ``set_video_ratio`` from the curriculum.
    require_min_schema : int
        a robot batch is only emitted from a schema with at least this many items,
        so the counterfactual permutation (§2.7) always has ≥2 rows.
    """

    def __init__(
        self,
        video_indices: list[int],
        robot_indices_by_schema: dict[int, list[int]],
        batch_size: int,
        video_ratio: float = 0.5,
        require_min_schema: int = 2,
        num_batches: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.video_indices = list(video_indices)
        self.robot_by_schema = {k: list(v) for k, v in robot_indices_by_schema.items()}
        self.batch_size = batch_size
        self.video_ratio = video_ratio
        self.require_min_schema = require_min_schema
        self.seed = seed
        self._epoch = 0
        # eligible robot schemas (enough rows for a full + counterfactual batch)
        self._robot_schemas = [
            s for s, idx in self.robot_by_schema.items()
            if len(idx) >= max(require_min_schema, 1)
        ]
        total = len(self.video_indices) + sum(len(v) for v in self.robot_by_schema.values())
        self.num_batches = num_batches if num_batches is not None else total // batch_size

    def set_video_ratio(self, ratio: float) -> None:
        self.video_ratio = float(ratio)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _gen(self) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        return g

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        g = self._gen()
        have_video = len(self.video_indices) >= self.batch_size
        have_robot = len(self._robot_schemas) > 0
        if not have_video and not have_robot:
            return
        for _ in range(self.num_batches):
            pick_video = (
                have_video
                and (not have_robot or float(torch.rand(1, generator=g)) < self.video_ratio)
            )
            if pick_video:
                yield self._sample_pool(self.video_indices, g)
            else:
                # choose a schema, then a batch within it (homogeneous schema)
                si = int(torch.randint(len(self._robot_schemas), (1,), generator=g))
                schema = self._robot_schemas[si]
                yield self._sample_pool(self.robot_by_schema[schema], g)

    def _sample_pool(self, pool: list[int], g: torch.Generator) -> list[int]:
        return _sample_pool(pool, self.batch_size, g)


def _sample_pool(pool: list[int], batch_size: int, g: torch.Generator) -> list[int]:
    n = len(pool)
    if n >= batch_size:
        perm = torch.randperm(n, generator=g)[:batch_size]
    else:
        # small pool: sample with replacement to fill the batch
        perm = torch.randint(n, (batch_size,), generator=g)
    return [pool[int(i)] for i in perm]
