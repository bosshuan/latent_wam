"""M5 mixed-batch sampler + curriculum tests (doc §4.2; CLAUDE.md §2.3/§2.9).

Asserts batches are homogeneous (all-video or all-one-robot-schema), robot schemas
with <2 items are excluded (so L_cf always has >=2 same-schema rows), and the
video ratio is honored.
"""

from __future__ import annotations

from data.mixed_batch_sampler import (
    CurriculumSchedule,
    CurriculumStage,
    MixedBatchSampler,
)


def _sampler(video_ratio, seed=0, num_batches=200):
    video = list(range(0, 100))            # ids 0..99 are video
    robot = {
        7: list(range(100, 110)),          # schema 7: 10 items
        9: list(range(110, 116)),          # schema 9: 6 items
        3: [200],                          # schema 3: singleton -> excluded
    }
    return MixedBatchSampler(
        video, robot, batch_size=4, video_ratio=video_ratio,
        require_min_schema=2, num_batches=num_batches, seed=seed,
    )


def test_batches_are_homogeneous():
    s = _sampler(video_ratio=0.5)
    video_set = set(range(100))
    schema_of = {}
    for sid, ids in {7: range(100, 110), 9: range(110, 116)}.items():
        for i in ids:
            schema_of[i] = sid
    for batch in s:
        assert len(batch) == 4
        is_video = all(i in video_set for i in batch)
        if not is_video:
            schemas = {schema_of[i] for i in batch}
            assert len(schemas) == 1, f"robot batch mixes schemas: {schemas}"


def test_singleton_schema_excluded():
    s = _sampler(video_ratio=0.0)  # robot-only
    seen = set()
    for batch in s:
        seen.update(batch)
    assert 200 not in seen  # singleton schema 3 never sampled (needs >=2 for L_cf)


def test_video_ratio_extremes():
    video_set = set(range(100))
    all_video = list(_sampler(video_ratio=1.0))
    assert all(all(i in video_set for i in b) for b in all_video)
    all_robot = list(_sampler(video_ratio=0.0))
    assert all(all(i not in video_set for i in b) for b in all_robot)


def test_curriculum_video_ratio_decreases():
    cur = CurriculumSchedule()
    assert cur.at(0.0).video_ratio >= cur.at(0.9).video_ratio
    assert isinstance(cur.at(0.5), CurriculumStage)
