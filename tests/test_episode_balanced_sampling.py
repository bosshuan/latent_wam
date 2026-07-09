from scripts.cache_robot_latents import (
    _episode_balanced_subset,
    _evenly_spaced_indices,
)


class _EpisodeDataset:
    def __len__(self):
        return 210

    def episode_ranges(self):
        return [
            {"episode": 0, "start": 0, "stop": 100, "length": 100},
            {"episode": 1, "start": 100, "stop": 200, "length": 100},
            {"episode": 2, "start": 200, "stop": 210, "length": 10},
        ]


def test_evenly_spaced_indices_are_centered():
    assert _evenly_spaced_indices(10, 90, 2) == [30, 70]


def test_episode_balanced_subset_round_robins_before_global_cap():
    subset, plan = _episode_balanced_subset(
        _EpisodeDataset(),
        {
            "episode_margin_frames": 10,
            "samples_per_episode": 2,
            "episode_stride": 1,
            "max_episodes_per_dataset": 0,
            "max_samples_per_dataset": 3,
        },
    )
    assert subset.indices == [30, 130, 70]
    assert plan["available_episodes"] == 3
    assert plan["selected_episodes"] == 2
    assert plan["skipped_short_episodes"] == 1
    assert plan["samples"] == 3


def test_episode_balanced_subset_filters_episode_remainders():
    subset, plan = _episode_balanced_subset(
        _EpisodeDataset(),
        {
            "episode_margin_frames": 10,
            "samples_per_episode": 2,
            "episode_modulus": 2,
            "episode_remainders": [1],
            "max_episodes_per_dataset": 0,
            "max_samples_per_dataset": 0,
        },
    )
    assert subset.indices == [130, 170]
    assert plan["available_episodes"] == 3
    assert plan["filtered_episodes"] == 1
    assert plan["episode_ids"] == [1]
