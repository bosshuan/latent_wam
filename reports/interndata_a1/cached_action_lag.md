# Cached Action-Lag Sweep

Current alignment: `lag=0`
Best held-out lag: `0`
Split: `same_dataset_episode_holdout`
Dataset: `interndata_a1__sim_dual_arm__sim__articulation_tasks__split_aloha__rotate_the_left_hearth_right_arm_counterclockwise`
Train episodes: `[0]`
Holdout episode: `1`

| lag | train pairs | val pairs | train R2 | val R2 | action mean shift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -2 | 628 | 396 | 0.971322 | -0.033029 | 1.081108 |
| -1 | 628 | 396 | 0.972169 | -0.020394 | 1.092392 |
| 0 | 628 | 396 | 0.969945 | -0.014291 | 1.120382 |

`lag=0` pairs each future latent transition with the action chunk currently used by the unified trainer. Negative lag uses an earlier action chunk.

Train and validation windows come from different episodes of the same dataset, so no overlapping video window crosses the split. Use the relative held-out lag ranking as the alignment diagnostic.
