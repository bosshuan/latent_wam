# Cached Action-Lag Sweep

Current alignment: `lag=0`
Best held-out lag: `-2`

| lag | train pairs | val pairs | train R2 | val R2 |
| ---: | ---: | ---: | ---: | ---: |
| -2 | 896 | 128 | 0.978321 | 0.966122 |
| -1 | 1344 | 192 | 0.964795 | 0.942381 |
| 0 | 1792 | 256 | 0.949943 | 0.930689 |
| 1 | 1344 | 192 | 0.965544 | 0.947571 |
| 2 | 896 | 128 | 0.976003 | 0.953715 |

`lag=0` pairs each future latent transition with the action chunk currently used by the unified trainer. Negative lag uses an earlier action chunk; positive lag uses a later action chunk.

Use the relative lag ranking as an alignment diagnostic. The manifest contains overlapping windows, so these R2 values are not a standalone generalization benchmark.
