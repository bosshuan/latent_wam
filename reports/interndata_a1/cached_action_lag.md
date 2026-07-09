# Cached Action-Lag Sweep

Current alignment: `lag=0`
Best held-out lag: `-2`

| lag | train pairs | val pairs | train R2 | val R2 |
| ---: | ---: | ---: | ---: | ---: |
| -2 | 1792 | 256 | 0.955303 | 0.942505 |
| -1 | 1792 | 256 | 0.953488 | 0.933435 |
| 0 | 1792 | 256 | 0.949943 | 0.930689 |

`lag=0` pairs each future latent transition with the action chunk currently used by the unified trainer. Negative lag uses an earlier action chunk; positive lag uses a later action chunk.

Use the relative lag ranking as an alignment diagnostic. The manifest contains overlapping windows, so these R2 values are not a standalone generalization benchmark.
