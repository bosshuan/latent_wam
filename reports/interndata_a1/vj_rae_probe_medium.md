# VJ-RAE Action-Discriminability Probe

Status: **PASS**

## Summary

| metric | value |
| --- | ---: |
| `retention_ok` | True |
| `predictive_ok` | True |
| `strong_pass` | True |
| `min_latent_r2` | 0.000000 |
| `max_relative_mse_increase` | 0.150000 |
| `action_std_mean` | 0.229342 |
| `action_std_min` | 0.006002 |

## Train Metrics

| metric | value |
| --- | ---: |
| `latent_mse` | 0.011928 |
| `pooled_vjepa_mse` | 0.001108 |
| `mean_baseline_mse` | 0.999971 |
| `checkpoint_probe_raw_mse` | 0.251538 |
| `latent_r2` | 0.988072 |
| `pooled_vjepa_r2` | 0.998892 |
| `relative_mse_increase` | 9.760628 |
| `eval_batches` | 256 |
| `valid_transition_count` | 1792 |

## Val Metrics

| metric | value |
| --- | ---: |
| `latent_mse` | 5.860706 |
| `pooled_vjepa_mse` | 5.515701 |
| `mean_baseline_mse` | 6.056055 |
| `checkpoint_probe_raw_mse` | 0.220311 |
| `latent_r2` | 0.032257 |
| `pooled_vjepa_r2` | 0.089225 |
| `relative_mse_increase` | 0.062550 |
| `eval_batches` | 128 |
| `valid_transition_count` | 896 |

`retention_ok` checks whether VJ-RAE latent deltas are no worse than pooled V-JEPA feature deltas by the configured relative MSE margin.

`predictive_ok` checks whether the standardized latent probe beats the train-mean action baseline on held-out data.
