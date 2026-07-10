# VJ-RAE Action-Discriminability Probe

Status: **PASS**

## Summary

| metric | value |
| --- | ---: |
| `retention_ok` | True |
| `predictive_ok` | True |
| `strong_pass` | True |
| `min_latent_r2` | 0.050000 |
| `max_relative_mse_increase` | 0.150000 |
| `action_std_mean` | 0.433719 |
| `action_std_min` | 0.023908 |

## Train Metrics

| metric | value |
| --- | ---: |
| `latent_mse` | 0.305895 |
| `pooled_vjepa_mse` | 0.052119 |
| `mean_baseline_mse` | 0.999997 |
| `checkpoint_probe_raw_mse` | 0.092658 |
| `latent_r2` | 0.694104 |
| `pooled_vjepa_r2` | 0.947881 |
| `relative_mse_increase` | 4.869171 |
| `eval_batches` | 768 |
| `valid_transition_count` | 5376 |

## Val Metrics

| metric | value |
| --- | ---: |
| `latent_mse` | 0.403015 |
| `pooled_vjepa_mse` | 0.472459 |
| `mean_baseline_mse` | 0.922909 |
| `checkpoint_probe_raw_mse` | 0.085919 |
| `latent_r2` | 0.563321 |
| `pooled_vjepa_r2` | 0.488076 |
| `relative_mse_increase` | -0.146983 |
| `eval_batches` | 256 |
| `valid_transition_count` | 1792 |

`retention_ok` checks whether VJ-RAE latent deltas are no worse than pooled V-JEPA feature deltas by the configured relative MSE margin.

`predictive_ok` checks whether the standardized latent probe beats the train-mean action baseline on held-out data.
