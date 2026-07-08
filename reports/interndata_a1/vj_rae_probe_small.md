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
| `action_std_mean` | 0.215836 |
| `action_std_min` | 0.006002 |

## Train Metrics

| metric | value |
| --- | ---: |
| `latent_mse` | 0.016376 |
| `pooled_vjepa_mse` | 0.000165 |
| `mean_baseline_mse` | 0.999954 |
| `checkpoint_probe_raw_mse` | 0.184427 |
| `latent_r2` | 0.983623 |
| `pooled_vjepa_r2` | 0.999835 |
| `relative_mse_increase` | 98.035168 |
| `eval_batches` | 128 |
| `valid_transition_count` | 896 |

## Val Metrics

| metric | value |
| --- | ---: |
| `latent_mse` | 17.403376 |
| `pooled_vjepa_mse` | 17.338757 |
| `mean_baseline_mse` | 17.635077 |
| `checkpoint_probe_raw_mse` | 0.257896 |
| `latent_r2` | 0.013139 |
| `pooled_vjepa_r2` | 0.016803 |
| `relative_mse_increase` | 0.003727 |
| `eval_batches` | 64 |
| `valid_transition_count` | 448 |

`retention_ok` checks whether VJ-RAE latent deltas are no worse than pooled V-JEPA feature deltas by the configured relative MSE margin.

`predictive_ok` checks whether the standardized latent probe beats the train-mean action baseline on held-out data.
