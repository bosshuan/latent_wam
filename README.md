# Latent-WAM

Stage A currently targets paired robot video+action data only, following the
DreamZero/VLA training setting. Actionless video support remains in the code for
later Stage-A+ ablations, but it is no longer the default training path.

The former "codec" module is now treated as a V-JEPA Representation Autoencoder
(VJ-RAE): frozen V-JEPA multi-layer dense features are normalized, pooled from
24x24 to 12x12, compressed to a 384-d latent grid, and reconstructed in V-JEPA
feature space rather than RGB pixel space.

## 8xA100 Debug Bringup

Run commands from this code root:

```bash
cd /mnt/sfs_turbo/fyy/latent_wam
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

On CUDA servers, it is usually better to keep the server's matching
`torch`/`torchvision` build if one is already installed, then install the rest of
the stack around it.

First verify VJ-RAE DDP training, normalizer all-reduce, probe loss, and
checkpoint save:

```bash
bash scripts/run_debug_vj_rae.sh
```

Then verify unified latent-action flow training, action counterfactual forwards,
collapse monitors, optimizer, validation, and checkpoint save:

```bash
bash scripts/run_debug_unified_robot.sh
```

Both scripts default to `NPROC_PER_NODE=8`. To run a one-GPU smoke job:

```bash
NPROC_PER_NODE=1 bash scripts/run_debug_unified_robot.sh
```

Checkpoints are written to:

```text
checkpoints/debug/vj_rae_synthetic.pt
checkpoints/debug/unified_robot_synthetic.pt
```

## Configs

Synthetic debug configs:

```text
configs/debug/vj_rae_synthetic.yaml
configs/debug/unified_robot_synthetic.yaml
```

Main Stage-A configs:

```text
configs/codec.yaml
configs/unified_pretrain.yaml
configs/model/latent_wam_dit.yaml
```

`codec.yaml` keeps the legacy filename for compatibility, but the preferred
config key is `vj_rae`.

## Real Data Hook

The debug scripts intentionally use `data.source: synthetic`; they do not claim
scientific training quality. They are meant to flush out CUDA, torchrun, DDP,
loss, monitor, and checkpoint issues on the 8xA100 server.

### InternData-A1 Schema Scan

The first real dataset target is InternData-A1 on the server:

```text
/mnt/sfs_turbo/rl/InternData-A1/sim/.../<LeRobot subset>/meta/info.json
```

Before wiring training dataloaders, recursively scan the LeRobot v2.1 subset
schemas under `sim/`:

```bash
bash scripts/scan_interndata_a1.sh
```

This writes:

```text
reports/interndata_a1/schema_report.md
reports/interndata_a1/schema_report.json
```

Inspect the report before training. In particular, confirm:

- `codebase_version` is `v2.1`
- each subset has at least one image/video key, e.g. `images.rgb.*`
- each subset has an `action*` key
- action/state dimensions and fps are sensible
- dual-arm articulation tasks and Franka tasks use different embodiments/schemas

Then smoke-test the real dual-arm dataloader using the shared head camera:

```bash
bash scripts/smoke_interndata_a1_dual_arm.sh
```

The smoke config uses `reader_backend: direct`, so it does not require the
official `lerobot` Python package. This avoids forcing a torch/CUDA upgrade on
debug servers. The dataset wrapper still supports the older
`lerobot.common.datasets.lerobot_dataset.LeRobotDataset` import path and the
newer `lerobot.datasets.lerobot_dataset.LeRobotDataset` path when you explicitly
select `reader_backend: auto` or `reader_backend: lerobot`.

Expected batch-level shapes:

```text
pixels  [B, T, 3, H, W]
actions [B, T_action, 14]
proprio [B, 14]
```

Then generate a tiny V-JEPA/VJ-RAE latent-cache smoke:

```bash
bash scripts/cache_interndata_a1_dual_arm_latents.sh
```

This default smoke uses the local sibling V-JEPA repo (`../vjepa2`) without
downloading pretrained weights, plus an explicitly random VJ-RAE. Both are
marked with random-smoke cache versions. It validates the real-data plumbing and
writes:

```text
.feature_cache/interndata_a1_dual_arm_random_smoke/*.pt
reports/interndata_a1/latent_cache_smoke_manifest.jsonl
```

Do not use the random-smoke cache for scientific training. Once real V-JEPA
target-encoder weights and a trained VJ-RAE checkpoint are available, use either
`encoder.pretrained: true` or the preferred offline-server path
`encoder.pretrained: false` + `encoder.checkpoint_path`, then set
`vj_rae.checkpoint_path`, set `allow_random_vj_rae: false`, and bump both cache
version fields.

After the V-JEPA 2.1 gigantic checkpoint has been downloaded to the server:

```text
/mnt/sfs_turbo/fyy/checkpoints/vjepa2/vjepa2_1_vitG_384.pt
```

run the local-checkpoint smoke:

```bash
bash scripts/cache_interndata_a1_dual_arm_vjepa_latents.sh
```

This uses `encoder.pretrained: false` plus `encoder.checkpoint_path` so
`torch.hub` builds the architecture from `../vjepa2` without online weight
download, then loads the local `target_encoder` weights. VJ-RAE remains
`random-smoke-v0` in this smoke until a real VJ-RAE checkpoint is trained.

Then run a tiny real-data VJ-RAE training smoke:

```bash
bash scripts/train_interndata_a1_dual_arm_vj_rae_smoke.sh
```

This fits the V-JEPA feature normalizer on a small dual-arm subset, trains
VJ-RAE for a few steps with the action-discriminability probe, and writes:

```text
checkpoints/interndata_a1/vj_rae_dual_arm_real_smoke.pt
```

This checkpoint is still a smoke artifact, but it is the first end-to-end
real-data VJ-RAE checkpoint. To immediately verify cache generation from that
checkpoint:

```bash
bash scripts/cache_interndata_a1_dual_arm_vjepa_vjrae_latents.sh
```

That config points `vj_rae.checkpoint_path` at the smoke checkpoint, sets
`allow_random_vj_rae: false`, and writes a separate
`vj-rae-dual-arm-real-smoke-v0` cache namespace.

Then smoke-test unified latent-action flow training on cached real latents:

```bash
bash scripts/train_interndata_a1_dual_arm_unified_cached_smoke.sh
```

This reads the VJ-RAE cache manifest, reloads actions/proprio from the original
LeRobot-v2.1-format parquet, builds `StepInputs(context_latent, r1, actions,
proprio)`, and runs the existing unified loss with counterfactual action
conditioning. It still uses the tiny DiT backbone, not Wan2.2.

This short smoke only checks the end-to-end training path. If the log ends with
`collapse=True`, inspect the two monitors:

```text
S_a             token-L2(r_hat_true_action, r_hat_permuted_action)
S_a_cos         cosine-distance reference for the same pair
delta_cond      token-L2(r_hat_no_action, target) - token-L2(r_hat_true_action, target)
cf_action_delta action-L2(true_action, permuted_action)
cf_valid_frac   fraction of rows with a meaningful counterfactual action
cf_inconclusive true when cf_valid_frac is too low to raise a collapse alarm
```

The small and medium cached configs keep the training loss on noisy
counterfactual actions (`cf_action_mode: noisy`) but use clean counterfactual
actions for the monitor (`monitor_cf_action_mode: clean`). This makes validation
`S_a` less sensitive to random low action timesteps. Validation batches are also
shuffled so the within-schema permutation is less likely to swap nearly
identical neighboring windows.

The cached small/medium configs use interleaved modulo train/validation splits
instead of taking the tail of the manifest as validation. This keeps validation
non-overlapping while spanning both selected InternData-A1 subsets. The
counterfactual source is also distance-aware:

```text
counterfactual_action_mode: farthest
min_counterfactual_action_delta: 0.05
min_counterfactual_valid_frac_for_alarm: 0.5
```

Rows whose best within-schema counterfactual action is still too similar are
excluded from `L_cf/S_a` through `cf_valid=0` instead of being counted as action
collapse.

If validation `S_a` is low while `cf_action_delta` is also low, the
counterfactual batch is not action-diverse enough and `S_a` is inconclusive. If
`cf_action_delta` is healthy but `S_a` stays below `0.01`, then the model is
really not using the specific action content on held-out windows.

Very early alarms are possible because the cache and VJ-RAE checkpoint above are
tiny smoke artifacts. The tiny DiT uses a small residual-gate warm start
(`model.adaln_gate_init`) so `S_a` should no longer be identically locked at zero
by initialization. To test whether the action path can actually overfit this
mini cache, run the diagnostic:

```bash
bash scripts/train_interndata_a1_dual_arm_unified_cached_action_overfit.sh
```

That diagnostic trains longer on the same two cached samples, lowers action-flow
weight so it does not dominate the tiny loss, increases `L_cf`, enables an
explicit action-to-latent residual conditioner, and uses clean-action
counterfactual forwards (`cf_action_mode: clean`). It is healthy if `S_a` rises
above the `0.01` alarm floor and `delta_cond` starts moving positive after the
early steps. `S_a_cos` can stay much smaller because cosine is less sensitive to
small high-dimensional latent changes.

The diagnostic also prints gradient probes:

```text
grad_action_encoder
grad_action_to_latent
grad_latent_head
```

If `grad_action_to_latent` is non-zero but `S_a` stays low, the path is connected
and the remaining issue is weak action/latent supervision in this tiny cache. If
it is zero, inspect the action-token path and computation graph before scaling.

The 2-sample action-overfit diagnostic is intentionally harsh. A typical
interpretation is:

```text
grad_action_to_latent > 0  -> action path is connected
delta_cond > 0             -> having action helps relative to no action
S_a near/below 0.01        -> specific action content is still weak
cf ~= lambda_cf * delta    -> true action is not yet better than permuted action
```

Once the path is connected, move to a larger but still debug-sized real run
instead of over-tuning the 2-sample cache:

```bash
bash scripts/train_interndata_a1_dual_arm_vj_rae_small.sh
bash scripts/cache_interndata_a1_dual_arm_vjepa_vjrae_small.sh
bash scripts/train_interndata_a1_dual_arm_unified_cached_small.sh
```

This writes:

```text
checkpoints/interndata_a1/vj_rae_dual_arm_small.pt
reports/interndata_a1/latent_cache_vjepa_vjrae_small_manifest.jsonl
checkpoints/interndata_a1/unified_cached_small.pt
```

The small run uses more cached windows, a less extreme `lambda_cf`, and the
main-training noisy-action counterfactual path. Track whether validation
`S_a`, `delta_cond`, and `cf` improve without relying on the clean-action
diagnostic.

Validation `collapse` is computed from averaged validation monitors. The log also
prints `batch_alarm_count`; this can be non-zero when individual validation
windows are hard even if the averaged validation state is healthy.

After a small VJ-RAE checkpoint exists, evaluate whether the compressed VJ-RAE
latent deltas still preserve action-discriminative information:

```bash
bash scripts/evaluate_interndata_a1_dual_arm_vj_rae_probe_small.sh
```

This trains two temporary inverse-dynamics probes with matched capacity:

```text
pooled V-JEPA feature delta -> action
VJ-RAE latent delta         -> action
```

The probe first collects train/validation windows, standardizes actions with
train-set statistics, flattens valid latent transitions, then trains on random
transition mini-batches. This avoids the high variance of training one probe
step on a single 7-transition window.

It writes:

```text
reports/interndata_a1/vj_rae_probe_small.json
reports/interndata_a1/vj_rae_probe_small.md
```

The probe report has two gates:

```text
retention_ok  = val.relative_mse_increase <= max_relative_mse_increase
predictive_ok = val.latent_r2 >= min_latent_r2
strong_pass   = retention_ok and predictive_ok
```

`retention_ok` means the VJ-RAE latent probe is no more than 15% worse than the
pooled V-JEPA feature probe under matched capacity. `predictive_ok` means the
standardized latent probe beats the train-mean action baseline on held-out data.

If `retention_ok=True` but `predictive_ok=False`, the VJ-RAE has not lost more
action information than pooled V-JEPA, but the current temporary inverse-dynamics
probe is not yet a reliable held-out predictor. In that case, treat the VJ-RAE
probe as inconclusive rather than a scientific pass; increase data/probe steps or
move to the medium VJ-RAE run.

Once the small path is healthy, run the medium debug chain:

```bash
bash scripts/train_interndata_a1_dual_arm_vj_rae_medium.sh
bash scripts/evaluate_interndata_a1_dual_arm_vj_rae_probe_medium.sh
bash scripts/cache_interndata_a1_dual_arm_vjepa_vjrae_medium.sh
bash scripts/train_interndata_a1_dual_arm_unified_cached_medium.sh
```

This writes:

```text
checkpoints/interndata_a1/vj_rae_dual_arm_medium.pt
reports/interndata_a1/vj_rae_probe_medium.json
reports/interndata_a1/vj_rae_probe_medium.md
reports/interndata_a1/latent_cache_vjepa_vjrae_medium_manifest.jsonl
checkpoints/interndata_a1/unified_cached_medium.pt
```

The medium cache uses 512 latent windows by default and the unified run splits
them into non-overlapping train/validation windows (`0:448` and `448:512`).
This is still a debug-scale run, but it is the first run where validation is not
just a same-window overfit check.

After the medium tiny-DiT cached run is healthy, run a Wan-style architecture
smoke over the same cached latents:

```bash
bash scripts/train_interndata_a1_dual_arm_unified_cached_wan_tiny_smoke.sh
```

This uses `model.type: wan_tiny`: Wan block names, timestep modulation, text
cross-attention, 3D RoPE, chunk-causal mask, action tokens, state register, and
latent/action heads, but with a small randomly initialized trunk. It does not
need the Wan2.2-5B weights. The goal is to catch M4 wiring issues before the
real 5B checkpoint and FSDP path are introduced.

You only need to download Wan weights before testing `model.type: wan` or a real
Wan2.2 initialization run. With the official `Wan-AI/Wan2.2-TI2V-5B` download,
use the server path:

```text
/mnt/sfs_turbo/fyy/checkpoints/Wan2.2-TI2V-5B
```

Then point `weights.wan_checkpoint` at either the checkpoint file or the
directory containing `.safetensors`/`.pt`/`.pth`/`.bin` shards. The loader prints
loaded, dropped, missing, and unexpected keys, and intentionally drops Wan VAE,
pixel head, CLIP image branch, and image cross-attention weights.

For the official `Wan-AI/Wan2.2-TI2V-5B` download path, first run a metadata-only
checkpoint inspection:

```bash
bash scripts/inspect_wan2_2_ti2v_5b.sh
```

Default path:

```text
/mnt/sfs_turbo/fyy/checkpoints/Wan2.2-TI2V-5B
```

This reads safetensors metadata and builds our 5B Wan-Latent-WAM model on the
`meta` device, so it checks key mapping and tensor shapes without loading the
34GB checkpoint into RAM/GPU. It writes:

```text
reports/wan2_2/wan2_2_ti2v_5b_checkpoint_report.json
reports/wan2_2/wan2_2_ti2v_5b_checkpoint_report.md
```

Healthy output should have `status=PASS`, `backbone_missing=0`,
`unexpected=0`, and `shape_mismatch=0`. Dropped VAE/pixel-head/image-branch keys
are expected and should not be treated as an error.

If the metadata inspection passes, run a real-weight single-forward smoke:

```bash
bash scripts/smoke_interndata_a1_dual_arm_unified_cached_wan_real_forward.sh
```

This loads the official Wan DiT shards, casts the model to bf16 on one GPU, and
runs one no-grad forward over a cached VJ-RAE robot batch. It does not build an
optimizer, does not run backward, and does not save a checkpoint. Healthy output
ends with:

```text
[wan-real-forward] output latent_velocity=(1, 4, 144, 384) action_velocity=(1, 4, 12, 14)
[wan-real-forward] ok
```

If this step runs out of memory, do not lower the scientific sequence geometry
yet; the next implementation step should be a true FSDP load/train smoke rather
than a single-process real-5B forward.

After the real-weight forward passes, run the multi-GPU FSDP backward smoke.
The server launcher currently defaults to the three available GPUs `4,5,6`:

```bash
cd /mnt/sfs_turbo/fyy/latent_wam
bash scripts/fit_interndata_a1_dual_arm_control_stats.sh
bash scripts/smoke_interndata_a1_dual_arm_unified_cached_wan_real_fsdp_backward.sh
```

The first command fits action statistics per `action_schema_id` and proprio
statistics per `embodiment_id` using only train split remainders `0..6`; remainder
`7` remains validation-only. It writes
`reports/interndata_a1/control_stats_train.json`. The cached dataset applies
these fixed statistics online, so the VJ-RAE latent cache is unchanged.

This runs exactly one real cached latent-action optimization step. Rank 0 loads
the official checkpoint; all other ranks construct on the meta device, then
FSDP synchronizes and shards the model. It uses bf16, full parameter/gradient/
optimizer sharding, per-Wan-block wrapping, and activation checkpointing. It
does not create EMA weights or save a 5B checkpoint. To use a different set of
GPUs, set `CUDA_VISIBLE_DEVICES`; the launcher derives `NPROC_PER_NODE` from it.

Success prints nonzero gradient norms for the Wan backbone, latent adapter/head,
and action encoder/bridge/head. It also prints the raw action RMS and rejects an
initial weighted action flow loss above `2`, which catches an unstable
from-scratch action head before a real run. Healthy output ends with:

```text
[wan-fsdp] total_grad_norm=... optimizer_update_max=...
[wan-fsdp] peak_cuda_alloc_gb=... peak_cuda_reserved_gb=...
[wan-fsdp] ok
```

Any non-finite loss/gradient, zero gradient in a required branch, missing
optimizer update, wrong world size, or CUDA/FSDP failure stops the script.

After the one-step normalized-control smoke passes, run the M5 short fixed-batch
overfit:

```bash
CUDA_VISIBLE_DEVICES=4,5,6 \
bash scripts/train_interndata_a1_dual_arm_wan_real_fsdp_short.sh
```

This keeps one different real train batch per rank for eight optimizer steps.
The pretrained Wan backbone uses `1e-6`; new latent/action/state modules use
`1e-4`. Fixed-noise train and independent validation probes run before and
after. Success requires train `total`, `z_fm`, and `a_fm` all to decrease and
ends with:

```text
[wan-short] train_overfit_ok=True ...
[wan-short] ok
```

Validation `S_a`, `delta_cond`, and `collapse` are diagnostic at this short
horizon; they are not used to claim generalization. This run intentionally does
not save a 5B checkpoint.

Before a longer pilot, verify the server's DCP API and run a full model+Adam
checkpoint round-trip:

```bash
bash scripts/inspect_distributed_runtime.sh

CUDA_VISIBLE_DEVICES=4,5,6 \
bash scripts/smoke_interndata_a1_wan_fsdp_checkpoint_roundtrip.sh
```

The second command writes a real sharded checkpoint under
`checkpoints/interndata_a1/wan_fsdp_dcp_smoke/step_000001_legacy`. It uses the
FSDP1-compatible `SHARDED_STATE_DICT` backend because the server's nested FSDP
layout is not supported correctly by its newer `get_state_dict()` path. It
trains one step, saves, deliberately trains another step, restores, and verifies
both fixed-noise model metrics and Adam step state return to the saved point.
Budget roughly `60 GB` of shared storage for this model+optimizer checkpoint.

For full 5B finetuning, use FSDP rather than plain DDP. With fp32 master
parameters/gradients and Adam moments, DDP approaches 80 GB per rank before
activations and communication buckets. For the expected 4-node x 8-GPU H800
topology, use `configs/distributed/h800_32gpu_hsdp.yaml`: `HYBRID_SHARD` shards
within each 8-GPU node and replicates across the four nodes. Keep `FULL_SHARD`
for the current single-node debug server. Confirm the production topology
before launch; if all 32 GPUs share one high-bandwidth fabric, benchmark
`FULL_SHARD` against HSDP instead of assuming the 4x8 layout.

After the checkpoint round-trip passes, run the first streaming pilot:

```bash
CUDA_VISIBLE_DEVICES=4,5,6 \
bash scripts/train_interndata_a1_dual_arm_wan_fsdp_pilot.sh
```

This uses `DistributedSampler` so ranks do not train on overlapping manifest
rows, streams 32 optimizer steps, validates every 8 steps with fixed noise, and
writes metrics to `reports/interndata_a1/wan_fsdp_pilot_32.jsonl`. It accepts
only a conclusive, non-collapsed final validation whose total loss is no more
than 1.10x its initial value. On success it saves exactly one full model+Adam
DCP checkpoint at `checkpoints/interndata_a1/wan_fsdp_pilot/step_000032`.

Continue from a saved pilot checkpoint with:

```bash
RESUME_FROM=checkpoints/interndata_a1/wan_fsdp_pilot/step_000032 \
bash scripts/train_interndata_a1_dual_arm_wan_fsdp_pilot.sh
```

Set `pilot.total_steps` above the saved step before resuming. The trainer restores
model, Adam, completed step, and reconstructs the distributed sampler epoch and
batch offset.

Real robot training should replace the server hooks in:

```text
train/train_codec.py::build_dataloaders
train/train_unified_flow.py::build_train_objects
```

The real path should produce paired robot-only batches:

```text
LeRobot v2.1 videos/actions
  -> frozen V-JEPA 2.1 features
  -> frozen VJ-RAE 12x12 latent cache
  -> StepInputs(context_latent, r1, actions, proprio, text)
  -> unified latent-action flow training
```

If `data.source` is set to a real dataset value before those hooks are wired, the
training scripts fail loudly instead of silently falling back to synthetic data.
