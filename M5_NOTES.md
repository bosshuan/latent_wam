# M5 — Unified latent-action flow training (Stage A)

> Local: `py_compile` + pure-Python checks of the curriculum/sampler/mask logic
> only (no torch here). The loss-decrease / S_a-threshold acceptances are server
> evals. Scope is **training-only** (option 1): the working KV cache + closed-loop
> receding-horizon are M8 — M5 leaves only a KV-cache *interface placeholder*.
> Context kept (no /clear); M1–M5 run together on the server.

## Deliverables
```
data/mixed_batch_sampler.py   homogeneous video/robot batches + CurriculumSchedule
flow/losses.py  (+)           UnifiedLossWeights + assemble_unified_loss (L_A)
flow/rollout.py               two_step_rollout_loss (rollout consistency)
flow/schedulers.py            coupled<->decoupled (reused; driven by curriculum)
train/train_unified_flow.py   unified_train_step + EmaModel + validate + main(FSDP)
data/feature_cache.py (+)     codec_version key -> 384-d codec-latent cache switch
models/kv_cache.py            KV-cache INTERFACE STUB (M8 fills it); forward fails
                              loud on a non-empty cache
```

## L_A assembly (doc §4.2)
`L_A = λ_z L_z_FM + m_a λ_a L_a_FM + λ_clean L_clean + λ_roll L_roll
       + m_a λ_cf L_cf + m_v λ_v L_value`  (`λ_v=0`, value term omitted).
`assemble_unified_loss` combines the term tensors; **`m_a` is structural** — a
video batch passes `v_a=None`/`l_cf=None` and the action/cf terms are simply absent
(never built-then-masked, CLAUDE.md §2.3). `m_v` term is dropped (value is a Stage-A
stub). `a_mask` is the action step/pad key mask for the present robot rows.

## unified_train_step (the testable core)
For one homogeneous batch:
1. sample `(t_z, t_a)` (coupled or decoupled per curriculum);
2. `make_noisy` the future latent (target `r1` = clean future tail) → `v_z`, `u_z`,
   `r̂1 = predict_x1`; robot also noises the action chunk → `v_a`, `u_a`;
3. **robot only** — counterfactual forward: re-noise a within-schema **permuted**
   action with the SAME noise/`t_a` (`interpolate(noise_a, ã, t_a)`), forward →
   `r̂1(ã)`; `L_cf = hinge[d(r̂1(a),r1) - d(r̂1(ã),r1) + δ]`; monitors
   `S_a = E d(r̂1(a),r̂1(ã))` and a no-action forward for
   `Δ_cond = d(r̂1^no-act,r1) - d(r̂1(a),r1)`. The no-action forward drops ONLY the
   action branch and **keeps proprio** — the state register is gated on proprio
   presence, not `has_action`, so `Δ_cond` isolates the *action* contribution. If
   it dropped proprio too, a strong state could mask an action collapse (false
   negative) — defeating the §2.7 alarm. (`test_state_register_kept_in_no_action_forward`.)
4. two-step rollout (curriculum-ramped): feed `r̂1[:,0]` back as the current
   context chunk, re-forward, constrain the later chunks (`flow/rollout.py`);
5. `assemble_unified_loss` → total; `metrics` carries the term scalars + `S_a`,
   `Δ_cond`, and `collapse_alarm` strings (`min_action_sensitivity=0.01`,
   `require_delta_cond_positive`). The no-action / monitor forwards run under
   `no_grad` (monitors, not loss).
The counterfactual / no-action / rollout forwards all **reuse the same sampled
noise + timestep**, so the only changing variable is the conditioning action
(doc §2.7).

## Sampler + curriculum (doc §4.2)
`MixedBatchSampler` yields **homogeneous** index batches (all-video or
all-one-robot-schema) via a per-batch video/robot coin flip; a robot schema must
have `>= require_min_schema (2)` items to be eligible, so `L_cf` always has ≥2
same-schema rows (a within-schema group that still ends up <2 is skipped with a
warning inside `permute_actions_within_schema`, never silently 0). `CurriculumSchedule`
anneals over training progress: video-heavy → joint → action/rollout ramp, and the
timestep schedule **coupled → decoupled** in the last stage (`scheduler.coupled =
stage.coupled` per step). Verified in pure Python (homogeneity, singleton exclusion,
ratio extremes, progression).

## FSDP / frozen modules (train main, server-only)
- Frozen modules (CLAUDE.md §2.2) stay **eval-only and out of the shard**: the
  V-JEPA encoder + codec are external (not submodules of the trained model); the
  umT5 `text_encoder` is handed to FSDP `ignored_modules` so its frozen params are
  neither sharded nor gradient-tracked.
- `MixedPrecision(bf16)` + activation checkpointing on `WanAttentionBlock`.
- Single-GPU / CPU degrade via `dist_utils` (no process group when not under
  torchrun) so the same code path smoke-tests on a laptop.
- EMA shadow of trainable params; rank-0 checkpoint of model + EMA; periodic
  `validate` prints `S_a/Δ_cond` and a `[ALARM]` on collapse.

## Feature-cache switch (CLAUDE.md §4)
`FeatureCache(codec_version=...)` folds the frozen-codec id into the key, so the
payload becomes the **384-d codec latent (fp16)** and codec-latent entries can never
be served for a raw-feature key (test: `test_codec_version_separates_raw_and_codec_latent`).

## KV-cache stub (M8 boundary)
`models/kv_cache.py` defines `KVCache`/`LayerKV` + the M8 design note (prefill the
clean context once, decode future against cached context K/V — exact because
context output is future-independent). The DiT `forward` takes `kv_cache=None` and
`WAMOutput.kv_cache` exists, but a **non-empty** cache in Stage A raises
`KVCacheNotImplemented` (fail loud, no silent no-op). No `inference/` dir, no real
read/write logic — that is M8.

## Acceptance mapping (server)
latent/action validation loss ↓ (`unified_train_step` overfit on CPU as a proxy);
`L_cf` in the loss; validation prints `S_a/Δ_cond` with no collapse alarm; cache is
384-d codec latent. Two-step rollout improves latent rollout (ramped late).
