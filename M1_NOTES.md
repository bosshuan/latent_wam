# M1 — Data layer + frozen V-JEPA: shape-flow walkthrough

> Local has no Python/CUDA — these tests are written but **not run** locally
> (CLAUDE.md §5). Run `pytest latent_wam/tests` on the server before training.
> M1 has **no attention mask** yet (that arrives in M3); this note is shape-flow.

## Verified-from-code encoder facts (not memory)
`../vjepa2/app/vjepa_2_1/models/vision_transformer.py` + `src/hub/backbones.py`:
`embed_dim=1664, depth=48, num_heads=26, patch_size=16, tubelet=2, img=384`;
hierarchical layers `[11,23,37,47]` (encoder `__init__` default
`n_output_distillation=4`, `:56,168-173`). `return_hierarchical=True` returns the
4-layer concat under `eval()` — the `hier.append` loop (`:328-330`) is **not**
gated on `training`, default `return_hierarchical=False` (`:181`). We read all
dims at runtime; the constants above are only fail-loud startup asserts.

## Encoder forward (gigantic, real dims)
```
pixels            [B, T, 3, 384, 384]          # T = (history+future)*tubelet = 16
  permute      -> [B, 3, T, 384, 384]
  backbone     -> [B, S=T_tok*576, L*D = 4*1664 = 6656]   # return_hierarchical
                  T_tok = T // tubelet = 8 ; 576 = (384/16)^2
  reshape      -> features [B, T_tok=8, N=576, L=4, D=1664]
token_grid = (8, 24, 24)
```
Dense 2D grid + time index preserved — never whole-frame pooled (CLAUDE.md §2.8).
`codec_in_dim = L*D = 6656` (computed, never hardcoded). 2×2 pool → 12×12=144 is
M2's job.

## Tiny mock (tests, CPU): embed_dim=16, depth=4, extract=(1,3) ⇒ L=2, img=384
```
pixels [2,4,3,384,384] -> [2,3,4,384,384] -conv3d(k=2,16,16)-> [2,16,2,24,24]
  flatten -> tokens [2, 1152, 16]   (1152 = 2*576)
  concat L=2 -> [2, 1152, 32]
  reshape -> features [2, 2, 576, 2, 16] ; token_grid (2,24,24)
```

## Temporal alignment (CLAUDE.md §3)
`actions_per_chunk(fps) = round(min(fps,30)*0.4)` → 30→12, 50→12, 10→4.
`build_delta_timestamps` → `num_chunks = history+future = 8`; obs chunk = tubelet
frame offsets, action chunk = `actions_per_chunk` step offsets; t=0 is the
history/future boundary (history offsets negative). fps always from
`meta/info.json` — passed in, never hardcoded.

## Missing-action invariant (CLAUDE.md §2.3) — enforced at 3 layers
1. `TrajectorySample(action_valid=False, actions=…)` → raises.
2. `collate_trajectory` of an all-video batch → `actions=None, proprio=None`
   (the sampler prefers homogeneous batches, so this is the common case).
3. Mixed batch → video rows zero-filled as **inert storage** (`action_valid=0`,
   `action_pad_mask=0`), **not** a pseudo-action. **Binding M1→M3 contract:** the
   M3 DiT **structurally omits the `Ak` tokens** for `action_valid=0` rows
   (gather valid rows) — *not* build-then-mask; a zero row never enters the
   action flow loss nor conditions latent prediction.

## Feature cache (user note 2)
Key = `(dataset_id, episode, frame_range, vjepa_version, extract_layers,
norm_version)` → sha256. Payload fp16. Raw multi-level feature ≈ 7.6 MB /
time-token, so the cache supports a **subset allowlist** + **LRU byte cap**. M5
swaps payload to 384-d codec latent; the `norm_version`/layer key invalidates raw
entries automatically.

## Schema scanner / registry
`scan_dataset(meta/info.json)` asserts `codebase_version=="v2.1"`, sums feature
shapes into state/action dims, allocates `embodiment_id`/`action_schema_id`, and
fails early on an OXE expected-dim mismatch. `write_schema_report` emits
`schema_report.md`. **Embodiment ids:** real embodiments get ids from 0;
`NEW_EMBODIMENT` (RoboTwin, untrained) gets a **trailing** id via
`new_embodiment_id()` — never id 0, so robot data can't silently route through
untrained weights. Video uses `INVALID_EMBODIMENT_ID=-1` (no action adapter).
`assert_trainable_batch(ids)` rejects -1 / NEW_EMBODIMENT ids in a training
batch.

## Files
- new: `models/vjepa_encoder.py`, `data/{schemas,temporal_alignment,collate,
  video_dataset,robot_dataset,feature_cache,schema_scanner,registry}.py`,
  `configs/{codec,unified_pretrain}.yaml`.
- tests: `tests/{_mock_vjepa,test_shapes,test_frozen_modules,test_missing_action,
  test_temporal_alignment,test_feature_cache,test_schema_scanner}.py`.
- not yet wired: real LeRobot / video decode backends (lazy-imported; server),
  `latent_target` (filled once codec frozen, M5), text embeddings (M4).
