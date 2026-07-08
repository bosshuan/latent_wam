# M2 — Latent codec: shape-flow walkthrough

> Local: shape + seeded-math tests only, **not run** (CLAUDE.md §5). The codec
> reconstruction-cosine and action-discriminability *accuracy* acceptances are
> server evals.

## Pipeline (gigantic dims: L=4, D=1664, grid 24×24=576, hidden=1024, latent=384)
DECISION (user Q1): the codec is the invertible `pooled-features <-> 384`
compressor at **12×12 = 144 tokens only**. The 2×2 pool is a fixed, intentional,
lossy downsample the codec does NOT invert; reconstruction target = the POOLED
normalized features (fixed), and the decoder stays at 144 tokens (no upsample).
```
raw features            [B, T, 576, 4, 1664]      (FrozenVJEPAEncoder)
  FixedFeatureNormalizer (per (L,D); offline stats ACROSS video+robot)
                       -> z [B, T, 576, 4, 1664]
  TokenReducer 2×2 (pool FIRST; keeps 2D grid+time; never whole-frame)
                       -> z_pooled [B, T, 144, 4, 1664]   # FIXED recon target
  MultiLevelFusion: per-layer LN(1664) -> concat 4*1664=6656 -> MLP
                       -> [B, T, 144, 1024]
  CodecEncoder MLP     -> R [B, T, 144, 384]        # LatentGrid grid=(T,12,12)
decode (NO upsample):
  R -> MLP(384 -> 1024 -> 4*1664) -> recon [B, T, 144, 4, 1664]
loss compares recon vs z_pooled (both 12×12) — the high-freq dropped by pooling
is excluded from BOTH sides, so cosine measures only the 384 compression fidelity.
```
`codec_in_dim = L*D = 6656` is read from `encoder.codec_in_dim`, never
hardcoded; `latent_dim=384/token`. Target is `pool(normalize(raw))` (fixed) — not
the trained fusion's output (which would be a circular/moving target).

## Tiny test dims (CPU): L=2, D=8, grid 4×4=16, hidden=16/32, latent=12
```
feats [2,3,16,2,8] -pool-> [2,3,4,2,8] -fuse/enc-> R [2,3,4,12] grid (3,2,2)
                   -decode-> [2,3,4,2,8]   (4 tokens, NOT 16 — no upsample)
TokenReducer (4,4)->(2,2): arange(16) -> 4 distinct block-means (not 1)
```

## Loss (doc §2.3) — `flow/losses.py::codec_loss`
`L1 + λ_cos(1-cos) + λ_var·VICReg-var + λ_cov·VICReg-cov + λ_dyn·m_a·MSE(g_φ(Δr),a)`.
- var/cov keep flow-space latent from collapsing (mean≈0, per-dim std≈1, decorrelated).
- `g_φ` = `ActionDiscriminabilityProbe` (training-only, **discarded** after codec
  training); fed `pooled_latent_delta` (spatial mean, consecutive-time diff)
  → `[B,T-1,latent]`. dyn term gated by `m_a` (robot only; never a fake action).
- **dyn gradient path (user Q2):** MSE(g_φ(Δr), a) backprops into BOTH the probe
  AND the codec (Δr derives from R, which carries codec grad). Intentional
  small-weight pressure (`λ_dyn=0.1`) to keep the codec from compressing away
  action info; NO stop-grad (that would train only the probe). Small weight
  avoids the codec overfitting to please the probe.

## Training (`train/train_codec.py`) — torchrun + DDP, single-GPU degrade
- `build_codec_from_encoder(encoder)` derives all dims from the live encoder.
- `fit_normalizer(codec, feature_iter)` MUST be fed a video+robot-mixed stream;
  accumulation buffers are all-reduced across ranks before `finalize`.
- `codec_train_step(...)` pure/unit-testable; `main()` (server only) wires the
  dataloader, DDP-wraps, trains, then `codec.freeze()` + checkpoint (codec +
  serialized normalizer stats with `norm_version`).
- `train/dist_utils.py` (new this milestone): process-group init from torchrun
  env (or single-process degrade), seeding (rank-offset), barrier/cleanup,
  rank-0 full checkpoint. FSDP sharded-state handling deferred to M5.

## Deferred to later milestones
- `models/latent_tokenizer.py` (the [C,Z,A,V] packing) is **M3**, not built here
  — TokenReducer (the only M2-relevant reducer) lives in `latent_codec.py`.
- per-transition action pooling for the probe: server-side data concern; the
  step takes a pre-pooled `[B,T-1,A]` so the probe never sees a fabricated action.
