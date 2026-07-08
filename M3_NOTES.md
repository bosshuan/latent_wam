# M3 — Minimal joint Latent-World-Action DiT: shape + mask walkthrough

> Local: shape + seeded-math tests only, **not run** (CLAUDE.md §5 — no
> Python/torch here; only `py_compile` + a pure-Python check of the mask
> predicate). The single-batch-overfit / S_a-threshold *training* acceptances are
> server evals. The tiny DiT is a CPU stand-in for the Wan2.2-5B backbone wired
> in M4 — it exercises every Stage-A invariant without the 5B weights.

## Files
```
flow/interpolation.py   project flow convention (t=0 noise, t=1 data)
flow/schedulers.py      coupled/decoupled (t_z,t_a) timestep sampling
flow/solver.py          forward-Euler 0->1 (round-trip + sign-flip guard)
flow/losses.py          + flow-matching / clean-target / L_cf / S_a / Δ_cond / permute
models/attention_mask.py     TokenLayout + chunk-causal boolean mask
models/latent_tokenizer.py   [C,Z,A,V] pack/unpack + modality/chunk/spatial embeds
models/adapters/{latent,action,state,condition}.py
models/heads/{latent_flow,action_flow,value}.py
models/outputs.py            WAMOutput
models/latent_world_action_dit.py   tiny LatentWorldActionDiT
tests/{test_flow_targets,test_attention_mask,test_dit,test_action_sensitivity}.py
```

## Flow convention (CLAUDE.md §2 invariant 1 — the #1 silent bug)
`x_t=(1-t)x0+t x1`, `u=x1-x0`, `x̂1=x_t+(1-t)v`. **Opposite** Wan/DreamZero
(`sample=(1-σ)data+σ·noise`, target `noise-sample`, σ↔(1-t)). Guards:
- algebra round-trip `predict_x1(x_t,u,t)==x1`;
- solver round-trip: pure noise + exact velocity → data for ANY step count;
- **sign-flip guard**: feeding `-u` to the solver integrates to `x0-u` (NOT `x1`),
  so a second flip smuggled into solver/loss turns the test red.

## Token org `[C0][Z1,A1,V1]...` (doc §2.2) and time indexing
Context covers the WHOLE timeline (`T_ctx` chunks, idx `0..T_ctx-1`); the noisy
future is the **tail** `T_fut` chunks, so future chunk `i` has global index
`T_ctx-T_fut+i` and its clean target is `context[:,T_ctx-T_fut+i]` = the `C_k`
paired with `Z_k`. Packing is modality-contiguous `[C | Z | A | V]` (unpack =
slice+reshape; attention order irrelevant — mask is index-driven).

### Shape flow (tiny: latent=8, action=4, H=16, emb=3, grid_n=2; B=2,T_ctx=3,T_fut=2,n_act=2)
```
context [2,3,2,8] --latent_adapter--> [2,3,2,16]
noisy_z [2,2,2,8] --latent_adapter--> [2,2,2,16]
noisy_a [2,2,2,4] --MultiEmbodimentActionEncoder(t_a,emb)--> [2,2,2,16]
value_query (param) -----------------> [2,2,1,16]
pack -> seq [2, 3*2 + 2*2 + 2*2 + 2*1 = 16, 16]   (C=6, Z=4, A=4, V=2)
backbone: 2x AdaLN-Zero DiT block (per-token t embed; t_z on Z, t_a on A, 0 on C/V)
          + text cross-attn (shared) + per-embodiment proprio state token
unpack -> Z[2,2,2,16] A[2,2,2,16] V[2,2,1,16]
heads -> v_z [2,2,2,8]  v_a [2,2,2,4]  value [2,2,1]
```
NOTE: `CategorySpecific*` use `bmm` → need a `[B,T,·]` 3-D input; the action
branch flattens `(T_fut,n_act)` before the encoder/decoder and reshapes back.

## Attention mask semantics — who attends whom (CLAUDE.md §2.7 / invariant 7)
`mask[q,k]=True` ⇒ q may attend k. Rules (pure index logic, verified in
`test_attention_mask` + a pure-Python replica):
1. **no future**: never read a key in a later chunk;
2. **Z/A bidirectional, current chunk only**: a noisy Z/A key is visible only
   within the same chunk (no cross-chunk noisy reads — futures condition on clean
   context, not on other noisy chunks);
3. **clean context strictly earlier**: a noisy/value query reads `C_j` only for
   `j < its chunk` — reading its own `C_k` would **leak the flow target**;
4. **context is a pure causal encoder**: a `C` query reads only `C` keys
   (`j<=k`), never the noisy Z/A (so context can't depend on the noise);
5. **value is a read-only sink**: `V` reads C/Z/A, but no Z/A/C query reads a `V`
   key (value never perturbs the latent/action distribution — doc §3.1).

Example mask (idx: C0,C1,Z1,A1,V1):
```
       C0  C1  Z1  A1  V1
 C0    1   .   .   .   .
 C1    1   1   .   .   .
 Z1    1   .   1   1   .
 A1    1   .   1   1   .
 V1    1   .   1   1   1
```
The leakage test perturbs a key and asserts every NON-attending query's output is
bit-identical (and an attending one changes): perturbing Z1 moves A1/V1 but not
C0/C1; perturbing V1 moves nothing in Z1/A1.

## Missing action = structural omission (CLAUDE.md §2.3) — NOT build-then-mask
The DiT decides the action branch from `noisy_action is None`:
- **video batch** (`noisy_action=None`): no A tokens packed, no action encoder/
  head call → action params get **zero gradient** (`test_video_batch_action_params_zero_grad`).
- **robot batch**: A tokens packed; per-embodiment encoder/decoder run.
- **mixed batch + actions**: **rejected** (`_assert_homogeneous_action`). The M5
  sampler delivers homogeneous batches; a mixed batch must be split upstream so
  omission stays structural rather than a mask over fabricated rows.
Also asserts no `embodiment_id<0` ever reaches the action adapter (CLAUDE.md §10).

## Anti-collapse: counterfactual path + monitors (doc §2.7)
- `permute_actions_within_schema`: within each `action_schema_id` group with **≥2**
  valid rows, build a **derangement** (no self); singleton/video rows get
  `cf_valid=0` + a warning (never silently permute-against-self).
- forward path: re-forward with `ã` under the SAME noise/`t_a` → `r̂1(ã)`; the
  conditioning action genuinely moves `r̂1` (`test_counterfactual_forward_changes_prediction`,
  `test_end_to_end_action_sensitivity_through_dit`).
- `L_cf = [d(r̂1(a),r1) - d(r̂1(ã),r1) + δ]_+` (hinge, masked by `cf_valid`).
- monitors `S_a=E d(r̂1(a),r̂1(ã))`, `Δ_cond=d(r̂1^no-act,r1)-d(r̂1(a),r1)`;
  `collapse_alarm` fires on `S_a<0.01` or `Δ_cond<=0`.

## Deferred to M4 (per plan)
- replace tiny backbone with Wan2.2-TI2V-5B blocks + **3D-RoPE remap to 12×12**
  (here position is a learned modality/chunk/spatial embed) + timestep + umT5
  text cross-attn weights + KV cache; `weight_loading.py` (strict=False, print
  missing/unexpected, never load VAE/pixel head). Two-step rollout loss + full
  `L_A` assembly + sampler land in M5.
