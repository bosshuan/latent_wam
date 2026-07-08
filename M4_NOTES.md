# M4 — Wan2.2-TI2V-5B backbone: RoPE order, weight map, causal, text

> Local: `py_compile` + pure-Python checks of the RoPE token-order and the weight
> remap only (no torch here). Tiny dims + **mock** weight loading run on the
> server; the real 5B checkpoint loads server-side. M1–M4 will be run together
> under torch on the server (context kept, not /clear-ed).

## What changed vs M3
M3's tiny AdaLN-Zero backbone → real **Wan2.2-5B trunk** (`wan_blocks.WanBackbone`,
names mirror the official checkpoint). M3's `attention_mask`, `latent_tokenizer`,
adapters, heads, flow are REUSED unchanged. New: `rope.py`, `wan_blocks.py`,
`wan_config.py`, `text_encoder.py`, `weight_loading.py`,
`wan_latent_world_action_dit.py`.

Geometry from `WanConfig` (read from `configs/model/latent_wam_dit.yaml` /official
config; never hardcoded): dim=3072, layers=30, heads=24 (head_dim=128), ffn=14336,
freq_dim=256, eps=1e-6, text_dim=4096 (umT5). VAE in/out=48 kept only for sanity.

## 【MOST CRITICAL】3D-RoPE coordinate order (user M4 point 1)
RoPE rotates q/k by an angle from each token's `(t,h,w)` coordinate, so the
coordinate handed to each token MUST equal its slot in the packed sequence — a
mismatch corrupts position silently (no error).

**Wan side** (`CausalWanModel._create_freqs`, dreamzero
`wan_video_dit_action_casual_chunk.py:2211-2219`):
```
freqs = cat([ time[f].view(f,1,1,-1).expand(f,h,w,-1),
              h[:h].view(1,h,1,-1).expand(f,h,w,-1),
              w[:w].view(1,1,w,-1).expand(f,h,w,-1) ], -1).reshape(f*h*w, ...)
```
→ token order is **time-major, then height, then width (w fastest)**, row-major.

**Our side** (`latent_tokenizer.pack`): latent/context blocks are
`[B, T, N, H] -> [B, T*N, H]` with `N = grid_h*grid_w` already row-major (h outer,
w inner) — the codec `TokenReducer` reshapes `[B,T,h,w,...]`. So per-token order is
`(t, h, w)` with `w` fastest — **IDENTICAL to Wan**. We therefore reuse Wan's grid
construction (`rope.Rope3D.grid_freqs`) unchanged, and `assemble` walks the packed
blocks `[C | Z | A | V]` in order:
  * `C`: `grid_freqs(chunks 0..T_ctx-1, h, w)`  (`C_k` and `Z_k` share the time idx)
  * `Z`: `grid_freqs(chunks T_ctx-T_fut..T_ctx-1, h, w)`
  * `A`/`V`: **1-D** RoPE, unique position per token enumerated in packing order
    (`i*n_act + step`) = DreamZero's per-block `freqs_action` slice.

Freq split (head_dim `d`, Wan): time `d-4·(d//6)`, h/w `2·(d//6)` each (5B: 44/42/42
→ 128). `test_rope` asserts the split sums to `d`, the `(t,h,w)` ordering, the
`assemble`↔pack block correspondence, and the rotary relative-position property.

## Weight loading (user M4 point 2) — `weight_loading.py`
`remap_wan_key`: drop-rules checked FIRST, then load-prefixes.
  * **LOAD** (→ `backbone.<key>`): `blocks.*` (self_attn q/k/v/o + norm_q/k,
    cross_attn q/k/v/o + norm_q/k, norm1/2/3, ffn.0/2, modulation),
    `time_embedding.*`, `time_projection.*`, `text_embedding.*`.
  * **DROP, never loaded** (asserted): `patch_embedding` (VAE patch-embed),
    `head.*` (pixel velocity head), `img_emb` (CLIP i2v), any `vae`, and the ti2v
    image cross-attn `cross_attn.{k_img,v_img,norm_k_img}` (we condition on V-JEPA
    latent context in-sequence, not on a CLIP image token).
`load_state_dict(strict=False)`, prints missing/unexpected. **Expected missing
(from scratch)** = `latent_adapter` (VJEPALatentInputAdapter, replaces VAE
patch-embed), `latent_head` (VJEPALatentFlowHead, replaces pixel head), action
branch (`action_encoder`/`action_head`), `state_adapter`, tokenizer embeds, and the
frozen umT5 `text_encoder` (separate module). A hard assert forbids any
VAE/pixel/img weight from reaching the load set, and asserts the from-scratch
prefixes land in `missing`.

## Bidirectional → causal (user M4 point 3) — mask only
`WanSelfAttention` consumes the M3 chunk-causal additive mask (the 5 rules: Z/A
in-chunk bidirectional, cross-chunk causal, clean-context strictly-earlier =
no target leak, context pure causal encoder, value read-only sink). **Only the
mask changes** — q/k/v/o weights load from Wan unchanged; RoPE is applied to q/k
inside the attention (Wan does the same).

## Text (user M4 point 4)
Frozen **umT5-XXL** (`text_encoder.FrozenUMT5TextEncoder`: eval +
`requires_grad_(False)`, stays eval through parent `.train()`, no_grad encode →
zero grad; covered by `test_frozen_modules`). Pipeline: umT5 → Wan
`text_embedding` (loaded) → Wan text **cross-attention** (loaded) per block.
Task-string **cache** (`encode` memoizes; `precache` for `meta/tasks.jsonl`),
fixed **null-text** for caption-less video (unified with `action_valid=0`), and
**CFG** `text_cfg_dropout=0.1` (`encode_with_cfg`; no-op in eval →
`test_cfg_dropout_is_noop_in_eval`). The mock encoder is a frozen deterministic
embedding so caching/frozen tests run on CPU; the real umT5 loads on the server.

## Shape flow (tiny: dim=24, heads=2, head_dim=12; latent=8, action=4; B=2,T_ctx=3,T_fut=2,grid 2x2,n_act=2)
```
context [2,3,4,8] --latent_adapter--> [2,3,4,24]   (Wan VAE patch-embed REPLACED)
noisy_z [2,2,4,8] --latent_adapter--> [2,2,4,24]
noisy_a [2,2,2,4] --action_encoder--> [2,2,2,24]
pack [C|Z|A|V] (positional=False; modality embed only) -> seq [2, 12+8+4+2=26, 24]
freqs = Rope3D.assemble(layout,(2,2))  -> [26, 6] complex
mask  = chunk-causal additive          -> [26, 26]
text "..." --umT5(frozen)--> [2,4,16] --backbone.text_embedding--> [2,4,24] (+state [2,1,24])
backbone: 2x Wan block (self_attn+RoPE+mask, text cross-attn, ffn, 6-way AdaLN)
unpack -> heads -> v_z [2,2,4,8]  v_a [2,2,2,4]  value [2,2,1]   (pixel head REPLACED)
```

## Proprio/state placement — settled against DreamZero source (post-review)
Decision: proprio is an **in-sequence STATE register token**, NOT a cross-attn
condition. Evidence in `../dreamzero/.../wan_video_dit_action_casual_chunk.py`:
  * the DiT sequence is `[clean_image | noisy_image | noisy_action | noisy_state]`
    — state is part of the self-attention **action_register** (one state token per
    block, `freqs_state` 1-D RoPE); not in cross-attn (cross-attn there is text/CLIP);
  * noisy image block `i` and action block `i` both put `noisy_state[i]` in their
    k/v context (`:710-714`, `:769-773`) → **Z and A read their chunk's state**;
  * `_process_state_blocks` (`:626-659`) → **state attends only itself** (its
    representation never depends on the noisy tokens).

So the cross-attn stays **pure text** (umT5, all-embodiment-shared — keeps the
per-embodiment proprio out of the shared text semantics), and proprio becomes a
6th modality `S` in the mask + tokenizer (omitted for video; single token at the
current/first-future chunk `l`, carrying `q_l`). Updated mask rules (now 6):
  1. cross-chunk causal (no future);
  2. Z/A bidirectional, current chunk only;
  3. clean context strictly-earlier (no own-chunk target leak);
  4. context = pure causal encoder (C reads only C);
  5. value = read-only sink (V reads C/Z/A/S; nobody reads V);
  6. **state = read-only conditioning register**: Z/A/V read `S` of the
     same-or-earlier chunk; `S` reads only itself; `C` never reads `S`.
**STATE token chunk index** (= its `freqs_state` 1-D position): `cur_chunk =
t_ctx - t_fut`, i.e. the **first future chunk = history/future boundary = current
time l** (`q_l` lives exactly there). This is the *minimum* future-chunk index, so
under rule 6 (`ck_S <= cq`) **every** future Z/A/V (all at chunk `>= t_ctx-t_fut`)
can read it — matching "q_l is visible to all of the future". History-only context
chunks `[0 .. t_ctx-t_fut-1]` precede it (and context never reads S anyway). The
RoPE position equals this same chunk index, so the register is temporally placed
at the boundary, consistent with the mask semantics.

Verified in pure Python + `test_attention_mask::test_state_register_boundaries`;
`test_wan_dit::test_proprio_is_insequence_register` checks proprio changes r̂1 and
the state adapter is on the gradient path. Both M3 and M4 models updated (shared
tokenizer) — the change is mask + tokenizer + RoPE (`S` gets 1-D `freqs_state`),
not a new attention path.

## Deferred (M5)
KV cache + receding-horizon inference; two-step rollout loss + full `L_A` assembly;
mixed-batch sampler (M4 rejects mixed action batches — must be split upstream).
