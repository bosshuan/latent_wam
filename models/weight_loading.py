"""Load Wan2.2-5B DiT weights into our model (CLAUDE.md §3 / §5; user M4 point 2).

Contract:
  * map official Wan DiT keys -> our module names (the Wan trunk lives under
    ``backbone.``; names otherwise match so the map is a prefix);
  * ``load_state_dict(strict=False)`` and **print missing / unexpected** — never
    silently drop (CLAUDE.md §5);
  * **assert the VAE and the pixel output head are NEVER loaded**: ``patch_embedding``
    (VAE patch-embed), ``head.*`` (pixel velocity head), ``img_emb`` (CLIP i2v), and
    any ``vae`` key are dropped, plus the ti2v image cross-attn
    (``cross_attn.{k_img,v_img,norm_k_img}``) we don't use;
  * **expected missing** (from-scratch) = ``latent_adapter`` (VJEPALatentInputAdapter),
    ``latent_head`` (VJEPALatentFlowHead), the action branch + action-to-latent
    conditioner + state adapter, the tokenizer embeddings, and the frozen umT5
    ``text_encoder`` (a separate module, not part of the DiT checkpoint).

The DiT geometry is taken from ``WanConfig`` (read from the official config), so
this loader never hardcodes dim/depth/heads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Wan key prefixes/fragments we must NEVER load into our (semantic-latent) model.
_DROP_RULES = {
    "vae_patch_embed": lambda k: k.startswith("patch_embedding"),
    "pixel_head": lambda k: k.startswith("head."),
    "clip_img_emb": lambda k: k.startswith("img_emb"),
    "vae": lambda k: "vae" in k,
    "image_cross_attn": lambda k: (".k_img" in k or ".v_img" in k or ".norm_k_img" in k),
}

# Wan trunk prefixes we DO load (mapped under ``backbone.``).
_LOAD_PREFIXES = ("blocks.", "time_embedding.", "time_projection.", "text_embedding.")


@dataclass
class WeightLoadReport:
    loaded: list[str] = field(default_factory=list)
    dropped: dict[str, list[str]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    def summary(self) -> str:
        drop_counts = ", ".join(f"{k}={len(v)}" for k, v in self.dropped.items())
        return (
            f"[wan-load] loaded={len(self.loaded)} | dropped({drop_counts}) | "
            f"missing(from-scratch/frozen)={len(self.missing)} | unexpected={len(self.unexpected)}"
        )


def remap_wan_key(key: str) -> tuple[str | None, str | None]:
    """Return ``(our_key, drop_category)``: exactly one is non-None.

    A loadable key -> ``(backbone.<key>, None)``; a forbidden key ->
    ``(None, category)``. Drop rules are checked FIRST so a dropped sub-key inside
    ``blocks.`` (e.g. image cross-attn) is never mapped.
    """
    for cat, pred in _DROP_RULES.items():
        if pred(key):
            return None, cat
    if any(key.startswith(p) for p in _LOAD_PREFIXES):
        return "backbone." + key, None
    # unknown top-level key (not a trunk weight, not a known VAE/head) -> drop,
    # categorized so it surfaces in the report rather than vanishing.
    return None, "other_unmapped"


def load_wan_backbone(
    model: torch.nn.Module,
    wan_state_dict: dict[str, torch.Tensor],
    verbose: bool = True,
) -> WeightLoadReport:
    """Remap + load Wan DiT weights into ``model`` (strict=False), with asserts."""
    report = WeightLoadReport()
    remapped: dict[str, torch.Tensor] = {}
    model_keys = set(model.state_dict().keys())

    for k, v in wan_state_dict.items():
        our_key, drop_cat = remap_wan_key(k)
        if drop_cat is not None:
            report.dropped.setdefault(drop_cat, []).append(k)
            continue
        remapped[our_key] = v

    # Hard guard: NOTHING from the VAE / pixel head / image branch may have made
    # it into the load set (CLAUDE.md §2.2 — never load VAE/pixel head).
    forbidden = ("patch_embedding", "img_emb", "vae", ".k_img", ".v_img", ".norm_k_img")
    for tgt in remapped:
        assert tgt.startswith("backbone."), f"non-backbone key reached load set: {tgt}"
        assert not tgt.startswith("backbone.head."), (
            f"pixel head key reached the load set: {tgt} (must be dropped)"
        )
        assert not any(f in tgt for f in forbidden), f"forbidden weight in load set: {tgt}"

    incompatible = model.load_state_dict(remapped, strict=False)
    report.loaded = sorted(remapped.keys())
    report.missing = list(incompatible.missing_keys)
    report.unexpected = list(incompatible.unexpected_keys)

    # The from-scratch input/output adapters and action branch MUST be missing
    # (they have no Wan counterpart) — assert they were not silently overwritten.
    expect_missing_prefixes = (
        "latent_adapter",
        "latent_head",
        "action_encoder",
        "action_to_latent",
        "action_head",
    )
    for pref in expect_missing_prefixes:
        if any(mk.startswith(pref) for mk in model_keys):
            assert any(mk.startswith(pref) for mk in report.missing), (
                f"expected from-scratch module '{pref}' to be in missing keys but it "
                "was not — check the Wan key map is not loading into it."
            )

    if verbose:
        print(report.summary())
        if report.unexpected:
            print("[wan-load] unexpected (in ckpt remap, absent in model):")
            for k in report.unexpected:
                print(f"    {k}")
        for cat, keys in report.dropped.items():
            print(f"[wan-load] dropped[{cat}]: {len(keys)} keys (NOT loaded)")

    return report
