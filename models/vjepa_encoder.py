"""Frozen V-JEPA 2.1 gigantic encoder wrapper.

Reads the dense multi-level token features used as Stage A targets, then keeps
the encoder permanently frozen (CLAUDE.md §2.2 frozen-module invariant).

Design decisions (user-confirmed, see plan):
  * **Runtime-read dims as single source of truth.** ``embed_dim / depth /
    num_heads / patch_size / tubelet / img_size`` are read off the *instantiated*
    backbone — never hand-written here or duplicated in yaml. The four gigantic
    constants confirmed from ``../vjepa2/`` code (embed_dim=1664, depth=48,
    num_heads=26, patch_size=16, tubelet=2, img=384) are only used as *startup
    assertions* against the live module, so a wrong build fails loud.
  * **``extract_layers`` is an explicit config choice** (default the gigantic
    hierarchical layers [11,23,37,47]) so the §6.2 "final-layer vs 4-layer
    fusion" ablation swaps a layer via config, not code.
  * **Hierarchical features under eval().** Verified in code: setting
    ``backbone.return_hierarchical = True`` makes the encoder return the
    per-layer-LayerNorm'd 4-layer concat **without** needing the ``training=True``
    forward path (``vision_transformer.py:328-340``; the ``hier.append`` loop is
    not gated on ``training`` and ``return_hierarchical`` defaults False). We run
    the encoder in ``eval()`` + ``no_grad`` always.

Reference (read-only): ``../vjepa2/src/hub/backbones.py`` (factory, weight load,
``_clean_backbone_key``) and ``../vjepa2/app/vjepa_2_1/models/vision_transformer.py``
(forward, hierarchical extraction).

The wrapper accepts an injected ``backbone`` (or ``build_fn``) so the unit tests
exercise the reshape/assert logic on a tiny CPU mock without downloading the 2B
checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

from data.schemas import MultiLevelFeatures

# Confirmed-from-code expectations for the gigantic encoder. Used ONLY as
# fail-loud startup assertions against the live module (never as values to
# compute with). depth=48 implies hierarchical layers [11,23,37,47].
_GIGANTIC_EXPECT = {
    "embed_dim": 1664,
    "depth": 48,
    "num_heads": 26,
    "patch_size": 16,
    "tubelet": 2,
    "img_size": 384,
}
_GIGANTIC_HIER_LAYERS = (11, 23, 37, 47)


def _default_build(
    hub_name: str,
    pretrained: bool,
    hub_repo: str = "facebookresearch/vjepa2",
    hub_source: str = "github",
) -> nn.Module:
    """Build the real backbone via torch.hub (server path; not run locally)."""
    import torch.hub  # local import keeps module import cheap for CPU tests

    result = torch.hub.load(
        hub_repo,
        hub_name,
        pretrained=pretrained,
        source=hub_source,
    )
    # CONFIRMED from ../vjepa2/src/hub/backbones.py: the v2.1 factory
    # ``_make_vjepa2_1_model`` builds ``encoder`` (:245), loads
    # ``state_dict[checkpoint_key]`` into it where ``checkpoint_key`` defaults to
    # "target_encoder" (:213) — and the gigantic factory (:330) does NOT override
    # it — then ``return encoder, predictor`` (:285). So ``result[0]`` is the
    # encoder carrying the EMA **target_encoder** weights, which is exactly the
    # clean-target branch we want to freeze.
    backbone = result[0] if isinstance(result, (tuple, list)) else result
    return backbone


class FrozenVJEPAEncoder(nn.Module):
    def __init__(
        self,
        hub_name: str = "vjepa2_1_vit_gigantic_384",
        hub_repo: str = "facebookresearch/vjepa2",
        hub_source: str = "github",
        extract_layers: tuple[int, ...] = _GIGANTIC_HIER_LAYERS,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
        checkpoint_key: str = "target_encoder",
        checkpoint_strict: bool = True,
        backbone: Optional[nn.Module] = None,
        build_fn: Optional[Callable[[str, bool], nn.Module]] = None,
        assert_gigantic: bool = True,
    ) -> None:
        super().__init__()
        self.hub_name = hub_name
        self.hub_repo = hub_repo
        self.hub_source = hub_source
        self.extract_layers = tuple(int(i) for i in extract_layers)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_key = checkpoint_key

        if checkpoint_path and pretrained:
            raise ValueError(
                "Pass either pretrained=True for torch.hub loading OR "
                "encoder.checkpoint_path for local loading, not both. "
                "Use pretrained=False with checkpoint_path on offline servers."
            )

        if backbone is None:
            if build_fn is not None:
                backbone = build_fn(hub_name, pretrained)
            else:
                backbone = _default_build(hub_name, pretrained, hub_repo, hub_source)
        self.backbone = backbone

        # --- freeze + enable hierarchical output (single time) ----------
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        if hasattr(self.backbone, "return_hierarchical"):
            self.backbone.return_hierarchical = True
        # Honor the config-chosen extraction layers when the real encoder
        # exposes its layer registry (enables the layer-fusion ablation).
        if hasattr(self.backbone, "hierarchical_layers") and hasattr(
            self.backbone, "out_layers_distillation"
        ):
            hier = set(int(x) for x in self.backbone.hierarchical_layers)
            missing = set(self.extract_layers) - hier
            if missing:
                raise ValueError(
                    f"extract_layers {self.extract_layers} not all in backbone "
                    f"hierarchical_layers {sorted(hier)}; cannot index norms_block."
                )
            self.backbone.out_layers_distillation = list(self.extract_layers)

        # --- read dims at runtime (single source of truth) --------------
        # Attribute names VERIFIED against ../vjepa2/app/vjepa_2_1/models/
        # vision_transformer.py: embed_dim(:60), num_heads(:61), patch_size(:70),
        # tubelet_size(:72), img_height/img_width(:69), blocks(:114),
        # hierarchical_layers/out_layers_distillation(:148-178),
        # return_hierarchical(:181). Not guessed.
        self._embed_dim = int(self.backbone.embed_dim)
        self._depth = self._read_depth(self.backbone)
        self._num_heads = int(getattr(self.backbone, "num_heads"))
        self._patch_size = int(getattr(self.backbone, "patch_size"))
        self._tubelet = int(getattr(self.backbone, "tubelet_size"))
        h = int(getattr(self.backbone, "img_height"))
        w = int(getattr(self.backbone, "img_width"))
        if h != w:
            raise ValueError(f"non-square img_size {h}x{w} unsupported")
        self._img_size = h

        # --- fail-loud startup asserts ----------------------------------
        if not all(layer < self._depth for layer in self.extract_layers):
            raise ValueError(
                f"extract_layers {self.extract_layers} must all be < depth "
                f"{self._depth}"
            )
        if self._img_size % self._patch_size != 0:
            raise ValueError(
                f"img_size {self._img_size} not divisible by patch_size "
                f"{self._patch_size}"
            )
        if assert_gigantic and hub_name == "vjepa2_1_vit_gigantic_384":
            self._assert_gigantic()

        if checkpoint_path:
            checkpoint = _load_checkpoint_file(checkpoint_path)
            self.load_pretrained(
                checkpoint,
                checkpoint_key=checkpoint_key,
                strict=checkpoint_strict,
                source=str(checkpoint_path),
            )

    @staticmethod
    def _read_depth(backbone: nn.Module) -> int:
        if hasattr(backbone, "blocks"):
            return len(backbone.blocks)
        if hasattr(backbone, "depth"):
            return int(backbone.depth)
        raise AttributeError("backbone exposes neither `blocks` nor `depth`")

    def _assert_gigantic(self) -> None:
        got = {
            "embed_dim": self._embed_dim,
            "depth": self._depth,
            "num_heads": self._num_heads,
            "patch_size": self._patch_size,
            "tubelet": self._tubelet,
            "img_size": self._img_size,
        }
        for k, expect in _GIGANTIC_EXPECT.items():
            if got[k] != expect:
                raise AssertionError(
                    f"gigantic encoder {k}={got[k]} != expected {expect}; "
                    "the loaded backbone does not match the confirmed gigantic "
                    "config — refusing to proceed."
                )

    # --- read-only dim properties (derived quantities computed here) ----
    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def tubelet(self) -> int:
        return self._tubelet

    @property
    def img_size(self) -> int:
        return self._img_size

    @property
    def num_layers(self) -> int:
        return len(self.extract_layers)

    @property
    def grid_hw(self) -> tuple[int, int]:
        g = self._img_size // self._patch_size
        return (g, g)

    @property
    def codec_in_dim(self) -> int:
        """Channel width the codec consumes: len(extract_layers) * embed_dim."""
        return self.num_layers * self._embed_dim

    # --- forward --------------------------------------------------------
    @torch.no_grad()
    def forward(self, pixels: torch.Tensor) -> MultiLevelFeatures:
        """Encode a clip into dense multi-level features.

        Args:
            pixels: [B, T, 3, H, W] (history+future frames, T = #raw frames).

        Returns:
            MultiLevelFeatures with features [B, T_tok, N, L, D] and
            token_grid (T_tok, grid_h, grid_w). T_tok = T // tubelet.
        """
        if pixels.ndim != 5:
            raise ValueError(
                f"expected pixels [B,T,3,H,W]; got {tuple(pixels.shape)}"
            )
        b, t, c, h, w = pixels.shape
        if c != 3:
            raise ValueError(f"expected 3 channels; got {c}")
        if h != self._img_size or w != self._img_size:
            raise ValueError(
                f"input {h}x{w} != encoder img_size {self._img_size} "
                "(no resize/RoPE interpolation — feed native 384)."
            )

        # backbone expects channel-first video [B, 3, T, H, W]
        vid = pixels.permute(0, 2, 1, 3, 4).contiguous()
        out = self.backbone(vid)  # hierarchical concat: [B, S, L*D]
        if out.ndim != 3:
            raise ValueError(
                f"backbone returned ndim {out.ndim}; expected [B, S, L*D]. "
                "Is return_hierarchical set?"
            )

        bb, seq, ld = out.shape
        d = self._embed_dim
        l = self.num_layers
        if ld != l * d:
            raise ValueError(
                f"channel dim {ld} != num_layers*embed_dim = {l}*{d}={l * d}; "
                "extract_layers / return_hierarchical mismatch."
            )
        gh, gw = self.grid_hw
        n = gh * gw
        if seq % n != 0:
            raise ValueError(
                f"token seq {seq} not divisible by spatial N {n} ({gh}x{gw})"
            )
        t_tok = seq // n
        # Layout VERIFIED against ../vjepa2/ source (not assumed):
        #  * Token order is time-major, row-major spatial. PatchEmbed3D.forward
        #    (utils/patch_embed.py:69-72): Conv3d -> [B,embed,T_tok,Hp,Wp] ->
        #    flatten(2) (C-contiguous: idx = t*Hp*Wp + h*Wp + w) -> transpose.
        #    No cls/register tokens prepended for gigantic (n_registers=0,
        #    cls_token=None, vision_transformer.py:180). So seq splits as
        #    (t_tok, n) with t_tok outermost.
        #  * Channel concat is layer-major. vision_transformer.py:328-336 appends
        #    norms_block[idx](x) in ascending block index, then cat(hier, dim=2).
        #    So the L*D channel axis splits as (l, d) with l outermost.
        # Hence this reshape is content-correct, not just shape-correct
        # (guarded by test_shapes::test_reshape_content_layout).
        feats = out.reshape(bb, t_tok, n, l, d)
        return MultiLevelFeatures(features=feats, token_grid=(t_tok, gh, gw))

    # --- explicit, auditable weight loading -----------------------------
    def load_pretrained(
        self,
        state_dict: dict,
        checkpoint_key: str = "target_encoder",
        strict: bool = False,
        source: str = "<memory>",
    ) -> None:
        """Load encoder weights, printing missing/unexpected keys (no silence).

        ``strict=False`` by design: the v2.1 checkpoint carries a pos_embed the
        RoPE encoder ignores, so we report rather than crash (CLAUDE.md §5).
        """
        selected = _select_checkpoint_state_dict(state_dict, checkpoint_key)
        cleaned = _clean_backbone_key(selected)
        result = self.backbone.load_state_dict(cleaned, strict=strict)
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        print(
            f"[FrozenVJEPAEncoder] loaded checkpoint source={source} "
            f"key={checkpoint_key} strict={strict}",
            flush=True,
        )
        print(f"[FrozenVJEPAEncoder] missing_keys ({len(missing)}): {missing}")
        print(f"[FrozenVJEPAEncoder] unexpected_keys ({len(unexpected)}): {unexpected}")
        # re-assert frozen/eval after load
        self.backbone.eval()
        self.backbone.requires_grad_(False)


def _load_checkpoint_file(path: str | Path) -> dict:
    """Load a local checkpoint without allowing arbitrary object execution."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"V-JEPA checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


def _looks_like_state_dict(obj) -> bool:
    return (
        isinstance(obj, dict)
        and bool(obj)
        and all(isinstance(k, str) for k in obj.keys())
        and any(torch.is_tensor(v) for v in obj.values())
    )


def _select_checkpoint_state_dict(checkpoint: dict, checkpoint_key: str) -> dict:
    """Select encoder weights from V-JEPA checkpoint wrappers.

    Official V-JEPA 2.1 checkpoints store the clean EMA branch under
    ``target_encoder`` for gigantic. We also tolerate common wrappers so local
    downloaded files fail only when no plausible encoder state exists.
    """

    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(checkpoint)!r}")

    if checkpoint_key and checkpoint_key in checkpoint:
        return checkpoint[checkpoint_key]

    for key in ("target_encoder", "encoder", "ema_encoder", "state_dict", "model"):
        value = checkpoint.get(key)
        if _looks_like_state_dict(value):
            return value

    if _looks_like_state_dict(checkpoint):
        return checkpoint

    available = ", ".join(str(k) for k in sorted(checkpoint.keys()))
    raise KeyError(
        f"could not find encoder state dict in checkpoint. "
        f"requested key={checkpoint_key!r}; available keys=[{available}]"
    )


def _clean_backbone_key(state_dict: dict) -> dict:
    """Strip ``module.`` / ``backbone.`` prefixes (mirrors vjepa2 helper).

    Source: ``../vjepa2/src/hub/backbones.py:_clean_backbone_key`` (rewritten).
    """
    cleaned = {}
    prefixes = (
        "module.",
        "backbone.",
        "target_encoder.",
        "encoder.",
        "ema_encoder.",
    )
    for key, val in state_dict.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        cleaned[key] = val
    return cleaned
