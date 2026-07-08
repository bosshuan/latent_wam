"""Wan2.2-TI2V-5B backbone config (read, never hardcoded — CLAUDE.md §3).

The DiT geometry (``dim`` / ``num_layers`` / ``num_heads`` / ``ffn_dim`` / ...) is
loaded from the official Wan2.2-5B config at build time; the model code reads it
off this object so a backbone swap is a config change, not a code edit. The
numbers below are NOT defaults baked into the model — ``WanConfig`` requires them
explicitly; ``configs/model/latent_wam_dit.yaml`` holds the authoritative values
for the server and ``from_yaml`` loads them.

Sanity (per DreamZero ``WAN22_BACKBONE.md`` / ``wan22`` head config):
    dim=3072, num_layers=30, num_heads=24, ffn_dim=14336, freq_dim=256,
    eps=1e-6, in_dim=48, out_dim=48  (head_dim = 3072/24 = 128).
``in_dim``/``out_dim`` are the Wan *VAE* latent channels — we NEVER use them (the
VAE patch-embed / pixel head are not loaded); they are kept only to assert the
checkpoint we got is the 5B we expect.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WanConfig:
    dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    freq_dim: int = 256
    text_dim: int = 4096          # umT5-XXL hidden (cross-attn context dim)
    eps: float = 1e-6
    qk_norm: bool = True
    cross_attn_norm: bool = True
    # VAE-side dims kept only for checkpoint sanity (never used in our forward)
    in_dim: int = 48
    out_dim: int = 48

    def __post_init__(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim {self.dim} not divisible by num_heads {self.num_heads}")
        if self.head_dim % 2 != 0:
            # Wan asserts (dim//num_heads) % 2 == 0; this guarantees each 3-D RoPE
            # split (time/h/w) is even (see models/rope.py::Rope3D).
            raise ValueError(f"head_dim {self.head_dim} must be even (Wan RoPE)")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @classmethod
    def from_dict(cls, d: dict) -> "WanConfig":
        """Build from a flat mapping (e.g. the ``diffusion_model_cfg`` block of the
        official Wan config). Unknown keys are ignored so the full Wan config can
        be passed through verbatim.
        """
        fields = {
            "dim", "num_layers", "num_heads", "ffn_dim", "freq_dim", "text_dim",
            "eps", "qk_norm", "cross_attn_norm", "in_dim", "out_dim",
        }
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def from_yaml(cls, path: str, key_path: tuple[str, ...] = ()) -> "WanConfig":
        """Load from a YAML file (server: the official Wan config). ``key_path``
        descends into nested blocks (e.g. ``("diffusion_model_cfg",)``). Requires
        PyYAML (present on the server); tests use :meth:`from_dict` directly.
        """
        import yaml  # local import: keep the package importable without PyYAML

        with open(path) as f:
            data = yaml.safe_load(f)
        for k in key_path:
            data = data[k]
        return cls.from_dict(data)


# The known-good Wan2.2-TI2V-5B geometry, exposed as a *named constant* used ONLY
# for fail-loud assertions (not as a silent default fed to the model).
WAN22_TI2V_5B = WanConfig(
    dim=3072, num_layers=30, num_heads=24, ffn_dim=14336, freq_dim=256,
    eps=1e-6, in_dim=48, out_dim=48,
)
