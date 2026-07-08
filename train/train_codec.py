"""Train the V-JEPA Representation Autoencoder (VJ-RAE) — torchrun + DDP.

Run (multi-GPU):   torchrun --nproc_per_node=8 -m train.train_vj_rae --config ...
Run (single/CPU):  python -m train.train_vj_rae --config ...  (degrades, smoke)

Pipeline: frozen V-JEPA features (cached) -> FixedFeatureNormalizer (stats fit
on paired robot video+action data for Stage A) -> VJ-RAE encode/decode ->
reconstruction/variance/covariance loss (+ training-only action-discriminability
probe). The VJ-RAE is **frozen** at the end (CLAUDE.md §2.2 / doc §4.1).

The numerically-heavy bits (`fit_normalizer`, `codec_train_step`) are factored
out as pure functions so they are unit-testable on CPU with tiny tensors. For
debug-server bringup, ``data.source: synthetic`` runs an actual DDP training
loop on random robot-shaped features without touching V-JEPA/LeRobot. Real data
wiring remains isolated behind ``build_dataloaders``.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.schemas import MultiLevelFeatures
from flow.losses import CodecLossWeights, codec_loss
from models.vj_rae import (
    ActionDiscriminabilityProbe,
    FixedFeatureNormalizer,
    VJEPRepresentationAutoencoder,
    pooled_latent_delta,
)
from models.vjepa_encoder import FrozenVJEPAEncoder
from train import dist_utils


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_codec_from_encoder(
    encoder: FrozenVJEPAEncoder,
    hidden_dim: int = 1024,
    latent_dim: int = 384,
    pool: int = 2,
    norm_version: str = "v0",
) -> VJEPRepresentationAutoencoder:
    """Construct the VJ-RAE using the encoder's runtime dims (no hardcoding).

    `codec_in_dim = L*D` is implied by (num_layers, embed_dim) read from the
    encoder — never written by hand (CLAUDE.md §3 / user note).
    """
    normalizer = FixedFeatureNormalizer(
        encoder.num_layers, encoder.embed_dim, norm_version=norm_version
    )
    assert encoder.codec_in_dim == encoder.num_layers * encoder.embed_dim
    return VJEPRepresentationAutoencoder(
        num_layers=encoder.num_layers,
        embed_dim=encoder.embed_dim,
        grid_hw=encoder.grid_hw,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        pool=pool,
        normalizer=normalizer,
    )


# ---------------------------------------------------------------------------
# Normalizer fit
# ---------------------------------------------------------------------------


def fit_normalizer(
    codec: VJEPRepresentationAutoencoder,
    feature_iter: Iterable[MultiLevelFeatures],
    ctx: Optional[dist_utils.DistContext] = None,
) -> None:
    """Accumulate normalizer stats over a Stage-A robot feature stream.

    For compatibility, ``feature_iter`` may yield either ``MultiLevelFeatures``
    or the training tuple ``(features, action_per_transition, m_a)``. Across ranks
    the accumulation buffers are all-reduced before finalize.
    """
    norm = codec.normalizer
    for feats in feature_iter:
        if isinstance(feats, tuple):
            feats = feats[0]
        norm.update(feats.features)

    if ctx is not None and ctx.distributed and dist.is_initialized():
        for buf in (norm._sum, norm._sumsq, norm._count):
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    norm.finalize()


# ---------------------------------------------------------------------------
# Train step (pure, unit-testable)
# ---------------------------------------------------------------------------


def codec_train_step(
    codec: torch.nn.Module,
    feats: MultiLevelFeatures,
    optimizer: torch.optim.Optimizer,
    weights: CodecLossWeights,
    probe: Optional[ActionDiscriminabilityProbe] = None,
    action_per_transition: Optional[torch.Tensor] = None,
    m_a: Optional[torch.Tensor] = None,
) -> dict:
    """One optimization step. Returns detached scalar loss terms.

    ``action_per_transition`` is [B, T_tok-1, A] — the action aligned to each
    latent time-token transition (server pools the action chunk to per-transition
    before calling; the probe never sees a fabricated action — gate with m_a).
    """
    core = dist_utils.unwrap(codec)
    # Recon target = POOLED normalized features at 12x12 (user Q1): the VJ-RAE
    # only inverts the 144-token channel compression, not the lossy 2×2 pool.
    target = core.pooled_target(feats)  # [B,T,144,L,D]
    latent, recon = codec(feats)

    probe_pred = None
    if probe is not None and action_per_transition is not None:
        # dyn-term gradient path (user Q2): `latent.latent` (=R) carries codec
        # gradient, so MSE(probe(Δr), a) backprops into BOTH the probe AND the
        # VJ-RAE — intentional small-weight pressure (lambda_dyn=0.1) to keep it
        # from compressing away action-discriminative info. NO stop-grad (that
        # would train only the probe and remove the pressure). The probe is
        # discarded after VJ-RAE training.
        delta = pooled_latent_delta(latent.latent)  # [B, T-1, latent_dim]
        probe_pred = probe(delta)

    terms = codec_loss(
        recon=recon,
        target=target,
        latent=latent.latent,
        weights=weights,
        probe_pred=probe_pred,
        action_target=action_per_transition,
        m_a=m_a,
    )

    optimizer.zero_grad(set_to_none=True)
    terms["total"].backward()
    optimizer.step()
    return {k: float(v.detach()) for k, v in terms.items()}


# ---------------------------------------------------------------------------
# Entry point (server; not run locally)
# ---------------------------------------------------------------------------


class SyntheticVJRAELoader:
    """Robot-shaped random VJ-RAE batches for distributed debug bringup."""

    def __init__(self, cfg: dict, ctx: Optional[dist_utils.DistContext] = None) -> None:
        syn = _cfg_get(cfg, "synthetic", default={})
        self.batch_size = int(_cfg_get(syn, "batch_size", default=2))
        self.num_steps = int(_cfg_get(syn, "num_steps", default=8))
        self.tok_time = int(_cfg_get(syn, "time_tokens", default=8))
        self.grid_hw = tuple(_cfg_get(syn, "grid_hw", default=[24, 24]))
        self.num_layers = int(_cfg_get(syn, "num_layers", default=4))
        self.embed_dim = int(_cfg_get(syn, "embed_dim", default=64))
        self.action_dim = int(_cfg_get(syn, "action_dim", default=7))
        seed = int(_cfg_get(cfg, "seed", default=0))
        rank = 0 if ctx is None else ctx.rank
        self.generator = torch.Generator()
        self.generator.manual_seed(seed + 1009 * rank)

    def __iter__(self):
        h, w = self.grid_hw
        token_grid = (self.tok_time, h, w)
        n = h * w
        for _ in range(self.num_steps):
            features = torch.randn(
                self.batch_size,
                self.tok_time,
                n,
                self.num_layers,
                self.embed_dim,
                generator=self.generator,
            )
            action_pt = torch.randn(
                self.batch_size,
                self.tok_time - 1,
                self.action_dim,
                generator=self.generator,
            )
            m_a = torch.ones(self.batch_size, dtype=torch.bool)
            yield MultiLevelFeatures(features=features, token_grid=token_grid), action_pt, m_a


def build_dataloaders(cfg, ctx: Optional[dist_utils.DistContext] = None):  # pragma: no cover - server hook
    """Return VJ-RAE feature batches.

    ``data.source: synthetic`` is fully wired for debug-server bringup. Real
    LeRobot/V-JEPA feature-cache construction is intentionally left as the server
    integration point, because dataset mounts and cache paths are environment
    specific.
    """
    source = _cfg_get(cfg, "data", "source", default="synthetic")
    if source == "synthetic":
        return SyntheticVJRAELoader(cfg, ctx)
    raise NotImplementedError(
        "real VJ-RAE feature batches need the server dataset/cache wiring; "
        "use data.source=synthetic for distributed debug bringup"
    )


def _cfg_get(cfg, *path, default=None):
    cur = cfg
    for key in path:
        if cur is None:
            return default
        if hasattr(cur, "get"):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def _features_to_device(feats: MultiLevelFeatures, device: torch.device) -> MultiLevelFeatures:
    return MultiLevelFeatures(feats.features.to(device), feats.token_grid)


def main():  # pragma: no cover - requires GPUs/data
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="force data.source=synthetic for 8-GPU debug bringup",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.synthetic:
        cfg.setdefault("data", {})["source"] = "synthetic"

    ctx = dist_utils.init_distributed(
        backend=_cfg_get(cfg, "distributed", "backend", default="nccl")
    )
    dist_utils.set_seed(int(_cfg_get(cfg, "seed", default=0)), ctx)

    vj_cfg = _cfg_get(cfg, "vj_rae", default=_cfg_get(cfg, "codec", default={}))
    source = _cfg_get(cfg, "data", "source", default="synthetic")
    if source == "synthetic":
        syn = _cfg_get(cfg, "synthetic", default={})
        num_layers = int(_cfg_get(syn, "num_layers", default=4))
        embed_dim = int(_cfg_get(syn, "embed_dim", default=64))
        grid_hw = tuple(_cfg_get(syn, "grid_hw", default=[24, 24]))
        normalizer = FixedFeatureNormalizer(
            num_layers,
            embed_dim,
            norm_version=_cfg_get(cfg, "feature_cache", "norm_version", default="v0"),
        )
        codec = VJEPRepresentationAutoencoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            grid_hw=grid_hw,
            hidden_dim=int(_cfg_get(vj_cfg, "hidden_dim", default=256)),
            latent_dim=int(_cfg_get(vj_cfg, "latent_dim", default=128)),
            pool=int(_cfg_get(vj_cfg, "pool", default=2)),
            normalizer=normalizer,
        ).to(ctx.device)
    else:
        encoder = FrozenVJEPAEncoder(
            hub_name=_cfg_get(cfg, "encoder", "hub_name", default="vjepa2_1_vit_gigantic_384"),
            hub_repo=_cfg_get(cfg, "encoder", "hub_repo", default="facebookresearch/vjepa2"),
            hub_source=_cfg_get(cfg, "encoder", "hub_source", default="github"),
            extract_layers=tuple(_cfg_get(cfg, "encoder", "extract_layers", default=[11, 23, 37, 47])),
            pretrained=bool(_cfg_get(cfg, "encoder", "pretrained", default=True)),
            checkpoint_path=_cfg_get(cfg, "encoder", "checkpoint_path", default=None),
            checkpoint_key=_cfg_get(cfg, "encoder", "checkpoint_key", default="target_encoder"),
            checkpoint_strict=bool(_cfg_get(cfg, "encoder", "checkpoint_strict", default=True)),
            assert_gigantic=bool(_cfg_get(cfg, "encoder", "assert_gigantic", default=True)),
        ).to(ctx.device)

        codec = build_codec_from_encoder(
            encoder,
            hidden_dim=int(_cfg_get(vj_cfg, "hidden_dim", default=1024)),
            latent_dim=int(_cfg_get(vj_cfg, "latent_dim", default=384)),
            pool=int(_cfg_get(vj_cfg, "pool", default=2)),
            norm_version=_cfg_get(cfg, "feature_cache", "norm_version", default="v0"),
        ).to(ctx.device)

    train_iter = build_dataloaders(cfg, ctx)
    fit_normalizer(codec, train_iter, ctx)

    probe = ActionDiscriminabilityProbe(
        int(_cfg_get(vj_cfg, "latent_dim", default=384)),
        int(_cfg_get(vj_cfg, "probe_action_dim", default=7)),
    ).to(ctx.device)

    ddp_device_ids = [ctx.local_rank] if ctx.distributed and torch.cuda.is_available() else None
    model = DDP(codec, device_ids=ddp_device_ids) if ctx.distributed else codec
    probe_model = DDP(probe, device_ids=ddp_device_ids) if ctx.distributed else probe
    params = list(model.parameters()) + list(probe_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(_cfg_get(vj_cfg, "lr", default=1e-4)))
    weights = CodecLossWeights(**_cfg_get(vj_cfg, "loss", default={}))

    for step, (feats, action_pt, m_a) in enumerate(build_dataloaders(cfg, ctx)):
        feats = _features_to_device(feats, ctx.device)
        action_pt = action_pt.to(ctx.device)
        m_a = m_a.to(ctx.device)
        logs = codec_train_step(model, feats, optimizer, weights, probe_model, action_pt, m_a)
        if ctx.is_main and step % int(_cfg_get(vj_cfg, "log_every", default=50)) == 0:
            print(f"[vj-rae] step {step}: {logs}", flush=True)

    dist_utils.unwrap(model).freeze()  # freeze VJ-RAE after training
    dist_utils.save_checkpoint(
        {"vj_rae": dist_utils.unwrap(model).state_dict(),
         "codec": dist_utils.unwrap(model).state_dict(),  # backward-compatible key
         "normalizer_stats": dist_utils.unwrap(model).normalizer.stats_state()},
        _cfg_get(vj_cfg, "out_path", default="./checkpoints/vj_rae.pt"),
        ctx,
    )
    dist_utils.cleanup(ctx)


if __name__ == "__main__":  # pragma: no cover
    main()
