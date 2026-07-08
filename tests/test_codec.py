"""M2 codec tests: normalizer stats, VICReg var/cov, probe shape, overfit step.

Local scope is shape + seeded-math only (CLAUDE.md §5). The action-discriminability
*accuracy* acceptance (post-codec Δr->a drop <= threshold) is a server eval.
"""

from __future__ import annotations

import torch

from data.schemas import MultiLevelFeatures
from flow.losses import (
    CodecLossWeights,
    covariance_loss,
    variance_loss,
)
from models.latent_codec import (
    ActionDiscriminabilityProbe,
    FixedFeatureNormalizer,
    VJEPALatentCodec,
)
from train.train_codec import codec_train_step


def test_normalizer_standardizes():
    torch.manual_seed(0)
    norm = FixedFeatureNormalizer(num_layers=2, embed_dim=4)
    # per-channel different scale/shift to check per-(L,D) stats
    x = torch.randn(4096, 2, 4) * 3.0 + 5.0
    norm.update(x)
    norm.finalize()
    z = norm.normalize(x)
    flat = z.reshape(-1, 2, 4)
    assert torch.allclose(flat.mean(dim=0), torch.zeros(2, 4), atol=1e-1)
    assert torch.allclose(flat.var(dim=0, unbiased=False), torch.ones(2, 4), atol=1e-1)
    # round-trip
    assert torch.allclose(norm.denormalize(z), x, atol=1e-3)
    assert bool(norm.fitted.item()) is True


def test_normalizer_has_no_grad():
    norm = FixedFeatureNormalizer(2, 4)
    assert all(not p.requires_grad for p in norm.parameters()) or not list(norm.parameters())


def test_variance_loss_penalizes_collapse():
    torch.manual_seed(0)
    collapsed = torch.randn(2000, 16) * 0.05  # tiny std
    healthy = torch.randn(2000, 16)            # std ~ 1
    assert variance_loss(collapsed) > variance_loss(healthy)


def test_covariance_loss_penalizes_correlation():
    torch.manual_seed(0)
    base = torch.randn(2000, 1)
    correlated = base.repeat(1, 8)             # all dims identical
    independent = torch.randn(2000, 8)
    assert covariance_loss(correlated) > covariance_loss(independent)


def test_probe_shape():
    probe = ActionDiscriminabilityProbe(latent_dim=12, action_dim=7)
    delta = torch.randn(3, 5, 12)              # [B, T-1, latent_dim]
    out = probe(delta)
    assert tuple(out.shape) == (3, 5, 7)


def test_codec_train_step_overfits_single_batch():
    torch.manual_seed(0)
    codec = VJEPALatentCodec(
        num_layers=2, embed_dim=8, grid_hw=(4, 4), hidden_dim=32, latent_dim=12
    )
    feats = MultiLevelFeatures(features=torch.randn(2, 3, 16, 2, 8), token_grid=(3, 4, 4))
    opt = torch.optim.AdamW(codec.parameters(), lr=1e-2)
    weights = CodecLossWeights()

    first = codec_train_step(codec, feats, opt, weights)["total"]
    for _ in range(60):
        last = codec_train_step(codec, feats, opt, weights)["total"]
    assert last < first  # single-batch overfit drives the loss down


def test_codec_train_step_dyn_gated_by_mask():
    """Probe/dyn term only contributes for robot rows (m_a)."""
    torch.manual_seed(0)
    codec = VJEPALatentCodec(
        num_layers=2, embed_dim=8, grid_hw=(4, 4), hidden_dim=32, latent_dim=12
    )
    probe = ActionDiscriminabilityProbe(latent_dim=12, action_dim=7)
    feats = MultiLevelFeatures(features=torch.randn(2, 3, 16, 2, 8), token_grid=(3, 4, 4))
    action_pt = torch.randn(2, 2, 7)           # [B, T-1, A]
    m_a = torch.tensor([1.0, 0.0])             # one robot, one video
    opt = torch.optim.AdamW(
        list(codec.parameters()) + list(probe.parameters()), lr=1e-3
    )
    logs = codec_train_step(
        codec, feats, opt, CodecLossWeights(), probe, action_pt, m_a
    )
    assert "dyn" in logs and logs["dyn"] >= 0.0
