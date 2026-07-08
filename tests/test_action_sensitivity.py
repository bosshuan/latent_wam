"""M3 anti-collapse tests (CLAUDE.md §2.9 / doc §2.7).

Covers the counterfactual machinery + monitors:
  * within-schema permutation builds a derangement and flags singleton schemas
    (cf_valid=0) instead of permuting a row against itself;
  * the hinge L_cf rewards true-action predictions that beat the permuted one;
  * S_a and Δ_cond monitors + the collapse alarm thresholds;
  * end-to-end S_a > 0 through the tiny DiT (action actually moves r̂1).
"""

from __future__ import annotations

import warnings

import torch

from flow.interpolation import predict_x1
from flow.losses import (
    action_sensitivity,
    collapse_alarm,
    counterfactual_loss,
    delta_cond,
    flow_distance,
    permute_actions_within_schema,
)
from models.adapters.action import MultiEmbodimentActionEncoder
from models.latent_world_action_dit import LatentWorldActionDiT


def test_permutation_is_derangement_within_schema():
    g = torch.Generator().manual_seed(0)
    # 4 rows: schema 5 has rows {0,1,2}, schema 9 has singleton {3}
    actions = torch.arange(4).float().reshape(4, 1).repeat(1, 3)  # distinct per row
    schema = torch.tensor([5, 5, 5, 9])
    valid = torch.ones(4, dtype=torch.bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm, cf_valid = permute_actions_within_schema(actions, schema, valid, generator=g)
    # the 3-member schema is permuted with no fixed point
    assert cf_valid[:3].all() and not cf_valid[3]
    for i in range(3):
        assert not torch.equal(perm[i], actions[i])  # derangement: no self
    # permuted rows are still a rearrangement of the schema's own actions
    assert {tuple(perm[i].tolist()) for i in range(3)} == {tuple(actions[i].tolist()) for i in range(3)}
    # singleton schema untouched
    assert torch.equal(perm[3], actions[3])


def test_permutation_warns_when_schema_too_small():
    actions = torch.randn(2, 3)
    schema = torch.tensor([1, 2])  # both singletons
    valid = torch.ones(2, dtype=torch.bool)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _perm, cf_valid = permute_actions_within_schema(actions, schema, valid)
    assert not cf_valid.any()
    assert any("L_cf" in str(x.message) for x in w)


def test_farthest_counterfactual_prefers_action_distant_rows():
    actions = torch.tensor([0.0, 1.0, 10.0]).reshape(3, 1, 1)
    schema = torch.tensor([1, 1, 1])
    valid = torch.ones(3, dtype=torch.bool)
    perm, cf_valid = permute_actions_within_schema(
        actions,
        schema,
        valid,
        mode="farthest",
    )
    assert cf_valid.all()
    # Row 0 should receive row 2's action, not the nearby row 1 action.
    assert torch.equal(perm[0], actions[2])
    # Row 2 should receive row 0's action.
    assert torch.equal(perm[2], actions[0])


def test_counterfactual_min_action_delta_skips_weak_swaps():
    actions = torch.tensor([0.0, 0.01, 5.0]).reshape(3, 1, 1)
    schema = torch.tensor([1, 1, 2])
    valid = torch.ones(3, dtype=torch.bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _perm, cf_valid = permute_actions_within_schema(
            actions,
            schema,
            valid,
            mode="farthest",
            min_action_delta=0.05,
        )
    assert cf_valid.tolist() == [False, False, False]


def test_video_rows_excluded_from_cf():
    actions = torch.randn(3, 4)
    schema = torch.tensor([7, 7, 7])
    valid = torch.tensor([True, True, False])  # one video row
    perm, cf_valid = permute_actions_within_schema(actions, schema, valid)
    # only the 2 robot rows are permutable; video row never enters cf
    assert cf_valid.tolist() == [True, True, False]


def test_counterfactual_hinge_behavior():
    # true action closer to r1 (smaller d) than permuted by > delta -> no penalty
    d_true = torch.tensor([0.1, 0.1])
    d_perm = torch.tensor([0.5, 0.5])
    cf_valid = torch.ones(2, dtype=torch.bool)
    assert counterfactual_loss(d_true, d_perm, delta=0.05, cf_valid=cf_valid).item() == 0.0
    # if true is NOT better, hinge is positive
    bad = counterfactual_loss(torch.tensor([0.5, 0.5]), torch.tensor([0.1, 0.1]), 0.05, cf_valid)
    assert bad.item() > 0.0


def test_action_sensitivity_and_delta_cond():
    r1 = torch.randn(2, 4, 8)
    r_true = r1 + 0.01 * torch.randn(2, 4, 8)   # good prediction
    r_perm = r1 + 0.5 * torch.randn(2, 4, 8)    # permuted action -> worse
    r_noact = r1 + 0.4 * torch.randn(2, 4, 8)   # no-action prediction

    s_a = action_sensitivity(r_true, r_perm)
    assert s_a.item() > 0.0  # predictions differ -> action is used

    d_act = flow_distance(r_true, r1)
    d_noact = flow_distance(r_noact, r1)
    dc = delta_cond(d_noact, d_act)
    assert dc.item() > 0.0  # conditioning on action helps


def test_action_encoder_preserves_action_content_at_init():
    torch.manual_seed(0)
    enc = MultiEmbodimentActionEncoder(action_dim=4, hidden_size=16, num_embodiments=1)
    t = torch.full((2,), 0.5)
    emb = torch.zeros(2, dtype=torch.long)
    actions = torch.stack([torch.zeros(3, 4), torch.ones(3, 4)], dim=0)
    out = enc(actions, t, emb)
    diff = (out[0] - out[1]).abs().mean()
    assert diff.item() > 1e-2


def test_collapse_alarm_thresholds():
    # healthy: no alarms
    assert collapse_alarm(torch.tensor(0.2), torch.tensor(0.1)) == []
    # S_a too low
    a = collapse_alarm(torch.tensor(0.001), torch.tensor(0.1))
    assert len(a) == 1 and "S_a" in a[0]
    # Δ_cond non-positive
    b = collapse_alarm(torch.tensor(0.2), torch.tensor(-0.01))
    assert len(b) == 1 and "Δ_cond" in b[0]


def test_end_to_end_action_sensitivity_through_dit():
    """S_a computed from real DiT forwards with true vs permuted actions is > 0
    (the M3 'robot batch S_a > threshold' acceptance, scaled to the tiny model)."""
    torch.manual_seed(0)
    dit = LatentWorldActionDiT(
        latent_dim=8, action_dim=4, hidden_dim=16, depth=2, heads=2,
        num_embodiments=3, grid_n=2, max_chunks=8, max_actions=4, state_dim=5,
    )
    b, t_ctx, t_fut, n, n_act = 2, 3, 2, 2, 2
    ctx = torch.randn(b, t_ctx, n, 8)
    z = torch.randn(b, t_fut, n, 8)
    t_z = torch.rand(b)
    t_a = torch.rand(b)
    actions = torch.randn(b, t_fut, n_act, 4)
    emb = torch.tensor([0, 1])

    common = dict(
        context_latent=ctx, noisy_latent=z, latent_timestep=t_z, action_timestep=t_a,
        action_valid=torch.ones(b, dtype=torch.bool), embodiment_id=emb,
        proprio=torch.randn(b, 5),
    )
    out_a = dit(noisy_action=actions, **common)
    out_p = dit(noisy_action=actions.flip(0), **common)
    r_true = predict_x1(z, out_a.latent_velocity, t_z)
    r_perm = predict_x1(z, out_p.latent_velocity, t_z)

    s_a = action_sensitivity(r_true, r_perm)
    assert s_a.item() > 1e-4
