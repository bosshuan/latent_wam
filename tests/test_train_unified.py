"""M5 unified-training tests: L_A step (robot + video), monitors, EMA, KV stub.

Uses the tiny Wan-backed DiT (real M4 model). Tiny dims: dim=24, heads=2,
layers=2; latent=8, action=4, grid 2x2 (N=4); B=2, T_ctx=3, T_fut=2, n_act=2.
"""

from __future__ import annotations

import torch

from data.mixed_batch_sampler import CurriculumSchedule
from flow.schedulers import TimestepScheduler
from models.kv_cache import KVCache, KVCacheNotImplemented, LayerKV
from models.wan_config import WanConfig
from models.wan_latent_world_action_dit import WanLatentWorldActionDiT
from flow.losses import UnifiedLossWeights
from train.train_unified_flow import EmaModel, StepInputs, unified_train_step


def _model():
    cfg = WanConfig(dim=24, num_layers=2, num_heads=2, ffn_dim=48, freq_dim=16, text_dim=16)
    return WanLatentWorldActionDiT(
        cfg, latent_dim=8, action_dim=4, num_embodiments=3, grid_hw=(2, 2),
        max_chunks=8, max_actions=4, state_dim=5, text_seq_len=4,
    )


def _robot_inputs(b=2, t_ctx=3, t_fut=2, n=4, n_act=2):
    ctx = torch.randn(b, t_ctx, n, 8)
    r1 = torch.randn(b, t_fut, n, 8)
    return StepInputs(
        context_latent=ctx,
        r1=r1,
        action_valid=torch.ones(b, dtype=torch.bool),
        embodiment_id=torch.tensor([0, 1]),
        action_schema_id=torch.tensor([0, 0]),  # same schema -> counterfactual ok
        actions=torch.randn(b, t_fut, n_act, 4),
        a_mask=torch.ones(b, t_fut, n_act, dtype=torch.bool),
        proprio=torch.randn(b, 5),
        text=["pick", "place"],
    )


def _video_inputs(b=2, t_ctx=3, t_fut=2, n=4):
    ctx = torch.randn(b, t_ctx, n, 8)
    return StepInputs(
        context_latent=ctx,
        r1=torch.randn(b, t_fut, n, 8),
        action_valid=torch.zeros(b, dtype=torch.bool),
        embodiment_id=torch.tensor([0, 0]),
        action_schema_id=torch.tensor([-1, -1]),
        actions=None,
        text=["", ""],
    )


def _sched_weights():
    return TimestepScheduler(coupled=True), UnifiedLossWeights(lambda_roll=0.0)


def test_robot_step_has_all_terms_and_monitors():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    m = _model()
    sched, w = _sched_weights()
    loss, metrics = unified_train_step(m, _robot_inputs(), sched, w, generator=g)
    for k in ("z_fm", "clean", "a_fm", "cf", "total"):
        assert k in metrics, f"missing loss term {k}"
    assert "S_a" in metrics and "delta_cond" in metrics and "alarms" in metrics
    assert metrics["S_a"] > 0.0  # action moves the latent prediction
    assert loss.requires_grad


def test_clean_action_counterfactual_mode_runs():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    m = _model()
    sched = TimestepScheduler(coupled=True)
    w = UnifiedLossWeights(lambda_roll=0.0, cf_action_mode="clean")
    loss, metrics = unified_train_step(m, _robot_inputs(), sched, w, generator=g)
    assert metrics["cf_action_mode"] == "clean"
    assert "cf" in metrics and "S_a" in metrics
    assert loss.requires_grad


def test_video_step_omits_action_terms_zero_grad():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    m = _model()
    sched, w = _sched_weights()
    loss, metrics = unified_train_step(m, _video_inputs(), sched, w, generator=g)
    assert "a_fm" not in metrics and "cf" not in metrics and "S_a" not in metrics
    loss.backward()
    action_params = (
        list(m.action_encoder.parameters())
        + list(m.action_to_latent.parameters())
        + list(m.action_head.parameters())
    )
    assert all(p.grad is None for p in action_params)
    assert any(p.grad is not None for p in m.latent_head.parameters())


def test_single_batch_overfit():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    m = _model()
    sched, w = _sched_weights()
    inp = _robot_inputs()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = last = None
    for i in range(80):
        opt.zero_grad()
        loss, met = unified_train_step(m, inp, sched, w, generator=g, compute_monitors=False)
        loss.backward()
        opt.step()
        if i == 0:
            first = met["total"]
        last = met["total"]
    assert last < 0.7 * first, f"L_A did not drop enough: {first:.3f}->{last:.3f}"


def test_rollout_weight_adds_term():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(2)
    m = _model()
    sched = TimestepScheduler(coupled=True)
    w = UnifiedLossWeights(lambda_roll=1.0)
    _loss, metrics = unified_train_step(m, _robot_inputs(t_fut=2), sched, w, generator=g)
    assert "roll" in metrics  # T_fut=2 -> two-step rollout present


def test_rollout_none_when_single_future_chunk():
    from flow.rollout import two_step_rollout_loss

    torch.manual_seed(0)
    m = _model()
    b, t_ctx, t_fut, n = 2, 3, 1, 4  # only one future chunk -> no second step
    fwd = dict(
        context_latent=torch.randn(b, t_ctx, n, 8),
        noisy_latent=torch.randn(b, t_fut, n, 8),
        latent_timestep=torch.rand(b),
        action_valid=torch.zeros(b, dtype=torch.bool),
        text=["", ""],
    )
    r1 = torch.randn(b, t_fut, n, 8)
    assert two_step_rollout_loss(m, fwd, r1, fwd["latent_timestep"]) is None


def test_ema_tracks_params():
    torch.manual_seed(0)
    m = _model()
    ema = EmaModel(m, decay=0.9)
    # change a param, then EMA-update -> shadow moves toward it but not all the way
    with torch.no_grad():
        for p in m.latent_head.parameters():
            p.add_(1.0)
    before = next(iter(ema.shadow.values())).clone()
    ema.update(m)
    after = next(iter(ema.shadow.values()))
    assert not torch.equal(before, after)


def test_curriculum_progression():
    cur = CurriculumSchedule()
    early, mid, late = cur.at(0.05), cur.at(0.5), cur.at(0.95)
    assert early.video_ratio > late.video_ratio        # video-heavy -> less video
    assert late.action_weight > early.action_weight    # action weight ramps up
    assert early.coupled and not late.coupled           # coupled -> decoupled
    assert late.rollout_weight > 0.0                    # rollout enabled late


def test_kv_cache_stub_rejects_nonempty():
    torch.manual_seed(0)
    m = _model()
    inp = _video_inputs()
    cache = KVCache(layers=[LayerKV(k=torch.zeros(1), v=torch.zeros(1))], context_len=1)
    try:
        m(
            context_latent=inp.context_latent,
            noisy_latent=inp.r1,
            latent_timestep=torch.rand(2),
            action_valid=inp.action_valid,
            text=inp.text,
            kv_cache=cache,
        )
    except KVCacheNotImplemented:
        return
    raise AssertionError("non-empty KV cache must fail loud in Stage A (M8 feature)")
