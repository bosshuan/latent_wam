"""Stage A robot-only unified latent-action flow training (Milestone 5).

The pure, unit-testable core is :func:`unified_train_step` (build noisy
latent/action, forward, assemble L_A incl ``L_cf`` + two-step rollout, compute the
``S_a``/``Δ_cond`` collapse monitors). :func:`validate` runs the same forwards
under no_grad and raises the collapse alarm. :func:`main` is the server entry
point (torchrun DDP/FSDP, single-GPU degrade, EMA, checkpoint).

Flow direction is the PROJECT convention everywhere (``t=0`` noise, ``t=1`` data —
CLAUDE.md §2 invariant 1). The counterfactual / no-action forwards reuse the SAME
sampled noise + timestep so the only changing variable is the conditioning
action (doc §2.7). The Stage-A main loader is robot-only; the pure train step
still accepts ``actions=None`` so old actionless-video ablations remain testable.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Optional

import torch

from data.mixed_batch_sampler import CurriculumSchedule, CurriculumStage
from flow.interpolation import interpolate, make_noisy, predict_x1
from flow.losses import (
    UnifiedLossWeights,
    action_sensitivity,
    assemble_unified_loss,
    collapse_alarm,
    counterfactual_loss,
    counterfactual_action_delta,
    delta_cond,
    flow_distance,
    permute_actions_within_schema,
)
from flow.rollout import two_step_rollout_loss
from flow.schedulers import TimestepScheduler
from train import dist_utils

# Collapse-alarm thresholds (CLAUDE.md §2.9 / configs/unified_pretrain.yaml).
MIN_ACTION_SENSITIVITY = 0.01


@dataclass
class StepInputs:
    """Already-shaped model inputs for one homogeneous batch.

    ``context_latent`` [B,T_ctx,N,C] clean frozen VJ-RAE history latents;
    ``r1`` [B,T_fut,N,C] future clean latents = the flow target; ``actions``
    [B,T_fut,n_act,A] clean future action chunk (robot) or None (video);
    ``a_mask`` [B,T_fut,n_act] step/pad mask; routing/condition fields as in the
    DiT forward.
    """

    context_latent: torch.Tensor
    r1: torch.Tensor
    action_valid: torch.Tensor
    embodiment_id: torch.Tensor
    action_schema_id: torch.Tensor
    actions: Optional[torch.Tensor] = None
    a_mask: Optional[torch.Tensor] = None
    proprio: Optional[torch.Tensor] = None
    text: Optional[list[str]] = None

    def to(self, device: torch.device | str) -> "StepInputs":
        def mv(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if x is None else x.to(device)

        return StepInputs(
            context_latent=mv(self.context_latent),
            r1=mv(self.r1),
            action_valid=mv(self.action_valid),
            embodiment_id=mv(self.embodiment_id),
            action_schema_id=mv(self.action_schema_id),
            actions=mv(self.actions),
            a_mask=mv(self.a_mask),
            proprio=mv(self.proprio),
            text=None if self.text is None else list(self.text),
        )


def _apply_curriculum(weights: UnifiedLossWeights, stage: CurriculumStage) -> UnifiedLossWeights:
    w = copy.copy(weights)
    w.lambda_a = weights.lambda_a * stage.action_weight
    w.lambda_roll = weights.lambda_roll * stage.rollout_weight
    return w


def unified_train_step(
    model,
    inp: StepInputs,
    scheduler: TimestepScheduler,
    weights: UnifiedLossWeights,
    stage: Optional[CurriculumStage] = None,
    generator: Optional[torch.Generator] = None,
    compute_monitors: bool = True,
) -> tuple[torch.Tensor, dict]:
    """One forward/loss step. Returns ``(total_loss, metrics)``.

    ``metrics`` carries the scalar loss terms plus ``S_a``/``Δ_cond`` and any
    collapse-alarm strings (robot batches only).
    """
    if stage is not None:
        weights = _apply_curriculum(weights, stage)

    device = inp.context_latent.device
    b = inp.context_latent.shape[0]
    robot = inp.actions is not None

    t_z, t_a = scheduler.sample(b, device=device, generator=generator)

    x_t_z, noise_z, u_z = make_noisy(inp.r1, t_z, generator=generator)

    fwd = dict(
        context_latent=inp.context_latent,
        noisy_latent=x_t_z,
        latent_timestep=t_z,
        action_valid=inp.action_valid,
        embodiment_id=inp.embodiment_id,
        proprio=inp.proprio,
        text=inp.text,
    )

    v_a = u_a = a_mask = l_cf = None
    metrics: dict = {}

    if robot:
        x_t_a, noise_a, u_a = make_noisy(inp.actions, t_a, generator=generator)
        fwd.update(noisy_action=x_t_a, action_timestep=t_a)

    out = model(**fwd)
    r1_hat = predict_x1(x_t_z, out.latent_velocity, t_z)

    if robot:
        v_a = out.action_velocity
        a_mask = inp.a_mask
        # --- counterfactual forward: true action vs permuted action ---
        a_perm, cf_valid = permute_actions_within_schema(
            inp.actions,
            inp.action_schema_id,
            inp.action_valid,
            generator=generator,
            mode=weights.counterfactual_action_mode,
            min_action_delta=weights.min_counterfactual_action_delta,
        )
        cf_mode = str(weights.cf_action_mode)
        if cf_mode == "noisy":
            # Same action noise/timestep as the main forward. This matches joint
            # flow training, but the action semantic signal can be weak when t_a
            # is low.
            r1_hat_true_cf = r1_hat
            x_t_a_cf = interpolate(noise_a, a_perm, t_a)
            fwd_cf = dict(fwd)
            fwd_cf["noisy_action"] = x_t_a_cf
        elif cf_mode == "clean":
            # Diagnostic / strong anti-collapse path: compare clean true action
            # to clean permuted action while keeping latent noise/t_z fixed. This
            # asks whether the latent branch can use action semantics at all.
            t_a_cf = torch.ones_like(t_a)
            fwd_true_cf = dict(fwd)
            fwd_true_cf.update(noisy_action=inp.actions, action_timestep=t_a_cf)
            out_true_cf = model(**fwd_true_cf)
            r1_hat_true_cf = predict_x1(x_t_z, out_true_cf.latent_velocity, t_z)
            fwd_cf = dict(fwd_true_cf)
            fwd_cf["noisy_action"] = a_perm
        else:
            raise ValueError(f"unknown cf_action_mode {cf_mode!r}; expected 'noisy' or 'clean'")
        out_cf = model(**fwd_cf)
        r1_hat_cf = predict_x1(x_t_z, out_cf.latent_velocity, t_z)

        d_true = flow_distance(r1_hat_true_cf, inp.r1, kind=weights.cf_distance)
        d_perm = flow_distance(r1_hat_cf, inp.r1, kind=weights.cf_distance)
        l_cf = counterfactual_loss(d_true, d_perm, weights.cf_delta, cf_valid)
        metrics["cf_action_mode"] = cf_mode

        if compute_monitors:
            # no-action forward for Δ_cond: drop ONLY the action branch, KEEP proprio
            # so Δ_cond = d(r̂1^no-act, r1) - d(r̂1(a), r1) isolates the ACTION
            # contribution (doc §2.7). Dropping proprio here would let a strong state
            # mask an action collapse -> false-negative alarm. proprio/state is gated
            # on proprio-presence (not has_action) in the DiT, so it survives.
            with torch.no_grad():
                monitor_mode = str(weights.monitor_cf_action_mode or cf_mode)
                mon_true = r1_hat_true_cf
                mon_perm = r1_hat_cf
                mon_d_true = d_true
                if monitor_mode != cf_mode:
                    if monitor_mode == "noisy":
                        mon_true = r1_hat
                        x_t_a_mon = interpolate(noise_a, a_perm, t_a)
                        fwd_mon = dict(fwd)
                        fwd_mon["noisy_action"] = x_t_a_mon
                        out_mon = model(**fwd_mon)
                        mon_perm = predict_x1(x_t_z, out_mon.latent_velocity, t_z)
                    elif monitor_mode == "clean":
                        t_a_mon = torch.ones_like(t_a)
                        fwd_mon_true = dict(fwd)
                        fwd_mon_true.update(noisy_action=inp.actions, action_timestep=t_a_mon)
                        out_mon_true = model(**fwd_mon_true)
                        mon_true = predict_x1(x_t_z, out_mon_true.latent_velocity, t_z)
                        fwd_mon_perm = dict(fwd_mon_true)
                        fwd_mon_perm["noisy_action"] = a_perm
                        out_mon_perm = model(**fwd_mon_perm)
                        mon_perm = predict_x1(x_t_z, out_mon_perm.latent_velocity, t_z)
                    else:
                        raise ValueError(
                            f"unknown monitor_cf_action_mode {monitor_mode!r}; "
                            "expected '', 'noisy', or 'clean'"
                        )
                    mon_d_true = flow_distance(mon_true, inp.r1, kind=weights.cf_distance)

                fwd_na = dict(fwd)
                fwd_na.update(noisy_action=None, action_timestep=None)
                fwd_na["action_valid"] = torch.zeros_like(inp.action_valid)
                # NOTE: proprio kept (do NOT set None)
                out_na = model(**fwd_na)
                r1_hat_na = predict_x1(x_t_z, out_na.latent_velocity, t_z)
                d_noact = flow_distance(r1_hat_na, inp.r1, kind=weights.cf_distance)
                s_a = action_sensitivity(
                    mon_true, mon_perm, kind=weights.cf_distance, cf_valid=cf_valid
                )
                s_a_cos = action_sensitivity(
                    mon_true, mon_perm, kind="cos", cf_valid=cf_valid
                )
                dcond = delta_cond(d_noact, mon_d_true)
                cf_act_delta = counterfactual_action_delta(
                    inp.actions, a_perm, a_mask=inp.a_mask, cf_valid=cf_valid
                )
            metrics["S_a"] = float(s_a)
            metrics["S_a_cos"] = float(s_a_cos)
            metrics["delta_cond"] = float(dcond)
            cf_valid_frac = float(cf_valid.float().mean())
            metrics["cf_valid_frac"] = cf_valid_frac
            metrics["cf_action_delta"] = float(cf_act_delta)
            metrics["monitor_cf_action_mode"] = monitor_mode
            metrics["cf_alarm_inconclusive"] = float(
                cf_valid_frac < weights.min_counterfactual_valid_frac_for_alarm
            )
            if cf_valid_frac >= weights.min_counterfactual_valid_frac_for_alarm:
                metrics["alarms"] = collapse_alarm(
                    s_a, dcond,
                    min_action_sensitivity=MIN_ACTION_SENSITIVITY,
                    require_delta_cond_positive=True,
                )
            else:
                metrics["alarms"] = []

    # --- two-step rollout (curriculum-ramped) ---
    l_roll = None
    if weights.lambda_roll > 0.0:
        l_roll = two_step_rollout_loss(model, fwd, inp.r1, t_z, weights.lambda_cos)

    terms = assemble_unified_loss(
        out.latent_velocity, u_z, r1_hat, inp.r1, weights,
        v_a=v_a, u_a=u_a, a_mask=a_mask, l_cf=l_cf, l_roll=l_roll,
    )
    for k, v in terms.items():
        metrics[k] = float(v.detach())
    return terms["total"], metrics


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class EmaModel:
    """Exponential moving average of the trainable params (eval/checkpoint copy)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, p in model.named_parameters():
            if p.requires_grad and k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for k, p in model.named_parameters():
            if k in self.shadow:
                p.copy_(self.shadow[k])


# ---------------------------------------------------------------------------
# Validation (collapse monitor)
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    model,
    batches: list[StepInputs],
    scheduler: TimestepScheduler,
    weights: UnifiedLossWeights,
) -> dict:
    """Average validation losses + S_a/Δ_cond, with a collapse-alarm flag (§2.9).

    ``collapse`` is based on the averaged validation monitors. Per-batch alarms
    are still reported separately as ``batch_alarm_count`` so a single hard
    validation window does not make the printed average look contradictory.
    """
    model.eval()
    agg: dict = {
        "total": 0.0,
        "S_a": 0.0,
        "S_a_cos": 0.0,
        "delta_cond": 0.0,
        "cf_valid_frac": 0.0,
        "cf_action_delta": 0.0,
        "n": 0,
        "n_robot": 0,
        "alarms": [],
    }
    for inp in batches:
        loss, m = unified_train_step(model, inp, scheduler, weights, compute_monitors=True)
        agg["total"] += m["total"]
        agg["n"] += 1
        if "S_a" in m:
            agg["S_a"] += m["S_a"]
            agg["S_a_cos"] += m.get("S_a_cos", 0.0)
            agg["delta_cond"] += m["delta_cond"]
            agg["cf_valid_frac"] += m.get("cf_valid_frac", 0.0)
            agg["cf_action_delta"] += m.get("cf_action_delta", 0.0)
            agg["n_robot"] += 1
            agg["alarms"].extend(m.get("alarms", []))
    agg["total"] /= max(agg["n"], 1)
    if agg["n_robot"]:
        agg["S_a"] /= agg["n_robot"]
        agg["S_a_cos"] /= agg["n_robot"]
        agg["delta_cond"] /= agg["n_robot"]
        agg["cf_valid_frac"] /= agg["n_robot"]
        agg["cf_action_delta"] /= agg["n_robot"]
    avg_alarms = []
    if agg["n_robot"] and agg["cf_valid_frac"] >= weights.min_counterfactual_valid_frac_for_alarm:
        avg_alarms = collapse_alarm(
            torch.tensor(agg["S_a"]),
            torch.tensor(agg["delta_cond"]),
            min_action_sensitivity=MIN_ACTION_SENSITIVITY,
            require_delta_cond_positive=True,
        )
    agg["cf_alarm_inconclusive"] = bool(
        agg["n_robot"] and agg["cf_valid_frac"] < weights.min_counterfactual_valid_frac_for_alarm
    )
    agg["batch_alarm_count"] = len(agg["alarms"])
    agg["avg_alarms"] = avg_alarms
    agg["collapse"] = len(avg_alarms) > 0
    return agg


# ---------------------------------------------------------------------------
# Server entry point (not exercised on CPU)
# ---------------------------------------------------------------------------


class SyntheticRobotStepLoader:
    """Random robot-only StepInputs for 8-GPU debug-server bringup.

    This exercises the same model/loss/optimizer/checkpoint path as real Stage A
    without requiring LeRobot, V-JEPA, Wan weights, or cached latents. It is not a
    scientific training run; it is an infrastructure smoke train.
    """

    def __init__(self, cfg: dict, ctx: Optional[dist_utils.DistContext] = None, split: str = "train") -> None:
        syn = _cfg_get(cfg, "synthetic", default={})
        self.batch_size = int(_cfg_get(syn, "batch_size", default=2))
        self.num_steps = int(_cfg_get(syn, "num_steps", default=8 if split == "train" else 2))
        self.context_chunks = int(_cfg_get(syn, "context_chunks", default=4))
        self.future_chunks = int(_cfg_get(syn, "future_chunks", default=4))
        self.grid_n = int(_cfg_get(syn, "grid_n", default=144))
        self.latent_dim = int(_cfg_get(syn, "latent_dim", default=384))
        self.actions_per_chunk = int(_cfg_get(syn, "actions_per_chunk", default=4))
        self.action_dim = int(_cfg_get(syn, "action_dim", default=7))
        self.state_dim = int(_cfg_get(syn, "state_dim", default=8))
        self.num_embodiments = int(_cfg_get(syn, "num_embodiments", default=2))
        seed = int(_cfg_get(cfg, "seed", default=0)) + (17 if split == "val" else 0)
        rank = 0 if ctx is None else ctx.rank
        self.generator = torch.Generator()
        self.generator.manual_seed(seed + 1009 * rank)

    def __iter__(self):
        for _ in range(self.num_steps):
            b = self.batch_size
            context = torch.randn(
                b, self.context_chunks, self.grid_n, self.latent_dim, generator=self.generator
            )
            actions = torch.randn(
                b,
                self.future_chunks,
                self.actions_per_chunk,
                self.action_dim,
                generator=self.generator,
            )
            # A simple action-dependent latent target makes the synthetic run less
            # degenerate and gives S_a/Δ_cond something real to react to.
            action_signal = actions.mean(dim=(2, 3), keepdim=False).view(b, self.future_chunks, 1, 1)
            noise = torch.randn(
                b, self.future_chunks, self.grid_n, self.latent_dim, generator=self.generator
            )
            r1 = 0.8 * noise + 0.2 * action_signal
            proprio = None
            if self.state_dim > 0:
                proprio = torch.randn(b, self.state_dim, generator=self.generator)
            yield StepInputs(
                context_latent=context,
                r1=r1,
                action_valid=torch.ones(b, dtype=torch.bool),
                embodiment_id=torch.zeros(b, dtype=torch.long),
                action_schema_id=torch.zeros(b, dtype=torch.long),
                actions=actions,
                a_mask=torch.ones(b, self.future_chunks, self.actions_per_chunk, dtype=torch.bool),
                proprio=proprio,
                text=[""] * b,
            )

    def __len__(self) -> int:
        return self.num_steps


def build_train_objects(cfg, ctx: Optional[dist_utils.DistContext] = None):  # pragma: no cover - server hook
    """Build model, sampler, train loader, validation batches, and frozen VJ-RAE.

    ``data.source: synthetic`` is fully wired for distributed debug. Real data
    training should replace this hook with LeRobot + cached frozen VJ-RAE latents.
    """
    source = _cfg_get(cfg, "data", "source", default="synthetic")
    if source != "synthetic":
        raise NotImplementedError(
            "real robot dataset wiring is server-specific: build LeRobot v2.1 "
            "datasets, encode/cache frozen VJ-RAE latents, and yield StepInputs; "
            "use data.source=synthetic to debug the 8-GPU training path now"
        )

    from models.latent_world_action_dit import LatentWorldActionDiT

    syn = _cfg_get(cfg, "synthetic", default={})
    model_cfg = _cfg_get(cfg, "model", default={})
    model = LatentWorldActionDiT(
        latent_dim=int(_cfg_get(syn, "latent_dim", default=384)),
        action_dim=int(_cfg_get(syn, "action_dim", default=7)),
        hidden_dim=int(_cfg_get(model_cfg, "hidden_dim", default=128)),
        depth=int(_cfg_get(model_cfg, "depth", default=2)),
        heads=int(_cfg_get(model_cfg, "heads", default=4)),
        num_embodiments=int(_cfg_get(syn, "num_embodiments", default=2)),
        grid_n=int(_cfg_get(syn, "grid_n", default=144)),
        max_chunks=int(_cfg_get(model_cfg, "max_chunks", default=16)),
        max_actions=int(_cfg_get(syn, "actions_per_chunk", default=4)),
        state_dim=int(_cfg_get(syn, "state_dim", default=8)),
        text_dim=0,
        adaln_gate_init=float(_cfg_get(model_cfg, "adaln_gate_init", default=0.05)),
        action_token_scale=float(_cfg_get(model_cfg, "action_token_scale", default=1.0)),
        action_latent_bridge_scale=float(
            _cfg_get(model_cfg, "action_latent_bridge_scale", default=0.0)
        ),
    )
    train_loader = SyntheticRobotStepLoader(cfg, ctx, split="train")
    val_batches = list(SyntheticRobotStepLoader(cfg, ctx, split="val"))
    return model, None, train_loader, val_batches, None


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


def main():  # pragma: no cover - requires GPUs/data
    from train.dist_utils import (
        barrier,
        cleanup,
        init_distributed,
        save_checkpoint,
        set_seed,
        unwrap,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unified_pretrain.yaml")
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

    dcfg = cfg.get("distributed", {})
    ctx = init_distributed(backend=dcfg.get("backend", "nccl"))
    set_seed(int(cfg.get("seed", 0)), ctx)
    if ctx.is_main:
        print(
            f"[unified] start config={args.config} "
            f"world_size={ctx.world_size} device={ctx.device}",
            flush=True,
        )

    model, sampler, train_loader, val_batches, vj_rae = build_train_objects(cfg, ctx)
    model = model.to(ctx.device)
    if val_batches is not None:
        val_batches = [b.to(ctx.device) for b in val_batches]

    # Frozen modules (CLAUDE.md §2.2) stay eval-only and are EXCLUDED from the
    # FSDP shard: the V-JEPA encoder + VJ-RAE are external (not submodules of
    # ``model``), and the umT5 text encoder lives at ``model.text_encoder`` — we
    # hand it to FSDP ``ignored_modules`` so its (frozen) params aren't sharded or
    # gradient-tracked. activation checkpointing + bf16 mixed precision per cfg.
    if ctx.distributed and dcfg.get("unified_strategy", "fsdp") == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        mp = None
        if dcfg.get("precision", "bf16") == "bf16":
            mp = MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
            )
        ignored = []
        if hasattr(model, "text_encoder"):
            ignored.append(model.text_encoder)  # frozen umT5 — never shard/train
        if dcfg.get("activation_checkpointing", True):
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                apply_activation_checkpointing,
            )
            from models.wan_blocks import WanAttentionBlock

            apply_activation_checkpointing(
                model, check_fn=lambda m: isinstance(m, WanAttentionBlock)
            )
        model = FSDP(model, mixed_precision=mp, ignored_modules=ignored or None)
    elif ctx.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[ctx.local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )

    import dataclasses

    loss_cfg = cfg.get("loss", {})
    _wfields = {f.name for f in dataclasses.fields(UnifiedLossWeights)}
    weights = UnifiedLossWeights(**{k: v for k, v in loss_cfg.items() if k in _wfields})
    # initial coupling; the curriculum overrides ``scheduler.coupled`` per step.
    scheduler = TimestepScheduler(coupled=cfg.get("timestep", {}).get("coupled", True))
    curriculum = CurriculumSchedule()
    ema = EmaModel(unwrap(model))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("lr", 1e-4)),
    )

    total_steps = int(cfg.get("total_steps", 100000))
    for step, batch in enumerate(train_loader):
        progress = step / max(total_steps, 1)
        stage = curriculum.at(progress)
        if sampler is not None and hasattr(sampler, "set_video_ratio"):
            sampler.set_video_ratio(stage.video_ratio)
        scheduler.coupled = stage.coupled  # coupled -> decoupled timestep schedule
        inp = batch.to(ctx.device)
        loss, metrics = unified_train_step(model, inp, scheduler, weights, stage)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(unwrap(model))
        if ctx.is_main and step % int(cfg.get("log_every", 50)) == 0:
            print(f"step {step} loss {metrics['total']:.4f} "
                  f"S_a {metrics.get('S_a', float('nan')):.4f} "
                  f"Δcond {metrics.get('delta_cond', float('nan')):.4f} "
                  f"cf_action_delta {metrics.get('cf_action_delta', float('nan')):.4f}", flush=True)
        if step % int(cfg.get("val_every", 1000)) == 0 and val_batches is not None:
            vm = validate(model, val_batches, scheduler, weights)
            if ctx.is_main:
                print(f"[val] total {vm['total']:.4f} S_a {vm['S_a']:.4f} "
                      f"S_a_cos {vm['S_a_cos']:.4f} Δcond {vm['delta_cond']:.4f} "
                      f"cf_valid {vm['cf_valid_frac']:.4f} "
                      f"cf_action_delta {vm['cf_action_delta']:.4f} "
                      f"cf_inconclusive {vm['cf_alarm_inconclusive']} "
                      f"batch_alarms {vm['batch_alarm_count']} "
                      f"collapse={vm['collapse']}", flush=True)
                if vm["collapse"]:
                    print("[ALARM] action-conditioning collapse:", *vm["avg_alarms"], sep="\n  ", flush=True)
            model.train()
        if step + 1 >= total_steps:
            break

    barrier(ctx)
    ckpt_path = save_checkpoint(
        {"model": unwrap(model).state_dict(), "ema": ema.shadow}, cfg.get("out_path", "./ckpt.pt"), ctx
    )
    if ctx.is_main:
        print(f"[unified] saved checkpoint: {ckpt_path}", flush=True)
    cleanup(ctx)


if __name__ == "__main__":  # pragma: no cover
    main()
