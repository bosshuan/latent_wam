"""Losses — codec (M2) + unified flow (M3).

M2 codec pretraining loss (doc §2.3):
    L_codec = ||D(C(z)) - z||_1 + λ_cos (1 - cos(D(C(z)), z))
              + λ_var L_var + λ_cov L_cov + λ_dyn · m_a · ||g_φ(Δr) - a||²

M3 unified flow loss ``L_A`` (doc §4.2):
    L_A = λ_z L_z_FM + m_a λ_a L_a_FM + λ_clean L_clean
          + λ_roll L_roll + m_a λ_cf L_cf + m_v λ_v L_value   (λ_v = 0 this stage)

The variance/covariance terms are VICReg-style (Bardes et al., 2022) — they keep
the flow-space latent from collapsing. The flow-matching terms regress the
velocity ``v_theta`` onto ``u = x1 - x0`` (PROJECT convention, CLAUDE.md §2
invariant 1 — NOT the Wan/DreamZero ``noise - sample``). ``L_cf`` is the
anti-collapse counterfactual contrast (§2.7): the true-action latent prediction
must beat a within-schema permuted action by margin ``δ``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance hinge: encourage per-dim std >= gamma. z: [..., C]."""
    z = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance: push off-diagonal covariance to 0. z: [..., C]."""
    z = z.reshape(-1, z.shape[-1])
    n, c = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / c


def reconstruction_loss(
    recon: torch.Tensor, target: torch.Tensor, lambda_cos: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """L1 + cosine on the (normalized) feature reconstruction. [..., D] cos."""
    l1 = F.l1_loss(recon, target)
    cos = (1.0 - F.cosine_similarity(recon, target, dim=-1)).mean()
    return l1, lambda_cos * cos


@dataclass
class CodecLossWeights:
    lambda_cos: float = 1.0
    lambda_var: float = 1.0
    lambda_cov: float = 0.04
    # Small by design (user Q2): the dyn term backprops into the codec to keep
    # action-discriminative info, but a small weight avoids the codec overfitting
    # to please the probe.
    lambda_dyn: float = 0.1
    var_gamma: float = 1.0


def codec_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    latent: torch.Tensor,
    weights: CodecLossWeights,
    probe_pred: torch.Tensor | None = None,
    action_target: torch.Tensor | None = None,
    m_a: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Assemble the codec loss; returns a dict of named scalar terms + 'total'.

    ``probe_pred``/``action_target`` are the training-only action-discriminability
    probe outputs (doc §2.3); the dyn term is gated by ``m_a`` (robot only) and is
    skipped entirely when no probe is wired.
    """
    l1, cos = reconstruction_loss(recon, target, weights.lambda_cos)
    var = weights.lambda_var * variance_loss(latent, weights.var_gamma)
    cov = weights.lambda_cov * covariance_loss(latent)

    terms = {"recon_l1": l1, "recon_cos": cos, "var": var, "cov": cov}
    total = l1 + cos + var + cov

    if probe_pred is not None and action_target is not None:
        per = F.mse_loss(probe_pred, action_target, reduction="none").mean(dim=-1)
        if m_a is not None:
            # broadcast per-sample m_a over the time axis if needed
            mask = m_a.float()
            while mask.ndim < per.ndim:
                mask = mask.unsqueeze(-1)
            denom = mask.sum().clamp_min(1.0)
            dyn = weights.lambda_dyn * (per * mask).sum() / denom
        else:
            dyn = weights.lambda_dyn * per.mean()
        terms["dyn"] = dyn
        total = total + dyn

    terms["total"] = total
    return terms


# ===========================================================================
# M3 — unified latent-action flow losses
# ===========================================================================


def _masked_mean(per_elem: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over all elements, optionally restricted by a broadcastable mask.

    ``per_elem`` is a per-element loss; ``mask`` (float/bool) is broadcast up to
    its rank and gates which elements count. Empty mask -> 0 (keeps the graph
    connected without NaNs).
    """
    if mask is None:
        return per_elem.mean()
    mask = mask.to(per_elem.dtype)
    while mask.ndim < per_elem.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return (per_elem * mask).sum() / denom


def flow_matching_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE ``||v_theta - u||²`` (doc §4.2), optionally masked.

    For the action term the caller passes the action key-padding / step mask so
    padded steps never contribute; the *structural* per-sample ``m_a`` gating
    (omit Ak entirely for video) happens upstream in the DiT, NOT here.
    """
    per = (v_pred - v_target).pow(2)
    return _masked_mean(per, mask)


def clean_target_loss(
    x1_hat: torch.Tensor,
    x1: torch.Tensor,
    lambda_cos: float = 1.0,
) -> dict[str, torch.Tensor]:
    """L_clean = ||x̂1 - x1||_1 + λ_cos (1 - cos(x̂1, x1))  (doc §2.5).

    ``x1_hat`` comes from :func:`flow.interpolation.predict_x1`. Returns the two
    named terms plus their sum so monitors can log them apart.
    """
    l1 = F.l1_loss(x1_hat, x1)
    cos = (1.0 - F.cosine_similarity(x1_hat, x1, dim=-1)).mean()
    total = l1 + lambda_cos * cos
    return {"clean_l1": l1, "clean_cos": lambda_cos * cos, "total": total}


def flow_distance(a: torch.Tensor, b: torch.Tensor, kind: str = "cos") -> torch.Tensor:
    """Per-sample distance ``d`` used by L_cf and the S_a/Δ_cond monitors.

    ``cos`` -> ``1 - cos`` averaged over the token grid; ``l1`` -> mean abs.
    Reduces everything except the batch dim so the result is ``[B]``.
    """
    if kind == "cos":
        d = 1.0 - F.cosine_similarity(a, b, dim=-1)  # [..., (no C)]
        return d.flatten(1).mean(dim=1) if d.ndim > 1 else d
    if kind == "l1":
        d = (a - b).abs()
        return d.flatten(1).mean(dim=1)
    if kind == "l2_token":
        # Token-level norm avoids diluting counterfactual gradients by the 384
        # latent channels. Shape [..., C] -> per-token L2 -> per-sample mean.
        d = torch.linalg.vector_norm((a - b).float(), ord=2, dim=-1)
        return d.flatten(1).mean(dim=1) if d.ndim > 1 else d
    raise ValueError(f"unknown distance kind {kind!r}")


def counterfactual_loss(
    d_true: torch.Tensor,
    d_perm: torch.Tensor,
    delta: float = 0.05,
    cf_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hinge ``[d(r̂1(a), r1) - d(r̂1(ã), r1) + δ]_+`` (doc §2.7), averaged over
    the rows that were actually permuted (``cf_valid``).

    Pushes the true-action prediction to be *closer* to ``r1`` than the permuted
    action by at least ``δ`` — i.e. "change action -> change prediction". Rows
    that could not be permuted (schema with <2 members) carry ``cf_valid=0`` and
    are excluded so the loss never silently collapses to 0.
    """
    hinge = F.relu(d_true - d_perm + delta)  # [B]
    return _masked_mean(hinge, cf_valid)


@torch.no_grad()
def action_sensitivity(
    r_hat_true: torch.Tensor,
    r_hat_perm: torch.Tensor,
    kind: str = "cos",
    cf_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """S_a = E d(r̂1(a), r̂1(ã)) (§2.7 monitor). Near 0 => action ignored."""
    d = flow_distance(r_hat_true, r_hat_perm, kind)  # [B]
    return _masked_mean(d, cf_valid)


@torch.no_grad()
def delta_cond(
    d_no_act: torch.Tensor,
    d_act: torch.Tensor,
    m_a: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Δ_cond = d(r̂1^no-act, r1) - d(r̂1(a), r1) (§2.7). <= 0 => action useless."""
    return _masked_mean(d_no_act - d_act, m_a)


@torch.no_grad()
def counterfactual_action_delta(
    actions: torch.Tensor,
    actions_perm: torch.Tensor,
    a_mask: Optional[torch.Tensor] = None,
    cf_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean per-sample action L2 distance between true and permuted actions.

    Low values make ``S_a`` inconclusive: if the counterfactual action is almost
    the same as the true action, the predicted latents should not be expected to
    move much. This is a monitor, not a training loss.
    """
    per_sample = _per_sample_action_delta(actions, actions_perm, a_mask)
    return _masked_mean(per_sample, cf_valid)


def collapse_alarm(
    s_a: torch.Tensor,
    delta_cond_val: torch.Tensor,
    min_action_sensitivity: float = 0.01,
    require_delta_cond_positive: bool = True,
) -> list[str]:
    """Return a (possibly empty) list of collapse-alarm messages (§2.9 monitor).

    Caller logs/raises at validation. Kept as plain strings so the training loop
    decides severity (warn vs stop) without this util importing a logger.
    """
    alarms: list[str] = []
    if float(s_a) < min_action_sensitivity:
        alarms.append(
            f"S_a={float(s_a):.4g} < {min_action_sensitivity} — action sensitivity "
            "collapsing (latent prediction ignoring the action)."
        )
    if require_delta_cond_positive and float(delta_cond_val) <= 0.0:
        alarms.append(
            f"Δ_cond={float(delta_cond_val):.4g} <= 0 — conditioning on action gives "
            "no improvement over the no-action prediction."
        )
    return alarms


@dataclass
class UnifiedLossWeights:
    """Weights for L_A (doc §4.2 / CLAUDE.md §2.4). ``lambda_v=0`` this stage."""

    lambda_z: float = 1.0
    lambda_a: float = 1.0
    lambda_clean: float = 1.0
    lambda_cos: float = 1.0
    lambda_roll: float = 0.0
    lambda_cf: float = 0.1
    cf_delta: float = 0.05
    # Token-level L2 avoids diluting counterfactual gradients by the 384 latent
    # channels, while still averaging over time/spatial tokens.
    cf_distance: str = "l2_token"
    # "noisy": reuse the main noisy action tokens for true-action CF.
    # "clean": run the CF pair with clean action tokens and action_timestep=1.
    cf_action_mode: str = "noisy"
    # Optional monitor-only override. Empty means use ``cf_action_mode`` for both
    # training loss and S_a/Δ_cond monitors. For debug validation, "clean" avoids
    # noisy action timesteps making action sensitivity look artificially small.
    monitor_cf_action_mode: str = ""
    # Counterfactual source selection. "random" gives a schema-local derangement;
    # "farthest" chooses the most different action row in the same schema, which
    # is more useful when contiguous robot windows have nearly identical actions.
    counterfactual_action_mode: str = "random"
    min_counterfactual_action_delta: float = 0.0
    min_counterfactual_valid_frac_for_alarm: float = 0.5
    lambda_v: float = 0.0


def assemble_unified_loss(
    v_z: torch.Tensor,
    u_z: torch.Tensor,
    r1_hat: torch.Tensor,
    r1: torch.Tensor,
    weights: UnifiedLossWeights,
    v_a: Optional[torch.Tensor] = None,
    u_a: Optional[torch.Tensor] = None,
    a_mask: Optional[torch.Tensor] = None,
    l_cf: Optional[torch.Tensor] = None,
    l_roll: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Combine the L_A terms into a named dict (+ ``total``).

    ``m_a`` is **structural** (CLAUDE.md §2.3): for a video batch the caller passes
    ``v_a=None`` (the action terms are simply absent — never built-then-masked),
    so the action / counterfactual terms do not appear. ``a_mask`` is the action
    step/pad mask for the present robot rows. ``l_cf`` / ``l_roll`` are precomputed
    scalars (they need extra forwards; see ``counterfactual_loss`` /
    ``flow.rollout``). The value term is a Stage-A stub (``lambda_v=0``).
    """
    terms: dict[str, torch.Tensor] = {}
    l_z = flow_matching_loss(v_z, u_z)
    clean = clean_target_loss(r1_hat, r1, weights.lambda_cos)
    terms["z_fm"] = weights.lambda_z * l_z
    terms["clean"] = weights.lambda_clean * clean["total"]
    total = terms["z_fm"] + terms["clean"]

    if v_a is not None:
        l_a = flow_matching_loss(v_a, u_a, a_mask)
        terms["a_fm"] = weights.lambda_a * l_a
        total = total + terms["a_fm"]
    if l_cf is not None:
        terms["cf"] = weights.lambda_cf * l_cf
        total = total + terms["cf"]
    if l_roll is not None:
        terms["roll"] = weights.lambda_roll * l_roll
        total = total + terms["roll"]

    terms["total"] = total
    return terms


def permute_actions_within_schema(
    actions: torch.Tensor,
    schema_ids: torch.Tensor,
    action_valid: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    warn: bool = True,
    mode: str = "random",
    min_action_delta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the counterfactual action tensor ``ã`` (doc §2.7).

    Within each ``action_schema_id`` group that has **>= 2** valid rows, copy a
    different member's action onto each row. ``mode="random"`` produces a
    schema-local derangement; ``mode="farthest"`` chooses the most action-distant
    row in the same schema. Rows in singleton/absent schemas, video rows, and rows
    whose counterfactual action is too similar are flagged ``cf_valid=0``.

    Returns ``(actions_perm, cf_valid)`` with ``actions_perm`` same shape as
    ``actions`` and ``cf_valid`` a ``[B]`` bool mask of rows that were permuted.
    """
    b = actions.shape[0]
    actions_perm = actions.clone()
    cf_valid = torch.zeros(b, dtype=torch.bool, device=actions.device)

    valid = action_valid.to(torch.bool)
    skipped_any = False
    low_delta_any = False
    # group rows by schema id, robot-only
    schema_vals = torch.unique(schema_ids[valid]) if valid.any() else schema_ids.new_empty(0)
    for sv in schema_vals.tolist():
        members = torch.nonzero((schema_ids == sv) & valid, as_tuple=False).flatten()
        if members.numel() < 2:
            skipped_any = True
            continue
        if mode == "random":
            # derangement of the member positions
            perm = _derangement(members.numel(), device=actions.device, generator=generator)
            src = members[perm]
        elif mode == "farthest":
            src = _farthest_action_sources(actions, members)
        else:
            raise ValueError(f"unknown counterfactual action mode {mode!r}")
        actions_perm[members] = actions[src]
        cf_valid[members] = True

        if min_action_delta > 0.0:
            delta = _per_sample_action_delta(actions[members], actions[src], None)
            keep = delta >= float(min_action_delta)
            if not bool(keep.all()):
                low_delta_any = True
                cf_valid[members[~keep]] = False

    if skipped_any and warn:
        warnings.warn(
            "counterfactual L_cf: some action_schema groups had <2 members in the "
            "batch; those rows are skipped (cf_valid=0) instead of silently "
            "permuting against themselves (doc §2.7).",
            stacklevel=2,
        )
    if low_delta_any and warn:
        warnings.warn(
            "counterfactual L_cf: some within-schema action swaps were below "
            f"min_action_delta={min_action_delta}; those rows are skipped "
            "(cf_valid=0) so low-diversity batches do not fake a collapse.",
            stacklevel=2,
        )
    return actions_perm, cf_valid


def _per_sample_action_delta(
    actions: torch.Tensor,
    actions_perm: torch.Tensor,
    a_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    per_step = torch.linalg.vector_norm((actions - actions_perm).float(), ord=2, dim=-1)
    if a_mask is not None:
        mask = a_mask.to(per_step.dtype)
        return (per_step * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return per_step.flatten(1).mean(dim=1)


def _farthest_action_sources(actions: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
    """For each member row, choose the row with max mean action L2 distance."""
    group = actions[members].float().flatten(1)
    pairwise = torch.cdist(group, group, p=2) / (group.shape[1] ** 0.5)
    pairwise.fill_diagonal_(-1.0)
    return members[pairwise.argmax(dim=1)]


def _derangement(
    n: int, device, generator: Optional[torch.Generator] = None, max_tries: int = 32
) -> torch.Tensor:
    """A permutation of ``range(n)`` with no fixed point (n >= 2)."""
    for _ in range(max_tries):
        perm = torch.randperm(n, device=device, generator=generator)
        if bool((perm != torch.arange(n, device=device)).all()):
            return perm
    # deterministic fallback: cyclic shift by 1 (always a derangement for n>=2)
    return torch.roll(torch.arange(n, device=device), shifts=1, dims=0)
