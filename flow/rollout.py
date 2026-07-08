"""Two-step rollout consistency loss (doc §4.2 ``L_two-step``).

Beyond teacher forcing, this constrains short-horizon rollout stability: predict
the future once with the clean (ground-truth) context, then feed the **predicted**
first future chunk back in as context and re-predict, requiring the later chunks
to still match the truth. It penalizes error accumulation when the model conditions
on its own latent imagination rather than ground truth — the failure mode an
open-loop world model must avoid (CLAUDE.md §2.4 rollout).

Ramped in late by the curriculum (``CurriculumSchedule``: ``rollout_weight`` rises
in the last stage), so early training is pure teacher forcing.

``model`` is any DiT exposing ``forward(...) -> WAMOutput`` and
``predict_clean_latent(out, noisy_latent, t_z)``; ``fwd_kwargs`` are the standard
forward inputs (``context_latent`` is clean history, while ``r1`` is the separate
future target).
"""

from __future__ import annotations

from typing import Optional

import torch

from flow.losses import clean_target_loss


def two_step_rollout_loss(
    model,
    fwd_kwargs: dict,
    r1: torch.Tensor,            # [B, T_fut, N, C] clean future latents (targets)
    t_z: torch.Tensor,           # [B] latent timestep used for the noisy future
    lambda_cos: float = 1.0,
    detach_imagined: bool = False,
) -> Optional[torch.Tensor]:
    """Return the two-step rollout scalar, or ``None`` if T_fut < 2 (no 2nd chunk).

    Step 1: forward with the clean context -> ``r̂1`` for all future chunks.
    Step 2: slide the context window forward by one chunk, append ``r̂1[:,0]``
    (the imagined first future latent), and re-forward with the SAME noise ->
    ``r̂1'``; constrain ``r̂1'`` on the chunks AFTER the first to match ``r1``.
    """
    context = fwd_kwargs["context_latent"]
    noisy_latent = fwd_kwargs["noisy_latent"]
    t_ctx, t_fut = context.shape[1], noisy_latent.shape[1]
    if t_fut < 2:
        return None

    out1 = model(**fwd_kwargs)
    r1_hat = model.predict_clean_latent(out1, noisy_latent, t_z)  # [B,T_fut,N,C]

    imagined = r1_hat[:, 0]
    if detach_imagined:
        imagined = imagined.detach()
    context2 = torch.cat([context[:, 1:], imagined.unsqueeze(1)], dim=1)

    kwargs2 = dict(fwd_kwargs)
    kwargs2["context_latent"] = context2
    out2 = model(**kwargs2)
    r1_hat2 = model.predict_clean_latent(out2, noisy_latent, t_z)

    # constrain the chunks the imagined context actually influences (chunk >= 1)
    return clean_target_loss(r1_hat2[:, 1:], r1[:, 1:], lambda_cos)["total"]
