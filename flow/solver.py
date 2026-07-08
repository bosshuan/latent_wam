"""ODE solver for the unified flow (project convention: integrate t: 0 -> 1).

Under ``x_t = (1-t)x0 + t x1`` the exact trajectory has constant derivative
``dx/dt = x1 - x0 = u``, and the network regresses ``v_theta ≈ u``. So a plain
forward-Euler integration from ``t=0`` (pure noise) to ``t=1`` recovers the data
sample, and with an *exact* velocity it is exact for **any** step count — that is
the round-trip property ``tests/test_flow_targets.py`` asserts (a sign flip
smuggled into the solver would break it even though shapes stay valid).

``velocity_fn(x_t, t_scalar)`` returns ``v`` with the same shape as ``x_t``.
"""

from __future__ import annotations

from typing import Callable

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def euler_solve(
    velocity_fn: VelocityFn,
    x0: torch.Tensor,
    num_steps: int = 10,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """Forward-Euler from ``t_start`` (noise end) to ``t_end`` (data end).

    Steps ``x <- x + v(x, t) * dt`` with uniform ``dt``. Returns the integrated
    sample (≈ x1 when ``t_start=0, t_end=1``).
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    x = x0
    dt = (t_end - t_start) / num_steps
    for i in range(num_steps):
        t = t_start + i * dt
        t_vec = torch.full((x.shape[0],), float(t), dtype=x.dtype, device=x.device)
        v = velocity_fn(x, t_vec)
        x = x + v * dt
    return x


@torch.no_grad()
def euler_solve_steps(
    velocity_fn: VelocityFn,
    x0: torch.Tensor,
    num_steps: int = 10,
) -> list[torch.Tensor]:
    """Same as :func:`euler_solve` but returns the full trajectory list (len
    ``num_steps+1``), used by the two-step rollout loss to grab intermediate
    states without re-integrating.
    """
    x = x0
    dt = 1.0 / num_steps
    traj = [x]
    for i in range(num_steps):
        t = i * dt
        t_vec = torch.full((x.shape[0],), float(t), dtype=x.dtype, device=x.device)
        x = x + velocity_fn(x, t_vec) * dt
        traj.append(x)
    return traj
