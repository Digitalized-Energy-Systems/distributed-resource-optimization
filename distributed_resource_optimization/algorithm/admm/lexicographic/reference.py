"""Centralised LP reference for the lexicographic cascade.

Solves exactly the per-tier LP that
:func:`~.coordinator.solve_cp_distributed_lexicographic_cascade` approaches by
sum-sharing ADMM, with cvxpy/HiGHS:

.. math::

    \\begin{aligned}
    \\text{maximise} \\quad & \\sum_{s,k} \\sigma_{s,\\tau}^k \\\\
    \\text{subject to} \\quad & r_i \\in [0,1]^H,\\;
        \\sigma_{s,\\tau}^k \\in [0, D_{s,\\tau}^k] \\\\
    & \\sigma_{s,\\tau}^k + \\sum_i r_i^k c_{i,s} \\;\\le\\;
        B_s^k - \\theta_{s,\\tau-1}^k \\quad \\forall s, k,
    \\end{aligned}

one tier at a time in ascending order, then
:math:`\\theta \\leftarrow \\theta + \\sigma^\\star`. Earlier tiers stay served
without extra constraints: :math:`\\sigma \\ge 0` makes each round's capacity
row imply the previous rounds' rows.

Only the served :math:`\\sigma` is unique; the optimal ``r`` usually is not
(any point of the optimal face), so compare the distributed result against
``served_by_sector_tier``, not ``factor_by_cp``.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from ..types import CPAdmmResult, CPSpec, SectorDemand


def solve_cp_lexicographic_cascade(
    cps: list[CPSpec],
    demands: list[SectorDemand],
    *,
    horizon: int = 1,
) -> CPAdmmResult:
    """Solve the lexicographic cascade centrally, one LP per priority tier.

    Same inputs and result layout as
    :func:`~.coordinator.solve_cp_distributed_lexicographic_cascade`;
    ``iterations`` counts the LPs solved and both residuals are 0.

    :raises ValueError: on a non-positive horizon or a mis-shaped profile.
    :raises RuntimeError: if a tier's LP is not solved to optimality.
    """
    H = int(horizon)
    if H < 1:
        raise ValueError("horizon must be >= 1")
    sectors = sorted({s for c in cps for s in c.capacity_by_sector} | {d.sector for d in demands})
    sec_idx = {s: i for i, s in enumerate(sectors)}
    tiers = sorted({t for d in demands for t in d.demand_by_tier})

    cap = np.zeros((len(cps), len(sectors)))
    for i, c in enumerate(cps):
        for s, value in c.capacity_by_sector.items():
            cap[i, sec_idx[s]] = float(value)
    base = np.zeros((len(sectors), H))
    D = {t: np.zeros((len(sectors), H)) for t in tiers}
    for d in demands:
        bs = np.asarray(d.base_supply, dtype=float)
        if bs.shape != (H,):
            raise ValueError(
                f"base_supply for sector {d.sector!r} must have shape ({H},), got {bs.shape}"
            )
        base[sec_idx[d.sector]] = bs
        for t, arr in d.demand_by_tier.items():
            a = np.asarray(arr, dtype=float)
            if a.shape != (H,):
                raise ValueError(
                    f"demand_by_tier[{t}] for sector {d.sector!r} must have shape ({H},), got {a.shape}"
                )
            D[t][sec_idx[d.sector]] = a

    theta = np.zeros((len(sectors), H))
    r_value = np.zeros((len(cps), H))
    sigma_by_tier: dict[int, np.ndarray] = {}
    for t in tiers:
        sigma = cp.Variable((len(sectors), H))
        constraints = [sigma >= 0, sigma <= D[t]]
        load = 0.0
        if cps:
            r = cp.Variable((len(cps), H))
            constraints += [r >= 0, r <= 1]
            load = cap.T @ r
        constraints.append(sigma + load <= base - theta)
        problem = cp.Problem(cp.Maximize(cp.sum(sigma)), constraints)
        problem.solve(solver=cp.HIGHS)
        if problem.status != cp.OPTIMAL:
            raise RuntimeError(f"tier {t}: reference LP status {problem.status!r}")
        served = np.clip(sigma.value, 0.0, D[t])
        sigma_by_tier[t] = served
        theta = theta + served
        if cps:
            r_value = np.clip(r.value, 0.0, 1.0)

    return CPAdmmResult(
        factor_by_cp={c.cp_id: r_value[i].copy() for i, c in enumerate(cps)},
        served_by_sector_tier={
            d.sector: {t: sigma_by_tier[t][sec_idx[d.sector]].copy() for t in tiers}
            for d in demands
        },
        iterations=len(tiers),
        primal_residual=0.0,
        dual_residual=0.0,
        converged=True,
    )
