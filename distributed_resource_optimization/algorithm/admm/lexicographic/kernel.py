"""Shared closed-form primitives for the lexicographic-cascade variants.

The closed-form ``(z, sigma)`` cell update is the one piece of math
common to both the coordinator-driven realisation
(:mod:`.coordinator`) and the coordinator-free gossip realisation
(:mod:`.gossip`).  It lives here so neither variant has to import the
other, along with the two scale-normalisation helpers both apply to
their inputs so the pair stays iterate-for-iterate identical.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _characteristic_scale(demands: Iterable[Any]) -> float:
    """Characteristic MW magnitude of one cascade round.

    Every quantity the kernel compares against an absolute constant --
    ``inner_abs_tol``, ``rho``, both residuals -- carries MW, while ``r`` is
    dimensionless and the coupling constraint
    :math:`\\sigma + \\sum_i r_i c_i \\le B - \\theta` is homogeneous.  The
    optimal ``r`` is therefore invariant under a uniform rescaling of the data,
    but the *solver* is not: on an LV feeder every residual sits below a
    ``1e-3`` tolerance from the first iteration, so the loop stops before ``r``
    has moved.  Dividing the round's data by this scale puts those constants on
    O(1) numbers and restores the invariance.

    Derived from ``demands`` alone -- which the coordinator and every gossip
    peer hold identically -- so both realisations normalise by the same number
    without exchanging capacities.
    """
    peak = 0.0
    for d in demands:
        bs = np.abs(np.asarray(d.base_supply, dtype=float))
        if bs.size:
            peak = max(peak, float(bs.max()))
        total: np.ndarray | None = None
        for arr in d.demand_by_tier.values():
            a = np.abs(np.asarray(arr, dtype=float))
            total = a if total is None else total + a
        if total is not None and total.size:
            peak = max(peak, float(total.max()))
    return peak if peak > 0.0 else 1.0


def _local_regularization(alpha: float, relative: bool, cap_norm_sq: float) -> float:
    """Effective ridge weight :math:`\\alpha` for one CP's ``r``-projection.

    The projection denominator is :math:`\\rho\\|c_i\\|^2 + \\alpha`, so
    :math:`\\alpha` carries MW\\ :sup:`2`.  Held absolute it swamps
    :math:`\\rho\\|c_i\\|^2` whenever the CP is small -- 7300x for a 3.6 kW LV
    converter -- which shrinks the step to a fraction of its proper length.
    ``relative`` reads ``alpha`` as a dimensionless multiple of the CP's *own*
    :math:`\\|c_i\\|^2`, which is scale-free and needs no knowledge of any
    peer's capacity, so the gossip peers and the in-process kernel compute
    identical values.
    """
    return alpha * cap_norm_sq if relative else alpha


def _z_sigma_cell_update(
    target_z: float,
    slack_max: float,
    demand: float,
    N: int,
    rho: float,
) -> tuple[float, float]:
    """Closed-form 2D QP for one ``(sector, step)`` cell.

    In Boyd §7.3 the shared variable is :math:`z = s / N` where
    :math:`s = \\sum_i x_i`, so the original sum-coupling
    :math:`\\sigma + s \\le \\text{slack\\_max}` becomes
    :math:`\\sigma + N z \\le \\text{slack\\_max}`.  The cell-wise
    sub-problem is therefore

    .. math::

        \\min_{z, \\sigma}\\;
            -\\sigma + \\tfrac{N\\rho}{2}(z - \\text{target})^2
        \\quad\\text{s.t.}\\;
        \\sigma + N z \\le \\text{slack\\_max},\\;
        \\sigma \\in [0, \\text{demand}].

    Stationarity gives interior :math:`z = \\text{target} - 1/\\rho`
    when the inequality binds; the KKT conditions partition the
    optimum into four regions (non-binding / binding-corner-upper /
    binding-interior / binding-corner-lower) handled by the case
    analysis below.

    :returns: ``(z*, sigma*)``.
    """
    if demand <= 0.0:
        # No tier-tau demand here -> sigma is forced to 0; constraint
        # collapses to N z <= slack_max.
        z_max = slack_max / N
        return min(target_z, z_max), 0.0

    # Non-binding region: the inequality has slack to spare so the
    # unconstrained quadratic minimum z = target and the linear-in-
    # sigma objective pushes sigma to its upper box D.  Feasible iff
    # demand + N target <= slack_max.
    if demand + N * target_z <= slack_max:
        return target_z, demand

    # Binding region: sigma + N z = slack_max.  The unconstrained
    # interior optimum for sigma (no box) is
    # sigma_b = slack_max - N target + N/rho.
    sigma_b = slack_max - N * target_z + N / rho
    if sigma_b >= demand:
        # sigma at upper box -> sigma = D, z = (slack_max - D) / N.
        return (slack_max - demand) / N, demand
    if sigma_b <= 0.0:
        # sigma at lower box -> sigma = 0, z = slack_max / N.
        return slack_max / N, 0.0
    # Interior sigma, z = target - 1/rho.
    return target_z - 1.0 / rho, sigma_b
