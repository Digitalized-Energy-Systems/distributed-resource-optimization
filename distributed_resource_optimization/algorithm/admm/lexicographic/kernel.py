"""Shared closed-form primitive for the lexicographic-cascade variants.

The closed-form ``(z, sigma)`` cell update is the one piece of math
common to both the coordinator-driven realisation
(:mod:`.coordinator`) and the coordinator-free gossip realisation
(:mod:`.gossip`).  It lives here so neither variant has to import the
other.
"""

from __future__ import annotations


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
