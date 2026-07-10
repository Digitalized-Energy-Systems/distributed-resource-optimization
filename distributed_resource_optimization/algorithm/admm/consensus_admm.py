"""Consensus ADMM — participants' solutions sum to *target* (exchange ADMM).

The global actor implements the variant where z and u are lists of
per-participant vectors (one entry per agent).

The z-update is:

.. math::

    \\delta = \\frac{\\text{target} - \\sum_i (x_i + u_i)}{N + \\rho / \\alpha}

    z_i \\leftarrow x_i + u_i + \\delta

With ``alpha = 0`` (the default) the denominator is exactly *N*, which is the
projection of :math:`(x_i + u_i)` onto the hard constraint
:math:`\\sum_i z_i = \\text{target}` — this makes the iteration **exactly
Boyd's exchange ADMM** (Boyd et al. 2011, §7.3.2, shifted by *target*): the
scaled dual accumulates :math:`u \\leftarrow u + \\bar{x} - \\text{target}/N`
and converges to the (scaled) market-clearing price.

With ``alpha > 0`` the hard constraint is relaxed to a soft quadratic penalty
:math:`\\frac{\\alpha}{2}\\|\\sum_i z_i - \\text{target}\\|^2`, whose exact
minimiser has denominator :math:`N + \\rho/\\alpha`.  Note that a soft penalty
biases the fixed point: with cost-bearing local objectives the converged sum
misses *target* by :math:`(\\rho/\\alpha)\\,\\delta^*`, so keep ``alpha = 0``
for economic dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import ADMMGenericCoordinator, ADMMGlobalActor, ADMMStart


@dataclass
class ADMMConsensusGlobalActor(ADMMGlobalActor):
    """Global actor for the consensus (exchange) ADMM variant.

    :param alpha: Soft-penalty weight on the sum constraint.  ``0`` (default)
                  enforces ``Σ zᵢ = target`` exactly — Boyd's exchange ADMM.
                  ``alpha > 0`` relaxes it to a quadratic penalty
                  ``(alpha/2)·‖Σz − target‖²`` and biases the converged sum
                  by ``(rho/alpha)·δ*`` when local objectives carry costs.
    """

    alpha: float = 0.0

    def z_update(
        self,
        input_data: np.ndarray,
        x: list[np.ndarray],
        u: list[np.ndarray],
        z: list[np.ndarray],
        rho: float,
        n: int,
    ) -> list[np.ndarray]:
        """Project ``x + u`` onto ``Σ zᵢ = target`` (softened when ``alpha > 0``)."""
        m = len(z[0])
        S = np.zeros(m)
        for xi, ui in zip(x, u):
            S += xi + ui
        denom = n + (rho / self.alpha if self.alpha > 0.0 else 0.0)
        delta = (np.asarray(input_data, dtype=float) - S) / denom
        return [xi + ui + delta for xi, ui in zip(x, u)]

    def u_update(
        self,
        x: list[np.ndarray],
        u: list[np.ndarray],
        z: list[np.ndarray],
        rho: float,
        n: int,
    ) -> list[np.ndarray]:
        """Scaled dual ascent: ``uᵢ ← uᵢ + xᵢ − zᵢ`` per participant."""
        return [ui + xi - zi for ui, xi, zi in zip(u, x, z)]

    def init_z(self, n: int, m: int) -> list[np.ndarray]:
        """Start every participant's z at ones (arbitrary feasible-ish point)."""
        return [np.ones(m) for _ in range(n)]

    def init_u(self, n: int, m: int) -> list[np.ndarray]:
        """Start every participant's scaled dual at zero."""
        return [np.zeros(m) for _ in range(n)]

    def actor_correction(
        self,
        x: list[np.ndarray],
        z: list[np.ndarray],
        u: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """Correction ``v = −zᵢ + uᵢ`` so the local QP centres on ``zᵢ − uᵢ``."""
        return -z[i] + u[i]

    def primal_residual(self, x: list[np.ndarray], z: list[np.ndarray]) -> float:
        """Largest per-participant ``‖xᵢ − zᵢ‖`` (consensus violation)."""
        return float(max(np.linalg.norm(xi - zi) for xi, zi in zip(x, z)))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_consensus_target_reach_admm_coordinator(
    rho: float = 1.0,
    max_iters: int = 1000,
    alpha: float = 0.0,
) -> ADMMGenericCoordinator:
    """Create an :class:`ADMMGenericCoordinator` for the consensus variant.

    :param rho: ADMM penalty parameter.
    :param max_iters: Maximum number of iterations.
    :param alpha: Sum-constraint softness (``0`` = exact exchange ADMM).
    """
    return ADMMGenericCoordinator(
        global_actor=ADMMConsensusGlobalActor(alpha=alpha),
        rho=rho,
        max_iters=max_iters,
    )


def create_admm_start_consensus(target: list | np.ndarray) -> ADMMStart:
    """Create an :class:`~.core.ADMMStart` for a consensus run.

    :param target: The target vector that the sum of all *x* values must reach.
    """
    t = np.asarray(target, dtype=float)
    return ADMMStart(data=t, solution_length=len(t))
