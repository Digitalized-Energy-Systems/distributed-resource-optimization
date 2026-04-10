"""Consensus ADMM — all participants reach the same value summing to *target*.

The global actor implements the consensus variant where z and u are lists of
per-participant vectors (one entry per agent).

The z-update is:

.. math::

    \\delta = \\frac{\\text{target} - \\sum_i (x_i + u_i)}{N + \\alpha / \\rho}

    z_i \\leftarrow x_i + u_i + \\delta


"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import ADMMGenericCoordinator, ADMMGlobalActor, ADMMStart


@dataclass
class ADMMConsensusGlobalActor(ADMMGlobalActor):
    """Global actor for the consensus ADMM variant.

    :param alpha: Regularisation weight that penalises deviation from the
                  consensus (default 100).
    """

    alpha: int = 100

    def z_update(
        self,
        input_data: np.ndarray,
        x: list[np.ndarray],
        u: list[np.ndarray],
        z: list[np.ndarray],
        rho: float,
        n: int,
    ) -> list[np.ndarray]:
        m = len(z[0])
        S = np.zeros(m)
        for xi, ui in zip(x, u):
            S += xi + ui
        delta = (np.asarray(input_data, dtype=float) - S) / (n + self.alpha / rho)
        return [xi + ui + delta for xi, ui in zip(x, u)]

    def u_update(
        self,
        x: list[np.ndarray],
        u: list[np.ndarray],
        z: list[np.ndarray],
        rho: float,
        n: int,
    ) -> list[np.ndarray]:
        return [ui + xi - zi for ui, xi, zi in zip(u, x, z)]

    def init_z(self, n: int, m: int) -> list[np.ndarray]:
        return [np.ones(m) for _ in range(n)]

    def init_u(self, n: int, m: int) -> list[np.ndarray]:
        return [np.zeros(m) for _ in range(n)]

    def actor_correction(
        self,
        x: list[np.ndarray],
        z: list[np.ndarray],
        u: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        return -z[i] + u[i]

    def primal_residual(self, x: list[np.ndarray], z: list[np.ndarray]) -> float:
        return float(max(np.linalg.norm(xi - zi) for xi, zi in zip(x, z)))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_consensus_target_reach_admm_coordinator() -> ADMMGenericCoordinator:
    """Create an :class:`ADMMGenericCoordinator` for the consensus variant."""
    return ADMMGenericCoordinator(global_actor=ADMMConsensusGlobalActor())


def create_admm_start_consensus(target: list | np.ndarray) -> ADMMStart:
    """Create an :class:`~.core.ADMMStart` for a consensus run.

    :param target: The target vector that the sum of all *x* values must reach.
    """
    t = np.asarray(target, dtype=float)
    return ADMMStart(data=t, solution_length=len(t))
