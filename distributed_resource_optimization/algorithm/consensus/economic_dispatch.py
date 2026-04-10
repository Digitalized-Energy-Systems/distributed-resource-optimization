"""Economic dispatch consensus actor.

A :class:`~.averaging.ConsensusActor` that computes a linearised inverted
quadratic cost response.  During each consensus iteration the actor updates its
local power output *P* to minimise cost given the current price signal λ, and
returns a gradient correction that pushes λ toward balancing supply and demand.

The gradient term is:

.. math::

    \\nabla_\\lambda = -\\rho \\left( P(\\lambda) - \\frac{P_{\\text{target}}}{N} \\right)

where

.. math::

    P(\\lambda) = \\text{clip}\\left(\\frac{\\lambda - c}{\\epsilon},\\; P_{\\min},\\; P_{\\max}\\right)


"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .averaging import ConsensusActor


@dataclass
class LinearCostEconomicDispatchConsensusActor(ConsensusActor):
    """Economic dispatch via linearised inverted quadratic cost function.

    :param cost: Marginal cost coefficient *c* in the cost function ``cP + εP²``.
    :param p_max: Maximum power output.
    :param rho: Gradient step size (consensus price sensitivity).
    :param epsilon: Sensitivity of power response to price (default 0.1).
    :param p_min: Minimum power output (default 0).
    :param n_guess: Estimated number of participants for target normalisation.
    """

    cost: float
    p_max: float
    rho: float = 0.05
    epsilon: float = 0.1
    p_min: float = 0.0
    n_guess: int = 10

    # Updated each iteration
    P: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def gradient_term(
        self,
        lam: np.ndarray,
        p_target: Any,
    ) -> np.ndarray:
        """Compute the gradient correction for the current price signal *lam*.

        :param lam: Current price/λ vector.
        :param p_target: Total target power (scalar or array); normalised by
                         :attr:`n_guess` to get the per-participant share.
        :returns: Additive gradient correction (same shape as *lam*).
        """
        self.P = np.clip(
            (lam - self.cost) / self.epsilon,
            self.p_min,
            self.p_max,
        )
        p_target_arr = np.asarray(p_target if p_target is not None else 0.0)
        return -self.rho * (self.P - p_target_arr / self.n_guess)
