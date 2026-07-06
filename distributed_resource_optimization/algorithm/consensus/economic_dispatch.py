"""Economic dispatch consensus actors (Jian et al. 2020, eq. 23).

:class:`~.averaging.ConsensusActor` subclasses that project the shared price
signal λ onto a local power output. The power-balancing correction itself
lives in :class:`~.averaging.AveragingConsensusAlgorithm` (the leader's
ΔP pinning term, eq. 22); these actors only implement the price-to-power
projection:

.. math::

    P(\\lambda) = \\text{clip}\\left(\\frac{\\lambda - c}{\\epsilon},\\; P_{\\min},\\; P_{\\max}\\right)

which is eq. (23)'s ``(λ - bi)/(2ai)`` clip for a quadratic cost
``Fi(PGi) = ci + bi*PGi + ai*PGi**2``, with ``cost`` ↔ ``bi`` and
``epsilon`` ↔ ``2*ai``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...misc.util import project_storage_schedule_from_price
from .averaging import ConsensusActor


@dataclass
class LinearCostEconomicDispatchConsensusActor(ConsensusActor):
    """Economic dispatch via linearised inverted quadratic cost function.

    :param cost: Marginal cost coefficient *c* in the cost function ``cP + εP²``.
    :param p_max: Maximum power output.
    :param epsilon: Sensitivity of power response to price (default 0.1).
    :param p_min: Minimum power output (default 0).
    """

    cost: float
    p_max: float
    epsilon: float = 0.1
    p_min: float = 0.0

    # Updated each iteration
    P: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def project_power(
        self,
        lam: np.ndarray,
        data: Any,
    ) -> np.ndarray:
        """Project the current price signal *lam* onto a local power output.

        :param lam: Current price/λ vector.
        :param data: Unused (total demand is only needed by the leader).
        :returns: Projected local power (same shape as *lam*).
        """
        self.P = np.clip(
            (np.asarray(lam, dtype=float) - self.cost) / self.epsilon,
            self.p_min,
            self.p_max,
        )
        return self.P


@dataclass
class ReservoirStorageConsensusActor(ConsensusActor):
    """Reservoir/battery storage actor for consensus-based dispatch.

    The storage actor responds to price signals λ(t) across the full time horizon,
    charging when prices are low and discharging when prices are high,
    while respecting energy capacity and power limits.

    :param e_max: Maximum energy capacity (MWh).
    :param p_charge_max: Maximum charging power (MW).
    :param p_discharge_max: Maximum discharging power (MW).
    :param eta_charge: Charging efficiency (default 0.95).
    :param eta_discharge: Discharging efficiency (default 0.95).
    :param e_initial: Initial energy level as a fraction of ``e_max``.
    :param e_final: Target final energy level as a fraction of ``e_max``.
    :param soc_min: Minimum state of charge (fraction).
    :param soc_max: Maximum state of charge (fraction).
    :param charge_cost: Marginal cost for charging.
    :param discharge_cost: Marginal benefit for discharging.
    :param epsilon: Sensitivity of power response to price (default 0.1).
    """

    e_max: float
    p_charge_max: float
    p_discharge_max: float
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    e_initial: float = 0.5
    e_final: float = 0.5
    soc_min: float = 0.0
    soc_max: float = 1.0
    charge_cost: float = 0.0
    discharge_cost: float = 0.0
    epsilon: float = 0.1

    P: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    E: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def project_power(self, lam: np.ndarray, data: Any) -> np.ndarray:
        """Project the price-driven desired schedule onto a feasible one.

        :param lam: Current price/λ vector.
        :param data: Unused (total demand is only needed by the leader).
        :returns: Projected local power, biased toward the target final SOC.
        """
        lam = np.asarray(lam, dtype=float)
        self.P, self.E, _ = project_storage_schedule_from_price(self, lam)
        return self.P
