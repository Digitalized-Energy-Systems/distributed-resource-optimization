"""Economic-dispatch actors for the diffusion algorithm.

Two :class:`~.diffusion.DiffusionActor` implementations:

* :class:`LinearCostEconomicDispatchDiffusionActor` — linearised inverted
  quadratic cost response.
* :class:`ReservoirStorageDiffusionActor` — battery/reservoir storage whose
  charge/discharge schedule responds to a time-varying price signal λ(t).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...misc.util import project_storage_schedule_from_price
from .diffusion import DiffusionActor

# ---------------------------------------------------------------------------
# Linear cost economic dispatch
# ---------------------------------------------------------------------------


@dataclass
class LinearCostEconomicDispatchDiffusionActor(DiffusionActor):
    """Economic dispatch via a linearised inverted quadratic cost function.

    :param cost: Marginal cost coefficient *c* in the cost function ``cP + εP²``.
    :param p_max: Maximum power output.
    :param epsilon: Sensitivity of power response to price (default 0.1).
    :param p_min: Minimum power output (default 0).
    :param n_guess: Estimated number of participants for target normalisation.
    """

    cost: float
    # Can be a scalar (same limit for all timesteps) or a vector
    # (one max value per timestep).
    p_max: float | np.ndarray
    epsilon: float = 0.1
    p_min: float = 0.0
    n_guess: int = 10

    P: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def gradient_term(self, lam: np.ndarray, p_target: Any) -> np.ndarray:
        """Compute the gradient for the adapt step.

        :param lam: Current price vector λ.
        :param p_target: Total target power (scalar or array); normalised by
                         :attr:`n_guess` to get the per-participant share.
        :returns: ``P(λ) - p_target / n_guess`` where ``P(λ)`` is the optimal
                  local power response.
        """
        p_max_arr = np.asarray(self.p_max, dtype=float)
        # If p_max is a vector, it should match the number of λ dimensions.
        # Broadcasting is fine for scalar p_max.
        if p_max_arr.ndim > 0 and p_max_arr.shape != lam.shape:
            if p_max_arr.size == lam.size:
                p_max_arr = p_max_arr.reshape(lam.shape)
            else:
                raise ValueError(
                    f"p_max shape {p_max_arr.shape} does not match lam shape {lam.shape}"
                )

        self.P = np.clip((lam - self.cost) / self.epsilon, self.p_min, p_max_arr)
        p_target_arr = np.asarray(p_target if p_target is not None else 0.0)
        return self.P - p_target_arr / self.n_guess


# ---------------------------------------------------------------------------
# Reservoir / battery storage
# ---------------------------------------------------------------------------


@dataclass
class ReservoirStorageDiffusionActor(DiffusionActor):
    """Reservoir/battery storage actor for diffusion-based dispatch.

    The storage actor wants to **discharge** (positive power) when
    ``λ(t) > discharge_cost + charge_cost`` and **charge** (negative power) when
    ``λ(t) < charge_cost``.  It respects energy-capacity and power limits,
    charge/discharge efficiencies, a target terminal energy level, and
    state-of-charge bounds.

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
    :param n_guess: Estimated number of participants for target normalisation.
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
    n_guess: int = 10

    P: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    E: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def gradient_term(self, lam: np.ndarray, p_target: Any) -> np.ndarray:
        lam = np.asarray(lam, dtype=float)

        # Project the price-driven desired schedule onto power+SOC-feasible
        # trajectories, biasing it to reach the target final SOC.
        self.P, self.E, _ = project_storage_schedule_from_price(self, lam)

        p_target_arr = np.asarray(p_target if p_target is not None else 0.0)
        return self.P - p_target_arr / self.n_guess
