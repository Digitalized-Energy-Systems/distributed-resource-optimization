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
    p_max: float
    epsilon: float = 0.1
    p_min: float = 0.0
    n_guess: int = 10

    P: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def gradient_term(
        self,
        lam: np.ndarray,
        p_target: Any,
    ) -> np.ndarray:
        """Compute the gradient for the adapt step.

        :param lam: Current price vector λ.
        :param p_target: Total target power (scalar or array); normalised by
                         :attr:`n_guess` to get the per-participant share.
        :returns: ``P(λ) - p_target / n_guess`` where ``P(λ)`` is the optimal
                  local power response.
        """
        self.P = np.clip(
            (lam - self.cost) / self.epsilon,
            self.p_min,
            self.p_max,
        )
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

    def gradient_term(
        self,
        lam: np.ndarray,
        p_target: Any,
    ) -> np.ndarray:
        lam = np.asarray(lam, dtype=float)
        T = len(lam)

        if len(self.E) != T:
            self.E = np.zeros(T)
        if len(self.P) != T:
            self.P = np.zeros(T)

        # Step 1: desired power per time step from local λ
        for t in range(T):
            lam_t = lam[t]
            if lam_t > self.discharge_cost + self.charge_cost:
                desired_P = min(
                    (lam_t - self.charge_cost) / self.epsilon,
                    self.p_discharge_max,
                )
                self.P[t] = desired_P
            elif lam_t < self.charge_cost:
                desired_P = max(
                    (lam_t - self.charge_cost) / self.epsilon,
                    -self.p_charge_max,
                )
                self.P[t] = desired_P
            else:
                self.P[t] = 0.0

        # Step 2: forward integration of energy state (dt = 1h)
        self.E[0] = self.e_initial * self.e_max
        for t in range(1, T):
            if self.P[t - 1] >= 0:
                # Discharging: energy decreases
                self.E[t] = self.E[t - 1] - self.P[t - 1] / self.eta_discharge
            else:
                # Charging: energy increases
                self.E[t] = self.E[t - 1] - self.P[t - 1] * self.eta_charge

        # Step 3: clip energy to SOC limits
        e_min = self.soc_min * self.e_max
        e_max_limit = self.soc_max * self.e_max
        self.E = np.clip(self.E, e_min, e_max_limit)

        # Step 4: backward pass to meet final energy target
        e_target_final = self.e_final * self.e_max
        e_error = e_target_final - self.E[T - 1]

        if abs(e_error) > 0.001:
            total_energy_change = float(np.sum(self.P))
            if abs(total_energy_change) > 0.001:
                scale_factor = 1.0 + e_error / total_energy_change
                self.P = self.P * scale_factor
            else:
                adjustment = e_error / T
                self.P = self.P + adjustment

        # Step 5: re-clamp power limits
        self.P = np.clip(self.P, -self.p_charge_max, self.p_discharge_max)

        # Step 6: enforce SOC limits per time step
        for t in range(T):
            soc_t = self.E[t] / self.e_max
            if self.P[t] > 0:  # Discharging
                max_discharge = min(
                    self.p_discharge_max,
                    (soc_t - self.soc_min) * self.e_max * self.eta_discharge,
                )
                self.P[t] = min(self.P[t], max_discharge)
            else:  # Charging
                max_charge = min(
                    self.p_charge_max,
                    (self.soc_max - soc_t) * self.e_max / self.eta_charge,
                )
                self.P[t] = max(self.P[t], -max_charge)

        p_target_arr = np.asarray(p_target if p_target is not None else 0.0)
        return self.P - p_target_arr / self.n_guess
