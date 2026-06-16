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
        T = len(lam)

        if len(self.E) != T:
            self.E = np.zeros(T)
        if len(self.P) != T:
            self.P = np.zeros(T)

        e_min = self.soc_min * self.e_max
        e_max_limit = self.soc_max * self.e_max
        e_target_final = self.e_final * self.e_max
        e_initial_abs = float(np.clip(self.e_initial * self.e_max, e_min, e_max_limit))

        # Step 1: desired signed power from local λ.
        desired = np.zeros(T, dtype=float)
        discharge_threshold = self.discharge_cost + self.charge_cost
        for t in range(T):
            lam_t = lam[t]
            if lam_t > discharge_threshold:
                desired[t] = (lam_t - discharge_threshold) / self.epsilon
            elif lam_t < self.charge_cost:
                desired[t] = (lam_t - self.charge_cost) / self.epsilon

        def _project_schedule(power_request: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            """Project a requested schedule onto power+SOC-feasible trajectories."""
            p = np.zeros(T, dtype=float)
            e_path = np.zeros(T + 1, dtype=float)
            e_path[0] = e_initial_abs

            for t in range(T):
                e_t = e_path[t]
                max_discharge_by_soc = (e_t - e_min) * self.eta_discharge
                max_charge_by_soc = (e_max_limit - e_t) / self.eta_charge
                p_low = max(-self.p_charge_max, -max_charge_by_soc)
                p_high = min(self.p_discharge_max, max_discharge_by_soc)
                p[t] = float(np.clip(power_request[t], p_low, p_high))

                if p[t] >= 0.0:
                    e_next = e_t - p[t] / self.eta_discharge
                else:
                    e_next = e_t - p[t] * self.eta_charge
                e_path[t + 1] = float(np.clip(e_next, e_min, e_max_limit))

            return p, e_path[:-1], float(e_path[-1])

        # Step 2: choose an affine bias on desired power to meet terminal energy.
        def _final_energy_with_bias(bias: float) -> tuple[np.ndarray, np.ndarray, float]:
            return _project_schedule(desired + bias)

        p0, e0, e_end0 = _final_energy_with_bias(0.0)
        f0 = e_end0 - e_target_final

        best_p, best_e, best_err = p0, e0, abs(f0)
        if best_err > 1e-3 and T > 0:
            lo, hi = -1.0, 1.0
            p_lo, e_lo, e_end_lo = _final_energy_with_bias(lo)
            p_hi, e_hi, e_end_hi = _final_energy_with_bias(hi)
            f_lo = e_end_lo - e_target_final
            f_hi = e_end_hi - e_target_final

            if abs(f_lo) < best_err:
                best_p, best_e, best_err = p_lo, e_lo, abs(f_lo)
            if abs(f_hi) < best_err:
                best_p, best_e, best_err = p_hi, e_hi, abs(f_hi)

            expansions = 0
            while f_lo * f_hi > 0 and expansions < 20:
                lo *= 2.0
                hi *= 2.0
                p_lo, e_lo, e_end_lo = _final_energy_with_bias(lo)
                p_hi, e_hi, e_end_hi = _final_energy_with_bias(hi)
                f_lo = e_end_lo - e_target_final
                f_hi = e_end_hi - e_target_final
                if abs(f_lo) < best_err:
                    best_p, best_e, best_err = p_lo, e_lo, abs(f_lo)
                if abs(f_hi) < best_err:
                    best_p, best_e, best_err = p_hi, e_hi, abs(f_hi)
                expansions += 1

            if f_lo * f_hi <= 0:
                for _ in range(35):
                    mid = 0.5 * (lo + hi)
                    p_mid, e_mid, e_end_mid = _final_energy_with_bias(mid)
                    f_mid = e_end_mid - e_target_final
                    if abs(f_mid) < best_err:
                        best_p, best_e, best_err = p_mid, e_mid, abs(f_mid)
                    if f_lo * f_mid <= 0:
                        hi, f_hi = mid, f_mid
                    else:
                        lo, f_lo = mid, f_mid

        self.P = best_p
        self.E = best_e

        p_target_arr = np.asarray(p_target if p_target is not None else 0.0)
        return self.P - p_target_arr / self.n_guess
