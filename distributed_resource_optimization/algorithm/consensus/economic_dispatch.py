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
    :param n_guess: Estimated number of participants for target normalisation.
    :param rho: Gradient step size for consensus (default 0.05).
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
    rho: float = 0.05

    P: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    E: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def gradient_term(self, lam: np.ndarray, p_target: Any) -> np.ndarray:
        lam = np.asarray(lam, dtype=float)
        T = len(lam)

        # Ensure internal arrays match the time horizon T
        if len(self.E) != T:
            self.E = np.zeros(T)
        if len(self.P) != T:
            self.P = np.zeros(T)

        # Compute absolute SOC bounds and targets in energy units
        e_min = self.soc_min * self.e_max
        e_max_limit = self.soc_max * self.e_max
        e_target_final = self.e_final * self.e_max
        e_initial_abs = float(np.clip(self.e_initial * self.e_max, e_min, e_max_limit))

        # Step 1: build a desired signed power schedule from local prices.
        # Positive = discharge, Negative = charge.
        desired = np.zeros(T, dtype=float)
        discharge_threshold = self.discharge_cost + self.charge_cost
        for t in range(T):
            lam_t = lam[t]
            if lam_t > discharge_threshold:
                desired[t] = (lam_t - discharge_threshold) / self.epsilon
            elif lam_t < self.charge_cost:
                desired[t] = (lam_t - self.charge_cost) / self.epsilon

        def _project_schedule(power_request: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            """Project a requested schedule onto power+SOC-feasible trajectories.
            Returns:
            - p: feasible power schedule clipped by power limits and SOC-driven bounds
            - e_path[:-1]: SOC at each timestep (before applying next step)
            - e_path[-1]: final SOC after executing schedule
            """
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

        # Step 2: choose an affine bias on desired power so the projected schedule reaches the target final SOC.
        def _final_energy_with_bias(bias: float) -> tuple[np.ndarray, np.ndarray, float]:
            return _project_schedule(desired + bias)

        # Evaluate un-biased schedule and how far final SOC is from target
        p0, e0, e_end0 = _final_energy_with_bias(0.0)
        f0 = e_end0 - e_target_final

        # Keep best feasible schedule found (minimises absolute terminal error)
        best_p, best_e, best_err = p0, e0, abs(f0)

        # If terminal error is larger than tolerance, search for a bias that reduces it.
        if best_err > 1e-3 and T > 0:
            lo, hi = -1.0, 1.0
            p_lo, e_lo, e_end_lo = _final_energy_with_bias(lo)
            p_hi, e_hi, e_end_hi = _final_energy_with_bias(hi)
            f_lo = e_end_lo - e_target_final
            f_hi = e_end_hi - e_target_final

            # Track any improvement from the initial bracket endpoints
            if abs(f_lo) < best_err:
                best_p, best_e, best_err = p_lo, e_lo, abs(f_lo)
            if abs(f_hi) < best_err:
                best_p, best_e, best_err = p_hi, e_hi, abs(f_hi)

            # Expand bracket if both sides have same sign (no root inside)
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

            # If we have a sign change, perform bisection to find bias with final SOC close to target
            if f_lo * f_hi <= 0:
                for _ in range(35):
                    mid = 0.5 * (lo + hi)
                    p_mid, e_mid, e_end_mid = _final_energy_with_bias(mid)
                    f_mid = e_end_mid - e_target_final
                    if abs(f_mid) < best_err:
                        best_p, best_e, best_err = p_mid, e_mid, abs(f_mid)
                    # Narrow bracket based on sign
                    if f_lo * f_mid <= 0:
                        hi, f_hi = mid, f_mid
                    else:
                        lo, f_lo = mid, f_mid

        # Store the best feasible schedule and SOC path found
        self.P = best_p
        self.E = best_e

        # Convert p_target to array for elementwise operations; default to 0 if None.
        p_target_arr = np.asarray(p_target if p_target is not None else 0.0)

        # Return consensus correction: negative scaled difference between local feasible schedule
        # and the per-agent share of the global target. This drives prices so agents' schedules match target.
        return -self.rho * (self.P - p_target_arr / self.n_guess)

