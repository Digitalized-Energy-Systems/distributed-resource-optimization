"""Shared numeric helpers used across the algorithm implementations."""

from __future__ import annotations

from typing import Any

import numpy as np


def project_storage_schedule_from_price(
    actor: Any,
    price: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Project a price signal onto an SOC-feasible battery/reservoir schedule.

    Shared by the averaging-consensus and diffusion (incl. exact diffusion)
    storage actors, which both price-drive a desired power request and then search (via
    bisection over an additive bias) for the schedule whose projected
    trajectory ends closest to the target terminal SOC ``actor.e_final``.

    *actor* must expose ``e_max``, ``p_charge_max``, ``p_discharge_max``,
    ``eta_charge``, ``eta_discharge``, ``e_initial``, ``e_final``,
    ``soc_min``, ``soc_max``, ``charge_cost``, ``discharge_cost``, and
    ``epsilon`` (all storage actors already carry these as attributes). ``e_initial``/``e_final``/``soc_min``/``soc_max`` are
    fractions of ``e_max``; ``price`` is a per-timestep price/λ vector.

    :returns: ``(p, e_path, e_terminal)`` — the feasible power schedule,
        the absolute energy level at the start of each timestep, and the final
        energy level reached.
    """
    e_max = actor.e_max
    p_charge_max = actor.p_charge_max
    p_discharge_max = actor.p_discharge_max
    eta_charge = actor.eta_charge
    eta_discharge = actor.eta_discharge
    soc_min = actor.soc_min
    soc_max = actor.soc_max
    charge_cost = actor.charge_cost
    discharge_cost = actor.discharge_cost
    epsilon = actor.epsilon

    price = np.asarray(price, dtype=float)
    T = len(price)
    e_min = soc_min * e_max
    e_max_limit = soc_max * e_max
    e_initial_abs = float(np.clip(actor.e_initial * e_max, e_min, e_max_limit))
    e_final_abs = float(np.clip(actor.e_final * e_max, e_min, e_max_limit))
    discharge_threshold = discharge_cost + charge_cost

    desired = np.zeros(T, dtype=float)
    for t in range(T):
        if price[t] > discharge_threshold:
            desired[t] = (price[t] - discharge_threshold) / epsilon
        elif price[t] < charge_cost:
            desired[t] = (price[t] - charge_cost) / epsilon

    def _project(power_request: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        p = np.zeros(T, dtype=float)
        e_path = np.zeros(T + 1, dtype=float)
        e_path[0] = e_initial_abs

        for t in range(T):
            e_t = e_path[t]
            max_discharge_by_soc = (e_t - e_min) * eta_discharge
            max_charge_by_soc = (e_max_limit - e_t) / eta_charge
            p_low = max(-p_charge_max, -max_charge_by_soc)
            p_high = min(p_discharge_max, max_discharge_by_soc)
            p[t] = float(np.clip(power_request[t], p_low, p_high))

            if p[t] >= 0.0:
                e_next = e_t - p[t] / eta_discharge
            else:
                e_next = e_t - p[t] * eta_charge
            e_path[t + 1] = float(np.clip(e_next, e_min, e_max_limit))

        return p, e_path[:-1], float(e_path[-1])

    best: tuple[np.ndarray, np.ndarray, float] | None = None
    best_err = float("inf")

    def _error_with_bias(bias: float) -> float:
        """Terminal-SOC error at *bias*; keeps the closest schedule seen in *best*."""
        nonlocal best, best_err
        result = _project(desired + bias)
        err = result[2] - e_final_abs
        if abs(err) < best_err:
            best, best_err = result, abs(err)
        return err

    _error_with_bias(0.0)

    if best_err > 1e-3 and T > 0:
        lo, hi = -1.0, 1.0
        f_lo = _error_with_bias(lo)
        f_hi = _error_with_bias(hi)

        expansions = 0
        while f_lo * f_hi > 0 and expansions < 20:
            lo *= 2.0
            hi *= 2.0
            f_lo = _error_with_bias(lo)
            f_hi = _error_with_bias(hi)
            expansions += 1

        if f_lo * f_hi <= 0:
            for _ in range(35):
                mid = 0.5 * (lo + hi)
                f_mid = _error_with_bias(mid)
                if f_lo * f_mid <= 0:
                    hi, f_hi = mid, f_mid
                else:
                    lo, f_lo = mid, f_mid

    assert best is not None
    return best
