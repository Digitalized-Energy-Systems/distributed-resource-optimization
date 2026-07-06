"""Shared numeric helpers used across the algorithm implementations."""

from __future__ import annotations

from typing import Any

import numpy as np


def project_storage_schedule_from_price(
    actor: Any,
    price: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Project a price signal onto an SOC-feasible battery/reservoir schedule.

    Shared by the ADMM, averaging-consensus, and diffusion storage actors,
    which all price-drive a desired power request and then search (via
    bisection over an additive bias) for the schedule whose projected
    trajectory ends closest to the target terminal SOC ``actor.e_final``.

    *actor* must expose ``e_max``, ``p_charge_max``, ``p_discharge_max``,
    ``eta_charge``, ``eta_discharge``, ``e_initial``, ``e_final``,
    ``soc_min``, ``soc_max``, ``charge_cost``, ``discharge_cost``, and
    ``epsilon`` (all storage actors already carry these as dataclass
    fields). ``e_initial``/``e_final``/``soc_min``/``soc_max`` are
    fractions of ``e_max``; ``price`` is a per-timestep price/λ vector.

    :returns: ``(p, e_path, e_terminal)`` — the feasible power schedule,
        the SOC at the start of each timestep, and the final SOC reached.
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

    def _final_energy_with_bias(bias: float) -> tuple[np.ndarray, np.ndarray, float]:
        return _project(desired + bias)

    p0, e0, e_end0 = _final_energy_with_bias(0.0)
    f0 = e_end0 - e_final_abs

    best_p, best_e, best_e_end, best_err = p0, e0, e_end0, abs(f0)

    if best_err > 1e-3 and T > 0:
        lo, hi = -1.0, 1.0
        p_lo, e_lo, e_end_lo = _final_energy_with_bias(lo)
        p_hi, e_hi, e_end_hi = _final_energy_with_bias(hi)
        f_lo = e_end_lo - e_final_abs
        f_hi = e_end_hi - e_final_abs

        if abs(f_lo) < best_err:
            best_p, best_e, best_e_end, best_err = p_lo, e_lo, e_end_lo, abs(f_lo)
        if abs(f_hi) < best_err:
            best_p, best_e, best_e_end, best_err = p_hi, e_hi, e_end_hi, abs(f_hi)

        expansions = 0
        while f_lo * f_hi > 0 and expansions < 20:
            lo *= 2.0
            hi *= 2.0
            p_lo, e_lo, e_end_lo = _final_energy_with_bias(lo)
            p_hi, e_hi, e_end_hi = _final_energy_with_bias(hi)
            f_lo = e_end_lo - e_final_abs
            f_hi = e_end_hi - e_final_abs
            if abs(f_lo) < best_err:
                best_p, best_e, best_e_end, best_err = p_lo, e_lo, e_end_lo, abs(f_lo)
            if abs(f_hi) < best_err:
                best_p, best_e, best_e_end, best_err = p_hi, e_hi, e_end_hi, abs(f_hi)
            expansions += 1

        if f_lo * f_hi <= 0:
            for _ in range(35):
                mid = 0.5 * (lo + hi)
                p_mid, e_mid, e_end_mid = _final_energy_with_bias(mid)
                f_mid = e_end_mid - e_final_abs
                if abs(f_mid) < best_err:
                    best_p, best_e, best_e_end, best_err = p_mid, e_mid, e_end_mid, abs(f_mid)
                if f_lo * f_mid <= 0:
                    hi, f_hi = mid, f_mid
                else:
                    lo, f_lo = mid, f_mid

    return best_p, best_e, best_e_end
