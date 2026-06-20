"""Economic-dispatch actors for the sharing ADMM algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np

from .core import ADMMAnswer, ADMMMessage
from .flex_actor import ADMMFlexActor, create_admm_flex_actor_box_bounded

if TYPE_CHECKING:
    from ...carrier.core import Carrier


def _price_from_message(message_data: ADMMMessage, n_participants: int) -> np.ndarray:
    """Map the global consensus *z* to a uniform marginal-price signal."""
    if message_data.z is None:
        raise ValueError("ADMMMessage.z is required for economic-dispatch actors.")
    # Nz ≈ target at convergence → use ρNz as the clearing price.
    return message_data.rho * n_participants * np.asarray(message_data.z, dtype=float)


class LinearCostEconomicDispatchADMMFlexActor(ADMMFlexActor):
    """Marginal-cost generator for sharing ADMM.

    Uses the global consensus vector ``z`` as a uniform price signal (like
    diffusion/consensus λ) and responds with

    .. math::

        x = \\mathrm{clip}\\left(\\frac{\\pi - c}{\\varepsilon},\\; x_{\\min},\\; x_{\\max}\\right),
        \\quad \\pi = \\rho N z.

    where *N* is the number of participants.  At convergence
    :math:`Nz \\approx` target, so :math:`\\pi \\approx \\rho \\cdot` target and
    merit-order dispatch fills demand across heterogeneous units.
    """

    def __init__(
        self,
        lb: np.ndarray,
        u: np.ndarray,
        cost: float | np.ndarray,
        *,
        n_participants: int,
        epsilon: float = 0.1,
    ) -> None:
        cost_arr = np.asarray(cost, dtype=float)
        super().__init__(
            lb=np.asarray(lb, dtype=float),
            u=np.asarray(u, dtype=float),
            C=np.zeros((0, len(lb))),
            d=np.zeros(0),
            S=cost_arr,
        )
        self.n_participants = n_participants
        self.epsilon = epsilon

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: ADMMMessage,
        meta: Any,
    ) -> None:
        if not isinstance(message_data, ADMMMessage):
            return
        pi = _price_from_message(message_data, self.n_participants)
        self.x = np.clip((pi - self.S) / self.epsilon, self.lb, self.u)
        carrier.reply_to_other(ADMMAnswer(x=self.x), meta)


def create_admm_economic_dispatch_actor(
    lb: list[float] | np.ndarray,
    u: list[float] | np.ndarray,
    cost: float | list[float] | np.ndarray,
    *,
    n_participants: int,
    epsilon: float = 0.1,
) -> LinearCostEconomicDispatchADMMFlexActor:
    """Create a box-bounded economic-dispatch ADMM participant."""
    lb_arr = np.asarray(lb, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    cost_arr = np.full(len(lb_arr), float(cost)) if np.isscalar(cost) else np.asarray(cost, dtype=float)
    return LinearCostEconomicDispatchADMMFlexActor(
        lb_arr, u_arr, cost_arr, n_participants=n_participants, epsilon=epsilon
    )


class StorageADMMFlexActor(ADMMFlexActor):
    """Reservoir storage participant for horizon ADMM sharing.

    Positive power means discharge; negative power means charge.  Uses the
    same price-driven heuristic as :class:`~.diffusion.economic_dispatch.ReservoirStorageDiffusionActor`,
    then projects the schedule onto SOC-feasible trajectories.
    """

    def __init__(
        self,
        *,
        horizon: int,
        e_max: float,
        p_charge_max: float,
        p_discharge_max: float,
        eta_charge: float = 0.95,
        eta_discharge: float = 0.95,
        e_initial: float = 0.5,
        e_final: float = 0.5,
        soc_min: float = 0.0,
        soc_max: float = 1.0,
        charge_cost: float = 0.0,
        discharge_cost: float = 0.0,
        epsilon: float = 0.1,
        n_participants: int = 1,
    ) -> None:
        super().__init__(
            lb=np.full(horizon, -p_charge_max),
            u=np.full(horizon, p_discharge_max),
            C=np.zeros((0, horizon)),
            d=np.zeros(0),
            S=np.zeros(horizon),
        )
        self.horizon = horizon
        self.e_max = e_max
        self.p_charge_max = p_charge_max
        self.p_discharge_max = p_discharge_max
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.e_initial = e_initial
        self.e_final = e_final
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.charge_cost = charge_cost
        self.discharge_cost = discharge_cost
        self.epsilon = epsilon
        self.n_participants = n_participants

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: ADMMMessage,
        meta: Any,
    ) -> None:
        if not isinstance(message_data, ADMMMessage):
            return
        pi = _price_from_message(message_data, self.n_participants)
        self.x = _storage_schedule_from_price(self, pi)
        carrier.reply_to_other(ADMMAnswer(x=self.x), meta)


def _storage_schedule_from_price(actor: StorageADMMFlexActor, pi: np.ndarray) -> np.ndarray:
    """Build a price-responsive storage schedule that meets the target final SOC.

    Uses the same bias-bisection approach as the consensus and diffusion storage
    actors: a uniform power bias is searched via bisection so that the projected
    schedule's final SOC matches ``actor.e_final``.
    """
    pi = np.asarray(pi, dtype=float)
    T = len(pi)
    e_min = actor.soc_min * actor.e_max
    e_max_limit = actor.soc_max * actor.e_max
    e_initial_abs = float(np.clip(actor.e_initial * actor.e_max, e_min, e_max_limit))
    e_final_abs = float(np.clip(actor.e_final * actor.e_max, e_min, e_max_limit))
    discharge_threshold = actor.discharge_cost + actor.charge_cost

    desired = np.zeros(T, dtype=float)
    for t in range(T):
        if pi[t] > discharge_threshold:
            desired[t] = (pi[t] - discharge_threshold) / actor.epsilon
        elif pi[t] < actor.charge_cost:
            desired[t] = (pi[t] - actor.charge_cost) / actor.epsilon

    def _project(power_request: np.ndarray) -> tuple[np.ndarray, float]:
        p = np.zeros(T, dtype=float)
        e_t = e_initial_abs
        for t in range(T):
            max_discharge_by_soc = (e_t - e_min) * actor.eta_discharge
            max_charge_by_soc = (e_max_limit - e_t) / actor.eta_charge
            p_low = max(-actor.p_charge_max, -max_charge_by_soc)
            p_high = min(actor.p_discharge_max, max_discharge_by_soc)
            p[t] = float(np.clip(power_request[t], p_low, p_high))
            if p[t] >= 0.0:
                e_t = e_t - p[t] / actor.eta_discharge
            else:
                e_t = e_t - p[t] * actor.eta_charge
            e_t = float(np.clip(e_t, e_min, e_max_limit))
        return p, e_t

    p0, e_end0 = _project(desired)
    f0 = e_end0 - e_final_abs
    best_p, best_err = p0, abs(f0)

    if best_err > 1e-3 and T > 0:
        lo, hi = -1.0, 1.0
        p_lo, e_end_lo = _project(desired + lo)
        p_hi, e_end_hi = _project(desired + hi)
        f_lo = e_end_lo - e_final_abs
        f_hi = e_end_hi - e_final_abs

        if abs(f_lo) < best_err:
            best_p, best_err = p_lo, abs(f_lo)
        if abs(f_hi) < best_err:
            best_p, best_err = p_hi, abs(f_hi)

        expansions = 0
        while f_lo * f_hi > 0 and expansions < 20:
            lo *= 2.0
            hi *= 2.0
            p_lo, e_end_lo = _project(desired + lo)
            p_hi, e_end_hi = _project(desired + hi)
            f_lo = e_end_lo - e_final_abs
            f_hi = e_end_hi - e_final_abs
            if abs(f_lo) < best_err:
                best_p, best_err = p_lo, abs(f_lo)
            if abs(f_hi) < best_err:
                best_p, best_err = p_hi, abs(f_hi)
            expansions += 1

        if f_lo * f_hi <= 0:
            for _ in range(35):
                mid = 0.5 * (lo + hi)
                p_mid, e_end_mid = _project(desired + mid)
                f_mid = e_end_mid - e_final_abs
                if abs(f_mid) < best_err:
                    best_p, best_err = p_mid, abs(f_mid)
                if f_lo * f_mid <= 0:
                    hi, f_hi = mid, f_mid
                else:
                    lo, f_lo = mid, f_mid

    return best_p


def create_admm_storage_actor(
    *,
    horizon: int,
    e_max: float,
    p_charge_max: float,
    p_discharge_max: float,
    eta_charge: float = 0.95,
    eta_discharge: float = 0.95,
    e_initial: float = 0.5,
    e_final: float | None = None,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    charge_cost: float = 0.0,
    discharge_cost: float = 0.0,
    epsilon: float = 0.1,
    n_participants: int = 1,
) -> StorageADMMFlexActor:
    """Create a horizon-vector storage actor for sharing ADMM."""
    if e_final is None:
        e_final = e_initial
    return StorageADMMFlexActor(
        horizon=horizon,
        e_max=e_max,
        p_charge_max=p_charge_max,
        p_discharge_max=p_discharge_max,
        eta_charge=eta_charge,
        eta_discharge=eta_discharge,
        e_initial=e_initial,
        e_final=e_final,
        soc_min=soc_min,
        soc_max=soc_max,
        charge_cost=charge_cost,
        discharge_cost=discharge_cost,
        epsilon=epsilon,
        n_participants=n_participants,
    )


__all__ = [
    "LinearCostEconomicDispatchADMMFlexActor",
    "StorageADMMFlexActor",
    "create_admm_economic_dispatch_actor",
    "create_admm_flex_actor_box_bounded",
    "create_admm_storage_actor",
]