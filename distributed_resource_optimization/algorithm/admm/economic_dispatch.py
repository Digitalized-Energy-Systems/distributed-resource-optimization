"""Economic-dispatch actors for the sharing ADMM algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np

from ...misc.util import project_storage_schedule_from_price
from .core import ADMMAnswer, ADMMMessage
from .flex_actor import ADMMFlexActor

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
    cost_arr = (
        np.full(len(lb_arr), float(cost)) if np.isscalar(cost) else np.asarray(cost, dtype=float)
    )
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
    actors (see :func:`~distributed_resource_optimization.misc.util.project_storage_schedule_from_price`).
    """
    p, _, _ = project_storage_schedule_from_price(actor, pi)
    return p


def solve_battery_price_schedule(
    *,
    horizon: int,
    pi: np.ndarray,
    e_max: float,
    p_charge_max: float,
    p_discharge_max: float,
    eta_charge: float = 1.0,
    eta_discharge: float = 1.0,
    e_initial: float = 0.5,
    e_final: float | None = None,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    charge_cost: float = 0.0,
    discharge_cost: float = 0.0,
) -> np.ndarray:
    """Battery dispatch LP for a given per-timestep clearing-price vector *pi*.

    Returns net power (positive = discharge, negative = charge), respecting
    power limits, SOC bounds, and the terminal-SOC constraint.
    """
    pi = np.asarray(pi, dtype=float)
    T = horizon
    e_final_frac = e_initial if e_final is None else e_final
    e_min_abs = soc_min * e_max
    e_max_abs = soc_max * e_max
    e_init = float(np.clip(e_initial * e_max, e_min_abs, e_max_abs))
    e_end = float(np.clip(e_final_frac * e_max, e_min_abs, e_max_abs))

    p_d = cp.Variable(T, nonneg=True)  # discharge power
    p_c = cp.Variable(T, nonneg=True)  # charge power
    E = cp.Variable(T + 1)
    x = p_d - p_c  # net power (positive = discharge)

    # Discharge earns (pi - discharge_cost); charge costs (pi + charge_cost).
    objective = cp.Minimize((discharge_cost - pi) @ p_d + (charge_cost + pi) @ p_c)

    constraints = [
        p_d <= p_discharge_max,
        p_c <= p_charge_max,
        E[0] == e_init,
        E[1:] == E[:-1] - p_d / eta_discharge + p_c * eta_charge,
        E >= e_min_abs,
        E <= e_max_abs,
        E[-1] == e_end,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if x.value is None:
        # Terminal SOC constraint infeasible — relax it and find the best-effort schedule.
        constraints_relaxed = constraints[:-1]
        prob2 = cp.Problem(objective, constraints_relaxed)
        prob2.solve(solver=cp.OSQP, verbose=False)
        if x.value is None:
            raise RuntimeError(
                f"Battery price LP infeasible (status={prob2.status}). Check capacity parameters."
            )

    return np.asarray(x.value, dtype=float)


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
    "create_admm_storage_actor",
    "solve_battery_price_schedule",
]
