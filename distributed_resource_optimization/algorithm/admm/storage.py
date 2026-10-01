"""SOC-constrained storage participant for the iterative exchange ADMM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np

from ..core import DistributedAlgorithm
from .core import ADMMAnswer, ADMMMessage

if TYPE_CHECKING:
    from ...carrier.core import Carrier


class ProximalStorageADMMActor(DistributedAlgorithm):
    """SOC-constrained storage participant for the iterative exchange ADMM.

    Solves the ADMM local subproblem each round:

    .. math::

        \\min_{x}\\; c_d^\\top p_d + c_c^\\top p_c
                 + \\frac{\\rho}{2}\\|x + v\\|^2
        \\quad \\text{s.t. SOC dynamics, power and energy bounds,}

    with ``x = p_d − p_c`` (positive = discharge) and *v* the coordinator's
    correction vector, so it composes with
    :class:`~.consensus_admm.ADMMConsensusGlobalActor`.  If the terminal SOC
    ``e_final`` is unreachable within the horizon, that constraint is dropped.

    ``p_d`` and ``p_c`` have no complementarity constraint, so the QP may
    charge and discharge simultaneously, burning energy through the round-trip
    losses without moving the SOC.  This only pays off when surplus cannot be
    shed otherwise; a positive ``charge_cost`` makes it less attractive.

    :param horizon: Number of timesteps.
    :param e_max: Energy capacity.
    :param p_charge_max: Maximum charging power.
    :param p_discharge_max: Maximum discharging power.
    :param eta_charge: Charging efficiency.
    :param eta_discharge: Discharging efficiency.
    :param e_initial: Initial SOC as a fraction of ``e_max``.
    :param e_final: Terminal SOC as a fraction of ``e_max`` (default: ``e_initial``).
    :param soc_min: Minimum SOC fraction.
    :param soc_max: Maximum SOC fraction.
    :param charge_cost: Linear cost per unit of charging power.
    :param discharge_cost: Linear cost per unit of discharging power.
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
        e_final: float | None = None,
        soc_min: float = 0.0,
        soc_max: float = 1.0,
        charge_cost: float = 0.0,
        discharge_cost: float = 0.0,
    ) -> None:
        self.horizon = horizon
        self.e_max = e_max
        self.p_charge_max = p_charge_max
        self.p_discharge_max = p_discharge_max
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.e_initial = e_initial
        self.e_final = e_initial if e_final is None else e_final
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.charge_cost = charge_cost
        self.discharge_cost = discharge_cost
        self.x: np.ndarray = np.array([])

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: ADMMMessage,
        meta: Any,
    ) -> None:
        """Answer with the exact proximal solution of the storage subproblem."""
        if not isinstance(message_data, ADMMMessage):
            return
        self.x = self._solve_prox(np.asarray(message_data.v, dtype=float), message_data.rho)
        carrier.reply_to_other(ADMMAnswer(x=self.x), meta)

    def _solve_prox(self, v: np.ndarray, rho: float) -> np.ndarray:
        T = self.horizon
        e_min_abs = self.soc_min * self.e_max
        e_max_abs = self.soc_max * self.e_max
        e_init = float(np.clip(self.e_initial * self.e_max, e_min_abs, e_max_abs))
        e_end = float(np.clip(self.e_final * self.e_max, e_min_abs, e_max_abs))

        p_d = cp.Variable(T, nonneg=True)
        p_c = cp.Variable(T, nonneg=True)
        E = cp.Variable(T + 1)
        x = p_d - p_c

        objective = cp.Minimize(
            self.discharge_cost * cp.sum(p_d)
            + self.charge_cost * cp.sum(p_c)
            + (rho / 2.0) * cp.sum_squares(x + v)
        )
        constraints = [
            p_d <= self.p_discharge_max,
            p_c <= self.p_charge_max,
            E[0] == e_init,
            E[1:] == E[:-1] - p_d / self.eta_discharge + p_c * self.eta_charge,
            E >= e_min_abs,
            E <= e_max_abs,
            E[-1] == e_end,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if x.value is None:
            # Terminal SOC infeasible for this horizon — relax it.
            prob2 = cp.Problem(objective, constraints[:-1])
            prob2.solve(solver=cp.OSQP, verbose=False)
            if x.value is None:
                raise RuntimeError(
                    f"Storage prox QP infeasible (status={prob2.status}). "
                    "Check capacity parameters."
                )

        return np.asarray(x.value, dtype=float)


def create_admm_proximal_storage_actor(
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
) -> ProximalStorageADMMActor:
    """Create a SOC-constrained storage participant for iterative exchange ADMM.

    See :class:`ProximalStorageADMMActor` for the parameters.
    """
    return ProximalStorageADMMActor(
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
    )
