"""SOC-constrained storage participant for the iterative exchange ADMM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...misc.storage import StorageModel
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
    :class:`~.consensus_admm.ADMMConsensusGlobalActor`.  The terminal
    condition is ``E_T >= e_final`` (at least as full as required; decision
    Q15, same rule as central dispatch and DEED-ADMM), capped at what the
    horizon can reach.  The model and the QP are the shared
    :class:`~distributed_resource_optimization.misc.storage.StorageModel`
    (:meth:`~distributed_resource_optimization.misc.storage.StorageModel.prox`
    at center ``−v``), compiled once and re-solved with warm starts.

    :param horizon: Number of timesteps.
    :param e_max: Energy capacity.
    :param p_charge_max: Maximum charging power.
    :param p_discharge_max: Maximum discharging power.
    :param eta_charge: Charging efficiency.
    :param eta_discharge: Discharging efficiency.
    :param e_initial: Initial SOC as a fraction of ``e_max``.
    :param e_final: Minimum terminal SOC as a fraction of ``e_max`` (default:
        ``e_initial``).
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
        self.model = StorageModel(
            e_max=e_max,
            p_charge_max=p_charge_max,
            p_discharge_max=p_discharge_max,
            eta_charge=eta_charge,
            eta_discharge=eta_discharge,
            e_initial=e_initial,
            e_final=e_final,
            soc_min=soc_min,
            soc_max=soc_max,
            discharge_cost=discharge_cost,
            charge_cost=charge_cost,
        )
        self.x: np.ndarray = np.array([])
        #: Energy path ``E[0..T]`` of the last answer.
        self.E: np.ndarray = np.array([])

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
        # min cost(p) + (rho/2)||p + v||² = prox at center −v (shared model, Q15).
        p, self.E = self.model.prox(-np.asarray(v, dtype=float), rho)
        return p


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
