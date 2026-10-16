"""ADMMFlexActor — local ADMM participant for flexibility/resource allocation.

Solves a quadratic program (QP) at each ADMM iteration:

.. math::

   \\min_x \\;\\frac{\\rho}{2}\\|x + v\\|^2 + S_i^\\top x

   \\text{subject to} \\quad l \\le x \\le u, \\quad Cx \\le d

where *v* is the correction sent by the coordinator (``ADMMMessage.v``; the
QP is centred on ``-v``), *S_i* is a per-sector linear cost/priority vector
(positive entries penalise, negative entries reward), and the constraints
represent box and coupling feasibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np

from ..core import DistributedAlgorithm
from .core import ADMMAnswer, ADMMMessage

if TYPE_CHECKING:
    from ...carrier.core import Carrier


class ADMMFlexActor(DistributedAlgorithm):
    """Local ADMM actor that solves a box+coupling-constrained QP.

    :param lb: Lower-bound vector.
    :param u: Upper-bound vector.
    :param C: Coupling constraint matrix (rows: constraints, cols: variables).
    :param d: Coupling RHS vector.
    :param S: Priority/penalty vector (negative values act as rewards).
    """

    def __init__(
        self,
        lb: np.ndarray,
        u: np.ndarray,
        C: np.ndarray,
        d: np.ndarray,
        S: np.ndarray,
    ) -> None:
        self.lb = lb
        self.u = u
        self.C = C
        self.d = d
        self.S = S
        self.x: np.ndarray = np.array([])

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: ADMMMessage,
        meta: Any,
    ) -> None:
        """Solve the local QP for the coordinator's correction and reply."""
        if not isinstance(message_data, ADMMMessage):
            return
        self.x = _local_update(self, message_data.v, message_data.rho)
        carrier.reply_to_other(ADMMAnswer(x=self.x), meta)


def _local_update(actor: ADMMFlexActor, v: np.ndarray, rho: float) -> np.ndarray:
    m = len(v)

    if actor.C.size == 0:
        # Box-only case: the QP is separable with closed-form solution
        #   min (rho/2)||x||^2 + (rho*v + S)^T x  s.t.  lb <= x <= u
        #   =>  x* = clip(-v - S/rho, lb, u)
        center = -np.asarray(v, dtype=float) - np.asarray(actor.S, dtype=float) / rho
        return np.clip(center, actor.lb, actor.u)

    x_var = cp.Variable(m)

    h = rho * np.asarray(v, dtype=float) + np.asarray(actor.S, dtype=float)
    objective = cp.Minimize((rho / 2) * cp.sum_squares(x_var) + h @ x_var)

    constraints = [
        x_var >= actor.lb,
        x_var <= actor.u,
        actor.C @ x_var <= actor.d,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if x_var.value is None:
        raise RuntimeError(
            f"ADMM local QP did not converge (status={prob.status}). "
            "Check feasibility of box + coupling constraints."
        )
    return np.asarray(x_var.value, dtype=float)


def _create_C_and_d(tech_capacity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coupling constraints for a one-to-many conversion device.

    *tech_capacity* is the per-output capacity ``in_capacity * eta``.  All
    outputs of the device must run at the same utilisation fraction of their
    capacity, which is what a single shared input implies.  Every
    nonzero-capacity output *j* is therefore tied to a reference output *r*
    (the last one with nonzero capacity) via ``x_j / t_j == x_r / t_r``,
    encoded as two opposing ``≤`` rows.  Zero-capacity outputs get no rows:
    their ratio is undefined and the box bounds already pin them to 0.  A
    total-capacity row is unnecessary since the box bounds plus the ratio
    rows already imply it.
    """
    m = len(tech_capacity)
    nonzero = [j for j in range(m) if tech_capacity[j] != 0]
    if len(nonzero) < 2:
        return np.zeros((0, m)), np.zeros(0)

    r = nonzero[-1]
    tr = tech_capacity[r]
    C = np.zeros((2 * (len(nonzero) - 1), m))
    for k, j in enumerate(nonzero[:-1]):
        tj = tech_capacity[j]
        C[2 * k, j] = 1.0 / tj
        C[2 * k, r] = -1.0 / tr
        C[2 * k + 1, j] = -1.0 / tj
        C[2 * k + 1, r] = 1.0 / tr
    return C, np.zeros(C.shape[0])


def create_admm_flex_actor_box_bounded(
    lb: list[float] | np.ndarray,
    u: list[float] | np.ndarray,
    S: list[float] | np.ndarray | None = None,
) -> ADMMFlexActor:
    """Create a box-bounded :class:`ADMMFlexActor` with one variable per timestep.

    :param lb: Lower bounds per decision variable.
    :param u: Upper bounds per decision variable.
    :param S: Linear cost per decision variable (positive = penalised,
              negative = rewarded).
    :returns: Configured :class:`ADMMFlexActor`.
    """
    lb_arr = np.asarray(lb, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    m = len(lb_arr)
    if u_arr.shape != lb_arr.shape:
        raise ValueError("lb and u must have the same shape.")
    s_arr = np.zeros(m) if S is None else np.asarray(S, dtype=float)
    if s_arr.shape != lb_arr.shape:
        raise ValueError("S must match lb/u length when provided.")
    return ADMMFlexActor(
        lb=lb_arr,
        u=u_arr,
        C=np.zeros((0, m)),
        d=np.zeros(0),
        S=s_arr,
    )


def create_admm_flex_actor_one_to_many(
    in_capacity: float,
    eta: list[float] | np.ndarray,
    P: list[float] | np.ndarray | None = None,
) -> ADMMFlexActor:
    """Create an :class:`ADMMFlexActor` for a one-to-many resource scenario.

    A single input of capacity *in_capacity* is split to ``len(eta)`` outputs
    according to efficiency factors *η*.  The box constraints reflect the
    feasible range of each output, and the coupling constraints preserve the
    one-to-many conversion ratio.

    :param in_capacity: Input resource capacity (e.g. rated power in kW).
    :param eta: Efficiency factors for each output (may be negative for
                bidirectional devices).
    :param P: Per-output priorities, entering the local objective as the
              linear term ``S = -P``.  Positive = rewarded (pulls the output
              towards its capacity), negative = penalised.  Default: zeros
              (neutral).
    :returns: Configured :class:`ADMMFlexActor`.
    """
    eta_arr = np.asarray(eta, dtype=float)
    tech_cap = in_capacity * eta_arr

    p_arr = np.zeros(len(eta_arr)) if P is None else np.asarray(P, dtype=float)

    lb = np.minimum(np.zeros(len(tech_cap)), tech_cap)
    u = np.maximum(tech_cap, np.zeros(len(tech_cap)))
    C, d = _create_C_and_d(tech_cap)

    return ADMMFlexActor(lb=lb, u=u, C=C, d=d, S=-p_arr)
