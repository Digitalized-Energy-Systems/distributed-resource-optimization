"""ADMMFlexActor — local ADMM participant for flexibility/resource allocation.

Solves a quadratic program (QP) at each ADMM iteration:

.. math::

   \\min_x \\;\\frac{\\rho}{2}\\|x + v\\|^2 + S_i^\\top x

   \\text{subject to} \\quad l \\le x \\le u, \\quad Cx \\le d

where *v* = ``-correction`` (the signal sent by the coordinator), *S_i* is
a per-sector priority/penalty vector, and the constraints represent box and
coupling feasibility.


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
        if not isinstance(message_data, ADMMMessage):
            return
        self.x = _local_update(self, message_data.v, message_data.rho)
        carrier.reply_to_other(ADMMAnswer(x=self.x), meta)


def result(actor: ADMMFlexActor) -> np.ndarray:
    """Return the most recent local solution of *actor*."""
    return actor.x


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
    ]
    if actor.C.size > 0:
        constraints.append(actor.C @ x_var <= actor.d)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if x_var.value is None:
        raise RuntimeError(
            f"ADMM local QP did not converge (status={prob.status}). "
            "Check feasibility of box + coupling constraints."
        )
    return np.asarray(x_var.value, dtype=float)


def _create_C_and_d(tech_capacity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = len(tech_capacity)
    n_rows = 1 + 2 * (m - 1)
    C = np.zeros((n_rows, m))
    d = np.zeros(n_rows)

    for i in range(m):
        C[0, i] = -1.0 if tech_capacity[i] < 0 else 1.0
    d[0] = float(np.sum(np.abs(tech_capacity)))

    for j in range(m - 1):
        r1 = 1 + 2 * j
        r2 = r1 + 1
        tj = tech_capacity[j]
        tm = tech_capacity[m - 1]
        if tj == 0 or tm == 0:
            C[r1, j] = 0.0
            C[r1, m - 1] = 0.0
            C[r2, j] = 0.0
            C[r2, m - 1] = 0.0
        else:
            C[r1, j] = 1.0 / tj
            C[r1, m - 1] = -1.0 / tm
            C[r2, j] = -1.0 / tj
            C[r2, m - 1] = 1.0 / tm

    return C, d


def create_admm_flex_actor_box_bounded(
    lb: list[float] | np.ndarray,
    u: list[float] | np.ndarray,
    S: list[float] | np.ndarray | None = None,
) -> ADMMFlexActor:
    """Create a box-bounded :class:`ADMMFlexActor` with one variable per timestep.

    :param lb: Lower bounds per decision variable.
    :param u: Upper bounds per decision variable.
    :param S: Linear cost / priority penalties (positive = penalised).
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
    :param P: Per-output priority penalties.  Positive = penalised,
              negative = rewarded.  Default: zeros (neutral).
    :returns: Configured :class:`ADMMFlexActor`.
    """
    eta_arr = np.asarray(eta, dtype=float)
    tech_cap = in_capacity * eta_arr

    p_arr = np.zeros(len(eta_arr)) if P is None else np.asarray(P, dtype=float)

    lb = np.minimum(np.zeros(len(tech_cap)), tech_cap)
    u = np.maximum(tech_cap, np.zeros(len(tech_cap)))
    C, d = _create_C_and_d(tech_cap)

    return ADMMFlexActor(lb=lb, u=u, C=C, d=d, S=-p_arr)
