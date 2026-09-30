"""Sharing ADMM — distributed resource sharing with target-distance objective.

Here *z* and *u* are **global** (single arrays shared across all participants)
rather than per-participant lists; *z* tracks the participant *average*
(the sharing constraint is ``x̄ = z``, Boyd et al. 2011 §7.3).

The z-update minimises a weighted L1 distance of the aggregate ``N z`` to the
target:

.. math::

    \\min_{z,d} \\;\\frac{N\\rho}{2}\\|z - \\bar{x} - u\\|^2 + \\mathbf{1}^\\top d

    \\text{s.t.} \\quad d_i \\ge p_i(N z_i - t_i), \\;
                         d_i \\ge -p_i(N z_i - t_i), \\; d \\ge 0

where :math:`\\bar{x}` is the participant average, *p* the priorities, and
*t* the target vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .core import ADMMGenericCoordinator, ADMMGlobalActor, ADMMStart

# ---------------------------------------------------------------------------
# Sharing data
# ---------------------------------------------------------------------------


@dataclass
class ADMMSharingData:
    """Input data for the sharing ADMM variant.

    :param target: Desired sum vector (length *m*).
    :param priorities: Per-element priority weights.  They weight the L1
                       target-distance penalty inside symmetric
                       absolute-value constraints, so only the magnitude
                       matters.
    """

    target: np.ndarray
    priorities: np.ndarray


def create_admm_sharing_data(
    target: list | np.ndarray,
    priorities: list | np.ndarray | None = None,
) -> ADMMSharingData:
    """Build :class:`ADMMSharingData` from user-friendly inputs.

    :param target: Target sum vector.
    :param priorities: Per-element priority weights (larger = deviations from
                       that element's target are penalised more).
                       Default: all ones.
    :returns: :class:`ADMMSharingData`.
    """
    t = np.asarray(target, dtype=float)
    p = np.ones(len(t)) if priorities is None else np.asarray(priorities, dtype=float)
    return ADMMSharingData(target=t, priorities=p)


def create_sharing_admm_start(data: ADMMSharingData) -> ADMMStart:
    """Wrap :class:`ADMMSharingData` in an :class:`~.core.ADMMStart` message."""
    return ADMMStart(data=data, solution_length=len(data.target))


# ---------------------------------------------------------------------------
# Sharing global actor
# ---------------------------------------------------------------------------


class ADMMSharingGlobalActor(ADMMGlobalActor):
    """Global actor for the sharing ADMM variant with a target-distance objective."""

    def z_update(
        self,
        input_data: ADMMSharingData,
        x: list[np.ndarray],
        u: np.ndarray,
        z: np.ndarray,
        rho: float,
        n: int,
    ) -> np.ndarray:
        """Solve QP to find the optimal global *z*."""
        x_avg = sum(x) / len(x)
        m = len(x_avg)

        z_var = cp.Variable(m)
        d_var = cp.Variable(m, nonneg=True)

        # Weighted absolute-value constraints
        constraints = []
        for i in range(m):
            p = float(input_data.priorities[i])
            lhs = p * (n * z_var[i] - float(input_data.target[i]))
            constraints.append(d_var[i] >= lhs)
            constraints.append(d_var[i] >= -lhs)

        objective = cp.Minimize((n * rho / 2) * cp.sum_squares(z_var - u - x_avg) + cp.sum(d_var))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if z_var.value is None:
            raise RuntimeError(f"Sharing ADMM z-update QP did not converge (status={prob.status}).")
        return np.asarray(z_var.value, dtype=float)

    def u_update(
        self,
        x: list[np.ndarray],
        u: np.ndarray,
        z: np.ndarray,
        rho: float,
        n: int,
    ) -> np.ndarray:
        """Scaled dual ascent on the average: ``u ← u + x̄ − z`` (Boyd §7.3)."""
        x_avg = sum(x) / len(x)
        return u + x_avg - z

    def init_z(self, n: int, m: int) -> np.ndarray:
        """Start the global average variable at ones."""
        return np.ones(m)

    def init_u(self, n: int, m: int) -> np.ndarray:
        """Start the scaled dual at zero."""
        return np.zeros(m)

    def actor_correction(
        self,
        x: list[np.ndarray],
        z: np.ndarray,
        u: np.ndarray,
        i: int,
    ) -> np.ndarray:
        """Sharing correction ``v = −xᵢ + x̄ − z + u`` (Boyd §7.3 x-update)."""
        x_avg = sum(x) / len(x)
        return -x[i] + x_avg - z + u

    def primal_residual(self, x: list[np.ndarray], z: np.ndarray) -> float:
        """Max-norm violation of the averaged sharing constraint ``x̄ = z``."""
        x_avg = sum(x) / len(x)
        return float(np.max(np.abs(x_avg - z)))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_sharing_target_distance_admm_coordinator(
    rho: float = 1.0,
    max_iters: int = 1000,
) -> ADMMGenericCoordinator:
    """Create an :class:`~.core.ADMMGenericCoordinator` for target-distance sharing.

    :param rho: ADMM penalty parameter.
    :param max_iters: Maximum number of iterations.
    """
    return ADMMGenericCoordinator(
        global_actor=ADMMSharingGlobalActor(), rho=rho, max_iters=max_iters
    )
