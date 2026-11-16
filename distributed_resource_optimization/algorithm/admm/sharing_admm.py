"""Sharing ADMM — distributed resource sharing with target-distance objective.

Here *z* and *u* are **global** (single arrays shared across all participants)
rather than per-participant lists.

The z-update minimises a weighted L1 distance to the target:

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

from .core import ADMMGenericCoordinator, ADMMGlobalActor, ADMMGlobalObjective, ADMMStart, _norm

# ---------------------------------------------------------------------------
# Global objective (currently informational only)
# ---------------------------------------------------------------------------


class ADMMTargetDistanceObjective(ADMMGlobalObjective):
    """Quadratic target-distance objective (informational)."""

    def objective(
        self,
        x: list[np.ndarray],
        u: np.ndarray,
        z: np.ndarray,
        n: int,
    ) -> float:
        return float(np.sum((z - np.asarray(x).mean(axis=0)) ** 2))


# ---------------------------------------------------------------------------
# Sharing data
# ---------------------------------------------------------------------------


@dataclass
class ADMMSharingData:
    """Input data for the sharing ADMM variant.

    :param target: Desired sum vector (length *m*).
    :param priorities: Per-element priority weights (negated so that positive
                       input values become penalties).
    """

    target: np.ndarray
    priorities: np.ndarray


def create_admm_sharing_data(
    target: list | np.ndarray,
    priorities: list | np.ndarray | None = None,
) -> ADMMSharingData:
    """Build :class:`ADMMSharingData` from user-friendly inputs.

    :param target: Target sum vector.
    :param priorities: Per-element priority weights (positive = higher priority
                       for fulfilling that element).  Default: all ones.
    :returns: :class:`ADMMSharingData` with negated priorities (penalty form).
    """
    t = np.asarray(target, dtype=float)
    p = np.ones(len(t)) if priorities is None else np.asarray(priorities, dtype=float)
    return ADMMSharingData(target=t, priorities=-p)  # negate → penalty form


def create_admm_start(data: ADMMSharingData) -> ADMMStart:
    """Wrap :class:`ADMMSharingData` in an :class:`~.core.ADMMStart` message."""
    return ADMMStart(data=data, solution_length=len(data.target))


# ---------------------------------------------------------------------------
# Sharing global actor
# ---------------------------------------------------------------------------


class ADMMSharingGlobalActor(ADMMGlobalActor):
    """Global actor for the sharing ADMM variant.

    :param global_objective: Global objective (currently unused in updates).
    """

    def __init__(self, global_objective: ADMMGlobalObjective) -> None:
        self.global_objective = global_objective

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
        x_avg = sum(x) / len(x)
        return u + x_avg - z

    def init_z(self, n: int, m: int) -> np.ndarray:
        return np.ones(m)

    def init_u(self, n: int, m: int) -> np.ndarray:
        return np.zeros(m)

    def actor_correction(
        self,
        x: list[np.ndarray],
        z: np.ndarray,
        u: np.ndarray,
        i: int,
    ) -> np.ndarray:
        x_avg = sum(x) / len(x)
        return -x[i] + x_avg - z + u

    # 2026-11-13 Q-E5 (D-121): Boyd et al. 2011 eq. 3.12 for the sharing form (§7.3) with the
    # stacked z_i = x_i - x̄ + z̄: ||r|| = √N·||x̄ - z̄||, s_i = ρ(z_i - z_i_old); was a max-norm
    def primal_residual(self, x: list[np.ndarray], z: np.ndarray) -> float:
        return float(np.sqrt(len(x)) * np.linalg.norm(sum(x) / len(x) - z))

    def dual_residual(self, z, z_old, rho, x=None, x_old=None) -> float:
        return rho * _norm([_stacked(x, z, i) - _stacked(x_old, z_old, i) for i in range(len(x))])

    def eps_scales(self, x, z, u, rho) -> tuple[float, float]:
        stacked_z = [_stacked(x, z, i) for i in range(len(x))]
        return max(_norm(x), _norm(stacked_z)), rho * np.sqrt(len(x)) * float(np.linalg.norm(u))


def _stacked(x: list[np.ndarray], z: np.ndarray, i: int) -> np.ndarray:
    """Participant i's z_i of the sharing form (x̄ and z̄ are the averages)."""
    return x[i] - sum(x) / len(x) + z


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_sharing_target_distance_admm_coordinator() -> ADMMGenericCoordinator:
    """Create an :class:`~.core.ADMMGenericCoordinator` for target-distance sharing."""
    return ADMMGenericCoordinator(
        global_actor=ADMMSharingGlobalActor(ADMMTargetDistanceObjective())
    )


def create_sharing_admm_coordinator(
    objective: ADMMGlobalObjective,
) -> ADMMGenericCoordinator:
    """Create an :class:`~.core.ADMMGenericCoordinator` with a custom *objective*."""
    return ADMMGenericCoordinator(global_actor=ADMMSharingGlobalActor(objective))
