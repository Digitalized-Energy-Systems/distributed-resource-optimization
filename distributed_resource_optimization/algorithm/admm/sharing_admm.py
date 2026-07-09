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

from .core import ADMMGenericCoordinator, ADMMGlobalActor, ADMMGlobalObjective, ADMMStart

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
class ADMMGeneratorSpec:
    """Per-participant economic-dispatch parameters for merit-order clearing.

    :param cost: Marginal cost (scalar or per-timestep vector).
    :param lb: Lower power bound per timestep.
    :param ub: Upper power bound per timestep.
    :param epsilon: Per-generator price sensitivity override. A generator's
        price-response band (the range above its marginal cost needed to ramp
        from ``lb`` to ``ub``) is ``epsilon * ub`` — with one shared epsilon,
        large-capacity generators need a far wider price margin to reach full
        output than small ones, breaking merit order once capacities are
        heterogeneous (real networks) rather than similar (toy network).
        Defaults to ``None``, which falls back to the shared
        :attr:`ADMMSharingData.epsilon`.
    """

    cost: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    epsilon: float | None = None


@dataclass
class ADMMSharingData:
    """Input data for the sharing ADMM variant.

    :param target: Desired sum vector (length *m*).
    :param priorities: Per-element priority weights.  Only the magnitude
                       matters: the z-update uses them inside symmetric
                       absolute-value constraints, so the sign is immaterial
                       (:func:`create_admm_sharing_data` stores them negated
                       for historical reasons).
    :param generators: Optional specs for merit-order clearing in ``z_update``.
    :param epsilon: Price sensitivity used with :class:`~.economic_dispatch` actors.
    """

    target: np.ndarray
    priorities: np.ndarray
    generators: list[ADMMGeneratorSpec] | None = None
    epsilon: float = 0.1


def create_admm_sharing_data(
    target: list | np.ndarray,
    priorities: list | np.ndarray | None = None,
    generators: list[ADMMGeneratorSpec] | None = None,
    epsilon: float = 0.1,
) -> ADMMSharingData:
    """Build :class:`ADMMSharingData` from user-friendly inputs.

    :param target: Target sum vector.
    :param priorities: Per-element priority weights (positive = higher priority
                       for fulfilling that element).  Default: all ones.
    :param generators: Optional merit-order generator specs.
    :param epsilon: Price sensitivity for economic-dispatch actors.
    :returns: :class:`ADMMSharingData`.  Priorities are stored negated (a
              historical convention); the sign is immaterial to the z-update,
              which only uses them inside symmetric absolute-value constraints
              — only the magnitude weights the target-distance penalty.
    """
    t = np.asarray(target, dtype=float)
    p = np.ones(len(t)) if priorities is None else np.asarray(priorities, dtype=float)
    return ADMMSharingData(
        target=t,
        priorities=-p,
        generators=generators,
        epsilon=epsilon,
    )


def create_sharing_admm_start(data: ADMMSharingData) -> ADMMStart:
    """Wrap :class:`ADMMSharingData` in an :class:`~.core.ADMMStart` message."""
    return ADMMStart(data=data, solution_length=len(data.target))


# ---------------------------------------------------------------------------
# Merit-order clearing (economic dispatch)
# ---------------------------------------------------------------------------


def _supply_at_price(
    price: float,
    specs: list[ADMMGeneratorSpec],
    t: int,
    epsilon: float,
) -> float:
    total = 0.0
    for spec in specs:
        cost_t = float(np.asarray(spec.cost, dtype=float).ravel()[t])
        lb_t = float(np.asarray(spec.lb, dtype=float).ravel()[t])
        ub_t = float(np.asarray(spec.ub, dtype=float).ravel()[t])
        eps = spec.epsilon if spec.epsilon is not None else epsilon
        total += float(np.clip((price - cost_t) / eps, lb_t, ub_t))
    return total


def _clearing_price(
    target_t: float,
    specs: list[ADMMGeneratorSpec],
    t: int,
    epsilon: float,
) -> float:
    """Find the uniform price where merit-order supply meets ``target_t``."""
    if target_t <= 0.0:
        return min(float(np.asarray(spec.cost, dtype=float).ravel()[t]) for spec in specs)

    costs = [float(np.asarray(spec.cost, dtype=float).ravel()[t]) for spec in specs]
    ub_sum = sum(float(np.asarray(spec.ub, dtype=float).ravel()[t]) for spec in specs)
    max_eps = max(
        (spec.epsilon if spec.epsilon is not None else epsilon) for spec in specs
    )

    lo = min(costs) - 1.0
    hi = max(costs) + max_eps * ub_sum + 1.0

    expansions = 0
    while _supply_at_price(hi, specs, t, epsilon) < target_t and expansions < 30:
        hi = hi * 2.0 + max_eps * ub_sum
        expansions += 1

    if _supply_at_price(hi, specs, t, epsilon) < target_t:
        ub_sum_str = f"{ub_sum:.4g}"
        raise ValueError(
            f"Infeasible dispatch at timestep {t}: target {target_t:.4g} MW exceeds "
            f"total generation capacity {ub_sum_str} MW."
        )

    if _supply_at_price(lo, specs, t, epsilon) >= target_t:
        return lo

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _supply_at_price(mid, specs, t, epsilon) >= target_t:
            hi = mid
        else:
            lo = mid
    return hi


def _z_from_clearing_prices(
    input_data: ADMMSharingData,
    rho: float,
    n: int,
) -> np.ndarray:
    """Map per-timestep clearing prices to the global ADMM consensus vector."""
    specs = input_data.generators
    if specs is None:
        raise ValueError("generators are required for merit-order clearing.")
    horizon = len(input_data.target)
    epsilon = input_data.epsilon
    prices = np.array(
        [_clearing_price(float(input_data.target[t]), specs, t, epsilon) for t in range(horizon)],
        dtype=float,
    )
    return prices / max(rho * n, 1e-12)


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
        if input_data.generators is not None:
            return _z_from_clearing_prices(input_data, rho, n)

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

    def primal_residual(self, x: list[np.ndarray], z: np.ndarray) -> float:
        x_avg = sum(x) / len(x)
        return float(np.max(np.abs(x_avg - z)))


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
