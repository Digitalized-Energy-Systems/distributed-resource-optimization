"""Priority-cascaded sharing ADMM with an internal waterfall projection.

A specialised ADMM variant for cross-resource coupling-point coordination
where each participant (a *coupling point*) holds a single regulation
knob ``r_i in [0, 1]`` over an ``H``-step horizon and its commitment in
sector ``s`` is the fixed product ``x_i[s, k] = r_i[k] * cap_i[s]``
(``cap_i[s]`` carries the per-sector signed effective capacity in load
convention: positive = the CP consumes from sector ``s``, negative = it
produces into it).

The standard scaled sharing ADMM is augmented with a priority-marginal
linear penalty that is recomputed every iteration from the current
aggregate via a per-sector waterfall.  ``lambda_s[k]`` is the priority
weight of the highest-priority tier in sector ``s`` that is currently
under-served by the aggregate CP-plus-base supply, and zero otherwise.
Each participant's local subproblem then minimises

.. math::

    (\\rho / 2)\\,\\|x_i - (z - u_i)\\|^2 \\;+\\; \\sum_s \\lambda_s \\, x_i[s]

over ``r_i in [0, 1]``.  Because the linear term is signed by the
direction of ``cap_i[s]`` it penalises consumption from a scarce sector
and rewards production into one — the price-elastic response that
drives near-strict priority at convergence.

The waterfall projection itself is the dual mechanism: as iterations
proceed, every cell that is served drops out of the marginal and only
the next-highest unserved tier carries weight, so the system waterfalls
down the priority schedule.

Architecture
------------

The numerical kernel :func:`solve_cp_priority_admm` is fully synchronous
and deterministic.  It is wrapped by a :class:`WaterfallADMMCoordinator`
that gathers each participant's :class:`CPSpec` over the carrier, runs
the kernel locally, and dispatches the converged per-CP regulation
factor back to each participant.  This matches the
:class:`distributed_resource_optimization.algorithm.admm.core.ADMMGenericCoordinator`
pattern in which the global update is centralised on the coordinator
while every participant contributes its local feasibility data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core import Coordinator, DistributedAlgorithm, OptimizationMessage

if TYPE_CHECKING:
    from ...carrier.core import Carrier


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPSpec:
    """Per-CP specification consumed by :func:`solve_cp_priority_admm`.

    :param cp_id: Stable identifier for the coupling point.
    :param capacity_by_sector: Per-sector signed effective capacity (load
        convention).  A P2H with 10 MW input and η = 0.95 is described
        as ``{"electricity": 10.0, "heat": -9.5}``.  A CHP with 10 MW
        gas input, 3.5 MW electricity output, 4.5 MW heat output is
        ``{"gas": 10.0, "electricity": -3.5, "heat": -4.5}``.
    """

    cp_id: str
    capacity_by_sector: dict[str, float]


@dataclass(frozen=True)
class SectorDemand:
    """Per-sector demand profile over the horizon.

    :param sector: Sector identifier (e.g. ``"electricity"``).
    :param demand_by_tier: ``tier -> length-H array`` of MW per priority
        tier (lower tier index = higher priority).
    :param base_supply: Length-H array of MW available to this sector
        before any CP contribution (load convention: positive = supply
        from base generators).
    """

    sector: str
    demand_by_tier: dict[int, np.ndarray]
    base_supply: np.ndarray


@dataclass(frozen=True)
class CPAdmmResult:
    """Output of :func:`solve_cp_priority_admm`.

    :param factor_by_cp: ``cp_id -> length-H regulation factor in [0, 1]``.
    :param served_by_sector_tier: ``sector -> tier -> length-H served MW``.
    :param iterations: Number of ADMM iterations executed.
    :param primal_residual: Final primal residual ``||x - z||``.
    :param dual_residual: Final dual residual ``rho * ||z - z_prev||``.
    :param converged: ``True`` if the per-step damped factor move fell
        below ``abs_tol`` before ``max_iters``.
    :param history: Optional per-iteration diagnostics when the kernel
        was invoked with ``record_history=True``.
    """

    factor_by_cp: dict[str, np.ndarray]
    served_by_sector_tier: dict[str, dict[int, np.ndarray]]
    iterations: int
    primal_residual: float
    dual_residual: float
    converged: bool
    history: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tier_priority_weight(
    tier: int, *, priority_tiers: int = 4, base: float = 1.0e4
) -> float:
    """Strictly-monotone weight: tier 1 -> ``base ** P``, tier P -> ``base``.

    With ``base = 1e4`` and ``P = 4`` this yields ``[1e16, 1e12, 1e8, 1e4]``
    — the four-orders-of-magnitude separation that makes tier-1 demand
    effectively strict against any tier-2 deficit.  Tests can pass a
    tame base (e.g. ``10``) for human-readable assertions.
    """
    if tier < 1:
        return 0.0
    return base ** max(0, priority_tiers - tier + 1)


def waterfall_serve(supply: np.ndarray, demand: np.ndarray) -> np.ndarray:
    """Per-``(sector, tier, step)`` priority-waterfall served amount.

    :param supply: Array of shape ``(n_sec, H)``.
    :param demand: Array of shape ``(n_sec, n_tier, H)`` with tiers
        sorted ascending (tier index 0 = highest priority).
    :returns: Served array with the same shape as *demand*.  For each
        ``(sector, step)`` the algorithm walks tiers in ascending order
        and assigns ``min(demand_cell, remaining_pool)`` until the pool
        is exhausted.
    """
    n_sec, n_tier, H = demand.shape
    served = np.zeros_like(demand)
    for s in range(n_sec):
        for k in range(H):
            remaining = max(float(supply[s, k]), 0.0)
            if remaining <= 0.0:
                continue
            for t in range(n_tier):
                dem = float(demand[s, t, k])
                if dem <= 0.0:
                    continue
                take = min(dem, remaining)
                served[s, t, k] = take
                remaining -= take
                if remaining <= 1e-12:
                    break
    return served


def marginal_priority(
    served: np.ndarray,
    demand: np.ndarray,
    priorities: np.ndarray,
    *,
    tol: float = 1e-9,
) -> np.ndarray:
    """Per-sector marginal priority value ``lambda[s, k]``.

    Equal to the priority weight of the highest-priority tier in sector
    ``s`` at step ``k`` that is not fully served.  Zero when every tier
    with positive demand is fully served, meaning the CPs face no
    priority pressure on that sector.

    *priorities* has length ``n_tier`` and is sorted in the same
    ascending order as the *demand* array's tier axis.
    """
    n_sec, n_tier, H = demand.shape
    lam = np.zeros((n_sec, H))
    for s in range(n_sec):
        for k in range(H):
            for t in range(n_tier):
                dem = float(demand[s, t, k])
                if dem > tol and served[s, t, k] < dem - tol:
                    lam[s, k] = float(priorities[t])
                    break
    return lam


# ---------------------------------------------------------------------------
# Pure-compute kernel
# ---------------------------------------------------------------------------


def solve_cp_priority_admm(
    cps: list[CPSpec],
    demands: list[SectorDemand],
    *,
    horizon: int = 1,
    rho: float = 1.0,
    max_iters: int = 500,
    abs_tol: float = 1e-3,
    priority_tiers: int = 4,
    priority_weight_base: float = 1.0e4,
    r_damping: float = 0.3,
    record_history: bool = False,
) -> CPAdmmResult:
    """Solve the priority-cascaded sharing ADMM for a CP group.

    The kernel is fully synchronous and deterministic — given identical
    inputs every caller produces the same output.

    :param cps: Every coupling point in the cross-sector component that
        the coordination spans.
    :param demands: Per-sector demand profiles aggregated from the
        downstream load reporters.  Sectors absent from *demands* are
        treated as having zero demand and zero base supply.
    :param horizon: Number of horizon steps ``H``.  Must be ``>= 1``.
    :param rho: ADMM penalty.
    :param max_iters: Iteration cap.
    :param abs_tol: Convergence threshold on the per-step damped factor
        move (``max |r_new - r_curr|``).
    :param priority_tiers: ``P`` in the strict-monotone weight schedule
        (see :func:`tier_priority_weight`).
    :param priority_weight_base: Base of the strict-monotone weight.
    :param r_damping: Trust-region step toward the closed-form
        ``r*`` per iteration: ``r_new = (1 - damping) * r + damping * r*``.
    :param record_history: When ``True`` the returned result carries
        per-iteration primal/dual residuals and the final marginal-
        priority vector in :attr:`CPAdmmResult.history`.
    """
    H = int(horizon)
    if H < 1:
        raise ValueError("horizon must be >= 1")
    N = len(cps)
    if N == 0:
        return CPAdmmResult(
            factor_by_cp={},
            served_by_sector_tier={
                d.sector: {t: np.asarray(a, dtype=float).copy()
                           for t, a in d.demand_by_tier.items()}
                for d in demands
            },
            iterations=0,
            primal_residual=0.0,
            dual_residual=0.0,
            converged=True,
        )

    all_sectors = sorted(
        {s for c in cps for s in c.capacity_by_sector}
        | {d.sector for d in demands}
    )
    if not all_sectors:
        raise ValueError("no sectors found in CPs or demands")
    n_sec = len(all_sectors)
    sec_idx = {s: i for i, s in enumerate(all_sectors)}

    all_tiers = sorted({t for d in demands for t in d.demand_by_tier})
    if not all_tiers:
        return CPAdmmResult(
            factor_by_cp={c.cp_id: np.zeros(H) for c in cps},
            served_by_sector_tier={d.sector: {} for d in demands},
            iterations=0,
            primal_residual=0.0,
            dual_residual=0.0,
            converged=True,
        )
    n_tier = len(all_tiers)
    tier_idx = {t: i for i, t in enumerate(all_tiers)}

    cap = np.zeros((N, n_sec), dtype=float)
    for i, c in enumerate(cps):
        for s, c_val in c.capacity_by_sector.items():
            cap[i, sec_idx[s]] = float(c_val)
    cap_norm_sq = (cap ** 2).sum(axis=1)

    D = np.zeros((n_sec, n_tier, H), dtype=float)
    base_supply = np.zeros((n_sec, H), dtype=float)
    for d in demands:
        s = sec_idx[d.sector]
        bs = np.asarray(d.base_supply, dtype=float)
        if bs.shape != (H,):
            raise ValueError(
                f"base_supply for sector {d.sector!r} must have shape ({H},), "
                f"got {bs.shape}"
            )
        base_supply[s, :] = bs
        for tier, arr in d.demand_by_tier.items():
            if tier not in tier_idx:
                continue
            a = np.asarray(arr, dtype=float)
            if a.shape != (H,):
                raise ValueError(
                    f"demand_by_tier[{tier}] for sector {d.sector!r} must have "
                    f"shape ({H},), got {a.shape}"
                )
            D[s, tier_idx[tier], :] = a

    priorities = np.array(
        [tier_priority_weight(t, priority_tiers=priority_tiers,
                              base=priority_weight_base)
         for t in all_tiers],
        dtype=float,
    )

    x = np.zeros((N, n_sec, H), dtype=float)
    z = np.zeros((n_sec, H), dtype=float)
    u = np.zeros((N, n_sec, H), dtype=float)
    r_curr = np.zeros((N, H), dtype=float)

    # Seed lambda from the all-zero baseline; without it the first
    # x-update would see lambda = 0, commit r = 0, and the convergence
    # test would declare a spurious "converged at no allocation" fixed
    # point because primal = dual = 0.
    served_init = waterfall_serve(base_supply, D)
    lam = marginal_priority(served_init, D, priorities)

    history_primal: list[float] = []
    history_dual: list[float] = []
    history_r_change: list[float] = []

    primal_res = float("inf")
    dual_res = float("inf")
    converged = False
    iteration = 0
    served = waterfall_serve(base_supply - x.sum(axis=0), D)

    for iteration in range(max_iters):
        z_prev = z.copy()
        max_r_change = 0.0

        for i in range(N):
            if cap_norm_sq[i] < 1e-12:
                continue
            for k in range(H):
                target = z[:, k] - u[i, :, k]
                num = float(rho * cap[i] @ target - cap[i] @ lam[:, k])
                den = float(rho * cap_norm_sq[i])
                r_star = num / den
                if r_star < 0.0:
                    r_star = 0.0
                elif r_star > 1.0:
                    r_star = 1.0
                r_new = (1.0 - r_damping) * r_curr[i, k] + r_damping * r_star
                if abs(r_new - r_curr[i, k]) > max_r_change:
                    max_r_change = abs(r_new - r_curr[i, k])
                r_curr[i, k] = r_new
                x[i, :, k] = r_new * cap[i]

        z = (x + u).mean(axis=0)

        x_agg = x.sum(axis=0)
        supply_net = base_supply - x_agg
        served = waterfall_serve(supply_net, D)
        lam = marginal_priority(served, D, priorities)

        u = u + x - z[np.newaxis, :, :]

        primal_res = float(np.linalg.norm(x - z[np.newaxis, :, :]))
        dual_res = float(rho * np.linalg.norm(z - z_prev))
        if record_history:
            history_primal.append(primal_res)
            history_dual.append(dual_res)
            history_r_change.append(max_r_change)
        if max_r_change < abs_tol:
            converged = True
            break

    factor_by_cp: dict[str, np.ndarray] = {
        c.cp_id: r_curr[i].copy() for i, c in enumerate(cps)
    }

    served_by_sector_tier = {
        d.sector: {
            t: served[sec_idx[d.sector], tier_idx[t], :].copy()
            for t in all_tiers
            if t in tier_idx
        }
        for d in demands
    }

    history: dict[str, Any] = {}
    if record_history:
        history = {
            "primal_residuals": history_primal,
            "dual_residuals": history_dual,
            "r_changes": history_r_change,
            "marginal_priority": lam.copy(),
        }

    return CPAdmmResult(
        factor_by_cp=factor_by_cp,
        served_by_sector_tier=served_by_sector_tier,
        iterations=iteration + 1,
        primal_residual=primal_res,
        dual_residual=dual_res,
        converged=converged,
        history=history,
    )


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class WaterfallADMMStart(OptimizationMessage):
    """Start payload for :meth:`WaterfallADMMCoordinator.start_optimization`.

    :param demands: Per-sector demand profiles aggregated externally
        (the kernel does not query participants for demand).
    :param horizon: Horizon length ``H``.
    :param rho: ADMM penalty.
    :param max_iters: Iteration cap.
    :param abs_tol: Convergence tolerance.
    :param priority_tiers: ``P`` in the strict-monotone schedule.
    :param priority_weight_base: Base of the strict-monotone schedule.
    :param r_damping: Trust-region damping.
    :param record_history: Persist per-iteration diagnostics on the result.
    """

    demands: list[SectorDemand]
    horizon: int = 1
    rho: float = 1.0
    max_iters: int = 500
    abs_tol: float = 1e-3
    priority_tiers: int = 4
    priority_weight_base: float = 1.0e4
    r_damping: float = 0.3
    record_history: bool = False


@dataclass
class WaterfallADMMSpecRequest(OptimizationMessage):
    """Coordinator -> participant: request its :class:`CPSpec`."""


@dataclass
class WaterfallADMMSpecReply(OptimizationMessage):
    """Participant -> coordinator: the participant's :class:`CPSpec`."""

    spec: CPSpec


@dataclass
class WaterfallADMMResult(OptimizationMessage):
    """Coordinator -> participant: the converged regulation factor.

    :param r: Length-``H`` regulation factor in ``[0, 1]`` for this CP.
    """

    r: np.ndarray


# ---------------------------------------------------------------------------
# Participant
# ---------------------------------------------------------------------------


class WaterfallADMMParticipant(DistributedAlgorithm):
    """Coupling-point participant in a waterfall-ADMM coordination round.

    A participant holds a single :class:`CPSpec` and is otherwise
    passive: when the coordinator asks for the spec it replies, and
    when the coordinator dispatches the converged factor it stores it
    in :attr:`r`.

    The role is light by design because the kernel itself is centralised
    on the coordinator — the participant exists to encode the
    distributed-data-ownership invariant (each CP knows its own
    effective capacities; the coordinator only sees what is published).

    :param spec: The participant's :class:`CPSpec`.
    """

    def __init__(self, spec: CPSpec) -> None:
        self.spec = spec
        self.r: np.ndarray = np.array([])

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: Any,
        meta: Any,
    ) -> None:
        if isinstance(message_data, WaterfallADMMSpecRequest):
            carrier.reply_to_other(WaterfallADMMSpecReply(spec=self.spec), meta)
        elif isinstance(message_data, WaterfallADMMResult):
            self.r = np.asarray(message_data.r, dtype=float).copy()


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class WaterfallADMMCoordinator(Coordinator):
    """Coordinator that drives a waterfall-ADMM round.

    Each round:

    1. Request a :class:`CPSpec` from every registered participant in
       parallel and await all :class:`WaterfallADMMSpecReply` replies.
    2. Run :func:`solve_cp_priority_admm` locally on the gathered specs
       plus the demands carried in the start message.
    3. Dispatch the converged per-CP factor back to each participant
       via :class:`WaterfallADMMResult`.

    :returns: The full :class:`CPAdmmResult`.
    """

    async def start_optimization(
        self,
        carrier: "Carrier",
        message_data: WaterfallADMMStart,
        meta: Any,
    ) -> CPAdmmResult:
        participant_addrs = carrier.others("coordinator")

        spec_futures = [
            carrier.send_awaitable(WaterfallADMMSpecRequest(), addr)
            for addr in participant_addrs
        ]
        spec_replies = await asyncio.gather(*spec_futures)
        specs = [reply.spec for reply in spec_replies]

        result = solve_cp_priority_admm(
            cps=specs,
            demands=message_data.demands,
            horizon=message_data.horizon,
            rho=message_data.rho,
            max_iters=message_data.max_iters,
            abs_tol=message_data.abs_tol,
            priority_tiers=message_data.priority_tiers,
            priority_weight_base=message_data.priority_weight_base,
            r_damping=message_data.r_damping,
            record_history=message_data.record_history,
        )

        send_tasks = [
            carrier.send_to_other(
                WaterfallADMMResult(r=result.factor_by_cp[spec.cp_id]), addr
            )
            for addr, spec in zip(participant_addrs, specs)
        ]
        await asyncio.gather(*send_tasks)

        return result


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_waterfall_admm_participant(
    cp_id: str,
    capacity_by_sector: dict[str, float],
) -> WaterfallADMMParticipant:
    """Create a :class:`WaterfallADMMParticipant` from raw CP parameters.

    :param cp_id: Stable identifier (must match the participant addressed
        in :attr:`CPAdmmResult.factor_by_cp`).
    :param capacity_by_sector: Per-sector signed effective capacity in
        load convention.
    """
    spec = CPSpec(
        cp_id=cp_id,
        capacity_by_sector=dict(capacity_by_sector),
    )
    return WaterfallADMMParticipant(spec=spec)


def create_waterfall_admm_coordinator() -> WaterfallADMMCoordinator:
    """Create a :class:`WaterfallADMMCoordinator`."""
    return WaterfallADMMCoordinator()


def create_waterfall_admm_start(
    demands: list[SectorDemand],
    *,
    horizon: int = 1,
    rho: float = 1.0,
    max_iters: int = 500,
    abs_tol: float = 1e-3,
    priority_tiers: int = 4,
    priority_weight_base: float = 1.0e4,
    r_damping: float = 0.3,
    record_history: bool = False,
) -> WaterfallADMMStart:
    """Create a :class:`WaterfallADMMStart` message.

    All keyword arguments map directly onto :func:`solve_cp_priority_admm`.
    """
    return WaterfallADMMStart(
        demands=list(demands),
        horizon=horizon,
        rho=rho,
        max_iters=max_iters,
        abs_tol=abs_tol,
        priority_tiers=priority_tiers,
        priority_weight_base=priority_weight_base,
        r_damping=r_damping,
        record_history=record_history,
    )
