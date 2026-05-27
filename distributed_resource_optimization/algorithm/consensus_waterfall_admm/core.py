"""Primal-dual variant of the waterfall ADMM with a Robbins-Monro dual.

Where :mod:`distributed_resource_optimization.algorithm.waterfall_admm`
recomputes the priority marginal ``lambda_s[k]`` from a closed-form
per-sector waterfall every iteration — a step function of the iterate
that jumps by orders of magnitude when a cutoff tier shifts and therefore
forces a trust-region damping ``r_damping`` to keep the inner ADMM
stable — this module promotes the marginal to a free dual state
``mu_s[k]`` and updates it by projected dual ascent on the cutoff-tier
deficit

.. math::

    \\mu_s^{k,\\nu+1} = \\Pi_{[0, \\bar\\mu]}\\bigl(
        \\mu_s^{k,\\nu} + \\gamma_\\nu \\cdot
        (D_{s,t^\\star}^k - T_{s,t^\\star}^k(P^{k,\\nu}))
    \\bigr),

with Robbins-Monro stepsize :math:`\\gamma_\\nu = c / \\nu` satisfying
:math:`\\sum_\\nu \\gamma_\\nu = \\infty` and
:math:`\\sum_\\nu \\gamma_\\nu^2 < \\infty`.  The inner loop is the
unaltered sharing-ADMM kernel with frozen :math:`\\mu`; the outer loop
is projected subgradient ascent on the active cutoff tier's deficit.

Read together the two loops chase a saddle point of the
constrained-priority Lagrangian

.. math::

    L(r, \\mu) = J(r) + \\sum_{s,t,k} \\mu_{s,t}^k
        \\bigl(D_{s,t}^k - T_{s,t}^k(P(r))\\bigr),

where :math:`J(r)` is the inner ADMM's primal objective.  Only the
cutoff-tier multiplier is carried explicitly because the satisfied
higher tiers contribute a zero deficit and the lower tiers are not yet
binding — so the state has the same shape ``(n_sec, H)`` as the
closed-form :func:`marginal_priority` it replaces.

Note on boundary saddles
~~~~~~~~~~~~~~~~~~~~~~~~

The sharing-ADMM inner problem this module composes carries an
implicit :math:`J(r) \\equiv 0` — there is no per-CP operating cost
to break ties between the (often infinitely many) primal points that
satisfy the demand constraints.  When the feasible set has slack the
saddle is at a box corner: typically ``r_i = 1`` with the deficit
turned negative and :math:`\\mu` driven all the way back down to
zero.  That is a *valid* saddle (complementary slackness holds:
:math:`\\mu (D - T) = 0` is satisfied with :math:`\\mu = 0`), but the
solver does not minimise production — every CP runs as hard as the
ADMM consensus permits.  Embedding the kernel in a system that
exposes a non-trivial :math:`J(r)` (an operating cost, a deviation
penalty against a baseline schedule, …) restores the interior
saddle.

Architecture
------------

The numerical kernel :func:`solve_cp_consensus_waterfall_admm` is fully
synchronous and deterministic.  It is wrapped by a
:class:`ConsensusWaterfallADMMCoordinator` that gathers each
participant's :class:`CPSpec` over the carrier, runs the kernel
locally, and dispatches the converged per-CP regulation factor back to
each participant — mirroring the
:class:`~distributed_resource_optimization.algorithm.waterfall_admm.core.WaterfallADMMCoordinator`
that carries the closed-form variant.  Input dataclasses
(:class:`CPSpec`, :class:`SectorDemand`) and the result dataclass
(:class:`CPAdmmResult`) are reused from the closed-form module so that
call sites can switch kernels by swapping the solver.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core import Coordinator, DistributedAlgorithm, OptimizationMessage
from ..waterfall_admm.core import (
    CPAdmmResult,
    CPSpec,
    SectorDemand,
    waterfall_serve,
)

if TYPE_CHECKING:
    from ...carrier.core import Carrier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cutoff_tier_deficit(
    supply_net: np.ndarray,
    demand: np.ndarray,
    *,
    tol: float = 1e-9,
) -> np.ndarray:
    """Per-``(sector, step)`` *signed* cutoff-tier deficit.

    The cutoff tier ``t^star(s, k)`` is the highest-priority tier in
    sector ``s`` at step ``k`` whose cumulative demand through it
    (inclusive of all higher-priority tiers) exceeds the supply pool.
    Naming :math:`\\mathrm{cum}D = \\sum_{t' \\le t^\\star} D_{s,t',k}`
    and the pool :math:`P = \\mathrm{supply\\_net}_{s,k}`, the returned
    deficit is :math:`\\mathrm{cum}D - P`:

    * positive when the cutoff is under-served (the dual ascent then
      raises :math:`\\mu_s^k`);
    * zero when the supply just meets the cutoff;
    * negative when every positive-demand tier is fully served with
      surplus — the cutoff has lapsed and the same signal drives
      :math:`\\mu_s^k` back down toward zero via the projection.

    Two-sided signedness is required for the saddle-point math: the
    *true* dual subgradient is the signed cumulative balance
    :math:`\\mathrm{cum}D - P`, not the waterfall-capped
    :math:`D_{s,t^\\star} - \\min(D_{s,t^\\star}, \\text{pool at } t^\\star)`
    which is non-negative by construction and provides no downward
    force to recover from an over-shoot.

    :param supply_net: Array of shape ``(n_sec, H)`` with the net
        supply pool ``base_supply - sum_i r_i * cap_i`` in each
        ``(sector, step)``.
    :param demand: Array of shape ``(n_sec, n_tier, H)`` with tiers
        sorted ascending (tier index 0 = highest priority).
    :param tol: Cells with ``demand <= tol`` are skipped; the cutoff
        check ``cum_D > pool + tol`` keeps the cutoff sharply
        identified despite floating-point noise.
    :returns: Array of shape ``(n_sec, H)`` with the signed deficit.
    """
    n_sec, n_tier, H = demand.shape
    deficit = np.zeros((n_sec, H))
    for s in range(n_sec):
        for k in range(H):
            pool = float(supply_net[s, k])
            cum_d = 0.0
            cutoff_found = False
            for t in range(n_tier):
                dem = float(demand[s, t, k])
                if dem <= tol:
                    continue
                cum_d += dem
                if cum_d > pool + tol:
                    deficit[s, k] = cum_d - pool
                    cutoff_found = True
                    break
            if not cutoff_found and cum_d > tol:
                deficit[s, k] = cum_d - pool
    return deficit


# ---------------------------------------------------------------------------
# Pure-compute kernel
# ---------------------------------------------------------------------------


def solve_cp_consensus_waterfall_admm(
    cps: list[CPSpec],
    demands: list[SectorDemand],
    *,
    horizon: int = 1,
    rho: float = 1.0,
    outer_iters: int = 200,
    inner_iters: int = 8,
    gamma0: float = 1.0,
    mu_upper_bound: float = 1.0e6,
    abs_tol: float = 1e-3,
    record_history: bool = False,
) -> CPAdmmResult:
    """Solve the consensus-waterfall ADMM (primal-dual variant) for a CP group.

    The kernel composes an inner sharing-ADMM loop on
    ``(x, z, u, r)`` with :math:`\\mu` held fixed and an outer projected
    subgradient-ascent loop on :math:`\\mu` driven by the cutoff-tier
    deficit.  No trust-region damping is needed: with :math:`\\mu`
    frozen during the inner pass the linear coefficient
    :math:`\\mu \\cdot c_i` is a constant of the iterate and the
    sharing-ADMM operator is firmly non-expansive.

    The closed-form variant's ``priority_tiers`` / ``priority_weight_base``
    inputs do not appear here: B' carries :math:`\\mu` as state seeded
    at zero rather than at a tier-weight schedule, so the priority
    structure enters the algorithm only through the *order* in which
    the per-tier demands are evaluated by :func:`cutoff_tier_deficit`.

    :param cps: Every coupling point in the cross-sector component.
    :param demands: Per-sector demand profiles aggregated externally.
        Sectors absent from *demands* are treated as having zero demand
        and zero base supply.
    :param horizon: Number of horizon steps ``H``.  Must be ``>= 1``.
    :param rho: Inner ADMM penalty.
    :param outer_iters: Outer (dual) iteration cap.
    :param inner_iters: Inner (primal) ADMM passes per outer step.
        The inner :math:`O(1/\\nu)` rate makes a small constant
        (default eight) sufficient for the inner residual to enter the
        summable-error regime needed by the coupling proof.
    :param gamma0: Robbins-Monro stepsize coefficient ``c`` in
        :math:`\\gamma_\\nu = c / \\nu`.
    :param mu_upper_bound: Projection ceiling :math:`\\bar\\mu` on
        every dual coordinate.  Numerical safety only; for a feasible
        problem the saddle-point :math:`\\mu^\\star` is finite.
    :param abs_tol: Outer-loop convergence threshold on the per-step
        max change of :math:`\\mu` and of ``r`` (both must dip below
        ``abs_tol``).
    :param record_history: When ``True`` the returned result carries
        per-outer-iteration residuals and the final dual vector.
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

    # Seed mu at zero — the natural primal-dual cold start.  An
    # earlier draft seeded from the closed-form priority weights to
    # avoid a degenerate first inner pass with no linear pressure,
    # but the Robbins-Monro schedule gamma_nu = c / nu has total
    # cumulative motion O(log N) and so cannot walk a 1e4-magnitude
    # warm start back down to the (typically O(rho) sized) saddle-point
    # value within practical iteration counts.  Starting at zero costs
    # one "ramp-up" outer iteration but lands at the true saddle.
    mu = np.zeros((n_sec, H), dtype=float)
    mu_bar = float(mu_upper_bound)

    x = np.zeros((N, n_sec, H), dtype=float)
    z = np.zeros((n_sec, H), dtype=float)
    u = np.zeros((N, n_sec, H), dtype=float)
    r_curr = np.zeros((N, H), dtype=float)

    history_primal: list[float] = []
    history_dual: list[float] = []
    history_mu_change: list[float] = []
    history_r_change: list[float] = []

    primal_res = float("inf")
    dual_res = float("inf")
    converged = False
    outer = 0
    served = waterfall_serve(base_supply - x.sum(axis=0), D)

    for outer in range(outer_iters):
        r_prev_outer = r_curr.copy()
        mu_prev_outer = mu.copy()

        # ----- inner ADMM with mu frozen -----
        inner_primal = float("inf")
        inner_dual = float("inf")
        for _ in range(max(1, int(inner_iters))):
            z_prev = z.copy()
            for i in range(N):
                if cap_norm_sq[i] < 1e-12:
                    continue
                for k in range(H):
                    target = z[:, k] - u[i, :, k]
                    num = float(rho * cap[i] @ target - cap[i] @ mu[:, k])
                    den = float(rho * cap_norm_sq[i])
                    r_star = num / den
                    if r_star < 0.0:
                        r_star = 0.0
                    elif r_star > 1.0:
                        r_star = 1.0
                    r_curr[i, k] = r_star
                    x[i, :, k] = r_star * cap[i]

            z = (x + u).mean(axis=0)
            u = u + x - z[np.newaxis, :, :]

            inner_primal = float(np.linalg.norm(x - z[np.newaxis, :, :]))
            inner_dual = float(rho * np.linalg.norm(z - z_prev))

        # ----- outer dual ascent on cutoff-tier deficit -----
        supply_net = base_supply - x.sum(axis=0)
        served = waterfall_serve(supply_net, D)
        deficit = cutoff_tier_deficit(supply_net, D)
        gamma_nu = gamma0 / float(outer + 1)
        mu = mu + gamma_nu * deficit
        if mu_bar > 0.0:
            np.clip(mu, 0.0, mu_bar, out=mu)
        else:
            np.maximum(mu, 0.0, out=mu)

        mu_change = float(np.max(np.abs(mu - mu_prev_outer))) if mu.size else 0.0
        r_change = float(np.max(np.abs(r_curr - r_prev_outer))) if r_curr.size else 0.0
        primal_res = inner_primal
        dual_res = inner_dual

        if record_history:
            history_primal.append(inner_primal)
            history_dual.append(inner_dual)
            history_mu_change.append(mu_change)
            history_r_change.append(r_change)

        # Convergence is declared on the standard "primal+dual stable"
        # check (both r and mu stopped moving), with one KKT-aware
        # extension: when r is stable and every multiplier sits at the
        # lower projection boundary with non-positive subgradient,
        # complementary slackness holds even though the harmonic
        # gamma_nu = c / nu schedule is still drifting mu's tail
        # cumulant toward zero.  Without the extension, problems whose
        # saddle is the over-production corner ("J=0 boundary saddle"
        # discussed in the module docstring) never trip the standard
        # check inside a reasonable iteration cap.
        if r_change < abs_tol:
            if mu_change < abs_tol:
                converged = True
                break
            if bool(np.all((mu <= abs_tol) & (deficit <= abs_tol))):
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
            "mu_changes": history_mu_change,
            "r_changes": history_r_change,
            "mu": mu.copy(),
        }

    return CPAdmmResult(
        factor_by_cp=factor_by_cp,
        served_by_sector_tier=served_by_sector_tier,
        iterations=outer + 1,
        primal_residual=primal_res,
        dual_residual=dual_res,
        converged=converged,
        history=history,
    )


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class ConsensusWaterfallADMMStart(OptimizationMessage):
    """Start payload for :meth:`ConsensusWaterfallADMMCoordinator.start_optimization`.

    Mirrors the closed-form variant's start message but trades
    ``max_iters`` / ``r_damping`` for the primal-dual control set
    ``outer_iters``, ``inner_iters``, ``gamma0`` and ``mu_upper_bound``.

    :param demands: Per-sector demand profiles aggregated externally.
    :param horizon: Horizon length ``H``.
    :param rho: Inner ADMM penalty.
    :param outer_iters: Outer (dual) iteration cap.
    :param inner_iters: Inner (primal) ADMM passes per outer step.
    :param gamma0: Robbins-Monro stepsize coefficient.
    :param mu_upper_bound: Projection ceiling on every dual coordinate.
    :param abs_tol: Outer-loop convergence tolerance.
    :param record_history: Persist per-iteration diagnostics on the result.
    """

    demands: list[SectorDemand]
    horizon: int = 1
    rho: float = 1.0
    outer_iters: int = 200
    inner_iters: int = 8
    gamma0: float = 1.0
    mu_upper_bound: float = 1.0e6
    abs_tol: float = 1e-3
    record_history: bool = False


@dataclass
class ConsensusWaterfallADMMSpecRequest(OptimizationMessage):
    """Coordinator -> participant: request its :class:`CPSpec`."""


@dataclass
class ConsensusWaterfallADMMSpecReply(OptimizationMessage):
    """Participant -> coordinator: the participant's :class:`CPSpec`."""

    spec: CPSpec


@dataclass
class ConsensusWaterfallADMMResult(OptimizationMessage):
    """Coordinator -> participant: the converged regulation factor.

    :param r: Length-``H`` regulation factor in ``[0, 1]`` for this CP.
    """

    r: np.ndarray


# ---------------------------------------------------------------------------
# Participant
# ---------------------------------------------------------------------------


class ConsensusWaterfallADMMParticipant(DistributedAlgorithm):
    """Coupling-point participant in a consensus-waterfall-ADMM round.

    The participant role is identical to
    :class:`~distributed_resource_optimization.algorithm.waterfall_admm.core.WaterfallADMMParticipant`:
    publish the :class:`CPSpec` on request, store the converged factor
    when it arrives.  The dual ascent runs entirely on the coordinator
    so participants need no per-round state beyond their static spec.

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
        if isinstance(message_data, ConsensusWaterfallADMMSpecRequest):
            carrier.reply_to_other(
                ConsensusWaterfallADMMSpecReply(spec=self.spec), meta
            )
        elif isinstance(message_data, ConsensusWaterfallADMMResult):
            self.r = np.asarray(message_data.r, dtype=float).copy()


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class ConsensusWaterfallADMMCoordinator(Coordinator):
    """Coordinator that drives a consensus-waterfall-ADMM round.

    Each round:

    1. Request a :class:`CPSpec` from every registered participant in
       parallel and await all
       :class:`ConsensusWaterfallADMMSpecReply` replies.
    2. Run :func:`solve_cp_consensus_waterfall_admm` locally on the
       gathered specs plus the demands carried in the start message.
    3. Dispatch the converged per-CP factor back to each participant
       via :class:`ConsensusWaterfallADMMResult`.

    :returns: The full :class:`CPAdmmResult`.
    """

    async def start_optimization(
        self,
        carrier: "Carrier",
        message_data: ConsensusWaterfallADMMStart,
        meta: Any,
    ) -> CPAdmmResult:
        participant_addrs = carrier.others("coordinator")

        spec_futures = [
            carrier.send_awaitable(ConsensusWaterfallADMMSpecRequest(), addr)
            for addr in participant_addrs
        ]
        spec_replies = await asyncio.gather(*spec_futures)
        specs = [reply.spec for reply in spec_replies]

        result = solve_cp_consensus_waterfall_admm(
            cps=specs,
            demands=message_data.demands,
            horizon=message_data.horizon,
            rho=message_data.rho,
            outer_iters=message_data.outer_iters,
            inner_iters=message_data.inner_iters,
            gamma0=message_data.gamma0,
            mu_upper_bound=message_data.mu_upper_bound,
            abs_tol=message_data.abs_tol,
            record_history=message_data.record_history,
        )

        send_tasks = [
            carrier.send_to_other(
                ConsensusWaterfallADMMResult(r=result.factor_by_cp[spec.cp_id]),
                addr,
            )
            for addr, spec in zip(participant_addrs, specs)
        ]
        await asyncio.gather(*send_tasks)

        return result


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_consensus_waterfall_admm_participant(
    cp_id: str,
    capacity_by_sector: dict[str, float],
) -> ConsensusWaterfallADMMParticipant:
    """Create a :class:`ConsensusWaterfallADMMParticipant` from raw CP parameters.

    :param cp_id: Stable identifier (must match the participant addressed
        in :attr:`CPAdmmResult.factor_by_cp`).
    :param capacity_by_sector: Per-sector signed effective capacity in
        load convention.
    """
    spec = CPSpec(
        cp_id=cp_id,
        capacity_by_sector=dict(capacity_by_sector),
    )
    return ConsensusWaterfallADMMParticipant(spec=spec)


def create_consensus_waterfall_admm_coordinator() -> ConsensusWaterfallADMMCoordinator:
    """Create a :class:`ConsensusWaterfallADMMCoordinator`."""
    return ConsensusWaterfallADMMCoordinator()


def create_consensus_waterfall_admm_start(
    demands: list[SectorDemand],
    *,
    horizon: int = 1,
    rho: float = 1.0,
    outer_iters: int = 200,
    inner_iters: int = 8,
    gamma0: float = 1.0,
    mu_upper_bound: float = 1.0e6,
    abs_tol: float = 1e-3,
    record_history: bool = False,
) -> ConsensusWaterfallADMMStart:
    """Create a :class:`ConsensusWaterfallADMMStart` message.

    All keyword arguments map directly onto
    :func:`solve_cp_consensus_waterfall_admm`.
    """
    return ConsensusWaterfallADMMStart(
        demands=list(demands),
        horizon=horizon,
        rho=rho,
        outer_iters=outer_iters,
        inner_iters=inner_iters,
        gamma0=gamma0,
        mu_upper_bound=mu_upper_bound,
        abs_tol=abs_tol,
        record_history=record_history,
    )
