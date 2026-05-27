"""Replicated-kernel distributed sum-sharing ADMM for the lexicographic cascade.

A hand-rolled primal-dual realisation of the :math:`\\Pi`-round
lexicographic cascade that solves each round's LP via Boyd et al.
§7.3 sharing ADMM — *sum coupling*, not per-CP consensus.  The two
modules in this package that share the cascade outer structure differ
only in how a single round's LP is realised:

* :mod:`~distributed_resource_optimization.algorithm.lexicographic_cascade`
  hands the round's LP to ``cvxpy`` and relies on a centralised QP
  solver.
* :mod:`...distributed_lexicographic_cascade` (this module) runs the
  round as a finite sharing-ADMM inner loop with a hand-coded closed
  form, eliminating the dependency on the QP solver and giving every
  CP the option to *replicate the kernel locally* and arrive at the
  identical commitment from the gossiped peer view.

Algorithm
---------

For each priority tier :math:`\\tau = 1, \\dots, \\Pi` we solve

.. math::

    \\begin{aligned}
    \\text{maximise} \\quad & \\sum_{s,k} \\sigma_{s,\\tau}^k \\\\
    \\text{subject to} \\quad & r_i \\in [0,1]^K,\\;
        \\sigma_{s,\\tau}^k \\in [0, D_{s,\\tau}^k] \\\\
    & \\sigma_{s,\\tau}^k + \\sum_i r_i^k c_{i,s} \\;\\le\\;
        B_s^k - \\theta_{s,\\tau-1}^k \\quad \\forall s, k,
    \\end{aligned}

via the Boyd §7.3 sharing-ADMM template.  Setting
:math:`x_i^k = r_i^k\\, c_i` (CP :math:`i`'s contribution to the
sector-vector aggregate at step :math:`k`) and
:math:`\\bar x = \\frac{1}{N} \\sum_i x_i`, the three updates per
inner iteration are:

* **Per-CP projection** onto the ray
  :math:`\\{r\\, c_i : r \\in [0,1]\\}` (closed form):

  .. math::

      r_i^{\\,+} = \\Pi_{[0,1]}\\!\\left[
          \\frac{c_i^\\top\\,(x_i - \\bar x + z - u)}
                {(\\rho + \\alpha/\\rho)\\,\\|c_i\\|_2^2 /\\,\\rho}\\right]

  with :math:`\\alpha \\geq 0` a tiny regulariser that breaks the
  flat-objective degeneracy and biases ties toward the minimum-norm
  ``r``.  Setting :math:`\\alpha = 0` recovers the bare sharing-ADMM
  closed form.

* **Shared :math:`(z, \\sigma)` update per sector-step cell** (also
  closed form — a 2D box-constrained QP per cell):

  .. math::

      (z, \\sigma)^{\\,+} = \\arg\\min_{z, \\sigma \\in [0, D]}
          -\\sigma + \\frac{N\\rho}{2}\\|z - \\bar x - u\\|^2
          \\quad \\text{s.t.}\\ \\sigma + z \\le B - \\theta.

  Three-region case analysis (non-binding / binding-interior /
  binding-corner) gives the closed form implemented in
  :func:`_z_sigma_cell_update`.

* **Shared scaled-dual update** (one ``u`` vector for the whole
  cascade, not per-CP):

  .. math::

      u^{\\,+} = u + \\bar x^{\\,+} - z^{\\,+}.

After the inner loop converges (primal residual
:math:`\\|\\bar x - z\\|` and dual residual
:math:`\\rho\\|z - z_{\\text{prev}}\\|` both below
``inner_abs_tol``), the round's :math:`\\sigma^\\star` is the
final :math:`\\sigma`, the threshold updates as
:math:`\\theta \\leftarrow \\theta + \\sigma^\\star`, and the cascade
moves to round :math:`\\tau + 1`.

Why sum-sharing instead of consensus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An earlier draft of this module used the *consensus* form
(:math:`z = \\bar x + \\bar u`, :math:`u_i \\leftarrow u_i + x_i - z`),
whose fixed point requires :math:`x_i = z` for every CP — equivalent
to demanding that every contribution vector :math:`r_i c_i` lies on
the same ray in sector-space.  That formulation works only for
*parallel* capacity vectors and silently collapses to :math:`r = 0`
for mixed-direction caps (e.g. P2H units with different efficiencies,
P2H + CHP, mixed-technology fleets).

The sum-sharing formulation has no such alignment requirement.  The
only cross-CP coupling is the scalar inequality
:math:`\\sum_i r_i c_i + \\sigma \\le B - \\theta` per
``(sector, step)`` — different CPs are free to contribute different
vectors to the sum.  Convergence to the LP optimum follows from
standard convexity of the round LP and the Boyd §7.3 theorem,
without any prerequisite on the capacity vectors.

Cross-tier warm-starting
~~~~~~~~~~~~~~~~~~~~~~~~

The shared state :math:`(x, \\bar x, z, u)` persists across tier
transitions.  Round :math:`\\tau + 1` begins from round
:math:`\\tau`'s converged iterate, which is feasible for the new
round because the cleared service is now baked into :math:`\\theta`.
In practice the late rounds need far fewer inner iterations than the
early ones.

Replicated-kernel decentralisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The kernel is fully synchronous and deterministic — identical inputs
give identical iterate sequences.  Every CP can therefore run
:func:`solve_cp_distributed_lexicographic_cascade` locally on its
gossiped peer view and read off its own slot of the final ``r``,
without a coordinator.  In this module the kernel is also wrapped by
a :class:`DistributedLexicographicCascadeCoordinator` to fit the
repo's coordinator/participant transport pattern (mirroring
:class:`~distributed_resource_optimization.algorithm.waterfall_admm.core.WaterfallADMMCoordinator`),
but the kernel is the load-bearing piece — the coordinator is just
the in-process driver.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core import Coordinator, DistributedAlgorithm, OptimizationMessage
from ..waterfall_admm.core import CPAdmmResult, CPSpec, SectorDemand

if TYPE_CHECKING:
    from ...carrier.core import Carrier


# ---------------------------------------------------------------------------
# Closed-form (z, sigma) cell update
# ---------------------------------------------------------------------------


def _z_sigma_cell_update(
    target_z: float,
    slack_max: float,
    demand: float,
    N: int,
    rho: float,
) -> tuple[float, float]:
    """Closed-form 2D QP for one ``(sector, step)`` cell.

    In Boyd §7.3 the shared variable is :math:`z = s / N` where
    :math:`s = \\sum_i x_i`, so the original sum-coupling
    :math:`\\sigma + s \\le \\text{slack\\_max}` becomes
    :math:`\\sigma + N z \\le \\text{slack\\_max}`.  The cell-wise
    sub-problem is therefore

    .. math::

        \\min_{z, \\sigma}\\;
            -\\sigma + \\tfrac{N\\rho}{2}(z - \\text{target})^2
        \\quad\\text{s.t.}\\;
        \\sigma + N z \\le \\text{slack\\_max},\\;
        \\sigma \\in [0, \\text{demand}].

    Stationarity gives interior :math:`z = \\text{target} - 1/\\rho`
    when the inequality binds; the KKT conditions partition the
    optimum into four regions (non-binding / binding-corner-upper /
    binding-interior / binding-corner-lower) handled by the case
    analysis below.

    :returns: ``(z*, sigma*)``.
    """
    if demand <= 0.0:
        # No tier-tau demand here -> sigma is forced to 0; constraint
        # collapses to N z <= slack_max.
        z_max = slack_max / N
        return min(target_z, z_max), 0.0

    # Non-binding region: the inequality has slack to spare so the
    # unconstrained quadratic minimum z = target and the linear-in-
    # sigma objective pushes sigma to its upper box D.  Feasible iff
    # demand + N target <= slack_max.
    if demand + N * target_z <= slack_max:
        return target_z, demand

    # Binding region: sigma + N z = slack_max.  The unconstrained
    # interior optimum for sigma (no box) is
    # sigma_b = slack_max - N target + N/rho.
    sigma_b = slack_max - N * target_z + N / rho
    if sigma_b >= demand:
        # sigma at upper box -> sigma = D, z = (slack_max - D) / N.
        return (slack_max - demand) / N, demand
    if sigma_b <= 0.0:
        # sigma at lower box -> sigma = 0, z = slack_max / N.
        return slack_max / N, 0.0
    # Interior sigma, z = target - 1/rho.
    return target_z - 1.0 / rho, sigma_b


# ---------------------------------------------------------------------------
# Pure-compute kernel
# ---------------------------------------------------------------------------


def solve_cp_distributed_lexicographic_cascade(
    cps: list[CPSpec],
    demands: list[SectorDemand],
    *,
    horizon: int = 1,
    rho: float = 1.0,
    inner_iters_max: int = 500,
    inner_abs_tol: float = 1.0e-4,
    r_regularization: float = 1.0e-1,
    record_history: bool = False,
) -> CPAdmmResult:
    """Solve the lexicographic cascade via sum-sharing ADMM.

    Exactly :math:`\\Pi` outer rounds are executed (one per priority
    tier present in *demands*, in ascending tier order).  Each round
    runs the sharing-ADMM inner loop until both primal residual
    :math:`\\|\\bar x - z\\|` and dual residual
    :math:`\\rho\\|z - z_{\\text{prev}}\\|` fall below
    ``inner_abs_tol`` (or ``inner_iters_max`` is hit, in which case
    the round closes at a feasible suboptimum).  At round closure the
    threshold advances as :math:`\\theta \\leftarrow \\theta + \\sigma`
    in closed form from the inner-loop's converged :math:`\\sigma`.

    The kernel is fully synchronous and deterministic.  Two CPs given
    identical input data trace identical iterate sequences and produce
    identical ``r``, so each CP can run the kernel locally on its
    gossiped peer view (replicated-kernel decentralisation) and skip
    the coordinator entirely.

    :param cps: Every coupling point in the cross-sector component.
    :param demands: Per-sector demand profiles aggregated externally.
        Sectors absent from *demands* are treated as having zero
        demand and zero base supply.
    :param horizon: Number of horizon steps ``H``.  Must be ``>= 1``.
    :param rho: Sharing-ADMM penalty.
    :param inner_iters_max: Cap on the per-round inner iteration
        count.  If hit, the round closes with the current iterate's
        :math:`\\sigma`, which is feasible-suboptimal.
    :param inner_abs_tol: Convergence threshold on
        :math:`\\max(\\|\\bar x - z\\|_\\infty,
        \\rho\\,\\|z - z_{\\text{prev}}\\|_\\infty,
        \\max|\\Delta r|)`.
    :param r_regularization: Proximal damping coefficient
        :math:`\\alpha \\ge 0` applied to the *iterate step*, not to
        the iterate value: the per-CP local subproblem carries the
        added term :math:`(\\alpha / 2) \\|r - r_{\\text{prev}}\\|^2`.
        At every fixed point :math:`r = r_{\\text{prev}}`, so the
        :math:`\\alpha` term cancels and the asymptote is the bare
        sharing-ADMM optimum — :math:`\\alpha` controls step size,
        not the saddle.  Default ``0.1`` is the empirical sweet
        spot for convergence speed; at :math:`\\alpha = 0` the
        algorithm is mathematically correct but the iterate can
        oscillate on a degenerate optimal face (multiple
        :math:`r`-vectors give the same :math:`\\sigma`) and the
        per-iter convergence test will not trip.
    :param record_history: When ``True`` the returned result carries
        per-round inner-iteration counts and the final
        :math:`\\theta` matrix on :attr:`CPAdmmResult.history`.
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

    # ---- persistent shared state across tier transitions ----
    r_curr = np.zeros((N, H), dtype=float)
    x = np.zeros((N, n_sec, H), dtype=float)
    x_bar = np.zeros((n_sec, H), dtype=float)
    z = np.zeros((n_sec, H), dtype=float)
    u = np.zeros((n_sec, H), dtype=float)  # SHARED dual (single vector, not per-CP)
    theta = np.zeros((n_sec, H), dtype=float)

    alpha = float(r_regularization)
    rho_f = float(rho)

    sigma_per_tier: dict[int, np.ndarray] = {}
    per_round_iters: list[int] = []
    per_round_converged: list[bool] = []
    history_primal: list[float] = []
    history_dual: list[float] = []
    history_r_changes: list[float] = []

    for tier in all_tiers:
        t = tier_idx[tier]
        D_tau = D[:, t, :]
        slack_max = base_supply - theta  # B - theta_{tau-1}

        primal_res = float("inf")
        dual_res = float("inf")
        r_change = float("inf")
        inner_iter = 0
        converged_round = False
        sigma = np.zeros((n_sec, H), dtype=float)

        for inner_iter in range(inner_iters_max):
            r_prev = r_curr.copy()
            z_prev = z.copy()

            # ----- per-CP projection with proximal damping toward r_prev -----
            # Sharing-ADMM target: x_i - x_bar + z - u.
            # Adding (alpha/2) ||r - r_prev||^2 to the local subproblem
            # gives the closed form
            #   r_i = clip((rho c_i.target + alpha r_prev_i) / (rho ||c_i||^2 + alpha), 0, 1).
            # At any fixed point r_i = r_prev_i so the alpha terms cancel
            # and the asymptote is the bare sharing-ADMM optimum — alpha
            # damps step size, does not bias the saddle.
            for i in range(N):
                if cap_norm_sq[i] < 1e-12:
                    continue
                for k in range(H):
                    target_ik = x[i, :, k] - x_bar[:, k] + z[:, k] - u[:, k]
                    num = rho_f * float(cap[i] @ target_ik) + alpha * r_prev[i, k]
                    den = rho_f * float(cap_norm_sq[i]) + alpha
                    r_ik = num / den
                    if r_ik < 0.0:
                        r_ik = 0.0
                    elif r_ik > 1.0:
                        r_ik = 1.0
                    r_curr[i, k] = r_ik
                    x[i, :, k] = r_ik * cap[i]

            # ----- shared mean of contributions -----
            x_bar = x.sum(axis=0) / float(N)

            # ----- shared (z, sigma) update: closed-form 2D QP per cell -----
            target_z_mat = x_bar + u  # shape (n_sec, H)
            for s in range(n_sec):
                for k in range(H):
                    z_sk, sig_sk = _z_sigma_cell_update(
                        float(target_z_mat[s, k]),
                        float(slack_max[s, k]),
                        float(D_tau[s, k]),
                        N,
                        rho_f,
                    )
                    z[s, k] = z_sk
                    sigma[s, k] = sig_sk

            # ----- shared scaled-dual update -----
            u = u + x_bar - z

            # ----- residuals + convergence test -----
            primal_res = float(np.linalg.norm(x_bar - z, ord=np.inf)) if z.size else 0.0
            dual_res = rho_f * float(np.linalg.norm(z - z_prev, ord=np.inf)) if z.size else 0.0
            r_change = float(np.max(np.abs(r_curr - r_prev))) if r_curr.size else 0.0

            if (
                primal_res < inner_abs_tol
                and dual_res < inner_abs_tol
                and r_change < inner_abs_tol
            ):
                converged_round = True
                break

        # ----- close the tier -----
        sigma_per_tier[tier] = sigma.copy()
        theta = theta + sigma

        per_round_iters.append(inner_iter + 1)
        per_round_converged.append(converged_round)
        if record_history:
            history_primal.append(primal_res)
            history_dual.append(dual_res)
            history_r_changes.append(r_change)

    converged = bool(all(per_round_converged))
    total_iters = int(sum(per_round_iters))

    factor_by_cp: dict[str, np.ndarray] = {
        c.cp_id: r_curr[i].copy() for i, c in enumerate(cps)
    }
    served_by_sector_tier = {
        d.sector: {
            t: sigma_per_tier[t][sec_idx[d.sector], :].copy()
            for t in all_tiers
            if t in tier_idx
        }
        for d in demands
    }

    history: dict[str, Any] = {}
    if record_history:
        history = {
            "per_round_iters": list(per_round_iters),
            "per_round_primal_residuals": history_primal,
            "per_round_dual_residuals": history_dual,
            "per_round_r_changes": history_r_changes,
            "theta_final": theta.copy(),
            "sigma_per_tier": {t: v.copy() for t, v in sigma_per_tier.items()},
        }

    return CPAdmmResult(
        factor_by_cp=factor_by_cp,
        served_by_sector_tier=served_by_sector_tier,
        iterations=total_iters,
        primal_residual=primal_res,
        dual_residual=dual_res,
        converged=converged,
        history=history,
    )


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class DistributedLexicographicCascadeStart(OptimizationMessage):
    """External start payload handed to the coordinator.

    Carries the aggregate demand picture (which the coordinator already
    has — sector demands and base supply do *not* come from the CPs)
    plus the algorithm hyperparameters.  The coordinator never
    requests per-CP capacities; each participant keeps its own
    :class:`CPSpec` private and runs the per-CP closed-form update
    locally.

    :param demands: Per-sector demand profiles aggregated externally.
    :param horizon: Horizon length ``H``.
    :param rho: Sharing-ADMM penalty.
    :param inner_iters_max: Per-round inner iteration cap.
    :param inner_abs_tol: Per-step convergence tolerance.
    :param r_regularization: Quadratic ``r``-norm penalty.
    :param record_history: Persist per-round diagnostics on the result.
    """

    demands: list[SectorDemand]
    horizon: int = 1
    rho: float = 1.0
    inner_iters_max: int = 500
    inner_abs_tol: float = 1.0e-4
    r_regularization: float = 1.0e-1
    record_history: bool = False


@dataclass
class DistributedLexicographicCascadeInit(OptimizationMessage):
    """Coordinator -> participant: one-shot setup at the cascade start.

    Establishes the global ``(sector, step)`` indexing that participants
    use when interpreting per-iteration correction broadcasts and
    building their reply contributions.  The participant keeps its
    own capacity vector private; only the *coordinate frame* (sector
    ordering, horizon, rho, alpha) is shared.

    :param sectors: Ordered list of sector identifiers.  Participants
        construct their contribution vector ``x_i`` in this ordering;
        a participant's capacity for a sector outside this list is
        silently ignored (it does not appear anywhere in the cascade
        LP and does not influence the per-CP projection).
    :param horizon: Horizon length ``H``.
    :param rho: Sharing-ADMM penalty.
    :param r_regularization: Quadratic ``r``-norm penalty
        :math:`\\alpha` for the participant's local closed-form
        projection.
    """

    sectors: list[str]
    horizon: int
    rho: float
    r_regularization: float


@dataclass
class DistributedLexicographicCascadeInitAck(OptimizationMessage):
    """Participant -> coordinator: ready ack with the participant's id."""

    cp_id: str


@dataclass
class DistributedLexicographicCascadeIter(OptimizationMessage):
    """Coordinator -> participant: per-iteration correction broadcast.

    The correction is the *shared* part of the sharing-ADMM x-update
    target — every participant receives the same ``correction``.
    Each participant adds its own private ``x_i`` to form the local
    target ``x_i + correction`` and projects onto its own capacity
    ray to produce the new ``x_i``.

    :param correction: Shape ``(n_sec, H)``, equal to ``z - u - x_bar``
        in the coordinator's current shared state.
    """

    correction: np.ndarray


@dataclass
class DistributedLexicographicCascadeAnswer(OptimizationMessage):
    """Participant -> coordinator: this iteration's contribution.

    :param x: Shape ``(n_sec, H)``, equal to ``r_i * cap_i`` in the
        global sector ordering established by
        :class:`DistributedLexicographicCascadeInit`.
    :param r_change: ``max_k |r_i^k_new - r_i^k_prev|`` at this
        participant — sent up so the coordinator can run the same
        ``max-r-step`` convergence test as the in-process reference
        kernel without having to know the per-CP factors.
    """

    x: np.ndarray
    r_change: float


@dataclass
class DistributedLexicographicCascadeDone(OptimizationMessage):
    """Coordinator -> participant: cascade complete; report final ``r``."""


@dataclass
class DistributedLexicographicCascadeDoneReply(OptimizationMessage):
    """Participant -> coordinator: final regulation factor.

    Sent in response to :class:`DistributedLexicographicCascadeDone`
    so the coordinator can assemble the global ``factor_by_cp``
    result.  Participants also keep their own ``r`` on
    :attr:`DistributedLexicographicCascadeParticipant.r` and would
    function correctly without this final round-trip; the round-trip
    only exists so the coordinator's return value matches the other
    variants' result-dataclass shape.
    """

    cp_id: str
    r: np.ndarray


# ---------------------------------------------------------------------------
# Participant: runs the per-CP closed-form x-update locally
# ---------------------------------------------------------------------------


class DistributedLexicographicCascadeParticipant(DistributedAlgorithm):
    """Coupling-point participant in a true-distributed cascade round.

    The participant *never publishes* its capacity vector.  At cascade
    start the coordinator broadcasts the global sector ordering;
    the participant projects its private ``capacity_by_sector`` onto
    that ordering once and then responds to each per-iteration
    correction broadcast with its own closed-form ``r_i``-projection
    locally.  The shared state ``(x_bar, z, u, \\theta)`` lives only
    on the coordinator; the participant only knows its own
    ``x_i``, ``cap_i``, and the global ``(\\rho, \\alpha)``
    hyperparameters.

    :param cp_id: Stable identifier (used by the coordinator to
        index the final ``factor_by_cp`` result).
    :param capacity_by_sector: Per-sector signed effective capacity
        in load convention.  *Private* — never sent over the wire.
    """

    def __init__(self, cp_id: str, capacity_by_sector: dict[str, float]) -> None:
        self.cp_id = cp_id
        self._capacity_by_sector = dict(capacity_by_sector)
        # Lazy: built on receipt of Init.
        self._cap_vec: np.ndarray = np.zeros(0)
        self._cap_norm_sq: float = 0.0
        self._rho: float = 1.0
        self._alpha: float = 0.0
        self._horizon: int = 0
        self._x_i: np.ndarray = np.zeros((0, 0))
        self._last_r_change: float = 0.0
        # Public: final regulation factor exposed after the cascade ends.
        self.r: np.ndarray = np.array([])

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: Any,
        meta: Any,
    ) -> None:
        if isinstance(message_data, DistributedLexicographicCascadeInit):
            self._on_init(message_data)
            carrier.reply_to_other(
                DistributedLexicographicCascadeInitAck(cp_id=self.cp_id), meta
            )
        elif isinstance(message_data, DistributedLexicographicCascadeIter):
            self._on_iter(message_data.correction)
            carrier.reply_to_other(
                DistributedLexicographicCascadeAnswer(
                    x=self._x_i, r_change=self._last_r_change
                ),
                meta,
            )
        elif isinstance(message_data, DistributedLexicographicCascadeDone):
            carrier.reply_to_other(
                DistributedLexicographicCascadeDoneReply(
                    cp_id=self.cp_id, r=self.r
                ),
                meta,
            )

    # ------------------------------------------------------------------
    # Local kernel (private)
    # ------------------------------------------------------------------

    def _on_init(self, msg: DistributedLexicographicCascadeInit) -> None:
        sectors = list(msg.sectors)
        H = int(msg.horizon)
        cap_vec = np.array(
            [float(self._capacity_by_sector.get(s, 0.0)) for s in sectors],
            dtype=float,
        )
        self._cap_vec = cap_vec
        self._cap_norm_sq = float(cap_vec @ cap_vec)
        self._rho = float(msg.rho)
        self._alpha = float(msg.r_regularization)
        self._horizon = H
        self._x_i = np.zeros((cap_vec.size, H), dtype=float)
        self.r = np.zeros(H, dtype=float)

    def _on_iter(self, correction: np.ndarray) -> None:
        """Apply one Boyd §7.3 sharing-ADMM x-update locally with proximal damping.

        ``target = x_i + correction`` where ``correction = z - u - x_bar``
        is broadcast by the coordinator.  The new ``r_i`` follows from
        projecting onto the ray ``{r * cap_i : r ∈ [0, 1]}`` plus a
        proximal damping term ``(alpha/2)(r - r_prev)^2`` that biases
        the iterate step (not the iterate value): the closed form is

        .. code-block:: text

            r_k = clip((rho c.target + alpha r_prev_k)
                       / (rho ||c||^2 + alpha), 0, 1).

        At any fixed point ``r_k = r_prev_k`` so the alpha terms cancel
        exactly and the asymptote is the bare sharing-ADMM optimum —
        alpha damps step size but does not bias the saddle.

        Also records the per-step ``max_k |r_new - r_prev|`` so the
        reply carries the same convergence signal the reference
        kernel checks.
        """
        cap = self._cap_vec
        den = self._rho * self._cap_norm_sq + self._alpha
        H = self._horizon
        if den <= 0.0 or cap.size == 0:
            # Idle CP (zero capacity); contribute zero forever.
            self._last_r_change = 0.0
            return
        r_prev = self.r.copy()
        target = self._x_i + np.asarray(correction, dtype=float)
        for k in range(H):
            num = self._rho * float(cap @ target[:, k]) + self._alpha * r_prev[k]
            r_k = num / den
            if r_k < 0.0:
                r_k = 0.0
            elif r_k > 1.0:
                r_k = 1.0
            self.r[k] = r_k
            self._x_i[:, k] = r_k * cap
        self._last_r_change = float(np.max(np.abs(self.r - r_prev))) if H > 0 else 0.0


# ---------------------------------------------------------------------------
# Coordinator: holds shared state, never sees CP capacities
# ---------------------------------------------------------------------------


class DistributedLexicographicCascadeCoordinator(Coordinator):
    """Coordinator for the true-distributed cascade.

    Each cascade round runs a sharing-ADMM inner loop in which the
    coordinator only handles the *shared* state ``(x_bar, z, u, \\theta)``
    and the *per-cell* shared-variable update.  The per-CP closed-form
    projection runs at each participant.  The coordinator's per-
    iteration round-trip is:

    1. Compute ``correction = z - u - x_bar``.
    2. Broadcast a
       :class:`DistributedLexicographicCascadeIter` (single payload,
       identical for every participant) and await each participant's
       :class:`DistributedLexicographicCascadeAnswer`.
    3. Aggregate :math:`\\bar x = \\frac{1}{N} \\sum_i x_i`.
    4. Update ``(z, \\sigma)`` per cell in closed form and update
       the shared dual ``u``.
    5. Check the primal/dual residuals.

    At cascade end the coordinator dispatches
    :class:`DistributedLexicographicCascadeDone` and assembles
    ``factor_by_cp`` from the participants' final-reply messages.

    :returns: The full :class:`CPAdmmResult` (aggregate plus per-CP
        factors assembled from the final-reply phase).
    """

    async def start_optimization(
        self,
        carrier: "Carrier",
        message_data: DistributedLexicographicCascadeStart,
        meta: Any,
    ) -> CPAdmmResult:
        participant_addrs = carrier.others("coordinator")
        N = len(participant_addrs)

        # ----- build shared state from the start payload -----
        demands = message_data.demands
        H = int(message_data.horizon)
        if H < 1:
            raise ValueError("horizon must be >= 1")

        all_sectors = sorted({d.sector for d in demands})
        if not all_sectors:
            raise ValueError("no sectors found in demands")
        n_sec = len(all_sectors)
        sec_idx = {s: i for i, s in enumerate(all_sectors)}

        all_tiers = sorted({t for d in demands for t in d.demand_by_tier})

        D = np.zeros((n_sec, max(len(all_tiers), 1), H), dtype=float)
        base_supply = np.zeros((n_sec, H), dtype=float)
        tier_idx = {t: i for i, t in enumerate(all_tiers)}
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

        # ----- one-shot Init broadcast -----
        init_msg = DistributedLexicographicCascadeInit(
            sectors=all_sectors,
            horizon=H,
            rho=float(message_data.rho),
            r_regularization=float(message_data.r_regularization),
        )
        ack_futures = [
            carrier.send_awaitable(init_msg, addr) for addr in participant_addrs
        ]
        await asyncio.gather(*ack_futures)

        # ----- shared state -----
        x_bar = np.zeros((n_sec, H), dtype=float)
        z = np.zeros((n_sec, H), dtype=float)
        u = np.zeros((n_sec, H), dtype=float)
        theta = np.zeros((n_sec, H), dtype=float)

        rho_f = float(message_data.rho)
        inner_iters_max = int(message_data.inner_iters_max)
        inner_abs_tol = float(message_data.inner_abs_tol)
        record_history = bool(message_data.record_history)

        sigma_per_tier: dict[int, np.ndarray] = {}
        per_round_iters: list[int] = []
        per_round_converged: list[bool] = []
        history_primal: list[float] = []
        history_dual: list[float] = []

        if not all_tiers:
            # No demand at all -> short-circuit: tell participants we're done
            # and assemble an empty result.
            return await self._finalize(
                carrier=carrier,
                participant_addrs=participant_addrs,
                served_by_sector_tier={d.sector: {} for d in demands},
                iterations=0,
                primal_residual=0.0,
                dual_residual=0.0,
                converged=True,
                history={},
            )

        # ----- cascade rounds -----
        primal_res = 0.0
        dual_res = 0.0

        for tier in all_tiers:
            t = tier_idx[tier]
            D_tau = D[:, t, :]
            slack_max = base_supply - theta
            sigma = np.zeros((n_sec, H), dtype=float)
            inner_iter = 0
            converged_round = False

            for inner_iter in range(inner_iters_max):
                z_prev = z.copy()

                # ----- broadcast Iter, await Answers, aggregate x_bar -----
                correction = z - u - x_bar
                iter_msg = DistributedLexicographicCascadeIter(correction=correction)
                futures = [
                    carrier.send_awaitable(iter_msg, addr)
                    for addr in participant_addrs
                ]
                replies = await asyncio.gather(*futures)

                if N > 0:
                    x_sum = np.zeros((n_sec, H), dtype=float)
                    max_r_change = 0.0
                    for reply in replies:
                        x_sum += np.asarray(reply.x, dtype=float)
                        if float(reply.r_change) > max_r_change:
                            max_r_change = float(reply.r_change)
                    x_bar_new = x_sum / float(N)
                else:
                    x_bar_new = np.zeros((n_sec, H), dtype=float)
                    max_r_change = 0.0

                # ----- shared (z, sigma) update: closed-form per cell -----
                target_z_mat = x_bar_new + u
                for s in range(n_sec):
                    for k in range(H):
                        z_sk, sig_sk = _z_sigma_cell_update(
                            float(target_z_mat[s, k]),
                            float(slack_max[s, k]),
                            float(D_tau[s, k]),
                            N if N > 0 else 1,
                            rho_f,
                        )
                        z[s, k] = z_sk
                        sigma[s, k] = sig_sk

                # ----- shared dual update -----
                u = u + x_bar_new - z

                # ----- residuals: same criterion as the reference kernel -----
                primal_res = float(np.max(np.abs(x_bar_new - z))) if z.size else 0.0
                dual_res = rho_f * (
                    float(np.max(np.abs(z - z_prev))) if z.size else 0.0
                )
                x_bar = x_bar_new

                if (
                    primal_res < inner_abs_tol
                    and dual_res < inner_abs_tol
                    and max_r_change < inner_abs_tol
                ):
                    converged_round = True
                    break

            sigma_per_tier[tier] = sigma.copy()
            theta = theta + sigma
            per_round_iters.append(inner_iter + 1)
            per_round_converged.append(converged_round)
            if record_history:
                history_primal.append(primal_res)
                history_dual.append(dual_res)

        served_by_sector_tier = {
            d.sector: {
                t: sigma_per_tier[t][sec_idx[d.sector], :].copy()
                for t in all_tiers
                if t in tier_idx
            }
            for d in demands
        }

        history: dict[str, Any] = {}
        if record_history:
            history = {
                "per_round_iters": list(per_round_iters),
                "per_round_primal_residuals": history_primal,
                "per_round_dual_residuals": history_dual,
                "theta_final": theta.copy(),
                "sigma_per_tier": {t: v.copy() for t, v in sigma_per_tier.items()},
            }

        return await self._finalize(
            carrier=carrier,
            participant_addrs=participant_addrs,
            served_by_sector_tier=served_by_sector_tier,
            iterations=int(sum(per_round_iters)),
            primal_residual=primal_res,
            dual_residual=dual_res,
            converged=bool(all(per_round_converged)),
            history=history,
        )

    async def _finalize(
        self,
        *,
        carrier: "Carrier",
        participant_addrs: list[int],
        served_by_sector_tier: dict[str, dict[int, np.ndarray]],
        iterations: int,
        primal_residual: float,
        dual_residual: float,
        converged: bool,
        history: dict[str, Any],
    ) -> CPAdmmResult:
        """Send Done to every participant, gather final r, build the result."""
        done_futures = [
            carrier.send_awaitable(
                DistributedLexicographicCascadeDone(), addr
            )
            for addr in participant_addrs
        ]
        done_replies = await asyncio.gather(*done_futures)
        factor_by_cp = {
            reply.cp_id: np.asarray(reply.r, dtype=float).copy()
            for reply in done_replies
        }

        return CPAdmmResult(
            factor_by_cp=factor_by_cp,
            served_by_sector_tier=served_by_sector_tier,
            iterations=iterations,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            converged=converged,
            history=history,
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_distributed_lexicographic_cascade_participant(
    cp_id: str,
    capacity_by_sector: dict[str, float],
) -> DistributedLexicographicCascadeParticipant:
    """Create a :class:`DistributedLexicographicCascadeParticipant`.

    The participant keeps *capacity_by_sector* private — it never
    leaves the participant's address space.  The coordinator only
    sees the per-iteration sum-aggregate :math:`x_i = r_i c_i` and
    the final regulation factor.

    :param cp_id: Stable identifier (used by the coordinator to
        index the final ``factor_by_cp`` result).
    :param capacity_by_sector: Per-sector signed effective capacity
        in load convention.
    """
    return DistributedLexicographicCascadeParticipant(
        cp_id=cp_id, capacity_by_sector=capacity_by_sector
    )


def create_distributed_lexicographic_cascade_coordinator() -> DistributedLexicographicCascadeCoordinator:
    """Create a :class:`DistributedLexicographicCascadeCoordinator`."""
    return DistributedLexicographicCascadeCoordinator()


def create_distributed_lexicographic_cascade_start(
    demands: list[SectorDemand],
    *,
    horizon: int = 1,
    rho: float = 1.0,
    inner_iters_max: int = 500,
    inner_abs_tol: float = 1.0e-4,
    r_regularization: float = 1.0e-1,
    record_history: bool = False,
) -> DistributedLexicographicCascadeStart:
    """Create a :class:`DistributedLexicographicCascadeStart` message.

    All keyword arguments map directly onto the inner ADMM hyper-
    parameters that the coordinator broadcasts to its participants.
    """
    return DistributedLexicographicCascadeStart(
        demands=list(demands),
        horizon=horizon,
        rho=rho,
        inner_iters_max=inner_iters_max,
        inner_abs_tol=inner_abs_tol,
        r_regularization=r_regularization,
        record_history=record_history,
    )
