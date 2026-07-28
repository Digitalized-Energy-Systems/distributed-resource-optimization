"""Replicated-kernel distributed sum-sharing ADMM for the lexicographic cascade.

A hand-rolled primal-dual realisation of the :math:`\\Pi`-round
lexicographic cascade that solves each round's LP via Boyd et al.
§7.3 sharing ADMM — *sum coupling*, not per-CP consensus.  This module
runs each round as a finite sharing-ADMM inner loop with a hand-coded
closed form, eliminating any dependency on a QP solver and giving every
CP the option to *replicate the kernel locally* and arrive at the
identical commitment from the gossiped peer view.  (A centralised
cvxpy-backed reference solver, ``solve_cp_lexicographic_cascade``, was
planned for cross-validation but is not yet implemented.)

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

The consensus form (:math:`z = \\bar x + \\bar u`,
:math:`u_i \\leftarrow u_i + x_i - z`) has a fixed point requiring
:math:`x_i = z` for every CP — i.e. every :math:`r_i c_i` on the same
ray — so it works only for parallel capacity vectors and collapses to
:math:`r = 0` for mixed-direction caps (P2H + CHP, mixed efficiencies).
Sum-sharing has no such requirement: the only cross-CP coupling is the
scalar :math:`\\sum_i r_i c_i + \\sigma \\le B - \\theta` per
``(sector, step)``, and convergence follows from Boyd §7.3 regardless
of the capacity vectors.

Shared state :math:`(x, \\bar x, z, u)` persists across tiers, so round
:math:`\\tau + 1` warm-starts from round :math:`\\tau`'s converged
(and now-feasible, via :math:`\\theta`) iterate — late rounds need far
fewer inner iterations.

Because the kernel is deterministic, every CP can run
:func:`solve_cp_distributed_lexicographic_cascade` locally on its
gossiped peer view and read off its own ``r`` without a coordinator.

For the transported (leader-follower) variant the same math is split
across the repo's classic-ADMM machinery: a
:class:`LexicographicCascadeGlobalActor` carries the shared ``(z, sigma,
u)`` updates and :class:`DistributedLexicographicCascadeCoordinator`
drives :class:`~...admm.core.ADMMGenericCoordinator`'s leader-follower
loop once per priority tier, while each
:class:`DistributedLexicographicCascadeParticipant` (follower) does only
its private ``r``-projection in response to a generic
:class:`~...admm.core.ADMMMessage`. The distributed path reproduces the
in-process kernel iterate-for-iterate.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...core import Coordinator, DistributedAlgorithm, OptimizationMessage
from ..core import (
    ADMMAnswer,
    ADMMGenericCoordinator,
    ADMMGlobalActor,
    ADMMMessage,
)
from ..types import CPAdmmResult, CPSpec, SectorDemand
from .kernel import _characteristic_scale, _local_regularization, _z_sigma_cell_update

if TYPE_CHECKING:
    from ....carrier.core import Carrier


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
    adaptive_rho: bool = True,
    rho_mu: float = 10.0,
    rho_tau: float = 2.0,
    minimize_usage: bool = False,
    normalize: bool = False,
    r_regularization_relative: bool = False,
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
    :param r_regularization: Regularisation weight :math:`\\alpha \\ge 0`
        in the local subproblem (always present in the denominator
        :math:`\\rho\\|c_i\\|^2 + \\alpha`); its *role* depends on
        ``minimize_usage``. Default ``0.1``.
    :param minimize_usage: When ``True`` the local term is a ridge toward
        zero :math:`(\\alpha/2)\\|r\\|^2`, which does not cancel at the
        fixed point and so selects the **minimum-usage** point of the
        degenerate optimal face — no surplus overshoot, unique solution.
        When ``False`` (default) the term is proximal toward the previous
        iterate :math:`(\\alpha/2)\\|r - r_{\\text{prev}}\\|^2`, which
        cancels at the fixed point, leaving the bare (non-unique) LP
        optimum (a demand-meeting ``r`` that may overshoot surplus
        capacity). ``True`` keeps the served :math:`\\sigma` — hence
        priority — identical but costs **substantially** more inner
        iterations (the dual must ramp against the ridge); the effect is
        mild for the synchronous kernel/coordinator but severe for the
        asynchronous gossip variant, so it is off by default. A larger
        ``r_regularization`` accelerates the ridge when enabled.
    :param adaptive_rho: When ``True`` (default), rebalance :math:`\\rho`
        each iteration per Boyd §3.4.1 — :math:`\\times \\tau` when
        primal :math:`>` ``rho_mu`` :math:`\\times` dual,
        :math:`/ \\tau` in the opposite case, with ``u`` rescaled
        inversely. Essential for the all-CPs-at-:math:`r=0` binding case
        (e.g. idle surplus P2G while another sector's tier binds), where
        the dual must ramp linearly to flip the
        :func:`_z_sigma_cell_update` branch — :math:`O(N / \\rho)`
        iterations at fixed :math:`\\rho`. Balancing collapses this to
        ``O(1)`` (≈12 vs ≈270 for ``N = 9``) and leaves balanced
        active-CP cases untouched.
    :param rho_mu: Residual-imbalance ratio that triggers a
        :math:`\\rho` adjustment (Boyd's ``mu``; default 10).
    :param rho_tau: Multiplicative :math:`\\rho` step (Boyd's
        ``tau^{incr} = tau^{decr}``; default 2).
    :param normalize: Divide the round's MW data by
        :func:`~.kernel._characteristic_scale` before solving and scale the
        served :math:`\\sigma` back afterwards. ``rho``, ``inner_abs_tol`` and
        both residuals are absolute constants in MW, so without this the
        returned ``r`` depends on the grid's magnitude even though the problem
        is homogeneous in it. On an LV feeder every residual is below a
        ``1e-3`` tolerance from the first iteration and the cascade returns its
        ``r = 0`` initialisation with ``converged=True``; measured across six
        decades of scaling, ``normalize=True`` holds ``r`` fixed to ten digits
        while the default varies from 0% to 217% of the sector deficit.
        Default ``False`` for backwards compatibility.
    :param r_regularization_relative: Read ``r_regularization`` as a
        dimensionless multiple of each CP's own :math:`\\|c_i\\|^2` rather than
        as an absolute MW\\ :sup:`2` weight; see
        :func:`~.kernel._local_regularization`. Only bites when
        ``minimize_usage`` is set, since otherwise :math:`\\alpha` cancels at
        the fixed point. Default ``False``.
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
                d.sector: {
                    t: np.asarray(a, dtype=float).copy() for t, a in d.demand_by_tier.items()
                }
                for d in demands
            },
            iterations=0,
            primal_residual=0.0,
            dual_residual=0.0,
            converged=True,
        )

    all_sectors = sorted(
        {s for c in cps for s in c.capacity_by_sector} | {d.sector for d in demands}
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

    scale = _characteristic_scale(demands) if normalize else 1.0

    cap = np.zeros((N, n_sec), dtype=float)
    for i, c in enumerate(cps):
        for s, c_val in c.capacity_by_sector.items():
            cap[i, sec_idx[s]] = float(c_val) / scale
    cap_norm_sq = (cap**2).sum(axis=1)
    alpha_by_cp = [
        _local_regularization(float(r_regularization), r_regularization_relative, float(q))
        for q in cap_norm_sq
    ]

    D = np.zeros((n_sec, n_tier, H), dtype=float)
    base_supply = np.zeros((n_sec, H), dtype=float)
    for d in demands:
        s = sec_idx[d.sector]
        bs = np.asarray(d.base_supply, dtype=float)
        if bs.shape != (H,):
            raise ValueError(
                f"base_supply for sector {d.sector!r} must have shape ({H},), got {bs.shape}"
            )
        base_supply[s, :] = bs / scale
        for tier, arr in d.demand_by_tier.items():
            if tier not in tier_idx:
                continue
            a = np.asarray(arr, dtype=float)
            if a.shape != (H,):
                raise ValueError(
                    f"demand_by_tier[{tier}] for sector {d.sector!r} must have "
                    f"shape ({H},), got {a.shape}"
                )
            D[s, tier_idx[tier], :] = a / scale

    # ---- persistent shared state across tier transitions ----
    r_curr = np.zeros((N, H), dtype=float)
    x = np.zeros((N, n_sec, H), dtype=float)
    x_bar = np.zeros((n_sec, H), dtype=float)
    z = np.zeros((n_sec, H), dtype=float)
    u = np.zeros((n_sec, H), dtype=float)  # SHARED dual (single vector, not per-CP)
    theta = np.zeros((n_sec, H), dtype=float)

    rho_f = float(rho)
    # Bounds for the adaptive penalty: residual balancing can keep
    # cranking rho while the dual residual sits at zero (the binding /
    # all-r=0-boundary case), so clamp it to a finite window around the
    # caller's rho to stay numerically sane if a round never converges.
    rho_lo = float(rho) * 1.0e-6
    rho_hi = float(rho) * 1.0e6

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

            # ----- per-CP projection (closed form) -----
            # target = x_i - x_bar + z - u, den = rho ||c_i||^2 + alpha.
            # minimize_usage -> ridge toward 0 ((alpha/2)||r||^2): the alpha
            #   term does NOT cancel, biasing the degenerate optimal face to
            #   the minimum-usage r (no surplus overshoot), num = rho c.target.
            # else -> proximal toward r_prev ((alpha/2)||r - r_prev||^2): alpha
            #   cancels at the fixed point, leaving the bare (non-unique) LP
            #   optimum, num = rho c.target + alpha r_prev.
            for i in range(N):
                if cap_norm_sq[i] < 1e-12:
                    continue
                alpha = alpha_by_cp[i]
                for k in range(H):
                    target_ik = x[i, :, k] - x_bar[:, k] + z[:, k] - u[:, k]
                    num = rho_f * float(cap[i] @ target_ik)
                    if not minimize_usage:
                        num += alpha * r_prev[i, k]
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

            if primal_res < inner_abs_tol and dual_res < inner_abs_tol and r_change < inner_abs_tol:
                converged_round = True
                break

            # ----- adaptive penalty (Boyd 3.4.1 residual balancing) -----
            # Rebalance rho on lopsided residuals; rescale u = u_unscaled
            # / rho inversely so the unscaled dual is preserved.
            if adaptive_rho:
                if primal_res > rho_mu * dual_res and rho_f * rho_tau <= rho_hi:
                    rho_f *= rho_tau
                    u /= rho_tau
                elif dual_res > rho_mu * primal_res and rho_f / rho_tau >= rho_lo:
                    rho_f /= rho_tau
                    u *= rho_tau

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

    # r is dimensionless and comes back untouched; every other output is MW and
    # has to leave the normalised frame the solve ran in.
    factor_by_cp: dict[str, np.ndarray] = {c.cp_id: r_curr[i].copy() for i, c in enumerate(cps)}
    served_by_sector_tier = {
        d.sector: {
            t: sigma_per_tier[t][sec_idx[d.sector], :] * scale for t in all_tiers if t in tier_idx
        }
        for d in demands
    }

    history: dict[str, Any] = {}
    if record_history:
        history = {
            "per_round_iters": list(per_round_iters),
            "per_round_primal_residuals": [v * scale for v in history_primal],
            "per_round_dual_residuals": [v * scale for v in history_dual],
            "per_round_r_changes": history_r_changes,
            "theta_final": theta * scale,
            "sigma_per_tier": {t: v * scale for t, v in sigma_per_tier.items()},
            "rho_final": rho_f,
            "characteristic_scale": scale,
        }

    return CPAdmmResult(
        factor_by_cp=factor_by_cp,
        served_by_sector_tier=served_by_sector_tier,
        iterations=total_iters,
        primal_residual=primal_res * scale,
        dual_residual=dual_res * scale,
        converged=converged,
        history=history,
    )


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class DistributedLexicographicCascadeStart(OptimizationMessage):
    """External start payload handed to the coordinator.

    Carries the externally-aggregated demand picture plus hyperparameters.
    The coordinator never requests per-CP capacities — each participant
    keeps its :class:`CPSpec` private and runs the closed-form update
    locally.

    :param demands: Per-sector demand profiles aggregated externally.
    :param horizon: Horizon length ``H``.
    :param rho: Sharing-ADMM penalty.
    :param inner_iters_max: Per-round inner iteration cap.
    :param inner_abs_tol: Per-step convergence tolerance.
    :param r_regularization: Quadratic ``r``-norm penalty.
    :param adaptive_rho: Enable Boyd §3.4.1 residual balancing of the
        penalty (default on); see
        :func:`solve_cp_distributed_lexicographic_cascade`.
    :param rho_mu: Residual-imbalance ratio that triggers adaptation.
    :param rho_tau: Multiplicative penalty step.
    :param minimize_usage: Select the minimum-usage point of the optimal
        face (no surplus overshoot, unique solution); see
        :func:`solve_cp_distributed_lexicographic_cascade`. Default off
        (it costs many more inner iterations).
    :param record_history: Persist per-round diagnostics on the result.
    """

    demands: list[SectorDemand]
    horizon: int = 1
    rho: float = 1.0
    inner_iters_max: int = 500
    inner_abs_tol: float = 1.0e-4
    r_regularization: float = 1.0e-1
    adaptive_rho: bool = True
    rho_mu: float = 10.0
    rho_tau: float = 2.0
    minimize_usage: bool = False
    record_history: bool = False


@dataclass
class DistributedLexicographicCascadeInit(OptimizationMessage):
    """Leader -> follower: one-shot setup at cascade start.

    Shares the *coordinate frame* (sector ordering, horizon, rho, alpha)
    a follower uses to interpret the leader's :class:`~...admm.core.ADMMMessage`
    corrections and build its replies; capacity vectors stay private.

    :param sectors: Ordered sector identifiers. Followers build ``x_i``
        in this order; capacity for a sector outside the list is ignored.
    :param horizon: Horizon length ``H``.
    :param rho: Sharing-ADMM penalty.
    :param r_regularization: Quadratic ``r``-norm penalty :math:`\\alpha`
        for the follower's local closed-form projection.
    :param minimize_usage: Ridge-toward-zero (min usage) vs proximal-toward-
        previous in the follower's projection; see the kernel.
    """

    sectors: list[str]
    horizon: int
    rho: float
    r_regularization: float
    minimize_usage: bool = False


@dataclass
class DistributedLexicographicCascadeInitAck(OptimizationMessage):
    """Follower -> leader: ready ack with the follower's id."""

    cp_id: str


@dataclass
class DistributedLexicographicCascadeDone(OptimizationMessage):
    """Leader -> follower: cascade complete; report final ``r``."""


@dataclass
class DistributedLexicographicCascadeDoneReply(OptimizationMessage):
    """Follower -> leader: final regulation factor.

    Lets the leader assemble the global ``factor_by_cp``. Each follower
    already keeps its own ``r``, so this round-trip exists only to match
    the other variants' result-dataclass shape.
    """

    cp_id: str
    r: np.ndarray


# ---------------------------------------------------------------------------
# Global actor: the leader-side sum-sharing math (Boyd §7.3)
# ---------------------------------------------------------------------------


class LexicographicCascadeGlobalActor(ADMMGlobalActor):
    """Sum-sharing ADMM global update for one priority tier.

    Plugs the cascade's closed-form shared updates into
    :class:`~...admm.core.ADMMGenericCoordinator`. The leader owns only
    the shared state ``(x_bar, z, sigma, u)`` — the per-CP ``r``-projection
    runs at each follower. ``z``/``u`` persist across tiers (warm start),
    and ``theta`` accumulates the cleared service; the coordinator sets
    :attr:`D_tau`/:attr:`slack_max` and calls :meth:`begin_tier` before
    each per-tier run.

    The hooks reproduce :func:`solve_cp_distributed_lexicographic_cascade`
    exactly: closed-form ``(z, sigma)`` per cell, single shared dual, the
    3-way (primal / dual / max-``r``-step) stop test, and Boyd §3.4.1
    adaptive-:math:`\\rho` with the dual rescaled inversely.

    :param n_sec: Number of sectors in the round's frame.
    :param horizon: Horizon length ``H``.
    :param rho_init: Caller's penalty (anchors the adaptive-rho window).
    :param adaptive_rho: Enable residual balancing.
    :param rho_mu: Residual-imbalance trigger ratio.
    :param rho_tau: Multiplicative penalty step.
    """

    def __init__(
        self,
        *,
        n_sec: int,
        horizon: int,
        rho_init: float,
        adaptive_rho: bool,
        rho_mu: float,
        rho_tau: float,
    ) -> None:
        self.n_sec = n_sec
        self.H = horizon
        self.adaptive_rho = adaptive_rho
        self.rho_mu = rho_mu
        self.rho_tau = rho_tau
        self.rho_lo = rho_init * 1.0e-6
        self.rho_hi = rho_init * 1.0e6
        # Persistent shared state (carried across tiers).
        self.z = np.zeros((n_sec, horizon), dtype=float)
        self.u = np.zeros((n_sec, horizon), dtype=float)
        self.theta = np.zeros((n_sec, horizon), dtype=float)
        # Set by the coordinator before the first tier.
        self.base_supply = np.zeros((n_sec, horizon), dtype=float)
        # Per-tier inputs / outputs.
        self.D_tau = np.zeros((n_sec, horizon), dtype=float)
        self.slack_max = np.zeros((n_sec, horizon), dtype=float)
        self.sigma = np.zeros((n_sec, horizon), dtype=float)
        # Per-tier bookkeeping.
        self._x_bar = np.zeros((n_sec, horizon), dtype=float)
        self.iters = 0
        self.converged = False
        self.last_primal = 0.0
        self.last_dual = 0.0

    def begin_tier(self, D_tau: np.ndarray) -> None:
        """Reset per-tier outputs and pin this tier's demand + slack."""
        self.D_tau = D_tau
        self.slack_max = self.base_supply - self.theta
        self.sigma = np.zeros((self.n_sec, self.H), dtype=float)
        self.iters = 0
        self.converged = False

    # ---- ADMMGlobalActor hooks ----

    def init_z(self, n: int, m: int) -> np.ndarray:
        return self.z  # carried (warm start)

    def init_u(self, n: int, m: int) -> np.ndarray:
        return self.u  # carried (warm start)

    def actor_correction(
        self, x: list[np.ndarray], z: np.ndarray, u: np.ndarray, i: int
    ) -> np.ndarray:
        x_bar = sum(x) / len(x)
        return z - u - x_bar  # follower target = x_i + correction

    def z_update(
        self,
        input_data: Any,
        x: list[np.ndarray],
        u: np.ndarray,
        z: np.ndarray,
        rho: float,
        n: int,
    ) -> np.ndarray:
        x_bar = sum(x) / float(n)
        self._x_bar = x_bar
        target_z = x_bar + u
        z_new = np.empty((self.n_sec, self.H), dtype=float)
        for s in range(self.n_sec):
            for k in range(self.H):
                z_sk, sig_sk = _z_sigma_cell_update(
                    float(target_z[s, k]),
                    float(self.slack_max[s, k]),
                    float(self.D_tau[s, k]),
                    n if n > 0 else 1,
                    rho,
                )
                z_new[s, k] = z_sk
                self.sigma[s, k] = sig_sk
        self.iters += 1
        return z_new

    def u_update(
        self,
        x: list[np.ndarray],
        u: np.ndarray,
        z: np.ndarray,
        rho: float,
        n: int,
    ) -> np.ndarray:
        return u + self._x_bar - z

    def primal_residual(self, x: list[np.ndarray], z: np.ndarray) -> float:
        return float(np.max(np.abs(self._x_bar - z))) if z.size else 0.0

    def should_stop(
        self,
        primal_res: float,
        dual_res: float,
        aux: list[Any],
        abs_tol: float,
    ) -> bool:
        r_changes = [float(a) for a in aux if a is not None]
        max_r_change = max(r_changes) if r_changes else 0.0
        self.last_primal = primal_res
        self.last_dual = dual_res
        self.converged = bool(
            primal_res < abs_tol and dual_res < abs_tol and max_r_change < abs_tol
        )
        return self.converged

    def adapt_rho(
        self, primal_res: float, dual_res: float, rho: float, u: np.ndarray
    ) -> tuple[float, np.ndarray]:
        if not self.adaptive_rho:
            return rho, u
        if primal_res > self.rho_mu * dual_res and rho * self.rho_tau <= self.rho_hi:
            return rho * self.rho_tau, u / self.rho_tau
        if dual_res > self.rho_mu * primal_res and rho / self.rho_tau >= self.rho_lo:
            return rho / self.rho_tau, u * self.rho_tau
        return rho, u


# ---------------------------------------------------------------------------
# Follower: runs the per-CP closed-form x-update locally
# ---------------------------------------------------------------------------


class DistributedLexicographicCascadeParticipant(DistributedAlgorithm):
    """Coupling-point follower in the leader-follower cascade.

    Never publishes its capacity vector: it projects private
    ``capacity_by_sector`` onto the leader's broadcast sector ordering
    once (on :class:`DistributedLexicographicCascadeInit`), then answers
    each :class:`~...admm.core.ADMMMessage` with a local closed-form
    ``r_i``-projection. Shared state ``(x_bar, z, u, theta)`` lives only
    on the leader.

    :param cp_id: Stable identifier (indexes the final ``factor_by_cp``).
    :param capacity_by_sector: Per-sector signed effective capacity, load
        convention. *Private* — never sent over the wire.
    """

    def __init__(self, cp_id: str, capacity_by_sector: dict[str, float]) -> None:
        self.cp_id = cp_id
        self._capacity_by_sector = dict(capacity_by_sector)
        # Lazy: built on receipt of Init.
        self._cap_vec: np.ndarray = np.zeros(0)
        self._cap_norm_sq: float = 0.0
        self._alpha: float = 0.0
        self._minimize_usage: bool = False
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
            carrier.reply_to_other(DistributedLexicographicCascadeInitAck(cp_id=self.cp_id), meta)
        elif isinstance(message_data, ADMMMessage):
            self._x_update(message_data.v, message_data.rho)
            carrier.reply_to_other(ADMMAnswer(x=self._x_i.copy(), aux=self._last_r_change), meta)
        elif isinstance(message_data, DistributedLexicographicCascadeDone):
            carrier.reply_to_other(
                DistributedLexicographicCascadeDoneReply(cp_id=self.cp_id, r=self.r),
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
        self._alpha = float(msg.r_regularization)
        self._minimize_usage = bool(msg.minimize_usage)
        self._horizon = H
        self._x_i = np.zeros((cap_vec.size, H), dtype=float)
        self.r = np.zeros(H, dtype=float)

    def _x_update(self, correction: np.ndarray, rho: float) -> None:
        """Apply one Boyd §7.3 sharing-ADMM x-update locally.

        With ``target = x_i + correction`` (``correction = z - u - x_bar``)
        the projection onto ``{r * cap_i : r ∈ [0,1]}`` is

        .. code-block:: text

            r_k = clip((rho c.target [+ alpha r_prev_k])
                       / (rho ||c||^2 + alpha), 0, 1).

        With ``minimize_usage`` the ``alpha r_prev`` term is dropped (ridge
        toward 0 -> minimum-usage, no overshoot); otherwise it is kept
        (proximal toward r_prev -> bare LP optimum). Mirrors the kernel.
        Also records ``max_k |r_new - r_prev|`` for the reply's ``aux``.
        """
        cap = self._cap_vec
        den = rho * self._cap_norm_sq + self._alpha
        H = self._horizon
        if den <= 0.0 or cap.size == 0:
            self._last_r_change = 0.0  # idle CP: contributes zero
            return
        r_prev = self.r.copy()
        target = self._x_i + np.asarray(correction, dtype=float)
        for k in range(H):
            num = rho * float(cap @ target[:, k])
            if not self._minimize_usage:
                num += self._alpha * r_prev[k]
            r_k = num / den
            if r_k < 0.0:
                r_k = 0.0
            elif r_k > 1.0:
                r_k = 1.0
            self.r[k] = r_k
            self._x_i[:, k] = r_k * cap
        self._last_r_change = float(np.max(np.abs(self.r - r_prev))) if H > 0 else 0.0


# ---------------------------------------------------------------------------
# Coordinator (leader): drives ADMMGenericCoordinator once per tier
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _RoundFrame:
    """The coordinate frame + demand tensors shared by every tier."""

    sectors: list[str]
    sec_idx: dict[str, int]
    tiers: list[int]
    tier_idx: dict[int, int]
    n_sec: int
    horizon: int
    D: np.ndarray  # (n_sec, n_tier, H) per-tier demand
    base_supply: np.ndarray  # (n_sec, H)


@dataclass(eq=False)
class _TierSweep:
    """Aggregated outputs of running the inner loop over every tier."""

    sigma_per_tier: dict[int, np.ndarray]
    per_round_iters: list[int]
    per_round_converged: list[bool]
    theta_final: np.ndarray
    primal_res: float
    dual_res: float
    history_primal: list[float]
    history_dual: list[float]


class DistributedLexicographicCascadeCoordinator(Coordinator):
    """Leader for the lexicographic cascade, built on classic ADMM.

    One-shot Init handshake fixes the coordinate frame, then the cascade
    runs :class:`~...admm.core.ADMMGenericCoordinator`'s leader-follower
    loop once per priority tier (via a :class:`LexicographicCascadeGlobalActor`),
    advancing the threshold :math:`\\theta` between tiers and warm-starting
    each tier from the previous one's converged iterate. Finally it
    dispatches :class:`DistributedLexicographicCascadeDone` and assembles
    ``factor_by_cp`` from the followers' replies.

    :returns: The full :class:`CPAdmmResult`.
    """

    async def start_optimization(
        self,
        carrier: "Carrier",
        message_data: DistributedLexicographicCascadeStart,
        meta: Any,
    ) -> CPAdmmResult:
        participant_addrs = carrier.others("coordinator")
        frame = self._build_frame(message_data)
        await self._init_followers(carrier, participant_addrs, frame, message_data)

        if not frame.tiers or not participant_addrs:
            # No demand / no followers -> short-circuit to an empty result.
            return await self._finalize(
                carrier,
                participant_addrs,
                served_by_sector_tier={d.sector: {} for d in message_data.demands},
                iterations=0,
                primal_residual=0.0,
                dual_residual=0.0,
                converged=True,
                history={},
            )

        sweep = await self._run_tiers(carrier, participant_addrs, frame, message_data)
        return await self._finalize(
            carrier,
            participant_addrs,
            served_by_sector_tier=self._collect_served(message_data.demands, frame, sweep),
            iterations=int(sum(sweep.per_round_iters)),
            primal_residual=sweep.primal_res,
            dual_residual=sweep.dual_res,
            converged=bool(all(sweep.per_round_converged)),
            history=self._build_history(message_data, sweep),
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _build_frame(
        message_data: DistributedLexicographicCascadeStart,
    ) -> _RoundFrame:
        """Validate the demands and pack them into per-tier tensors."""
        demands = message_data.demands
        H = int(message_data.horizon)
        if H < 1:
            raise ValueError("horizon must be >= 1")

        sectors = sorted({d.sector for d in demands})
        if not sectors:
            raise ValueError("no sectors found in demands")
        sec_idx = {s: i for i, s in enumerate(sectors)}
        n_sec = len(sectors)

        tiers = sorted({t for d in demands for t in d.demand_by_tier})
        tier_idx = {t: i for i, t in enumerate(tiers)}

        D = np.zeros((n_sec, max(len(tiers), 1), H), dtype=float)
        base_supply = np.zeros((n_sec, H), dtype=float)
        for d in demands:
            s = sec_idx[d.sector]
            bs = np.asarray(d.base_supply, dtype=float)
            if bs.shape != (H,):
                raise ValueError(
                    f"base_supply for sector {d.sector!r} must have shape ({H},), got {bs.shape}"
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

        return _RoundFrame(
            sectors=sectors,
            sec_idx=sec_idx,
            tiers=tiers,
            tier_idx=tier_idx,
            n_sec=n_sec,
            horizon=H,
            D=D,
            base_supply=base_supply,
        )

    @staticmethod
    async def _init_followers(
        carrier: "Carrier",
        participant_addrs: list,
        frame: _RoundFrame,
        message_data: DistributedLexicographicCascadeStart,
    ) -> None:
        """Broadcast the coordinate frame and await every follower's ack."""
        init_msg = DistributedLexicographicCascadeInit(
            sectors=frame.sectors,
            horizon=frame.horizon,
            rho=float(message_data.rho),
            r_regularization=float(message_data.r_regularization),
            minimize_usage=bool(message_data.minimize_usage),
        )
        await carrier.gather(
            *[carrier.send_awaitable(init_msg, addr) for addr in participant_addrs]
        )

    # ------------------------------------------------------------------
    # The cascade: one generic-ADMM run per priority tier
    # ------------------------------------------------------------------

    async def _run_tiers(
        self,
        carrier: "Carrier",
        participant_addrs: list,
        frame: _RoundFrame,
        message_data: DistributedLexicographicCascadeStart,
    ) -> _TierSweep:
        """Run the leader-follower loop per tier, advancing ``theta``."""
        N = len(participant_addrs)
        actor = LexicographicCascadeGlobalActor(
            n_sec=frame.n_sec,
            horizon=frame.horizon,
            rho_init=float(message_data.rho),
            adaptive_rho=bool(message_data.adaptive_rho),
            rho_mu=float(message_data.rho_mu),
            rho_tau=float(message_data.rho_tau),
        )
        actor.base_supply = frame.base_supply
        inner = ADMMGenericCoordinator(
            global_actor=actor,
            rho=float(message_data.rho),
            max_iters=int(message_data.inner_iters_max),
            abs_tol=float(message_data.inner_abs_tol),
            rel_tol=0.0,
        )

        m = frame.n_sec * frame.horizon
        # Warm-start primal, carried follower-for-follower across tiers.
        x_state = [np.zeros((frame.n_sec, frame.horizon)) for _ in range(N)]
        sweep = _TierSweep(
            sigma_per_tier={},
            per_round_iters=[],
            per_round_converged=[],
            theta_final=actor.theta,
            primal_res=0.0,
            dual_res=0.0,
            history_primal=[],
            history_dual=[],
        )
        record_history = bool(message_data.record_history)

        for tier in frame.tiers:
            actor.begin_tier(frame.D[:, frame.tier_idx[tier], :])
            x_state, z, u = await inner._run(carrier, None, m, x_init=x_state)
            actor.z, actor.u = z, u  # carry the iterate into the next tier
            actor.theta = actor.theta + actor.sigma  # advance the threshold

            sweep.sigma_per_tier[tier] = actor.sigma.copy()
            sweep.per_round_iters.append(actor.iters)
            sweep.per_round_converged.append(actor.converged)
            sweep.primal_res = actor.last_primal
            sweep.dual_res = actor.last_dual
            if record_history:
                sweep.history_primal.append(actor.last_primal)
                sweep.history_dual.append(actor.last_dual)

        sweep.theta_final = actor.theta
        return sweep

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_served(
        demands: list[SectorDemand], frame: _RoundFrame, sweep: _TierSweep
    ) -> dict[str, dict[int, np.ndarray]]:
        return {
            d.sector: {
                t: sweep.sigma_per_tier[t][frame.sec_idx[d.sector], :].copy()
                for t in frame.tiers
                if t in frame.tier_idx
            }
            for d in demands
        }

    @staticmethod
    def _build_history(
        message_data: DistributedLexicographicCascadeStart, sweep: _TierSweep
    ) -> dict[str, Any]:
        if not message_data.record_history:
            return {}
        return {
            "per_round_iters": list(sweep.per_round_iters),
            "per_round_primal_residuals": sweep.history_primal,
            "per_round_dual_residuals": sweep.history_dual,
            "theta_final": sweep.theta_final.copy(),
            "sigma_per_tier": {t: v.copy() for t, v in sweep.sigma_per_tier.items()},
        }

    async def _finalize(
        self,
        carrier: "Carrier",
        participant_addrs: list,
        *,
        served_by_sector_tier: dict[str, dict[int, np.ndarray]],
        iterations: int,
        primal_residual: float,
        dual_residual: float,
        converged: bool,
        history: dict[str, Any],
    ) -> CPAdmmResult:
        """Send Done to every follower, gather final r, build the result."""
        done_replies = await carrier.gather(
            *[
                carrier.send_awaitable(DistributedLexicographicCascadeDone(), addr)
                for addr in participant_addrs
            ]
        )
        factor_by_cp = {
            reply.cp_id: np.asarray(reply.r, dtype=float).copy() for reply in done_replies
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


def create_distributed_lexicographic_cascade_coordinator() -> (
    DistributedLexicographicCascadeCoordinator
):
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
    adaptive_rho: bool = True,
    rho_mu: float = 10.0,
    rho_tau: float = 2.0,
    minimize_usage: bool = False,
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
        adaptive_rho=adaptive_rho,
        rho_mu=rho_mu,
        rho_tau=rho_tau,
        minimize_usage=minimize_usage,
        record_history=record_history,
    )
