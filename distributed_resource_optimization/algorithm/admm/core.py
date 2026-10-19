"""ADMM core — generic coordinator and message types.

Provides the :class:`ADMMGenericCoordinator` which drives the standard
Alternating Direction Method of Multipliers iteration loop.  Concrete
global-actor implementations live in :mod:`.consensus_admm` and
:mod:`.sharing_admm`; the local actors live in :mod:`.flex_actor` and
:mod:`.storage`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core import Coordinator

if TYPE_CHECKING:
    from ...carrier.core import Carrier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class ADMMStart:
    """Sent to the coordinator to begin a new ADMM run.

    :param data: Algorithm-specific input (e.g. :class:`~.sharing_admm.ADMMSharingData`
                 or a target vector).
    :param solution_length: Number of decision variables per participant.
    """

    data: Any
    solution_length: int


@dataclass
class ADMMMessage:
    """Sent by the coordinator to each participant to request an x-update.

    :param v: Correction vector; the local update minimises
              ``f_i(x) + (rho/2)·‖x + v‖²``, i.e. it is pulled towards ``-v``.
    :param rho: ADMM penalty parameter.
    """

    v: np.ndarray
    rho: float


@dataclass
class ADMMAnswer:
    """Reply from a participant after solving its local update.

    :param x: Local solution vector.
    """

    x: np.ndarray


# ---------------------------------------------------------------------------
# Abstract global-actor interface
# ---------------------------------------------------------------------------


class ADMMGlobalActor(ABC):
    """Interface for the coordinator-side global update in ADMM variants."""

    @abstractmethod
    def z_update(
        self,
        input_data: Any,
        x: list[np.ndarray],
        u: Any,
        z: Any,
        rho: float,
        n: int,
    ) -> Any:
        """Compute the new global *z* from the current *x* and *u*."""

    @abstractmethod
    def u_update(
        self,
        x: list[np.ndarray],
        u: Any,
        z: Any,
        rho: float,
        n: int,
    ) -> Any:
        """Update the dual variable *u*."""

    @abstractmethod
    def init_z(self, n: int, m: int) -> Any:
        """Initialise *z* (called once before the iteration loop)."""

    @abstractmethod
    def init_u(self, n: int, m: int) -> Any:
        """Initialise *u* (called once before the iteration loop)."""

    @abstractmethod
    def actor_correction(
        self,
        x: list[np.ndarray],
        z: Any,
        u: Any,
        i: int,
    ) -> np.ndarray:
        """Compute the correction vector sent to participant *i* (0-indexed)."""

    @abstractmethod
    def primal_residual(self, x: list[np.ndarray], z: Any) -> float:
        """Compute the primal residual used for convergence checking."""

    def constraint_violation(
        self, input_data: Any, x: list[np.ndarray]
    ) -> tuple[float, float] | None:
        """Violation of the variant's hard coupling constraint by the actual *x*.

        Returns ``(violation, scale)``: the largest per-element violation and
        the magnitude it is judged against (e.g. the peak target), or ``None``
        if the variant has no hard constraint.  The standard residuals are
        per participant, so on their own they let the aggregate violation grow
        with the number of participants; the coordinator additionally requires
        ``violation <= abs_tol + feas_tol * scale`` before declaring convergence.
        """
        return None


# ---------------------------------------------------------------------------
# Helper: max-norm over list-of-arrays or single array
# ---------------------------------------------------------------------------


def _max_norm(v: Any) -> float:
    """Return ``max ||v_i||`` if *v* is a list, else ``max |v_j|`` for a vector."""
    if isinstance(v, list):
        return float(max(float(np.linalg.norm(vi)) for vi in v))
    return float(np.max(np.abs(v)))


def _max_diff_norm(a: Any, b: Any) -> float:
    """``max ||a_i - b_i||`` for lists or ``max |a_j - b_j|`` for arrays."""
    if isinstance(a, list):
        return float(max(float(np.linalg.norm(ai - bi)) for ai, bi in zip(a, b)))
    return float(np.max(np.abs(a - b)))


def _scale_dual(u: Any, factor: float) -> Any:
    """``factor · u`` for an array or a list of arrays (the scaled dual)."""
    if isinstance(u, list):
        return [np.asarray(v) * factor for v in u]
    return np.asarray(u) * factor


def _deepcopy_z(z: Any) -> Any:
    if isinstance(z, list):
        return [np.copy(zi) for zi in z]
    return np.copy(z)


# ---------------------------------------------------------------------------
# Generic ADMM coordinator
# ---------------------------------------------------------------------------


class ADMMGenericCoordinator(Coordinator):
    """Standard ADMM iteration loop.

    Each round:

    1. Send :class:`ADMMMessage` (correction + ρ) to all participants in
       parallel and await :class:`ADMMAnswer` from each.
    2. Global *z*-update via :meth:`~ADMMGlobalActor.z_update`.
    3. Dual *u*-update via :meth:`~ADMMGlobalActor.u_update`.
    4. Check primal and dual residuals against tolerances and, for variants
       with a hard coupling constraint, the aggregate violation of that
       constraint by the returned *x* (see
       :meth:`~ADMMGlobalActor.constraint_violation`); stop if all are met.

    After a run, :attr:`iterations` and :attr:`converged` describe how it
    ended (hitting ``max_iters`` also logs a warning).

    :param global_actor: Variant-specific global update logic.
    :param rho: ADMM penalty parameter (default: 1.0).
    :param max_iters: Maximum number of iterations (default: 1000).
    :param abs_tol: Absolute convergence tolerance (default: 1e-4).
    :param rel_tol: Relative convergence tolerance (default: 1e-3).
    :param feas_tol: Tolerance on the aggregate coupling-constraint violation,
        relative to the constraint's scale (default: 1e-4, i.e. ``Σxᵢ`` must
        be within 0.01 % of the peak target for exchange ADMM).
    :param round_timeout: Seconds (on the carrier's clock) to wait for the
        participants' answers each round; a participant that does not answer
        in time keeps its previous ``x``.  ``None`` (default) waits for every
        answer, which blocks forever if a message is lost.
    :param max_missing_rounds: Stop the run (with :attr:`converged` ``False``)
        once a participant has missed this many consecutive rounds.  ``None``
        (default) never stops for that reason.
    :param adaptive_rho: Residual balancing (Boyd et al. 2011, §3.4.1, eq. 3.13):
        after each round that did not converge, ρ ← τ·ρ if ‖r‖ > μ‖s‖ and
        ρ ← ρ/τ if ‖s‖ > μ‖r‖; the scaled dual u is divided by the same factor
        so the unscaled dual ρ·u is unchanged. Off by default (fixed ρ).
    :param rho_mu: μ of eq. 3.13 (Boyd: 10).
    :param rho_tau: τ_incr = τ_decr of eq. 3.13 (Boyd: 2).

    At least one participant must be registered with the coordinator; with
    none, the variant's global update fails (``IndexError`` /
    ``ZeroDivisionError``) instead of returning an empty result.

    Under message loss this is partially asynchronous ADMM: rounds use the
    latest answer of every participant.  Convergence is only declared after
    a round in which every participant answered, because a round with stale
    answers can leave *z* unchanged and fake a zero dual residual.
    :attr:`missed_answers` counts the missing answers of the last run.
    """

    def __init__(
        self,
        global_actor: ADMMGlobalActor,
        rho: float = 1.0,
        max_iters: int = 1000,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-3,
        feas_tol: float = 1e-4,
        round_timeout: float | None = None,
        max_missing_rounds: int | None = None,
        adaptive_rho: bool = False,
        rho_mu: float = 10.0,
        rho_tau: float = 2.0,
    ) -> None:
        self.global_actor = global_actor
        self.rho = rho
        self.adaptive_rho = adaptive_rho
        self.rho_mu = rho_mu
        self.rho_tau = rho_tau
        #: ρ at the end of the last run (differs from :attr:`rho` with adaptive ρ).
        self.final_rho = rho
        #: How often ρ was changed in the last run.
        self.rho_changes = 0
        self.max_iters = max_iters
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.feas_tol = feas_tol
        self.round_timeout = round_timeout
        self.max_missing_rounds = max_missing_rounds
        self.iterations = 0
        self.converged = False
        self.missed_answers = 0

    async def start_optimization(
        self,
        carrier: "Carrier",
        message_data: ADMMStart,
        meta: Any,
    ) -> list[np.ndarray]:
        return await self._run(carrier, message_data.data, message_data.solution_length)

    async def _x_update_round(
        self,
        carrier: "Carrier",
        participant_addrs: list[Any],
        x: list[np.ndarray],
        z: Any,
        u: Any,
        rho: float | None = None,
    ) -> list[bool]:
        """Send each participant its correction in parallel; store replies in *x*.

        A participant whose request or answer was lost within
        :attr:`round_timeout` keeps its previous ``x[i]`` (initially zeros).

        :returns: Per participant, whether it answered in this round.
        """
        requests = {
            addr: ADMMMessage(
                v=self.global_actor.actor_correction(x, z, u, i),
                rho=self.rho if rho is None else rho,
            )
            for i, addr in enumerate(participant_addrs)
        }
        replies = await carrier.request_all(
            requests, timeout=self.round_timeout, reply_type=ADMMAnswer
        )
        fresh: list[bool] = []
        for i, addr in enumerate(participant_addrs):
            reply = replies.get(addr)
            if reply is not None:
                x[i] = np.asarray(reply.x, dtype=float)
            fresh.append(reply is not None)
        return fresh

    async def _run(
        self,
        carrier: "Carrier",
        input_data: Any,
        m: int,
    ) -> list[np.ndarray]:
        """Core ADMM loop.

        :param carrier: Coordinator's carrier.
        :param input_data: Algorithm-specific data (target, priorities, …).
        :param m: Problem dimension (number of decision variables).
        :returns: The participants' ``x`` at convergence or max-iter.
        """
        actor = self.global_actor
        rho = self.rho
        participant_addrs = carrier.others()
        n = len(participant_addrs)

        x: list[np.ndarray] = [np.zeros(m) for _ in range(n)]
        u = actor.init_u(n, m)
        z = actor.init_z(n, m)
        self.iterations = 0
        self.converged = False
        self.missed_answers = 0
        self.rho_changes = 0
        missed_in_a_row = [0] * n

        for k in range(1, self.max_iters + 1):
            # 1. x-update round (all participants in parallel); a participant
            #    that did not answer keeps its previous x.
            fresh = await self._x_update_round(carrier, participant_addrs, x, z, u, rho)
            missed_in_a_row = [0 if ok else m_ + 1 for ok, m_ in zip(fresh, missed_in_a_row)]
            self.missed_answers += fresh.count(False)
            round_complete = all(fresh)

            # 2. z-update
            z_old = _deepcopy_z(z)
            z = actor.z_update(input_data, x, u, z, rho, n)

            # 3. u-update
            u = actor.u_update(x, u, z, rho, n)

            # 4. Convergence check
            r_norm = actor.primal_residual(x, z)
            s_norm = rho * _max_diff_norm(z, z_old)
            eps_pri = np.sqrt(m * n) * self.abs_tol + self.rel_tol * max(_max_norm(x), _max_norm(z))
            # u is the *scaled* dual, so the unscaled dual is rho·u (Boyd §3.3).
            eps_dual = np.sqrt(m * n) * self.abs_tol + self.rel_tol * rho * _max_norm(u)
            self.iterations = k

            # The residuals above are per participant; also bound the
            # aggregate constraint violation of the x we actually return.
            feasible = True
            violation = actor.constraint_violation(input_data, x)
            if violation is not None:
                gap, scale = violation
                feasible = gap <= self.abs_tol + self.feas_tol * scale

            if round_complete and r_norm < eps_pri and s_norm < eps_dual and feasible:
                self.converged = True
                logger.debug("ADMM converged in %d iterations.", k)
                break

            if (
                self.max_missing_rounds is not None
                and max(missed_in_a_row, default=0) >= self.max_missing_rounds
            ):
                silent = [
                    participant_addrs[i]
                    for i, missed in enumerate(missed_in_a_row)
                    if missed >= self.max_missing_rounds
                ]
                logger.warning(
                    "ADMM stopped after %d iterations: %d participant(s) missed %d "
                    "consecutive rounds (%s).",
                    k,
                    len(silent),
                    self.max_missing_rounds,
                    silent,
                )
                break

            if self.adaptive_rho:
                if r_norm > self.rho_mu * s_norm:
                    rho *= self.rho_tau
                    u = _scale_dual(u, 1.0 / self.rho_tau)
                    self.rho_changes += 1
                elif s_norm > self.rho_mu * r_norm:
                    rho /= self.rho_tau
                    u = _scale_dual(u, self.rho_tau)
                    self.rho_changes += 1

            if k == self.max_iters:
                logger.warning(
                    "ADMM reached max iterations (%d) without full convergence (r=%.4g, s=%.4g).",
                    self.max_iters,
                    r_norm,
                    s_norm,
                )

        self.final_rho = rho
        return x
