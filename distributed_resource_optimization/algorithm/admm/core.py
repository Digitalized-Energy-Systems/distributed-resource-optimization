"""ADMM core — generic coordinator and message types.

Provides the :class:`ADMMGenericCoordinator` which drives the standard
Alternating Direction Method of Multipliers iteration loop.  Concrete
global-actor implementations live in :mod:`.consensus_admm` and
:mod:`.sharing_admm`; the local actor lives in :mod:`.flex_actor`.


"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core import Coordinator, OptimizationMessage

if TYPE_CHECKING:
    from ...carrier.core import Carrier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class ADMMStart(OptimizationMessage):
    """Sent to the coordinator to begin a new ADMM run.

    :param data: Algorithm-specific input (e.g. :class:`ADMMSharingData` or a
                 target vector).
    :param solution_length: Number of decision variables per participant.
    """

    data: Any
    solution_length: int


@dataclass
class ADMMMessage(OptimizationMessage):
    """Sent by the coordinator to each participant to request an x-update.

    :param v: Scaled consensus/sharing vector (the local QP reference point).
    :param rho: ADMM penalty parameter.
    :param k: Iteration of the coordinator's loop (1, 2, …) — a participant that missed a
        lossy scatter still knows which iteration it answers.
    """

    v: np.ndarray
    rho: float
    k: int = 0


@dataclass
class ADMMAnswer(OptimizationMessage):
    """Reply from a participant after solving its local update.

    :param x: Local solution vector.
    :param aux: Optional follower-side scalar/data (e.g. a per-step move
        magnitude) surfaced to the global actor's convergence hook. ``None``
        for variants that converge on the primal/dual residuals alone.
    """

    x: np.ndarray
    aux: Any = None


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

    # ---- optional hooks (defaults preserve standard ADMM behaviour) ----

    def dual_residual(
        self,
        z: Any,
        z_old: Any,
        rho: float,
        x: list[np.ndarray] | None = None,
        x_old: list[np.ndarray] | None = None,
    ) -> float:
        """Dual residual for convergence. Default: ``rho * max||z - z_old||``.
        *x* / *x_old* (this and the previous iterate) are for variants whose stacked z
        depends on x (sharing)."""
        return float(rho * _max_diff_norm(z, z_old))

    def eps_scales(self, x: list[np.ndarray], z: Any, u: Any, rho: float) -> tuple[float, float]:
        """Scales of Boyd et al. 2011 eq. 3.12: ``(max(||x||, ||z||), ||rho·u||)``, stacked
        2-norms (A = I, B = -I, scaled dual u). Variants with a reduced z/u override it."""
        return max(_norm(x), _norm(z)), rho * _norm(u)

    def should_stop(
        self,
        primal_res: float,
        dual_res: float,
        aux: list[Any],
        abs_tol: float,
    ) -> bool | None:
        """Convergence override. Return ``True``/``False`` to decide directly,
        or ``None`` to fall back to the coordinator's eps_pri/eps_dual test."""
        return None

    def adapt_rho(
        self,
        primal_res: float,
        dual_res: float,
        rho: float,
        u: Any,
    ) -> tuple[float, Any]:
        """Optionally rebalance ``rho`` (rescaling the scaled dual ``u``
        inversely). Default: leave both unchanged."""
        return rho, u


class ADMMGlobalObjective(ABC):
    """Optional global objective (currently informational only)."""

    @abstractmethod
    def objective(
        self,
        x: list[np.ndarray],
        u: Any,
        z: Any,
        n: int,
    ) -> float:
        """Evaluate the global objective."""


# ---------------------------------------------------------------------------
# Helper: max-norm over list-of-arrays or single array
# ---------------------------------------------------------------------------


def _norm(v: Any) -> float:
    """2-norm of *v*, stacked over the entries if *v* is a list of arrays."""
    if isinstance(v, list):
        return float(np.sqrt(sum(float(np.sum(np.square(vi))) for vi in v)))
    return float(np.linalg.norm(v))


def _max_diff_norm(a: Any, b: Any) -> float:
    """``max ||a_i - b_i||`` for lists or ``max |a_j - b_j|`` for arrays."""
    if isinstance(a, list):
        return float(max(float(np.linalg.norm(ai - bi)) for ai, bi in zip(a, b)))
    return float(np.max(np.abs(a - b)))


def _scale(v: Any, f: float) -> Any:
    return [vi * f for vi in v] if isinstance(v, list) else v * f


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
    4. Check primal and dual residuals against tolerances; stop if met
       (Boyd et al. 2011 eq. 3.12 with stacked 2-norms, unless the actor decides).
    5. Optional penalty update: the actor's :meth:`~ADMMGlobalActor.adapt_rho`, or
       residual balancing eq. 3.13 when *mu* is set.

    :param global_actor: Variant-specific global update logic.
    :param rho: ADMM penalty parameter (default: 1.0).
    :param max_iters: Maximum number of iterations (default: 1000).
    :param abs_tol: Absolute convergence tolerance (default: 1e-4).
    :param rel_tol: Relative convergence tolerance (default: 1e-3).
    :param mu: Residual balancing (Boyd et al. 2011 eq. 3.13): ρ·τ when ||r|| > μ||s||,
        ρ/τ when ||s|| > μ||r||, the scaled dual u rescaled inversely. None = off.
    :param tau: Factor of the residual balancing (default 2).
    :param round_timeout: Lossy links: a round ends at the latest round_timeout after its
        scatter (carrier clock); a participant whose reply is missing then keeps its last x,
        and a round without any reply is not tested for convergence. None = wait for every
        reply. Give it slack over the longest round trip: a reply
        due exactly at the deadline is a tie (under mango stepping the step order decides).
    """

    def __init__(
        self,
        global_actor: ADMMGlobalActor,
        rho: float = 1.0,
        max_iters: int = 1000,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-3,
        mu: float | None = None,
        tau: float = 2.0,
        round_timeout: float | None = None,
    ) -> None:
        self.global_actor = global_actor
        self.rho = rho
        self.max_iters = max_iters
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.mu = mu
        self.tau = tau
        self.round_timeout = round_timeout
        self.iterations = 0
        self.converged = False  # the last run stopped by the residual test, not max_iters
        self.stop_time: float | None = None  # carrier time at which the last run stopped
        self.missing_replies = 0  # replies not in by their round deadline (round_timeout)

    async def start_optimization(
        self,
        carrier: "Carrier",
        message_data: ADMMStart,
        meta: Any,
    ) -> list[np.ndarray]:
        x, _z, _u = await self._run(carrier, message_data.data, message_data.solution_length)
        return x

    async def _run(
        self,
        carrier: "Carrier",
        input_data: Any,
        m: int,
        *,
        x_init: list[np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], Any, Any]:
        """Core ADMM loop.

        :param carrier: Coordinator's carrier.
        :param input_data: Algorithm-specific data (target, priorities, …).
        :param m: Problem dimension (number of decision variables).
        :param x_init: Optional warm-start primal list (e.g. carried over
            from a previous round); defaults to per-participant zeros.
        :returns: ``(x_list, z, u)`` at convergence or max-iter.
        """
        actor = self.global_actor
        rho = self.rho
        participant_addrs = carrier.others("coordinator")
        n = len(participant_addrs)

        x: list[np.ndarray] = (
            [np.array(xi, dtype=float) for xi in x_init]
            if x_init is not None
            else [np.zeros(m) for _ in range(n)]
        )
        z = actor.init_z(n, m)
        u = actor.init_u(n, m)

        for k in range(1, self.max_iters + 1):
            # 1. Send ADMMMessage to all participants in parallel, collect futures
            futures: list[asyncio.Future] = []
            for i, addr in enumerate(participant_addrs):
                correction = actor.actor_correction(x, z, u, i)
                fut = carrier.send_awaitable(ADMMMessage(v=correction, rho=rho, k=k), addr)
                futures.append(fut)

            # Await all replies simultaneously (carrier park point: the
            # replies arrive as messages, possibly in a later sim step)
            # 2026-11-16 D-134: lossy runs end the round at its deadline, a missing x_i stays;
            # D-137: the deadline counts from this round's scatter (a grid t0 + k·round banked
            # the slack of every clean round and one late loss then waited for all of it)
            deadline = None if self.round_timeout is None else carrier.now() + self.round_timeout
            replies = await carrier.gather(*futures, deadline=deadline)
            aux = [getattr(reply, "aux", None) for reply in replies]
            x_old = list(x)
            for i, reply in enumerate(replies):
                if reply is None:
                    self.missing_replies += 1
                else:
                    x[i] = np.asarray(reply.x, dtype=float)
            self.iterations = k

            # 2. z-update
            z_old = _deepcopy_z(z)
            z = actor.z_update(input_data, x, u, z, rho, n)

            # 3. u-update
            u = actor.u_update(x, u, z, rho, n)

            # 4. Convergence check — actor may override the eps_pri/eps_dual test
            r_norm = actor.primal_residual(x, z)
            s_norm = actor.dual_residual(z, z_old, rho, x=x, x_old=x_old)
            stop = actor.should_stop(r_norm, s_norm, aux, self.abs_tol)
            if replies and all(reply is None for reply in replies):
                stop = False  # nothing arrived: at the flat start r = s = 0 would "converge"
            if stop is None:
                # 2026-11-13 Q-E5 (D-121): eq. 3.12 with 2-norms (was max-norms)
                primal_scale, dual_scale = actor.eps_scales(x, z, u, rho)
                eps_pri = np.sqrt(m * n) * self.abs_tol + self.rel_tol * primal_scale
                eps_dual = np.sqrt(m * n) * self.abs_tol + self.rel_tol * dual_scale
                stop = bool(r_norm <= eps_pri and s_norm <= eps_dual)

            self.converged = bool(stop)
            if stop:
                logger.debug("ADMM converged in %d iterations.", k)
                break

            # 5. Optional penalty re-balancing (no-op unless the actor adapts).
            rho, u = actor.adapt_rho(r_norm, s_norm, rho, u)
            if self.mu is not None:  # eq. 3.13 (2026-11-13 Q-E6, D-122)
                if r_norm > self.mu * s_norm:
                    rho, u = rho * self.tau, _scale(u, 1 / self.tau)
                elif s_norm > self.mu * r_norm:
                    rho, u = rho / self.tau, _scale(u, self.tau)

            if k == self.max_iters:
                logger.warning(
                    "ADMM reached max iterations (%d) without full convergence (r=%.4g, s=%.4g).",
                    self.max_iters,
                    r_norm,
                    s_norm,
                )

        self.rho = rho  # persist adapted penalty for callers reusing this loop
        self.stop_time = carrier.now()
        return x, z, u


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_admm_start(data: Any, length: int | None = None) -> ADMMStart:
    """Create an :class:`ADMMStart` message.

    When *length* is omitted the length is inferred from ``data.solution_length``
    or from ``len(data.target)`` (for :class:`.sharing_admm.ADMMSharingData`).
    """
    if length is not None:
        return ADMMStart(data=data, solution_length=length)
    # Try to infer from data
    if hasattr(data, "solution_length"):
        return ADMMStart(data=data, solution_length=data.solution_length)
    if hasattr(data, "target"):
        return ADMMStart(data=data, solution_length=len(data.target))
    raise ValueError("Cannot infer solution_length; pass it explicitly as the second argument.")
