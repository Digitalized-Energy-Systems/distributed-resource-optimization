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

    :param data: Algorithm-specific input (e.g. :class:`ADMMSharingData` or a
                 target vector).
    :param solution_length: Number of decision variables per participant.
    """

    data: Any
    solution_length: int


@dataclass
class ADMMMessage:
    """Sent by the coordinator to each participant to request an x-update.

    :param v: Scaled consensus/sharing vector (the local QP reference point).
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
    4. Check primal and dual residuals against tolerances; stop if met.

    :param global_actor: Variant-specific global update logic.
    :param rho: ADMM penalty parameter (default: 1.0).
    :param max_iters: Maximum number of iterations (default: 1000).
    :param abs_tol: Absolute convergence tolerance (default: 1e-4).
    :param rel_tol: Relative convergence tolerance (default: 1e-3).
    """

    def __init__(
        self,
        global_actor: ADMMGlobalActor,
        rho: float = 1.0,
        max_iters: int = 1000,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-3,
    ) -> None:
        self.global_actor = global_actor
        self.rho = rho
        self.max_iters = max_iters
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    async def start_optimization(
        self,
        carrier: "Carrier",
        message_data: ADMMStart,
        meta: Any,
    ) -> list[np.ndarray]:
        x, _z, _u = await self._run(
            carrier, message_data.data, message_data.solution_length
        )
        return x

    async def _run(
        self,
        carrier: "Carrier",
        input_data: Any,
        m: int,
    ) -> tuple[list[np.ndarray], Any, Any]:
        """Core ADMM loop.

        :param carrier: Coordinator's carrier.
        :param input_data: Algorithm-specific data (target, priorities, …).
        :param m: Problem dimension (number of decision variables).
        :returns: ``(x_list, z, u)`` at convergence or max-iter.
        """
        actor = self.global_actor
        rho = self.rho
        participant_addrs = carrier.others("coordinator")
        n = len(participant_addrs)

        x: list[np.ndarray] = [np.zeros(m) for _ in range(n)]
        z = actor.init_z(n, m)
        u = actor.init_u(n, m)

        for k in range(1, self.max_iters + 1):
            # 1. Send ADMMMessage to all participants in parallel, collect futures
            futures: list[asyncio.Future] = []
            for i, addr in enumerate(participant_addrs):
                correction = actor.actor_correction(x, z, u, i)
                fut = carrier.send_awaitable(ADMMMessage(v=correction, rho=rho), addr)
                futures.append(fut)

            # Await all replies simultaneously
            replies = await asyncio.gather(*futures)
            for i, reply in enumerate(replies):
                x[i] = np.asarray(reply.x, dtype=float)

            # 2. z-update
            z_old = _deepcopy_z(z)
            z = actor.z_update(input_data, x, u, z, rho, n)

            # 3. u-update
            u = actor.u_update(x, u, z, rho, n)

            # 4. Convergence check
            r_norm = actor.primal_residual(x, z)
            s_norm = rho * _max_diff_norm(z, z_old)
            eps_pri = (
                np.sqrt(m * n) * self.abs_tol
                + self.rel_tol * max(_max_norm(x), _max_norm(z))
            )
            eps_dual = (
                np.sqrt(m * n) * self.abs_tol
                + self.rel_tol * _max_norm(u)
            )

            if r_norm < eps_pri and s_norm < eps_dual:
                logger.debug("ADMM converged in %d iterations.", k)
                break

            if k == self.max_iters:
                logger.warning(
                    "ADMM reached max iterations (%d) without full convergence "
                    "(r=%.4g, s=%.4g).",
                    self.max_iters,
                    r_norm,
                    s_norm,
                )

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
    raise ValueError(
        "Cannot infer solution_length; pass it explicitly as the second argument."
    )
