"""Diffusion algorithm (adapt-then-combine).

Each participant maintains a local price estimate λ and a power iterate φ over a
scheduling horizon.  At every iteration a participant

1. **adapts** its power iterate via a local gradient step
   ``φ = λ - ε · ∇J(λ, data)``,
2. broadcasts ``φ`` to its neighbours, and
3. **combines** its own φ with all received φ's by unweighted averaging to form
   the next λ.

The update rule is:

.. math::

    \\lambda^{k+1} = \\frac{1}{N+1} \\left( \\varphi^k +
                    \\sum_{j \\in \\mathcal{N}} \\varphi_j^k \\right),
    \\qquad
    \\varphi^{k+1} = \\lambda^{k+1} - \\varepsilon \\, \\nabla J(\\lambda^{k+1}, \\text{data}).

The optional :class:`DiffusionActor` plug-in supplies ``∇J``; the default
:class:`NoDiffusionActor` returns zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..core import DistributedAlgorithm, OptimizationMessage

if TYPE_CHECKING:
    from ...carrier.core import Carrier


# ---------------------------------------------------------------------------
# DiffusionActor hierarchy
# ---------------------------------------------------------------------------


class DiffusionActor:
    """Optional plug-in that supplies the gradient term for the adapt step.

    Subclass this to bias the diffusion iterates toward a local optimum
    (e.g. economic dispatch or battery storage scheduling).
    """

    def gradient_term(self, lam: np.ndarray, data: Any) -> np.ndarray | float:
        """Return the gradient ``∇J(λ, data)`` for the current iterate *lam*.

        :param lam: Current local price/λ estimate over the schedule.
        :param data: Auxiliary data forwarded from the start message.
        :returns: Additive gradient (default: 0).
        """
        return 0


class NoDiffusionActor(DiffusionActor):
    """Neutral diffusion actor — gradient is identically zero."""


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class DiffusionMessage(OptimizationMessage):
    """Message exchanged between diffusion participants.

    :param phi: Current power iterate φ of the sender.
    :param k: Current iteration counter.
    :param data: Auxiliary payload forwarded to :meth:`DiffusionActor.gradient_term`.
    :param initial: If ``True`` this is the kick-off message; recipients
                    (re-)initialise their state.
    """

    phi: np.ndarray
    k: int
    data: Any
    initial: bool = False


# ---------------------------------------------------------------------------
# DiffusionAlgorithm
# ---------------------------------------------------------------------------


class DiffusionAlgorithm(DistributedAlgorithm):
    """Distributed adapt-then-combine diffusion over a scheduling horizon.

    :param finish_callback: Called with ``(algorithm, carrier)`` when
                            :attr:`max_iter` is reached.
    :param diffusion_actor: Optional :class:`DiffusionActor` supplying the
                            gradient.  ``None`` → :class:`NoDiffusionActor`.
    :param initial_lam: Starting scalar (broadcast to all λ dimensions).
    :param epsilon: Gradient step size (ε).
    :param max_iter: Maximum number of diffusion iterations.
    :param horizon: Number of time steps in the schedule.
    """

    def __init__(
        self,
        finish_callback: Callable,
        diffusion_actor: DiffusionActor | None = None,
        initial_lam: float = 10.0,
        epsilon: float = 0.1,
        max_iter: int = 300,
        horizon: int = 24,
    ) -> None:
        self.finish_callback = finish_callback
        self.actor: DiffusionActor = (
            diffusion_actor if diffusion_actor is not None else NoDiffusionActor()
        )
        self.initial_lam = initial_lam
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.horizon = horizon

        self._message_queue: dict[int, list[DiffusionMessage]] = {}
        self._first_message: bool = True
        self._k: int = 0
        self._lam: np.ndarray = np.array([1.0])
        self._phi: np.ndarray = np.array([1.0])

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: DiffusionMessage,
        meta: Any,
    ) -> None:
        """Process one incoming diffusion message."""
        neighbours = carrier.others("")

        # --- Termination path ---
        if message_data.k >= self.max_iter:
            if self._first_message:
                return
            self.finish_callback(self, carrier)
            self._first_message = True
            self._message_queue.clear()
            return

        # --- Initialisation path ---
        if self._first_message or message_data.initial:
            self._first_message = False
            self._k = 0
            self._lam = np.ones(len(message_data.phi)) * self.initial_lam

            grad_J = self.actor.gradient_term(self._lam, message_data.data)
            self._phi = self._lam - self.epsilon * np.asarray(grad_J)

            for addr in neighbours:
                carrier.send_to_other(
                    DiffusionMessage(
                        phi=self._phi.copy(),
                        k=0,
                        data=message_data.data,
                    ),
                    addr,
                )

        # --- Queue the message ---
        queue = self._message_queue.setdefault(message_data.k, [])
        queue.append(message_data)

        # --- Advance if all neighbours have reported for this iteration ---
        if len(queue) == len(neighbours):
            # Combination: unweighted average of own φ and all received φ's
            n = len(queue) + 1
            lam_new = self._phi.copy()
            for m in queue:
                lam_new = lam_new + m.phi
            self._lam = lam_new / n

            del self._message_queue[message_data.k]

            # Adaptation
            grad_J = self.actor.gradient_term(self._lam, message_data.data)
            self._phi = self._lam - self.epsilon * np.asarray(grad_J)

            self._k += 1

            for addr in neighbours:
                carrier.send_to_other(
                    DiffusionMessage(
                        phi=self._phi.copy(),
                        k=self._k,
                        data=message_data.data,
                    ),
                    addr,
                )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_diffusion_participant(
    finish_callback: Callable,
    diffusion_actor: DiffusionActor | None = None,
    initial_lam: float = 10.0,
    epsilon: float = 0.1,
    max_iter: int = 300,
    horizon: int = 24,
) -> DiffusionAlgorithm:
    """Create a :class:`DiffusionAlgorithm` participant.

    :param finish_callback: ``(algorithm, carrier) -> None`` — called when done.
    :param diffusion_actor: Optional gradient actor.  ``None`` → no gradient.
    :param initial_lam: Initial λ scalar.
    :param epsilon: Gradient step size.
    :param max_iter: Maximum iterations.
    :param horizon: Number of schedule time steps.
    """
    return DiffusionAlgorithm(
        finish_callback=finish_callback,
        diffusion_actor=diffusion_actor,
        initial_lam=initial_lam,
        epsilon=epsilon,
        max_iter=max_iter,
        horizon=horizon,
    )


def create_diffusion_start(
    initial_lam: float,
    data: Any = None,
) -> DiffusionMessage:
    """Create the initial kick-off message for a diffusion run.

    :param initial_lam: Starting scalar; broadcast to all λ dimensions.
    :param data: Auxiliary payload forwarded to each participant's
                 :meth:`DiffusionActor.gradient_term`.
    :returns: A :class:`DiffusionMessage` with ``initial=True``.
    """
    return DiffusionMessage(
        phi=np.array([initial_lam]),
        k=0,
        data=data,
        initial=True,
    )
