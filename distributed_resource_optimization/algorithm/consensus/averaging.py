"""Averaging consensus algorithm.

Each participant maintains a local estimate λ and iteratively averages it
with its neighbours' estimates.  An optional :class:`ConsensusActor` can
add a gradient term to bias the consensus toward a local optimum.

The update rule is:

.. math::

    \\lambda^{k+1} = \\lambda^k + \\alpha (\\bar{\\lambda}^k - \\lambda^k)
                    + \\nabla f(\\lambda^k, \\text{data})

where :math:`\\bar{\\lambda}^k` is the average of all neighbours' estimates
at iteration *k*.


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..core import DistributedAlgorithm, OptimizationMessage

if TYPE_CHECKING:
    from ...carrier.core import Carrier


# ---------------------------------------------------------------------------
# ConsensusActor hierarchy
# ---------------------------------------------------------------------------

class ConsensusActor:
    """Optional plug-in that adds a gradient term to the averaging update.

    Subclass this to bias the consensus toward a local optimum (e.g. economic
    dispatch or price signals).
    """

    def gradient_term(self, lam: np.ndarray, data: Any) -> np.ndarray | float:
        """Return the gradient correction for the current iterate *lam*.

        :param lam: Current local estimate.
        :param data: Auxiliary data forwarded from the start message.
        :returns: Additive correction (default: 0).
        """
        return 0


class NoConsensusActor(ConsensusActor):
    """Neutral consensus actor — no gradient term (pure averaging)."""


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class AveragingConsensusMessage(OptimizationMessage):
    """Message exchanged between averaging-consensus participants.

    :param lam: Current λ estimate of the sender.
    :param k: Current iteration counter.
    :param data: Auxiliary payload forwarded to :meth:`ConsensusActor.gradient_term`.
    :param initial: If ``True`` this is the kick-off message; recipients
                    (re-)initialise their state.
    """

    lam: np.ndarray
    k: int
    data: Any
    initial: bool = False


@dataclass
class ConsensusFinishedMessage:
    """Emitted (internally) when a participant finishes the consensus run.

    :param lam: Final λ estimate.
    :param k: Iteration at which convergence / max_iter was reached.
    :param actor: The :class:`ConsensusActor` instance of this participant.
    """

    lam: np.ndarray
    k: int
    actor: ConsensusActor


# ---------------------------------------------------------------------------
# AveragingConsensusAlgorithm
# ---------------------------------------------------------------------------

class AveragingConsensusAlgorithm(DistributedAlgorithm):
    """Distributed averaging consensus with an optional gradient correction.

    :param finish_callback: Called with ``(algorithm, carrier)`` when the run
                            ends (either :attr:`max_iter` reached or all
                            neighbours signal convergence).
    :param consensus_actor: Optional :class:`ConsensusActor` for gradient terms.
    :param initial_lam: Starting scalar (broadcast to all λ dimensions).
    :param alpha: Averaging step size (0 < α ≤ 1).
    :param max_iter: Maximum number of consensus iterations.
    """

    def __init__(
        self,
        finish_callback: Callable,
        consensus_actor: ConsensusActor | None = None,
        initial_lam: float = 10.0,
        alpha: float = 0.3,
        max_iter: int = 50,
    ) -> None:
        self.finish_callback = finish_callback
        self.actor: ConsensusActor = (
            consensus_actor if consensus_actor is not None else NoConsensusActor()
        )
        self.initial_lam = initial_lam
        self.alpha = alpha
        self.max_iter = max_iter

        # Mutable iteration state (reset at the start of each consensus run)
        self._message_queue: dict[int, list[AveragingConsensusMessage]] = {}
        self._first_message: bool = True
        self._started: bool = False  # True once any round has begun
        self._k: int = 0
        self._lam: np.ndarray = np.array([initial_lam])

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: AveragingConsensusMessage,
        meta: Any,
    ) -> None:
        """Process one incoming averaging consensus message."""
        neighbours = carrier.others("")

        # --- Termination path ---
        if message_data.k >= self.max_iter:
            if self._first_message:
                # Negotiation already finished; ignore stale terminal messages
                return
            self.finish_callback(self, carrier)
            self._first_message = True
            self._message_queue.clear()
            return

        # After termination, ignore stale messages from the previous round.
        # Only an explicit initial=True message may start a new round.
        if self._first_message and self._started and not message_data.initial:
            return

        # --- Initialisation path ---
        if self._first_message or message_data.initial:
            self._first_message = False
            self._started = True
            self._k = 0
            self._lam = np.ones(len(message_data.lam)) * self.initial_lam
            for addr in neighbours:
                carrier.send_to_other(
                    AveragingConsensusMessage(lam=self._lam.copy(), k=0, data=message_data.data),
                    addr,
                )

        # --- Queue the message ---
        queue = self._message_queue.setdefault(message_data.k, [])
        queue.append(message_data)

        # --- Advance if we have all neighbours' messages for this iteration ---
        if len(queue) == len(neighbours) or self._k < message_data.k:
            avg_lam = sum(m.lam for m in queue) / len(queue)
            grad = self.actor.gradient_term(self._lam, message_data.data)
            self._lam = self._lam + self.alpha * (avg_lam - self._lam) + grad
            self._k = message_data.k + 1

            del self._message_queue[message_data.k]

            for addr in neighbours:
                carrier.send_to_other(
                    AveragingConsensusMessage(
                        lam=self._lam.copy(),
                        k=self._k,
                        data=message_data.data,
                    ),
                    addr,
                )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def create_averaging_consensus_participant(
    finish_callback: Callable,
    consensus_actor: ConsensusActor | None = None,
    initial_lam: float = 10.0,
    alpha: float = 0.3,
    max_iter: int = 50,
) -> AveragingConsensusAlgorithm:
    """Create an :class:`AveragingConsensusAlgorithm` participant.

    :param finish_callback: ``(algorithm, carrier) -> None`` — called when done.
    :param consensus_actor: Optional gradient actor.  ``None`` → pure averaging.
    :param initial_lam: Initial λ scalar.
    :param alpha: Step size.
    :param max_iter: Maximum iterations.
    """
    return AveragingConsensusAlgorithm(
        finish_callback=finish_callback,
        consensus_actor=consensus_actor,
        initial_lam=initial_lam,
        alpha=alpha,
        max_iter=max_iter,
    )


def create_averaging_consensus_start(
    initial_lam: float,
    data: Any = None,
) -> AveragingConsensusMessage:
    """Create the initial kick-off message for an averaging consensus run.

    :param initial_lam: Starting scalar broadcast to all λ dimensions.
    :param data: Auxiliary payload forwarded to gradient actors.
    :returns: An :class:`AveragingConsensusMessage` with ``initial=True``.
    """
    return AveragingConsensusMessage(
        lam=np.array([initial_lam]),
        k=0,
        data=data,
        initial=True,
    )
