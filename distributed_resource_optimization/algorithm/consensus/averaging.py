"""Averaging consensus algorithm — leader-follower economic dispatch.

Each participant maintains a local estimate λ and iteratively averages it
with its neighbours' estimates. This implements the leader-follower
incremental-cost consensus of Jian et al. 2020 (eqs. 20-23): exactly one
participant is the *leader*, the rest are *followers*.

The update rule is:

λ_i^{k+1} = λ_i^k + α * (λ̄^k - λ_i^k) + [ε * ΔP  if i is the leader, else 0]

where λ̄^k is the average of all neighbours' λ at iteration k, and
ΔP = P_target - Σ_j P_j(λ_j^k) is the real system-wide power imbalance,
recovered from the projected power that every participant (leader and
followers alike) attaches to its broadcast message each round.

Each participant derives its own dispatch from λ via
:meth:`ConsensusActor.project_power` (e.g. eq. 23's price-to-power clip).
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
    """Optional plug-in that derives a local power output from the price λ.

    Subclass this to project the shared price signal onto a local dispatch
    (e.g. economic dispatch via eq. 23 of Jian et al. 2020).
    """

    def project_power(self, lam: np.ndarray, data: Any) -> np.ndarray | float:
        """Return the local power output implied by the current price *lam*.

        :param lam: Current local price estimate.
        :param data: Auxiliary data forwarded from the start message (e.g.
                     the total demand vector).
        :returns: Projected local power (default: 0, i.e. no dispatch).
        """
        return 0.0


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
    :param data: Auxiliary payload forwarded to :meth:`ConsensusActor.project_power`.
    :param p: Sender's projected local power for this round (used by the
              leader to recover the real system-wide power imbalance).
    :param initial: If ``True`` this is the kick-off message; recipients
                    (re-)initialise their state.
    """

    lam: np.ndarray
    k: int
    data: Any
    p: np.ndarray | None = None
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
    """Leader-follower incremental-cost consensus (Jian et al. 2020, eqs. 20-23).

    Exactly one participant should be constructed with ``is_leader=True``;
    the rest are followers. The leader nudges its price by ``leader_gain *
    ΔP`` where ``ΔP`` is the real system-wide power imbalance, recovered
    each round from every participant's projected power. Followers do pure
    neighbour averaging.

    :param finish_callback: Called with ``(algorithm, carrier)`` when the run
                            ends (either :attr:`max_iter` reached or all
                            neighbours signal convergence).
    :param consensus_actor: Optional :class:`ConsensusActor` for the price
                            to power projection (eq. 23).
    :param initial_lam: Starting scalar (broadcast to all λ dimensions).
    :param alpha: Averaging step size (0 < α ≤ 1).
    :param max_iter: Maximum number of consensus iterations.
    :param is_leader: Whether this participant is the leader (eq. 22).
    :param leader_gain: Leader's power-imbalance pinning gain (ε in eq. 22).
                        Ignored for followers.
    """

    def __init__(
        self,
        finish_callback: Callable,
        consensus_actor: ConsensusActor | None = None,
        initial_lam: float = 10.0,
        alpha: float = 0.3,
        max_iter: int = 50,
        is_leader: bool = False,
        leader_gain: float = 0.0,
    ) -> None:
        self.finish_callback = finish_callback
        self.actor: ConsensusActor = (
            consensus_actor if consensus_actor is not None else NoConsensusActor()
        )
        self.initial_lam = initial_lam
        self.alpha = alpha
        self.max_iter = max_iter
        self.is_leader = is_leader
        self.leader_gain = leader_gain

        # Mutable iteration state (reset at the start of each consensus run)
        self._message_queue: dict[int, list[AveragingConsensusMessage]] = {}
        self._first_message: bool = True
        self._started: bool = False  # True once any round has begun
        self._k: int = 0
        self._lam: np.ndarray = np.array([initial_lam])
        self._P: np.ndarray = np.array([0.0])  # own last-projected power

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
            self._P = np.asarray(self.actor.project_power(self._lam, message_data.data))
            for addr in neighbours:
                carrier.send_to_other(
                    AveragingConsensusMessage(
                        lam=self._lam.copy(), k=0, data=message_data.data, p=self._P.copy()
                    ),
                    addr,
                )
            if message_data.initial:
                # This is the external kick-off trigger, not a neighbour's
                # round-0 contribution — don't queue it for averaging/ΔP.
                return

        # --- Queue the message ---
        queue = self._message_queue.setdefault(message_data.k, [])
        queue.append(message_data)

        # --- Advance if we have all neighbours' messages for this iteration ---
        if len(queue) == len(neighbours) or self._k < message_data.k:
            avg_lam = sum(m.lam for m in queue) / len(queue)
            if self.is_leader:
                pd = np.asarray(message_data.data, dtype=float)
                sigma_p = self._P + sum(np.asarray(m.p) for m in queue)
                delta_p = pd - sigma_p
                self._lam = (
                    self._lam + self.alpha * (avg_lam - self._lam) + self.leader_gain * delta_p
                )
            else:
                self._lam = self._lam + self.alpha * (avg_lam - self._lam)
            self._P = np.asarray(self.actor.project_power(self._lam, message_data.data))
            self._k = message_data.k + 1

            del self._message_queue[message_data.k]

            for addr in neighbours:
                carrier.send_to_other(
                    AveragingConsensusMessage(
                        lam=self._lam.copy(),
                        k=self._k,
                        data=message_data.data,
                        p=self._P.copy(),
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
    is_leader: bool = False,
    leader_gain: float = 0.0,
) -> AveragingConsensusAlgorithm:
    """Create an :class:`AveragingConsensusAlgorithm` participant.

    :param finish_callback: ``(algorithm, carrier) -> None`` — called when done.
    :param consensus_actor: Optional price-to-power actor.  ``None`` → pure averaging.
    :param initial_lam: Initial λ scalar.
    :param alpha: Step size.
    :param max_iter: Maximum iterations.
    :param is_leader: Whether this participant is the leader (eq. 22).
                      Exactly one participant per run should set this.
    :param leader_gain: Leader's power-imbalance pinning gain (ε in eq. 22).
    """
    return AveragingConsensusAlgorithm(
        finish_callback=finish_callback,
        consensus_actor=consensus_actor,
        initial_lam=initial_lam,
        alpha=alpha,
        max_iter=max_iter,
        is_leader=is_leader,
        leader_gain=leader_gain,
    )


def create_averaging_consensus_start(
    initial_lam: float,
    data: Any = None,
) -> AveragingConsensusMessage:
    """Create the initial kick-off message for an averaging consensus run.

    :param initial_lam: Starting scalar broadcast to all λ dimensions.
    :param data: Auxiliary payload forwarded to :meth:`ConsensusActor.project_power`.
    :returns: An :class:`AveragingConsensusMessage` with ``initial=True``.
    """
    return AveragingConsensusMessage(
        lam=np.array([initial_lam]),
        k=0,
        data=data,
        initial=True,
    )
