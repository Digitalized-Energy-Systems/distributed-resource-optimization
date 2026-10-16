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

from .._rounds import RoundBuffer, RoundDeadline, sender_key
from ..core import DistributedAlgorithm, OptimizationMessage

if TYPE_CHECKING:
    from ...carrier.core import Carrier


# ---------------------------------------------------------------------------
# ConsensusActor hierarchy
# ---------------------------------------------------------------------------


class ConsensusActor:
    """Optional plug-in that derives a local power output from the price λ.

    Subclass this to project the shared price signal onto a local dispatch
    (e.g. economic dispatch via eq. 23 of Jian et al. 2020).  Used as-is it is
    the neutral actor (pure averaging, no dispatch).
    """

    def project_power(self, lam: np.ndarray, data: Any) -> np.ndarray | float:
        """Return the local power output implied by the current price *lam*.

        :param lam: Current local price estimate.
        :param data: Auxiliary data forwarded from the start message (e.g.
                     the total demand vector).
        :returns: Projected local power (default: 0, i.e. no dispatch).
        """
        return 0.0


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class AveragingConsensusMessage(OptimizationMessage):
    """Message exchanged between averaging-consensus participants.

    :param lam: Current λ estimate of the sender.  In the kick-off message
                (``initial=True``) the values are ignored — every participant
                starts from its own ``initial_lam``; only the length is read.
    :param k: Current iteration counter.
    :param data: Auxiliary payload forwarded to :meth:`ConsensusActor.project_power`.
                 Only carried by the kick-off and round-0 messages; recipients
                 cache it, so later rounds send ``None``.
    :param p: Sender's projected local power for this round (used by the
              leader to recover the real system-wide power imbalance).
    :param initial: If ``True`` this is the kick-off message; recipients
                    (re-)initialise their state.
    :param stop: Sent by the leader once the power balance holds; ``k`` is
                 the round whose schedules the leader verified. Recipients
                 finish on that round's schedule.
    """

    lam: np.ndarray
    k: int
    data: Any
    p: np.ndarray | None = None
    initial: bool = False
    stop: bool = False


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

    The leader reconstructs ΔP from its own power plus the powers attached
    to the messages it receives, so it must be a direct neighbour of every
    other participant (complete or leader-star topology) — on a sparser
    graph it would pin λ against a partial, understated imbalance.

    Termination follows Jian et al. 2020 (p. 1428, after eq. 23): iterate
    "until the system satisfies the power balance, that is, ΔP = 0". As
    ΔP = 0 is never hit exactly, the leader stops once every timestep
    satisfies ``|ΔP_t| ≤ balance_tol · |P_D,t|`` and sends a ``stop``
    message; :attr:`max_iter` remains as a failsafe.

    :param finish_callback: Called with ``(algorithm, carrier)`` when the run
                            ends (power balance reached, or the
                            :attr:`max_iter` failsafe).
    :param consensus_actor: Optional :class:`ConsensusActor` for the price
                            to power projection (eq. 23).
    :param initial_lam: Starting scalar (broadcast to all λ dimensions).
    :param alpha: Averaging step size (0 < α ≤ 1).
    :param max_iter: Maximum number of consensus iterations (failsafe).
    :param is_leader: Whether this participant is the leader (eq. 22).
    :param leader_gain: Leader's power-imbalance pinning gain (ε in eq. 22).
                        Must be positive for the leader; ignored for followers.
    :param balance_tol: Relative power-balance tolerance for the leader's
                        stop criterion (default 1 % of demand per timestep).
                        Ignored for followers.
    :param round_timeout: Seconds (carrier clock) to wait for a round's
                          messages before advancing anyway.  A neighbour whose
                          message is missing contributes its last known λ and
                          power (λ = ``initial_lam`` and power 0 if never
                          heard); the leader applies its ΔP term only once it
                          has heard from every neighbour, and checks the power
                          balance only on rounds where every neighbour's value
                          arrived.  ``None`` (default) waits for every
                          neighbour, which stalls if a message is lost.
    :raises ValueError: If ``is_leader`` is set without a positive ``leader_gain``
                        (the leader would never correct the power imbalance).
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
        balance_tol: float = 0.01,
        round_timeout: float | None = None,
    ) -> None:
        if is_leader and leader_gain <= 0:
            raise ValueError(
                f"The leader needs a positive leader_gain to correct the power "
                f"imbalance (eq. 22), got leader_gain={leader_gain}."
            )
        self.finish_callback = finish_callback
        self.actor: ConsensusActor = (
            consensus_actor if consensus_actor is not None else ConsensusActor()
        )
        self.initial_lam = initial_lam
        self.alpha = alpha
        self.max_iter = max_iter
        self.is_leader = is_leader
        self.leader_gain = leader_gain
        self.balance_tol = balance_tol

        #: How the last run ended: ``True`` if the leader reached the power
        #: balance, ``False`` if it hit the max_iter failsafe.
        self.converged: bool = False
        #: Iterations completed when the last run terminated.
        self.iterations: int = 0

        # Mutable iteration state (reset at the start of each consensus run)
        self._rounds = RoundBuffer()  # (λ, p) per round and sender
        self._deadline = RoundDeadline(round_timeout)
        self._data: Any = None  # run payload, cached from the round-0 message
        self._first_message: bool = True
        self._started: bool = False  # True once any round has begun
        self._k: int = 0
        self._lam: np.ndarray = np.array([initial_lam])
        self._prev_lam: np.ndarray = self._lam.copy()  # λ before the last update
        self._P: np.ndarray = np.array([0.0])  # own last-projected power

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: AveragingConsensusMessage,
        meta: Any,
    ) -> None:
        """Process one incoming averaging consensus message."""
        neighbours = carrier.others()

        # --- Termination path (leader's power-balance stop) ---
        if message_data.stop:
            if self._first_message:
                return
            if message_data.k < self._k:
                # We already applied the next round's update before the stop
                # arrived (we can be at most one round ahead: advancing again
                # needs the leader's next message). Roll back so we finish on
                # the schedule the leader verified.
                self._lam = self._prev_lam
                self._P = np.asarray(self.actor.project_power(self._lam, self._data))
                self._k = message_data.k
            self._finish(carrier, converged=True)
            return

        # A neighbour's last-round message (k = max_iter) does not end this
        # run: every participant stops by its own counter (max_iter failsafe
        # in _advance), so one that is behind under loss still applies its
        # remaining rounds. The message is kept as that neighbour's freshest
        # value. Outside a run it is stale and ignored.
        if message_data.k >= self.max_iter and self._first_message:
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
            self._rounds.reset()
            self.converged = False
            # ``data`` is constant for the whole run, so it only travels with
            # the kick-off and round-0 messages. Under packet loss a follower
            # that misses every round-0 message never receives it; that is
            # harmless as only the leader (which gets the kick-off) needs it.
            self._data = message_data.data
            self._lam = np.ones(len(message_data.lam)) * self.initial_lam
            self._P = np.asarray(self.actor.project_power(self._lam, self._data))
            self._broadcast(carrier, neighbours)
            if message_data.initial:
                # This is the external kick-off trigger, not a neighbour's
                # round-0 contribution — don't queue it for averaging/ΔP.
                return

        sender = sender_key(carrier, meta)
        value = (np.asarray(message_data.lam), np.asarray(message_data.p))
        if message_data.k < self._k:
            # Too late for its round (never rewinds the counter), but still
            # the freshest value of that neighbour for later fallbacks.
            self._rounds.remember(message_data.k, sender, value)
            return
        self._rounds.add(message_data.k, sender, value)
        while not self._first_message and self._rounds.complete(self._k, len(neighbours)):
            self._advance(carrier, neighbours)

    def _advance(self, carrier: "Carrier", neighbours: list) -> None:
        """Run round ``self._k`` with the best values available and broadcast."""
        self._deadline.cancel()
        size = len(self._lam)
        values = self._rounds.values(
            self._k,
            neighbours,
            default=lambda _n: (np.full(size, self.initial_lam), np.zeros(size)),
        )
        avg_lam = sum(lam for lam, _ in values) / len(values)
        self._prev_lam = self._lam
        # The ΔP term needs every participant's power; until each neighbour
        # has been heard once, the default p = 0 would fake a huge deficit.
        round_complete = self._rounds.complete(self._k, len(neighbours))
        if self.is_leader and (self._rounds.heard_from_all(neighbours) or round_complete):
            pd = np.asarray(self._data, dtype=float)
            sigma_p = self._P + sum(p for _, p in values)
            delta_p = pd - sigma_p
            # Stop criterion: power balance ΔP ≈ 0 (see class docstring).
            # This round's schedules are the ones checked, so stop before
            # updating λ. Only judged on real round-k values, never on
            # last-known fallbacks.
            rel_gap = np.abs(delta_p) / np.maximum(np.abs(pd), 1e-9)
            if round_complete and np.max(rel_gap) <= self.balance_tol:
                for addr in neighbours:
                    carrier.send_to_other(
                        AveragingConsensusMessage(
                            lam=self._lam.copy(), k=self._k, data=None, stop=True
                        ),
                        addr,
                    )
                self._finish(carrier, converged=True)
                return
            self._lam = self._lam + self.alpha * (avg_lam - self._lam) + self.leader_gain * delta_p
        else:
            self._lam = self._lam + self.alpha * (avg_lam - self._lam)
        self._P = np.asarray(self.actor.project_power(self._lam, self._data))
        self._k += 1
        self._rounds.discard_before(self._k)
        self._broadcast(carrier, neighbours)
        if self._k >= self.max_iter:
            self._finish(carrier, converged=False)

    def _broadcast(self, carrier: "Carrier", neighbours: list) -> None:
        """Send this round's state and arm the round deadline."""
        for addr in neighbours:
            carrier.send_to_other(
                AveragingConsensusMessage(
                    lam=self._lam.copy(),
                    k=self._k,
                    data=self._data if self._k == 0 else None,
                    p=self._P.copy(),
                ),
                addr,
            )
        if self._k < self.max_iter:
            k = self._k
            self._deadline.arm(carrier, lambda: self._on_deadline(carrier, k))

    def _on_deadline(self, carrier: "Carrier", k: int) -> None:
        """Round *k* timed out: advance with what arrived, plus last values."""
        if self._first_message or self._k != k:
            return
        neighbours = carrier.others()
        self._advance(carrier, neighbours)
        while not self._first_message and self._rounds.complete(self._k, len(neighbours)):
            self._advance(carrier, neighbours)

    def _finish(self, carrier: "Carrier", converged: bool) -> None:
        self._deadline.cancel()
        self.converged = converged
        self.iterations = self._k
        self.finish_callback(self, carrier)
        self._first_message = True
        self._rounds.reset()


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


#: Create an :class:`AveragingConsensusAlgorithm` participant (same signature).
create_averaging_consensus_participant = AveragingConsensusAlgorithm


def create_averaging_consensus_start(
    initial_lam: float,
    data: Any = None,
) -> AveragingConsensusMessage:
    """Create the initial kick-off message for an averaging consensus run.

    :param initial_lam: Placeholder λ value for the message.  It does **not**
                        set the starting price: participants start from the
                        ``initial_lam`` they were constructed with.  The
                        message's λ has length 1; for a multi-timestep
                        ``data`` vector, λ takes on the horizon length via
                        broadcasting during the first update.
    :param data: Auxiliary payload forwarded to :meth:`ConsensusActor.project_power`
                 (the leader also reads it as the demand vector for ΔP).
    :returns: An :class:`AveragingConsensusMessage` with ``initial=True``.
    """
    return AveragingConsensusMessage(
        lam=np.array([initial_lam]),
        k=0,
        data=data,
        initial=True,
    )
