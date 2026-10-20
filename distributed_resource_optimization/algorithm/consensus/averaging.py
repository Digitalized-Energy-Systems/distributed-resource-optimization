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

from .._rounds import RoundBuffer, RoundDeadline, address_of, sender_key
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

    def restore_power(self, power: np.ndarray) -> None:
        """Reset the actor's schedule to *power* (an earlier round's answer).

        Used when a follower rolls back to the round the leader verified.
        The default keeps ``self.P`` in step if the actor has one.
        """
        if hasattr(self, "P"):
            self.P = np.array(power, dtype=float, copy=True)

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
    :param stop_ack: Under message loss (with a round deadline), a recipient's
                     acknowledgement of a ``stop``; the leader repeats the stop
                     to every neighbour that has not acknowledged it yet.
    :param report: A follower that is not the leader's neighbour reports its
                   round-``k`` power ``p`` to the leader directly (routed over
                   the communication graph), so the leader can still compute
                   the system-wide ΔP.
    :param run: Sender's run counter (round messages only); a finished
                participant starts a new run from a round-0 message of a
                later run, never from a stale one.
    """

    lam: np.ndarray
    k: int
    data: Any
    p: np.ndarray | None = None
    initial: bool = False
    stop: bool = False
    stop_ack: bool = False
    report: bool = False
    run: int = 0


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
    to its neighbours' messages. On a sparser graph the followers that are
    not its neighbours also send the leader a ``report`` with their round
    power (Jian et al. 2020, p. 1428: the leader "need[s] to obtain the global
    variable ΔP in real time"; they cite transmitting a PCC measurement to it).
    For that the leader needs :attr:`participants` and each follower
    :attr:`leader`; both may be set after construction, once addresses exist.
    The leader's ``stop`` goes to every participant.

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
    :param stop_retries: With a ``round_timeout``, how often the leader repeats
                         its power-balance ``stop`` (every ``round_timeout``)
                         to neighbours that have not acknowledged it, so a lost
                         stop cannot leave a follower iterating on an
                         unverified schedule.
    :param leader: Followers: the leader's address. A follower that is not
                   the leader's neighbour reports its power to it each round.
                   ``None`` (default): the leader neighbours everyone.
    :param participants: Leader: every participant's address (its own may be
                         included). ``None`` (default): only the neighbours.
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
        stop_retries: int = 20,
        leader: Any = None,
        participants: list[Any] | None = None,
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
        self.stop_retries = stop_retries
        self.leader = leader
        self.participants = participants

        #: How the last run ended: ``True`` if the leader reached the power
        #: balance, ``False`` if it hit the max_iter failsafe.
        self.converged: bool = False
        #: Iterations completed when the last run terminated.
        self.iterations: int = 0

        # Mutable iteration state (reset at the start of each consensus run)
        self._rounds = RoundBuffer()  # (λ, p) per round and sender
        self._reports = RoundBuffer()  # leader: p of non-neighbours
        self._deadline = RoundDeadline(round_timeout)
        self._data: Any = None  # run payload, cached from the round-0 message
        self._first_message: bool = True
        self._started: bool = False  # True once any round has begun
        self._run: int = 0  # runs started so far
        self._k: int = 0
        self._lam: np.ndarray = np.array([initial_lam])
        self._prev_lam: np.ndarray = self._lam.copy()  # λ before the last update
        # λ at the start of each round, so a stop that arrives several rounds
        # late (lost and repeated) can roll back to the verified round.
        self._lam_history: dict[int, np.ndarray] = {}
        # Power broadcast in each round, restored together with λ on a rollback.
        self._P_history: dict[int, np.ndarray] = {}
        self._P: np.ndarray = np.array([0.0])  # own last-projected power
        # Leader, under loss: the stop being repeated and who has not acked it.
        self._stop_msg: AveragingConsensusMessage | None = None
        self._stop_unacked: set[Any] = set()
        self._stop_handle: Any = None
        self._stop_repeats_left: int = 0

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: AveragingConsensusMessage,
        meta: Any,
    ) -> None:
        """Process one incoming averaging consensus message."""
        neighbours = carrier.others()

        # --- Leader: a neighbour acknowledged the stop (loss only) ---
        if message_data.stop_ack:
            self._stop_unacked.discard(sender_key(carrier, meta))
            if not self._stop_unacked:
                self._cancel_stop_repeats()
            return

        # --- Termination path (leader's power-balance stop) ---
        if message_data.stop:
            if self._deadline.enabled:
                # Acknowledge every copy (also after finishing), so the leader
                # stops repeating it.
                leader = sender_key(carrier, meta)
                if leader is not None:
                    carrier.send_to_other(
                        AveragingConsensusMessage(
                            lam=self._lam.copy(), k=message_data.k, data=None, stop_ack=True
                        ),
                        leader,
                    )
            if self._first_message:
                return
            if message_data.k < self._k:
                # We already applied later rounds before the stop arrived:
                # without loss at most one (advancing needs the leader's next
                # message); under loss a repeated stop can be several rounds
                # late. Roll back so we finish on the schedule the leader
                # verified.
                self._lam = self._lam_history.get(message_data.k, self._prev_lam)
                if message_data.k in self._P_history:
                    self._P = self._P_history[message_data.k]
                    self.actor.restore_power(self._P)
                else:
                    self._P = np.asarray(self.actor.project_power(self._lam, self._data))
                self._k = message_data.k
            self._finish(carrier, converged=True)
            return

        # --- Leader: power report of a participant that is not a neighbour ---
        if message_data.report:
            if not self.is_leader or self._first_message:
                return
            sender = sender_key(carrier, meta)
            power = np.asarray(message_data.p)
            if message_data.k < self._k:
                self._reports.remember(message_data.k, sender, power)
                return
            self._reports.add(message_data.k, sender, power)
            while not self._first_message and self._ready(carrier, neighbours):
                self._advance(carrier, neighbours)
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
        if (
            self._first_message
            and self._started
            and not message_data.initial
            and not (message_data.k == 0 and message_data.run > self._run)
        ):
            return
        # A kick-off (or a repeated one) for a run already started must not
        # restart it; a neighbour's round-0 message may start a run before the
        # kick-off arrives (it can take more hops on a sparse topology).
        if message_data.initial and not self._first_message:
            return

        # --- Initialisation path ---
        if self._first_message or message_data.initial:
            self._first_message = False
            self._started = True
            self._run = max(self._run + 1, int(getattr(message_data, "run", 0)))
            self._k = 0
            self._rounds.reset()
            self._reports.reset()
            self._lam_history.clear()
            self._P_history.clear()
            self._cancel_stop_repeats()
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
        while not self._first_message and self._ready(carrier, neighbours):
            self._advance(carrier, neighbours)

    def _non_neighbours(self, carrier: "Carrier", neighbours: list) -> list:
        """Leader: participants that report their power instead of being neighbours."""
        if not self.is_leader or not self.participants:
            return []
        me = address_of(carrier)
        return [a for a in self.participants if a != me and a not in neighbours]

    def _ready(self, carrier: "Carrier", neighbours: list) -> bool:
        """Round ``self._k`` is complete: every neighbour (and, for the leader,
        every reporting participant) sent its value."""
        if not self._rounds.complete(self._k, len(neighbours)):
            return False
        others = self._non_neighbours(carrier, neighbours)
        return not others or self._reports.complete(self._k, len(others))

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
        self._lam_history[self._k] = self._lam
        self._P_history[self._k] = self._P
        # The ΔP term needs every participant's power; until each neighbour
        # has been heard once, the default p = 0 would fake a huge deficit.
        others = self._non_neighbours(carrier, neighbours)
        round_complete = self._rounds.complete(self._k, len(neighbours)) and (
            not others or self._reports.complete(self._k, len(others))
        )
        heard_from_all = self._rounds.heard_from_all(neighbours) and self._reports.heard_from_all(
            others
        )
        if self.is_leader and (heard_from_all or round_complete):
            pd = np.asarray(self._data, dtype=float)
            sigma_p = self._P + sum(p for _, p in values)
            if others:
                sigma_p = sigma_p + sum(
                    self._reports.values(self._k, others, default=lambda _n: np.zeros(size))
                )
            delta_p = pd - sigma_p
            # Stop criterion: power balance ΔP ≈ 0 (see class docstring).
            # This round's schedules are the ones checked, so stop before
            # updating λ. Only judged on real round-k values, never on
            # last-known fallbacks.
            rel_gap = np.abs(delta_p) / np.maximum(np.abs(pd), 1e-9)
            if round_complete and np.max(rel_gap) <= self.balance_tol:
                stop = AveragingConsensusMessage(
                    lam=self._lam.copy(), k=self._k, data=None, stop=True
                )
                for addr in [*neighbours, *others]:
                    carrier.send_to_other(stop, addr)
                if self._deadline.enabled and self.stop_retries > 0:
                    self._stop_msg = stop
                    self._stop_unacked = set(neighbours) | set(others)
                    self._stop_repeats_left = self.stop_retries
                    self._schedule_stop_repeat(carrier)
                self._finish(carrier, converged=True)
                return
            self._lam = self._lam + self.alpha * (avg_lam - self._lam) + self.leader_gain * delta_p
        else:
            self._lam = self._lam + self.alpha * (avg_lam - self._lam)
        self._P = np.asarray(self.actor.project_power(self._lam, self._data))
        self._k += 1
        self._rounds.discard_before(self._k)
        self._reports.discard_before(self._k)
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
                    run=self._run,
                ),
                addr,
            )
        if not self.is_leader and self.leader is not None and self.leader not in neighbours:
            carrier.send_to_other(
                AveragingConsensusMessage(
                    lam=self._lam.copy(),
                    k=self._k,
                    data=None,
                    p=self._P.copy(),
                    report=True,
                    run=self._run,
                ),
                self.leader,
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
        while not self._first_message and self._ready(carrier, neighbours):
            self._advance(carrier, neighbours)

    def _schedule_stop_repeat(self, carrier: "Carrier") -> None:
        """Repeat the stop to unacknowledged neighbours after one round timeout."""

        def repeat() -> None:
            self._stop_handle = None
            if self._stop_msg is None or not self._stop_unacked:
                return
            for addr in list(self._stop_unacked):
                carrier.send_to_other(self._stop_msg, addr)
            self._stop_repeats_left -= 1
            if self._stop_repeats_left > 0:
                self._schedule_stop_repeat(carrier)

        self._stop_handle = carrier.schedule_after(self._deadline.timeout_s, repeat)

    def _cancel_stop_repeats(self) -> None:
        if self._stop_handle is not None:
            self._stop_handle.cancel()
            self._stop_handle = None
        self._stop_msg = None
        self._stop_unacked = set()

    def _finish(self, carrier: "Carrier", converged: bool) -> None:
        self._deadline.cancel()
        self.converged = converged
        self.iterations = self._k
        self.finish_callback(self, carrier)
        self._first_message = True
        self._rounds.reset()
        self._reports.reset()


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
