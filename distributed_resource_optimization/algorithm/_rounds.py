"""Round bookkeeping shared by the peer-to-peer algorithms (loss tolerance).

Consensus, diffusion, exact diffusion, FDGDM and DEED-ADMM run synchronous
rounds: every participant broadcasts its round-k state and advances once it
holds round-k state from every neighbour.  A single lost message used to
stall that forever.  Two helpers make the rounds tolerate loss:

* :class:`RoundBuffer` stores neighbour values per round **and per sender**.
  Round k uses each neighbour's round-k value when it arrived; a neighbour
  whose round-k message is missing contributes its most recent value, and a
  neighbour never heard from contributes a given default.  Without loss every
  round is complete, so results are identical to the strictly synchronous
  algorithm.
* :class:`RoundDeadline` arms a per-round timer on the carrier's clock
  (:meth:`~..carrier.core.Carrier.schedule_after`), so a round advances when it
  is complete *or* when its deadline passes, whichever is first.

If the sender of a message cannot be identified (test stand-in carriers that
pass no ``meta``), values are kept anonymously and rounds complete by count,
as before; the last-value fallback then has nothing to fall back on.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from typing import Any


def sender_key(carrier: Any, meta: Any) -> Any:
    """The sender of a message, or ``None`` if the carrier/meta cannot tell."""
    if not meta:
        return None
    sender_of = getattr(carrier, "sender_of", None)
    if sender_of is None:
        return None
    try:
        return sender_of(meta)
    except (KeyError, TypeError, AttributeError):
        return None


def address_of(carrier: Any) -> Any:
    """The carrier's own address, or ``None`` if it cannot tell."""
    address = getattr(carrier, "address", None)
    return address() if address is not None else None


class _Anonymous:
    """Key for a value whose sender is unknown (never matches a neighbour)."""

    _ids = itertools.count()

    def __init__(self) -> None:
        self.id = next(self._ids)

    def __repr__(self) -> str:
        return f"<anonymous {self.id}>"


class RoundBuffer:
    """Neighbour values per round with a last-known-value fallback."""

    def __init__(self) -> None:
        self._rounds: dict[int, dict[Any, Any]] = {}
        self._latest: dict[Any, tuple[int, Any]] = {}

    def reset(self) -> None:
        """Forget everything (start of a new run)."""
        self._rounds.clear()
        self._latest.clear()

    def add(self, k: int, sender: Any, value: Any) -> None:
        """Store *sender*'s value for round *k* (first value per sender wins)."""
        key = sender if sender is not None else _Anonymous()
        self._rounds.setdefault(k, {}).setdefault(key, value)
        self.remember(k, sender, value)

    def remember(self, k: int, sender: Any, value: Any) -> None:
        """Update *sender*'s last known value without filling a round.

        Used for messages of rounds already passed: too late for their round,
        but still the freshest information about that neighbour.
        """
        if sender is None:
            return
        latest = self._latest.get(sender)
        if latest is None or k >= latest[0]:
            self._latest[sender] = (k, value)

    def count(self, k: int) -> int:
        """Number of distinct senders that reported for round *k*."""
        return len(self._rounds.get(k, ()))

    def complete(self, k: int, n_neighbours: int) -> bool:
        """Whether every neighbour reported for round *k*."""
        return self.count(k) >= n_neighbours

    def heard_from_all(self, neighbours: Iterable[Any]) -> bool:
        """Whether every neighbour has sent at least one value so far."""
        return all(n in self._latest for n in neighbours)

    def missing(self, k: int, neighbours: Iterable[Any]) -> list[Any]:
        """Neighbours without a round-*k* value."""
        got = self._rounds.get(k, {})
        return [n for n in neighbours if n not in got]

    def values(self, k: int, neighbours: Iterable[Any], default: Callable[[Any], Any]) -> list[Any]:
        """One value per neighbour for round *k*.

        Round-*k* value if it arrived, else the neighbour's most recent value,
        else ``default(neighbour)``.  Anonymous round-*k* values (unknown
        sender) are included as they are.  Values that arrived come first, in
        arrival order, so a lossless run sums them in the same order as a plain
        message queue would (bit-identical floating-point results).
        """
        got = self._rounds.get(k, {})
        neighbours = list(neighbours)
        wanted = set(neighbours)
        out = [v for key, v in got.items() if isinstance(key, _Anonymous) or key in wanted]
        for n in neighbours:
            if n in got:
                continue
            if n in self._latest:
                out.append(self._latest[n][1])
            else:
                out.append(default(n))
        # Without sender ids the neighbour list cannot be matched: use the
        # round's values only (legacy, count-based behaviour).
        if out and all(isinstance(key, _Anonymous) for key in got) and got:
            return list(got.values())
        return out

    def all_recent(
        self,
        neighbours: Iterable[Any],
        k: int,
        max_age: int,
        predicate: Callable[[Any], bool],
    ) -> bool:
        """Whether every neighbour's last value is at most *max_age* rounds
        older than *k* and satisfies *predicate*."""
        for n in neighbours:
            latest = self._latest.get(n)
            if latest is None or k - latest[0] > max_age or not predicate(latest[1]):
                return False
        return True

    def value_map(
        self, k: int, neighbours: Iterable[Any], default: Callable[[Any], Any]
    ) -> dict[Any, Any]:
        """Like :meth:`values`, keyed by neighbour (identified senders only).

        Same order as :meth:`values`: arrived values first, in arrival order.
        """
        got = self._rounds.get(k, {})
        neighbours = list(neighbours)
        wanted = set(neighbours)
        out: dict[Any, Any] = {key: v for key, v in got.items() if key in wanted}
        for n in neighbours:
            if n in out:
                continue
            if n in self._latest:
                out[n] = self._latest[n][1]
            else:
                out[n] = default(n)
        return out

    def round_values(self, k: int) -> list[Any]:
        """Only the values that actually arrived for round *k*."""
        return list(self._rounds.get(k, {}).values())

    def discard_before(self, k: int) -> None:
        """Drop per-round storage for rounds older than *k*."""
        for old in [r for r in self._rounds if r < k]:
            del self._rounds[old]


class RoundDeadline:
    """At most one pending round deadline, on the carrier's clock."""

    def __init__(self, timeout_s: float | None) -> None:
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError(f"round_timeout must be positive, got {timeout_s!r}")
        self.timeout_s = timeout_s
        self._handle: Any = None
        self._token: object | None = None

    @property
    def enabled(self) -> bool:
        return self.timeout_s is not None

    def arm(self, carrier: Any, on_expiry: Callable[[], None]) -> None:
        """Replace any pending deadline by a new one (no-op without a timeout)."""
        self.cancel()
        if self.timeout_s is None:
            return
        token = object()
        self._token = token

        def fire() -> None:
            # The firing timer is no longer pending: forget it first, so the
            # expiry handler (which advances the round and re-arms) does not
            # cancel the very timer that is running it.
            if self._token is token:
                self._handle = None
            on_expiry()

        self._handle = carrier.schedule_after(self.timeout_s, fire)

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None


def _order_key(address: Any) -> str:
    """Total order on participant addresses (mango addresses by agent id)."""
    return str(getattr(address, "aid", address))


class EdgeFlows:
    """Pairwise flows ``w·(x_other − x_own)`` that both ends apply exactly once.

    Mixing steps such as ``x_i ← x_i + Σ_j w_ij (x_j − x_i)`` conserve
    ``Σ_i x_i`` only if the two ends of every edge apply opposite flows,
    i.e. use the *same* pair of values.  FDGDM (power balance), DEED-ADMM's
    ξ tracker and Exact Diffusion's correction rely on that invariant; with a
    lost message one end would use a stale value and the sum would drift for
    good.

    Running-sum protocol: each edge has an owner (the end with the smaller
    address).  Both ends compute the round's flow from their own view (so a
    lossless run is unchanged), and each records its cumulative flow per
    round, in the owner's sign.  The owner sends ``(r, S)`` — its cumulative
    flow through round ``r`` — with its messages to the other end (see
    :meth:`owner_sum`); the other end compares ``S`` with its own cumulative
    flow through the same round and applies the difference
    (:meth:`correction`).  A lost or stale message therefore only delays a
    contribution; the sum over both ends stays exact up to the flows of the
    rounds still in flight.
    """

    #: Rounds of per-round history kept for comparisons.
    history = 256

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._total: dict[Any, Any] = {}  # cumulative flow (owner sign)
        self._last_round: dict[Any, int] = {}
        self._prefix: dict[Any, dict[int, Any]] = {}  # round -> cumulative
        self._corrected: dict[Any, Any] = {}  # corrections applied (owner sign)
        self._reconciled: dict[Any, int] = {}  # last round reconciled
        self._offers: dict[Any, tuple[int, Any]] = {}  # owner's latest (r, S)

    @staticmethod
    def owns(me: Any, other: Any) -> bool:
        """Whether *me* owns the edge to *other*."""
        return _order_key(me) < _order_key(other)

    def flow(self, me: Any, other: Any, k: int, own_x: Any, other_x: Any, weight: float) -> Any:
        """Round-*k* flow into *me* along the edge to *other*; recorded."""
        into_me = weight * (other_x - own_x)
        owner_sign = into_me if self.owns(me, other) else -into_me
        total = self._total.get(other, 0) + owner_sign
        self._total[other] = total
        self._last_round[other] = k
        history = self._prefix.setdefault(other, {})
        history[k] = total
        for old in [r for r in history if r <= k - self.history]:
            del history[old]
        return into_me

    def owner_sum(self, me: Any, other: Any) -> tuple[int, Any] | None:
        """``(r, S)`` to send to *other* if *me* owns the edge, else ``None``."""
        if not self.owns(me, other) or other not in self._total:
            return None
        return self._last_round[other], self._total[other]

    def offered_round(self, other: Any) -> int:
        """Round of the newest owner sum received from *other* (``-1``: none)."""
        offer = self._offers.get(other)
        return -1 if offer is None else offer[0]

    def offer(self, other: Any, r: int, owner_total: Any) -> None:
        """Store the owner's ``(r, S)`` received from *other* (newest wins)."""
        current = self._offers.get(other)
        if current is None or r > current[0]:
            self._offers[other] = (r, owner_total)

    def correction(self, me: Any, other: Any) -> Any:
        """Extra flow into *me* that aligns this end with the owner's sum.

        Zero for owned edges, without a usable offer, and in lossless runs.
        """
        if self.owns(me, other):
            return 0.0
        offer = self._offers.get(other)
        if offer is None:
            return 0.0
        r, owner_total = offer
        if r <= self._reconciled.get(other, -1):
            return 0.0
        mine = self._prefix.get(other, {}).get(r)
        if mine is None:
            return 0.0  # round r not stepped yet (or forgotten): later
        extra = owner_total - mine - self._corrected.get(other, 0)
        self._corrected[other] = self._corrected.get(other, 0) + extra
        self._reconciled[other] = r
        return -extra


class FlowSettlement:
    """End-of-run confirmation of :class:`EdgeFlows` sums under message loss.

    Without it, the flows of the last rounds are lost when the owner's final
    running sum (sent with its round-``max_iter`` message) does not arrive
    before the other end finishes — the conserved sum then drifts by those
    flows (noticeable when a run is cut off by ``max_iter`` before it
    converged).

    When a participant reaches ``max_iter``:

    * **as owner** of an edge it keeps resending its final sum (via
      *resend*) every *timeout_s* until the other end acknowledges it, at most
      *retries* times;
    * **as non-owner** it waits until every owner neighbour's final sum
      arrived (or *retries* × *timeout_s* passed), then calls *on_settled* —
      the algorithm applies the outstanding :meth:`EdgeFlows.correction`\\ s
      and finishes.

    Owners never wait for non-owners, so the protocol cannot deadlock. A
    participant acknowledges every final sum it receives (also after it
    finished) via :meth:`ack_needed`.
    """

    def __init__(self, timeout_s: float | None, retries: int = 20) -> None:
        self.timeout_s = timeout_s
        self.retries = retries
        self.reset()

    def reset(self) -> None:
        self.active = False  # waiting for owners' final sums
        self.final_round = -1
        self._unacked: set[Any] = set()
        self._awaiting: set[Any] = set()
        self._on_settled: Callable[[], None] | None = None
        self._repeat_handle: Any = None
        self._wait_handle: Any = None

    @property
    def enabled(self) -> bool:
        return self.timeout_s is not None

    def start(
        self,
        carrier: Any,
        me: Any,
        neighbours: Iterable[Any],
        flows: EdgeFlows,
        final_round: int,
        resend: Callable[[Any], None],
        on_settled: Callable[[], None],
    ) -> None:
        """Begin settling after the last round (*final_round* = ``max_iter − 1``).

        *resend(addr)* sends this participant's final message (carrying its
        owner sum) to *addr* again; *on_settled()* is called exactly once.
        """
        self.cancel()
        self.final_round = final_round
        neighbours = list(neighbours)
        self._unacked = {
            n for n in neighbours if flows.owns(me, n) and flows.owner_sum(me, n) is not None
        }
        self._awaiting = {
            n for n in neighbours if not flows.owns(me, n) and flows.offered_round(n) < final_round
        }
        if self._unacked and self.retries > 0:
            self._schedule_repeat(carrier, resend, self.retries)
        if not self._awaiting:
            on_settled()
            return
        self.active = True
        self._on_settled = on_settled
        self._wait_handle = carrier.schedule_after(
            self.timeout_s * max(self.retries, 1), self._give_up
        )

    def _schedule_repeat(self, carrier: Any, resend: Callable[[Any], None], left: int) -> None:
        def repeat() -> None:
            self._repeat_handle = None
            for addr in list(self._unacked):
                resend(addr)
            if self._unacked and left > 1:
                self._schedule_repeat(carrier, resend, left - 1)

        self._repeat_handle = carrier.schedule_after(self.timeout_s, repeat)

    def ack_needed(self, flow_round: int, final_round: int) -> bool:
        """Whether a received owner sum for *flow_round* is a final one to acknowledge."""
        return self.enabled and flow_round >= final_round >= 0

    def acked(self, sender: Any) -> None:
        """*sender* confirmed it holds this participant's final owner sum."""
        self._unacked.discard(sender)
        if not self._unacked and self._repeat_handle is not None:
            self._repeat_handle.cancel()
            self._repeat_handle = None

    def offer_arrived(self, sender: Any, flow_round: int) -> None:
        """An owner sum from *sender* arrived (already passed to :meth:`EdgeFlows.offer`)."""
        if not self.active or flow_round < self.final_round:
            return
        self._awaiting.discard(sender)
        if not self._awaiting:
            self._settle()

    def _give_up(self) -> None:
        self._wait_handle = None
        if self.active:
            self._settle()

    def _settle(self) -> None:
        self.active = False
        if self._wait_handle is not None:
            self._wait_handle.cancel()
            self._wait_handle = None
        on_settled, self._on_settled = self._on_settled, None
        if on_settled is not None:
            on_settled()

    def cancel(self) -> None:
        """Stop all timers (new run)."""
        for handle in (self._repeat_handle, self._wait_handle):
            if handle is not None:
                handle.cancel()
        self.reset()


def convergence_radius(
    own_flag: bool, neighbour_states: Iterable[tuple[bool, int]], diameter: int
) -> int:
    """Hops around this agent known to have converged (-1: not even itself).

    *neighbour_states* are each neighbour's ``(converged flag, radius)`` as
    carried by its round message, where the radius is the one it computed in
    its previous round. A neighbour with its flag set certifies at least
    radius 0, so this agent's radius is ``1 + min`` over the neighbours'.
    Once the radius reaches the graph *diameter*, every agent's flag is
    known to be set and the run can end for everyone. On a complete graph
    (diameter 1) this is "own flag and every neighbour's flag".
    """
    if not own_flag:
        return -1
    reach = [max(radius, 0) if flag else -1 for flag, radius in neighbour_states]
    return 1 + min(reach, default=diameter)
