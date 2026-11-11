"""Consensus + innovations (Kar & Hug 2012; Hug, Kar & Wu 2015).

Every participant j holds a vector λ_j and a local response to it. Per iteration i (from 1):

.. math::

    \\lambda_j \\leftarrow \\lambda_j - \\beta_i \\sum_{l \\in \\omega_j} (\\lambda_j - \\lambda_l)
                                  - \\alpha_i \\, \\mathrm{innovation}_j(\\lambda_j)

over the participant's peers ω_j (topology neighbours that take part), then projected onto
λ_j ≥ ``lam_min`` (default −∞: no bound). The innovation is the
participant's local mismatch for its current λ (supplied by a :class:`ConsensusActor`); the
consensus term makes the λ_j agree, the innovation drives them to where the mismatches sum to 0.
Rounds are synchronous: iteration k waits for every peer's λ of iteration k.

Stop: a participant is *converged* when ``max |Δλ| < eps`` and its actor reports
:meth:`ConsensusActor.converged`. It reports flips of that flag to the coordinator
(:class:`ConsensusReport`); the coordinator ends the run when all participants are converged
(the carrier's :class:`OptimizationFinishedMessage` is the stop). Each participant also stops on
its own after ``max_iter`` iterations. On stopping it calls ``finish_callback(algorithm, carrier)``.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ...core import Coordinator, DistributedAlgorithm, OptimizationMessage

if TYPE_CHECKING:
    from ....carrier.core import Carrier


class ConsensusActor:
    """Local problem of one participant: its innovation for a λ and its own stop check."""

    def innovation(self, lam: np.ndarray) -> np.ndarray:
        """Local mismatch at *lam* (e.g. supply − demand); summed over all participants it is
        zero at the solution."""
        return np.zeros_like(lam)

    def converged(self) -> bool:
        """True when the actor's own decisions stopped moving (default: always)."""
        return True


def power_step(c: float, exponent: float) -> Callable[[int], float]:
    """Step schedule ``c / i**exponent`` (exponent 0 = constant step)."""
    return lambda i: c / i**exponent


@dataclass
class ConsensusStart(OptimizationMessage):
    """Kick-off from the coordinator: initial λ, who takes part, where to report."""

    lam: np.ndarray
    participants: list
    coordinator: Any
    t0: float = 0.0  # carrier time at which round 0 starts for everyone (round deadlines)


@dataclass
class ConsensusMessage(OptimizationMessage):
    """λ of the sender after *k* iterations."""

    lam: np.ndarray
    k: int


@dataclass
class ConsensusReport(OptimizationMessage):
    """Participant → coordinator: its converged flag flipped, or it hit ``max_iter`` (*done*)."""

    sender: Any
    converged: bool
    k: int
    done: bool = False


@dataclass
class ConsensusResult:
    """What the coordinator saw: ``stop_reason`` "converged" or "max_iter"."""

    stop_reason: str
    iterations: int


class ConsensusInnovationsAlgorithm(DistributedAlgorithm):
    """One consensus + innovations participant.

    :param finish_callback: ``(algorithm, carrier)``, called once when the participant stops.
    :param actor: The local problem (:class:`ConsensusActor`).
    :param alpha: Innovation step α_i as a function of the iteration i ≥ 1.
    :param beta: Consensus step β_i as a function of the iteration i ≥ 1.
    :param eps: Tolerance on ``max |Δλ|`` for the converged flag.
    :param max_iter: Iteration cap.
    :param lam_min: Lower bound on λ (scalar or per entry), applied after each update; default
        −∞ = unbounded (the papers' update). A floor keeps λ finite when no λ can make the
        mismatches sum to 0 from above (e.g. a supply surplus nobody can absorb).
    :param round_timeout: Round deadline (carrier clock) for lossy links: if some peers' λ of
        the current iteration have not arrived after this long, iterate with those that did —
        the missing ones are failed links of that iteration (Kar & Hug 2012 §IV, ω_q(t)); a λ of a
        finished iteration arriving later is dropped. ``None`` (default) = wait for every peer.
        ``rounds_by_deadline`` / ``missing_edges`` count how often and how many were missing.
    """

    def __init__(
        self,
        finish_callback: Callable,
        actor: ConsensusActor,
        alpha: Callable[[int], float],
        beta: Callable[[int], float],
        eps: float = 1e-4,
        max_iter: int = 10_000,
        lam_min: float | np.ndarray = -np.inf,
        round_timeout: float | None = None,
    ) -> None:
        self.finish_callback = finish_callback
        self.actor = actor
        self.alpha, self.beta = alpha, beta
        self.eps, self.max_iter = eps, max_iter
        self.lam_min = lam_min
        self.lam: np.ndarray | None = None
        self.k = 0
        self.stopped = False
        self._start: ConsensusStart | None = None
        self._peers: list = []
        self._inbox: dict[int, list[np.ndarray]] = defaultdict(list)
        self._converged = False
        self.round_timeout = round_timeout
        self.rounds_by_deadline = 0
        self.missing_edges = 0

    async def on_exchange_message(self, carrier: "Carrier", message_data: Any, meta: Any) -> None:
        if self.stopped:
            return
        if isinstance(message_data, ConsensusStart):
            self._start = message_data
            self.lam = np.array(message_data.lam, dtype=float)
            self._peers = [a for a in carrier.others("") if a in message_data.participants]
            self._send_lam(carrier)
            if self.round_timeout is not None and self._peers:
                carrier.spawn(self._deadlines(carrier))
        elif isinstance(message_data, ConsensusMessage):
            if message_data.k >= self.k:  # a λ of a finished (deadline) round is dropped
                self._inbox[message_data.k].append(message_data.lam)
        else:  # the coordinator's end of the run (OptimizationFinishedMessage)
            self._stop(carrier)
            return
        self._advance(carrier)

    def _advance(self, carrier: "Carrier") -> None:
        """Iterate while the current round is complete (every peer's λ is in)."""
        while (
            self._start is not None
            and not self.stopped
            and len(self._inbox[self.k]) == len(self._peers)
        ):
            self._iterate(carrier)

    def _iterate(self, carrier: "Carrier") -> None:
        i = self.k + 1
        neighbours = self._inbox.pop(self.k, [])  # [] = every peer's λ lost (deadline round)
        consensus = sum((self.lam - lam_l for lam_l in neighbours), np.zeros_like(self.lam))
        new = self.lam - self.beta(i) * consensus - self.alpha(i) * self.actor.innovation(self.lam)
        new = np.maximum(new, self.lam_min)
        converged = bool(np.max(np.abs(new - self.lam)) < self.eps) and self.actor.converged()
        self.lam, self.k = new, i
        if self.k >= self.max_iter:
            self._report(carrier, converged, done=True)
            self._stop(carrier)
            return
        if converged != self._converged:
            self._converged = converged
            self._report(carrier, converged)
            if converged and not self._peers:
                # alone: its own check is the global one, and without peers the loop in
                # on_exchange_message never yields, so the coordinator's stop could not arrive
                self._stop(carrier)
                return
        self._send_lam(carrier)

    def _send_lam(self, carrier: "Carrier") -> None:
        for peer in self._peers:
            carrier.send_to_other(ConsensusMessage(self.lam.copy(), self.k), peer)

    async def _deadlines(self, carrier: "Carrier") -> None:
        """One timer per participant: round k ends at the same carrier time for everyone,
        t0 + (k+1)·round_timeout (KH12's common iteration index t), so a λ that is only late is
        never cut off and participants do not drift apart. Asleep until the current round's end;
        if that round is still open then, iterate with the peers that arrived."""
        while not self.stopped:
            k = self.k
            end = self._start.t0 + (k + 1) * self.round_timeout
            await carrier.sleep(max(0.0, end - carrier.now()))
            if self.stopped or self.k != k:
                continue  # round finished in time (or the run ended): wait for the current one
            self.rounds_by_deadline += 1
            self.missing_edges += len(self._peers) - len(self._inbox[k])
            self._iterate(carrier)
            self._advance(carrier)

    def _report(self, carrier: "Carrier", converged: bool, done: bool = False) -> None:
        report = ConsensusReport(carrier.get_address(), converged, self.k, done)
        carrier.send_to_other(report, self._start.coordinator)

    def _stop(self, carrier: "Carrier") -> None:
        self.stopped = True
        self.finish_callback(self, carrier)


class ConsensusCoordinator(Coordinator):
    """Starts the participants and ends the run once all report converged (or all hit max_iter).

    :param participants: Addresses of all participants.
    :param lam0: Initial λ for every participant (e.g. a flat start).
    :param settle_s: When all flags say converged, wait this long (carrier clock) and stop only if
        they still do. Reports reach the coordinator after different delays (multi-hop routes), so a
        late "not converged" flip can still be on its way; set it to the longest report delay.
        0 (default) = stop at once (equal delays, e.g. every participant a neighbour).
    :param start_latency: Longest delay of the start message to a participant; round 0 of the
        participants' round deadlines (``round_timeout``) starts that long after sending.
    """

    def __init__(
        self,
        participants: list,
        lam0: np.ndarray,
        settle_s: float = 0.0,
        start_latency: float = 0.0,
    ) -> None:
        self.participants = list(participants)
        self.lam0 = np.asarray(lam0, dtype=float)
        self.settle_s = settle_s
        self.start_latency = start_latency
        self._settling = False
        self.result: ConsensusResult | None = None
        self._flags: dict = {}
        self._done: set = set()
        self._k = 0
        self._finished: asyncio.Future | None = None

    async def start_optimization(self, carrier: "Carrier", message_data: Any, meta: Any):
        self._finished = asyncio.get_event_loop().create_future()
        t0 = carrier.now() + self.start_latency  # round 0 starts once every start has arrived
        start = ConsensusStart(self.lam0, self.participants, carrier.get_address(), t0)
        for p in self.participants:
            carrier.send_to_other(start, p)
        await carrier.wait_for(self._finished)
        return self.result

    async def on_exchange_message(self, carrier: "Carrier", message_data: Any, meta: Any) -> None:
        """Collects each :class:`ConsensusReport` (carriers forward participant messages here)."""
        if not isinstance(message_data, ConsensusReport) or self._finished.done():
            return
        self._flags[message_data.sender] = message_data.converged
        if message_data.done:
            self._done.add(message_data.sender)
        self._k = max(self._k, message_data.k)
        if len(self._done) == len(self.participants):
            self._finish(ConsensusResult("max_iter", self._k))
        elif self._all_converged():
            if self.settle_s <= 0:
                self._finish(ConsensusResult("converged", self._k))
            elif not self._settling:
                self._settling = True
                carrier.spawn(self._settle(carrier))

    def _all_converged(self) -> bool:
        return all(self._flags.get(p, False) for p in self.participants)

    async def _settle(self, carrier: "Carrier") -> None:
        await carrier.sleep(self.settle_s)
        self._settling = False
        if not self._finished.done() and self._all_converged():
            self._finish(ConsensusResult("converged", self._k))

    def _finish(self, result: ConsensusResult) -> None:
        self.result = result
        self._finished.set_result(result)
