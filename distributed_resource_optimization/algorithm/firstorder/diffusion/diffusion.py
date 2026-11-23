"""Diffusion, combine-then-adapt, with a sum constraint (de Azevedo et al. 2017, §III).

Two passes over the same peer-to-peer rounds and combination weights a_ij (given per peer,
a_ii = 1 − Σ_j a_ij):

1. **Information sharing** (eq. 13): every participant starts from a local quantity x_i⁰
   (:meth:`DiffusionActor.measure`) and iterates φ_i = Σ_j a_ij x_j, x_i = φ_i − μ_IS·ŝ_i with ŝ_i the
   change of φ_i since the previous round; x_i tends to the mean. A participant is *converged* when
   ``max |Δx_i| ≤ eps_is``; once all are (coordinator), the target is n·x_i (eq. 10: P_N = x·n).
2. **Optimization** (eqs. 16, 19–21): every participant holds an estimate w_i of all n entries
   (one row per participant, one column per period), starts at 0 and per round combines
   φ_i = Σ_j a_ij w_j. A participant whose entry is *active* adapts its own entry by
   Δ = −μ·∇J_i(φ_i[own]) (:meth:`DiffusionActor.gradient`) and spreads the "nudge"
   target − Σ w_i,k−1 − Δ equally over the other active entries, so the estimate keeps the sum
   (eq. 19); an inactive one only combines. The run ends at ``max_iter`` (counted over both
   passes; the paper tracks forever).
"""

# 2026-11-23 D-225: generic CTA + nudge (de Azevedo 2017); replaces the Ces-based ATC (D-211)

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ...core import OptimizationMessage
from ..consensus.innovations import ConsensusCoordinator, ConsensusReport, ConsensusStart

if TYPE_CHECKING:
    from ....carrier.core import Carrier


class DiffusionActor:
    """Local problem of one participant."""

    def measure(self, periods: int) -> np.ndarray:
        """x_i⁰ of the information sharing pass (one value per period)."""
        return np.zeros(periods)

    def gradient(self, w_own: np.ndarray) -> np.ndarray:
        """∇J_i at the own entry *w_own* (one value per period)."""
        return np.zeros_like(w_own)


@dataclass
class DiffusionStart(ConsensusStart):
    """Kick-off of pass *pass_no* (0 = information sharing, 1 = optimization)."""

    pass_no: int = 0


@dataclass
class DiffusionMessage(OptimizationMessage):
    """State (x or w) of *sender* in round *k* of pass *pass_no*; *iterations* = the sender's
    iterations of both passes so far."""

    value: np.ndarray
    k: int
    sender: Any
    pass_no: int = 0
    iterations: int = 0


@dataclass
class DiffusionReport(ConsensusReport):
    """A consensus report plus the pass it belongs to."""

    pass_no: int = 0


class DiffusionAlgorithm:
    """One diffusion participant.

    :param finish_callback: ``(algorithm, carrier)``, called once when the participant stops.
    :param actor: The local problem (:class:`DiffusionActor`).
    :param weights: a_ij per peer address; a_ii = 1 − Σ.
    :param own: Index of the participant's entry in w.
    :param active: Indices of the entries that adapt and take nudges (e.g. dispatchable units).
    :param entries: Number of entries of w (n).
    :param step: μ of the optimization pass.
    :param step_is: μ of the information sharing pass (0 = plain averaging).
    :param eps_is: Precision that ends the information sharing pass (``max |Δx| ≤ eps_is``).
    :param max_iter: Iteration cap over both passes.
    :param round_timeout: Round deadline (carrier clock) for lossy links, as in consensus: a
        missing peer's weight goes to self. ``None`` = wait for every peer.
    """

    def __init__(
        self,
        finish_callback: Callable,
        actor: DiffusionActor,
        weights: dict,
        own: int,
        active: list[int],
        entries: int,
        step: float,
        step_is: float = 0.0,
        eps_is: float = 1e-4,
        max_iter: int = 10_000,
        round_timeout: float | None = None,
    ) -> None:
        self.finish_callback, self.actor, self.weights = finish_callback, actor, weights
        self.own, self.entries = own, entries
        self.others = [j for j in active if j != own]
        self.is_active = own in active
        self.step, self.step_is, self.eps_is = step, step_is, eps_is
        self.max_iter, self.round_timeout = max_iter, round_timeout
        self.value: np.ndarray | None = None  # x (pass 0) or w (pass 1)
        self.target: np.ndarray | None = None
        self._phi_prev: np.ndarray | None = None
        self.k = 0  # iterations of both passes
        self.r = 0  # round within the pass
        self.pass_no = -1
        self.stopped = False
        self._start: DiffusionStart | None = None
        self._peers: list = []
        self._inbox: dict[tuple[int, int], list[DiffusionMessage]] = defaultdict(list)
        self._converged = False
        self.rounds_by_deadline = 0
        self.missing_edges = 0

    async def on_exchange_message(self, carrier: "Carrier", message_data: Any, meta: Any) -> None:
        if self.stopped:
            return
        if isinstance(message_data, DiffusionStart):
            if message_data.pass_no <= self.pass_no:
                return
            first = self._start is None
            self._begin(carrier, message_data)
            if first and self.round_timeout is not None and self._peers:
                carrier.spawn(self._deadlines(carrier))
        elif isinstance(message_data, DiffusionMessage):
            # an older pass or a finished (deadline) round is dropped; a newer pass is kept
            if (message_data.pass_no, message_data.k) >= (self.pass_no, self.r):
                self._inbox[message_data.pass_no, message_data.k].append(message_data)
        elif isinstance(message_data, ConsensusReport):
            return  # a peer's report to a coordinator on this agent (see consensus)
        else:  # the coordinator's end of the run (OptimizationFinishedMessage)
            self._stop(carrier)
            return
        self._advance(carrier)

    def _begin(self, carrier: "Carrier", start: DiffusionStart) -> None:
        periods = len(start.lam)
        if start.pass_no == 0:
            self.value = np.asarray(self.actor.measure(periods), dtype=float)
        else:  # eq. (10): P_N = x·n; w⁰ = 0 (S8a plan, D-222)
            self.target = self.value * len(start.participants)
            self.value = np.zeros((self.entries, periods))
        self._start, self.pass_no, self.r = start, start.pass_no, 0
        self._peers = [a for a in carrier.others("") if a in start.participants]
        self._phi_prev, self._converged = None, False
        self._send(carrier)

    def _send(self, carrier: "Carrier") -> None:
        message = DiffusionMessage(
            self.value.copy(), self.r, carrier.get_address(), self.pass_no, self.k
        )
        for peer in self._peers:
            carrier.send_to_other(message, peer)

    def _advance(self, carrier: "Carrier") -> None:
        """Iterate while the current round is complete; alone, until the IS pass ends."""
        while (
            self._start is not None
            and not self.stopped
            and len(self._inbox[self.pass_no, self.r]) == len(self._peers)
        ):
            if self._iterate(carrier) and not self._peers:
                return  # alone: wait for the coordinator's next pass

    def _iterate(self, carrier: "Carrier") -> bool:
        """One round; returns True if it sent a report."""
        received = self._inbox.pop((self.pass_no, self.r), [])
        a = [self.weights[m.sender] for m in received]
        # combine; weights of missing peers (deadline rounds) go to self (as DEED-ADMM Z8)
        phi = (1.0 - sum(a)) * self.value + sum(
            (aj * m.value for aj, m in zip(a, received)), np.zeros_like(self.value)
        )
        old = self.value
        if self.pass_no == 0:  # eq. (13): ŝ = change of φ since the previous round
            s = 0.0 if self._phi_prev is None else phi - self._phi_prev
            self._phi_prev, self.value = phi, phi - self.step_is * s
        else:
            self.value = self._adapt(phi, old)
        # a new pass reaches the participants at different times; max keeps k equal (max_iter)
        self.k = max([self.k, *(m.iterations for m in received)]) + 1
        self.r += 1
        if self.k >= self.max_iter:
            self._report(carrier, False, done=True)
            self._stop(carrier)
            return True
        reported = False
        if self.pass_no == 0:
            converged = bool(np.max(np.abs(self.value - old)) <= self.eps_is)
            if converged != self._converged:
                self._converged, reported = converged, True
                self._report(carrier, converged)
        self._send(carrier)
        return reported

    def _adapt(self, phi: np.ndarray, old: np.ndarray) -> np.ndarray:
        """Eqs. (16), (20), (21): gradient step on the own entry, nudge on the other active ones."""
        gradient = np.asarray(self.actor.gradient(phi[self.own]))
        if not self.is_active:
            return phi  # uninformed: combine only (§III-B)
        w = phi.copy()
        delta = -self.step * gradient
        w[self.own] += delta
        # 2026-11-23 D-226: eq. (21) as printed (previous estimate's sum): the sum error follows
        # e_k = (A − I)e_{k−1}, so it grows where A has an eigenvalue < 0 (a result, not fixed)
        nudge = self.target - old.sum(axis=0) - delta
        # 2026-11-23 D-223: equal split (Fig. 8); alone, the own entry keeps the sum
        if self.others:
            w[self.others] += nudge / len(self.others)
        else:
            w[self.own] += nudge
        return w

    async def _deadlines(self, carrier: "Carrier") -> None:
        """Round r of a pass ends at t0 + (r+1)·round_timeout for everyone (as consensus)."""
        while not self.stopped:
            current = (self.pass_no, self.r)
            end = self._start.t0 + (self.r + 1) * self.round_timeout
            await carrier.sleep(max(0.0, end - carrier.now()))
            if self.stopped or (self.pass_no, self.r) != current:
                continue
            self.rounds_by_deadline += 1
            self.missing_edges += len(self._peers) - len(self._inbox[current])
            self._iterate(carrier)
            self._advance(carrier)

    def _report(self, carrier: "Carrier", converged: bool, done=False) -> None:
        report = DiffusionReport(carrier.get_address(), converged, self.k, done, self.pass_no)
        carrier.send_to_other(report, self._start.coordinator)

    def _stop(self, carrier: "Carrier") -> None:
        self.stopped = True
        self.finish_callback(self, carrier)


class DiffusionCoordinator(ConsensusCoordinator):
    """The consensus coordinator with two passes: when all participants have converged in the
    information sharing pass it starts the optimization pass; that pass has no stop rule, the
    run ends when all reach ``max_iter``."""

    pass_no = 0

    def _start_message(self, carrier: "Carrier", t0: float) -> DiffusionStart:
        return DiffusionStart(self.lam0, self.participants, carrier.get_address(), t0, self.pass_no)

    async def on_exchange_message(self, carrier: "Carrier", message_data: Any, meta: Any) -> None:
        if (
            isinstance(message_data, DiffusionReport)
            and message_data.pass_no != self.pass_no
            and not message_data.done
        ):
            return  # a late report of an earlier pass
        await super().on_exchange_message(carrier, message_data, meta)

    def _converged(self, carrier: "Carrier") -> None:
        # 2026-11-23 D-217: IS converged → optimization pass (Fig. 5, eq. 10)
        if self.pass_no == 0:
            self.pass_no = 1
            self._flags.clear()
            start = self._start_message(carrier, carrier.now() + self.start_latency)
            for p in self.participants:
                carrier.send_to_other(start, p)
