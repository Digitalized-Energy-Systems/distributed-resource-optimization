"""Diffusion, combine-then-adapt, with a sum constraint (de Azevedo et al. 2017, §III).

Every round each participant sends one packet to its neighbours (Fig. 10) and runs two diffusion
processes on what it received:

* **Information sharing** (eq. 13): x_i starts at a local quantity (:meth:`DiffusionActor.measure`);
  φ_i = Σ_j a_ij x_j, x_i = φ_i − μ_IS·ŝ_i with ŝ_i the change of φ_i since the previous round, so
  x_i tends to the mean. Convergence by flags among neighbours (§III): a participant's flag is
  ``max |Δx_i| ≤ eps_is``; its counter c_i = 1 + min(own and the neighbours' counters) while the flag
  holds, else 0. Past the network diameter d it proposes a reset round, the earliest proposal is
  flooded, and at that round every participant sets the target n·x_i (eq. 10) and restarts x_i from
  its measurement ("the information sharing process restarts", Fig. 5).
* **Optimization** (eqs. 16, 19–21), from the first reset on: an estimate w_i of all n entries
  (row per participant, column per period), w⁰ = 0; φ_i = Σ_j a_ij w_j, the own entry steps
  −μ·∇J_i(φ_i[own]) (:meth:`DiffusionActor.gradient`) and the "nudge" target − Σ w_i,k−1 − Δ is split
  equally over the other active entries; an inactive (uninformed) participant only combines.

Weights: Metropolis eq. (12) as printed, a_ij = 1/max(n_i, n_j) from the degrees in the packets,
a_ii = 1 − Σ_j a_ij, built on the fly (§III-A). n = the participants in the flooded ID database
(entries older than d rounds are dropped); a neighbour silent for more than d rounds is dropped
and the weights follow (§III-A). The run ends at ``max_iter`` (the paper tracks forever).
"""

# 2026-11-23 D-225: generic CTA + nudge (de Azevedo 2017); replaces the Ces-based ATC (D-211)

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ...core import OptimizationMessage
from ..consensus.innovations import ConsensusReport, ConsensusStart

if TYPE_CHECKING:
    from ....carrier.core import Carrier


def metropolis(n_i: int, n_j: int) -> float:
    """Eq. (12) as printed (2026-11-23 D-216): a_ij = 1/max(n_i, n_j), n = connections (no +1;
    even rings then have eigenvalue −1 — 1/(1+max) is the noted alternative)."""
    return 1 / max(n_i, n_j)


class DiffusionActor:
    """Local problem of one participant."""

    def measure(self, periods: int) -> np.ndarray:
        """x_i of the information sharing at every (re)start (one value per period)."""
        return np.zeros(periods)

    def gradient(self, w_own: np.ndarray) -> np.ndarray:
        """∇J_i at the own entry *w_own* (one value per period)."""
        return np.zeros_like(w_own)


@dataclass
class DiffusionMessage(OptimizationMessage):
    """One packet of round *k* (Fig. 10): the information sharing state *x*, the estimate *w*
    (None before the optimization starts), the sender's degree (eq. 12), convergence counter and
    proposed reset round (§III flags), and its ID database {id: last round seen} (§III-A)."""

    x: np.ndarray
    w: np.ndarray | None
    k: int
    sender: Any
    degree: int
    count: int
    reset: int | None
    seen: dict = field(default_factory=dict)


class DiffusionAlgorithm:
    """One diffusion participant.

    :param finish_callback: ``(algorithm, carrier)``, called once when the participant stops.
    :param actor: The local problem (:class:`DiffusionActor`).
    :param own: Index of the participant's entry in w.
    :param active: Indices of the entries that adapt and take nudges (e.g. dispatchable units).
    :param entries: Number of entries of w.
    :param step: μ of the optimization.
    :param step_is: μ of the information sharing (0 = plain averaging).
    :param eps_is: Precision of the information sharing flag (``max |Δx| ≤ eps_is``).
    :param diameter: d of the communication graph (flags, ID database, silent neighbours).
    :param max_iter: Iteration cap.
    :param round_timeout: Round deadline (carrier clock) for lossy links, as in consensus: a missing
        packet's weights go to self that round. ``None`` = wait for every neighbour.
    """

    def __init__(
        self,
        finish_callback: Callable,
        actor: DiffusionActor,
        own: int,
        active: list[int],
        entries: int,
        step: float,
        step_is: float = 0.0,
        eps_is: float = 1e-4,
        diameter: int = 1,
        max_iter: int = 10_000,
        round_timeout: float | None = None,
    ) -> None:
        self.finish_callback, self.actor = finish_callback, actor
        self.own, self.entries = own, entries
        self.others = [j for j in active if j != own]
        self.is_active = own in active
        self.step, self.step_is, self.eps_is = step, step_is, eps_is
        self.d, self.max_iter, self.round_timeout = diameter, max_iter, round_timeout
        self.x: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self.target: np.ndarray | None = None
        self.resets = 0  # information sharing convergences (each one sets the target)
        self._phi_prev: np.ndarray | None = None
        self.count, self.reset_at = 0, None
        self.seen: dict[str, int] = {}
        self.k = 0
        self.stopped = False
        self._start: ConsensusStart | None = None
        self._peers: list = []
        self._heard: dict = {}
        self._inbox: dict[int, list[DiffusionMessage]] = defaultdict(list)
        self.rounds_by_deadline = 0
        self.missing_edges = 0

    async def on_exchange_message(self, carrier: "Carrier", message_data: Any, meta: Any) -> None:
        if self.stopped:
            return
        if isinstance(message_data, ConsensusStart):
            if self._start is not None:
                return
            self._start = message_data
            self._me = str(carrier.get_address())
            self._peers = [a for a in carrier.others("") if a in message_data.participants]
            self._heard = dict.fromkeys(self._peers, 0)
            self.x = np.asarray(self.actor.measure(len(message_data.lam)), dtype=float)
            self.seen = {self._me: 0}
            self._send(carrier)
            if self.round_timeout is not None and self._peers:
                carrier.spawn(self._deadlines(carrier))
        elif isinstance(message_data, DiffusionMessage):
            if message_data.k >= self.k:  # a packet of a finished (deadline) round is dropped
                self._inbox[message_data.k].append(message_data)
        elif isinstance(message_data, ConsensusReport):
            return  # a peer's report to a coordinator on this agent (see consensus)
        else:  # the coordinator's end of the run (OptimizationFinishedMessage)
            self._stop(carrier)
            return
        self._advance(carrier)

    def _send(self, carrier: "Carrier") -> None:
        message = DiffusionMessage(
            self.x.copy(),
            None if self.w is None else self.w.copy(),
            self.k,
            carrier.get_address(),
            len(self._peers),
            self.count,
            self.reset_at,
            dict(self.seen),
        )
        for peer in self._peers:
            carrier.send_to_other(message, peer)

    def _advance(self, carrier: "Carrier") -> None:
        """Iterate while the current round is complete (alone: every round at once)."""
        while (
            self._start is not None
            and not self.stopped
            and len(self._current(self.k)) == len(self._peers)
        ):
            self._iterate(carrier)

    def _current(self, k: int) -> list[DiffusionMessage]:
        """Round *k*'s packets from current neighbours (a dropped one's are ignored)."""
        return [m for m in self._inbox[k] if m.sender in self._peers]

    @staticmethod
    def _combine(own: np.ndarray, values: list, weights: list) -> np.ndarray:
        # a missing packet's weight goes to self (deadline rounds; as DEED-ADMM Z8)
        return (1.0 - sum(weights)) * own + sum(
            (a * v for a, v in zip(weights, values)), np.zeros_like(own)
        )

    def _iterate(self, carrier: "Carrier") -> None:
        received = self._current(self.k)
        del self._inbox[self.k]
        n_i = len(self._peers)
        a = [metropolis(n_i, m.degree) for m in received]  # from the packets' degrees
        for m in received:
            self._heard[m.sender] = self.k
            for i, r in m.seen.items():
                self.seen[i] = max(self.seen.get(i, -1), r)
        # information sharing, eq. (13): ŝ = change of φ since the previous round
        phi = self._combine(self.x, [m.x for m in received], a)
        s = 0.0 if self._phi_prev is None else phi - self._phi_prev
        x, self._phi_prev = phi - self.step_is * s, phi
        flag = bool(np.max(np.abs(x - self.x)) <= self.eps_is)
        self.count = 1 + min([self.count, *(m.count for m in received)]) if flag else 0
        self.x = x
        # optimization, eqs. (16), (20), (21): with the packets that carry an estimate
        if self.w is not None:
            got = [(aj, m.w) for aj, m in zip(a, received) if m.w is not None]
            phi_w = self._combine(self.w, [v for _, v in got], [aj for aj, _ in got])
            self.w = self._adapt(phi_w, self.w)
        self.k += 1
        self.seen[self._me] = self.k
        self.seen = {i: r for i, r in self.seen.items() if self.k - r <= self.d}
        # 2026-11-23 D-228: flags among neighbours (§III); past d the earliest reset round wins
        proposals = [m.reset for m in received if m.reset is not None and m.reset >= self.k]
        if self.reset_at is not None and self.reset_at >= self.k:
            proposals.append(self.reset_at)
        if self.count > self.d:
            proposals.append(self.k + self.d + 1)
        self.reset_at = min(proposals, default=None)
        if self.reset_at == self.k:
            self._reset()
        if self.round_timeout is not None:  # §III-A: drop a neighbour silent for > d rounds
            self._peers = [p for p in self._peers if self.k - self._heard[p] <= self.d]
        if self.k >= self.max_iter:
            report = ConsensusReport(carrier.get_address(), False, self.k, True)
            carrier.send_to_other(report, self._start.coordinator)
            self._stop(carrier)
            return
        self._send(carrier)

    def _reset(self) -> None:
        """Information sharing converged everywhere: P_N = x·n (eq. 10), restart x (Fig. 5);
        the first one starts the optimization at w⁰ = 0 (D-222)."""
        # 2026-11-23 D-229: the IS restarts and runs alongside the optimization (Fig. 5/10)
        self.target = self.x * len(self.seen)
        self.x = np.asarray(self.actor.measure(len(self.x)), dtype=float)
        self._phi_prev, self.count, self.reset_at = None, 0, None
        self.resets += 1
        if self.w is None:
            self.w = np.zeros((self.entries, len(self.x)))

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
        """Round k ends at t0 + (k+1)·round_timeout for everyone (as consensus)."""
        while not self.stopped:
            k = self.k
            end = self._start.t0 + (k + 1) * self.round_timeout
            await carrier.sleep(max(0.0, end - carrier.now()))
            if self.stopped or self.k != k:
                continue
            self.rounds_by_deadline += 1
            self.missing_edges += len(self._peers) - len(self._current(k))
            self._iterate(carrier)
            self._advance(carrier)

    def _stop(self, carrier: "Carrier") -> None:
        self.stopped = True
        self.finish_callback(self, carrier)
