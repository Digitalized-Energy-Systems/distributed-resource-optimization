"""Diffusion, adapt-then-combine (Ces et al. 2025 §3.3; Yuan et al. 2019 Part I §II-A).

Every participant i holds a vector λ_i. Per iteration k:

.. math::

    \\varphi_i = \\lambda_i - \\varepsilon_i \\nabla J_i(\\lambda_i), \\qquad
    \\lambda_i \\leftarrow w_{ii} \\varphi_i + \\sum_{j \\in \\mathcal{N}_i} w_{ij} \\varphi_j

(adapt: Ces eq. 26 / Yuan eq. 6; combine over the peers' φ of the same round: Ces eq. 27 / Yuan
eq. 7), with the peer weights w_ij given and w_ii = 1 − Σ_j w_ij. ∇J_i and the start λ_i⁰ come from
a :class:`DiffusionActor`. Rounds are synchronous, as in consensus + innovations.

Stop (Ces §4): a participant is *converged* when ``max |Δλ| ≤ eps``; it reports flips to the
coordinator (:class:`DiffusionCoordinator`, the consensus protocol incl. ``settle_s``). Passes (Ces
§4, Fig. 1): a converged participant also reports whether its actor has new *flags* (e.g. a limit
violated); when all are converged and any has flags, the coordinator starts a new pass — every
actor's :meth:`DiffusionActor.restart`, then λ = :meth:`DiffusionActor.start` again — else it ends
the run. ``k`` counts the iterations of all passes; ``max_iter`` ends the run.
"""

# 2026-11-23 D-211: replaces the upstream DiffusionAlgorithm (plain mean, no stop) — S8a plan §4

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ...core import OptimizationMessage
from ..consensus.innovations import (
    ConsensusCoordinator,
    ConsensusReport,
    ConsensusStart,
)

if TYPE_CHECKING:
    from ....carrier.core import Carrier


class DiffusionActor:
    """Local problem of one participant: its gradient, start point and flags."""

    def gradient(self, lam: np.ndarray) -> np.ndarray:
        """∇J_i at *lam*; summed over all participants it is zero at the solution."""
        return np.zeros_like(lam)

    def start(self, lam0: np.ndarray) -> np.ndarray:
        """λ_i⁰ of a pass (default: the coordinator's *lam0*)."""
        return lam0

    def flags(self) -> bool:
        """True when, at the current λ, the actor would change its problem for a new pass."""
        return False

    def restart(self) -> None:
        """Apply the flags before a new pass (e.g. pin units at a violated limit)."""


@dataclass
class DiffusionStart(ConsensusStart):
    """Kick-off of pass *pass_no* (0 = the first)."""

    pass_no: int = 0


@dataclass
class DiffusionMessage(OptimizationMessage):
    """φ of *sender* in round *k* of pass *pass_no*; *iterations* = the sender's iterations of all
    passes so far."""

    phi: np.ndarray
    k: int
    sender: Any
    pass_no: int = 0
    iterations: int = 0


@dataclass
class DiffusionReport(ConsensusReport):
    """A consensus report plus the actor's flags and the pass it belongs to."""

    flags: bool = False
    pass_no: int = 0


class DiffusionAlgorithm:
    """One diffusion participant.

    :param finish_callback: ``(algorithm, carrier)``, called once when the participant stops.
    :param actor: The local problem (:class:`DiffusionActor`).
    :param step: ε_i, the adapt step.
    :param weights: w_ij per peer address (topology neighbours that take part); w_ii = 1 − Σ.
    :param eps: Tolerance on ``max |Δλ|`` for the converged flag ("≤", Ces §4).
    :param max_iter: Iteration cap over all passes.
    :param round_timeout: Round deadline (carrier clock) for lossy links, as in consensus: the
        round ends with the φ that arrived; a missing φ_j's weight goes to self. ``None`` = wait.
    """

    def __init__(
        self,
        finish_callback: Callable,
        actor: DiffusionActor,
        step: float,
        weights: dict,
        eps: float = 1e-4,
        max_iter: int = 10_000,
        round_timeout: float | None = None,
    ) -> None:
        self.finish_callback, self.actor = finish_callback, actor
        self.step, self.weights = step, weights
        self.eps, self.max_iter, self.round_timeout = eps, max_iter, round_timeout
        self.lam: np.ndarray | None = None
        self.phi: np.ndarray | None = None
        self.k = 0  # iterations of all passes
        self.r = 0  # round within the pass
        self.pass_no = -1
        self.stopped = False
        self._start: DiffusionStart | None = None
        self._peers: list = []
        self._inbox: dict[tuple[int, int], list[DiffusionMessage]] = defaultdict(list)
        self._state = (False, False)  # last reported (converged, flags)
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
        if self._start is not None:
            self.actor.restart()
        self._start, self.pass_no, self.r = start, start.pass_no, 0
        self._peers = [a for a in carrier.others("") if a in start.participants]
        self._state = (False, False)
        # φ⁰ = λ⁰ (Ces §3.3): iteration 1 adapts at λ⁰ and combines
        self.lam = np.array(self.actor.start(np.asarray(start.lam, dtype=float)), dtype=float)
        self._adapt_and_send(carrier)

    def _adapt_and_send(self, carrier: "Carrier") -> None:
        self.phi = self.lam - self.step * np.asarray(self.actor.gradient(self.lam))
        message = DiffusionMessage(
            self.phi.copy(), self.r, carrier.get_address(), self.pass_no, self.k
        )
        for peer in self._peers:
            carrier.send_to_other(message, peer)

    def _advance(self, carrier: "Carrier") -> None:
        """Iterate while the current round is complete; alone, iterate until a report."""
        while (
            self._start is not None
            and not self.stopped
            and len(self._inbox[self.pass_no, self.r]) == len(self._peers)
        ):
            if self._iterate(carrier) and not self._peers:
                # shortcut: alone with final_stop=False and no flags nobody answers and the run
                # stalls; only a one-bus network sweep hits it — then iterate on here
                return  # alone: a report was sent; wait for the coordinator's answer

    def _iterate(self, carrier: "Carrier") -> bool:
        """One combine step; returns True if it sent a report."""
        received = self._inbox.pop((self.pass_no, self.r), [])
        w = [self.weights[m.sender] for m in received]
        # weights of missing φ (deadline rounds) go to self (S8a plan §4, as DEED-ADMM Z8)
        new = (1.0 - sum(w)) * self.phi + sum(
            (wj * m.phi for wj, m in zip(w, received)), np.zeros_like(self.phi)
        )
        converged = bool(np.max(np.abs(new - self.lam)) <= self.eps)
        # a new pass reaches the participants at different times; max keeps k equal (max_iter)
        self.k = max([self.k, *(m.iterations for m in received)]) + 1
        self.lam, self.r = new, self.r + 1
        if self.k >= self.max_iter:
            self._report(carrier, converged, False, done=True)
            self._stop(carrier)
            return True
        state = (converged, converged and bool(self.actor.flags()))
        reported = state != self._state
        if reported:
            self._state = state
            self._report(carrier, *state)
        self._adapt_and_send(carrier)
        return reported

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

    def _report(self, carrier: "Carrier", converged: bool, flags: bool, done=False) -> None:
        report = DiffusionReport(
            carrier.get_address(), converged, self.k, done, flags, self.pass_no
        )
        carrier.send_to_other(report, self._start.coordinator)

    def _stop(self, carrier: "Carrier") -> None:
        self.stopped = True
        self.finish_callback(self, carrier)


class DiffusionCoordinator(ConsensusCoordinator):
    """The consensus coordinator plus passes: when all participants are converged (after
    ``settle_s``) and any reported flags, it starts the next pass instead of stopping.

    :param final_stop: False = never stop on convergence (the run ends at ``max_iter``); pass ends
        and their flag checks still happen (S8a plan, D-214).
    """

    def __init__(self, *args, final_stop: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.final_stop = final_stop
        self.pass_no = 0
        self._flagged: dict = {}

    def _start_message(self, carrier: "Carrier", t0: float) -> DiffusionStart:
        return DiffusionStart(self.lam0, self.participants, carrier.get_address(), t0, self.pass_no)

    async def on_exchange_message(self, carrier: "Carrier", message_data: Any, meta: Any) -> None:
        if isinstance(message_data, DiffusionReport):
            if message_data.pass_no != self.pass_no and not message_data.done:
                return  # a late report of an earlier pass
            self._flagged[message_data.sender] = message_data.flags
        await super().on_exchange_message(carrier, message_data, meta)

    def _converged(self, carrier: "Carrier") -> None:
        # 2026-11-23 D-212: new flags at a pass end → next pass (Ces §4, Fig. 1)
        if any(self._flagged.get(p, False) for p in self.participants):
            self.pass_no += 1
            self._flags.clear()
            self._flagged.clear()
            start = self._start_message(carrier, carrier.now() + self.start_latency)
            for p in self.participants:
                carrier.send_to_other(start, p)
        elif self.final_stop:
            super()._converged(carrier)
