"""Exact diffusion (Yuan, Ying, Zhao, Sayed 2019, Part I, Algorithm 1).

Each participant k holds an estimate w_k of a common vector and minimises Σ_k J_k(w) with its
neighbours only. Per round i (eqs. 98–100):

    ψ_i = w_{i−1} − μ∇J_k(w_{i−1})          (adaptation)
    φ_i = ψ_i + w_{i−1} − ψ_{i−1}           (correction; ψ_{−1} = w_{−1}, so φ_0 = ψ_0)
    w_i = Σ_ℓ ā_ℓk φ_ℓ,  Ā = (I + A)/2       (combination)

The correction removes plain diffusion's O(μ) bias: the iterates reach the exact minimiser for a
locally balanced A. A = Hastings rule with equal μ_k/q_k (Policy 1, eq. 18) = 1/max(n_k, n_ℓ),
n = |N| incl. the node (S8a's :func:`~..diffusion.diffusion.metropolis`), from the degrees in the
packets; it is doubly stochastic, so the Perron vector is uniform and one μ serves all (eq. 13).
No stop rule (the algorithm repeats for i = 0, 1, …): the run ends at ``max_iter``.
"""

# 2026-11-26 D-248: Yuan Alg. 1 (98)–(100); D-250: Hastings weights = S8a's; D-255: no stop rule

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ...core import DistributedAlgorithm, OptimizationMessage
from ..consensus.innovations import ConsensusReport, ConsensusStart
from ..diffusion.diffusion import DiffusionAlgorithm, metropolis

if TYPE_CHECKING:
    from ....carrier.core import Carrier


class ExactDiffusionActor(ABC):
    """Local problem J_k of one participant."""

    @abstractmethod
    def start(self) -> np.ndarray:
        """w_{k,−1} (arbitrary in the paper); its length sets the vector size."""

    @abstractmethod
    def gradient(self, w: np.ndarray) -> np.ndarray:
        """∇J_k at *w*."""


@dataclass
class ExactDiffusionMessage(OptimizationMessage):
    """φ of round *k* and the sender's degree (Hastings weights)."""

    phi: np.ndarray
    k: int
    sender: Any
    degree: int


class ExactDiffusionAlgorithm(DistributedAlgorithm):
    """One exact-diffusion participant.

    :param finish_callback: ``(algorithm, carrier)``, called once when the participant stops.
    :param actor: The local problem (:class:`ExactDiffusionActor`).
    :param step: μ (one for all participants, D-251).
    :param max_iter: Iteration cap = the end of the run.
    :param round_timeout: Round deadline (carrier clock) for lossy links, as diffusion: a missing
        packet's weight goes to self that round. ``None`` = wait for every neighbour.
    """

    def __init__(
        self,
        finish_callback: Callable,
        actor: ExactDiffusionActor,
        step: float,
        max_iter: int = 10_000,
        round_timeout: float | None = None,
    ) -> None:
        self.finish_callback, self.actor = finish_callback, actor
        self.step, self.max_iter, self.round_timeout = step, max_iter, round_timeout
        self.w: np.ndarray | None = None
        self._psi: np.ndarray | None = None
        self._phi: np.ndarray | None = None
        self.k = 0
        self.stopped = False
        self._start: ConsensusStart | None = None
        self._peers: list = []
        self._inbox: dict[int, list[ExactDiffusionMessage]] = defaultdict(list)
        self.rounds_by_deadline = 0
        self.missing_edges = 0

    async def on_exchange_message(self, carrier: "Carrier", message_data: Any, meta: Any) -> None:
        if self.stopped:
            return
        if isinstance(message_data, ConsensusStart):
            if self._start is not None:
                return
            self._start = message_data
            self._peers = [a for a in carrier.others("") if a in message_data.participants]
            self.w = np.asarray(self.actor.start(), dtype=float)
            self._psi = self.w  # ψ_{−1} = w_{−1}
            self._adapt_correct(carrier)
            if self.round_timeout is not None and self._peers:
                carrier.spawn(self._deadlines(carrier))
        elif isinstance(message_data, ExactDiffusionMessage):
            if message_data.k >= self.k:  # a packet of a finished (deadline) round is dropped
                self._inbox[message_data.k].append(message_data)
        elif isinstance(message_data, ConsensusReport):
            return  # a peer's report to a coordinator on this agent
        else:  # the coordinator's end of the run (OptimizationFinishedMessage)
            self._stop(carrier)
            return
        self._advance(carrier)

    def _adapt_correct(self, carrier: "Carrier") -> None:
        """Eqs. (98)/(99) at the current w, then send φ."""
        psi = self.w - self.step * np.asarray(self.actor.gradient(self.w))
        self._phi, self._psi = psi + self.w - self._psi, psi
        message = ExactDiffusionMessage(
            self._phi.copy(), self.k, carrier.get_address(), len(self._peers)
        )
        for peer in self._peers:
            carrier.send_to_other(message, peer)

    def _advance(self, carrier: "Carrier") -> None:
        """Iterate while the current round is complete (alone: every round at once)."""
        while (
            self._start is not None
            and not self.stopped
            and len(self._inbox[self.k]) == len(self._peers)
        ):
            self._iterate(carrier)

    def _iterate(self, carrier: "Carrier") -> None:
        received = self._inbox.pop(self.k, [])  # [] = every packet lost (deadline round)
        n_k = len(self._peers) + 1
        # eq. (100) with Ā = (I + A)/2: ā_ℓk = a_ℓk/2, ā_kk = 1 − Σ ā_ℓk = (1 + a_kk)/2
        a = [metropolis(n_k, m.degree + 1) / 2 for m in received]
        self.w = DiffusionAlgorithm._combine(self._phi, [m.phi for m in received], a)
        self.k += 1
        if self.k >= self.max_iter:
            report = ConsensusReport(carrier.get_address(), False, self.k, True)
            carrier.send_to_other(report, self._start.coordinator)
            self._stop(carrier)
            return
        self._adapt_correct(carrier)

    async def _deadlines(self, carrier: "Carrier") -> None:
        """Round k ends at t0 + (k+1)·round_timeout for everyone (as consensus)."""
        while not self.stopped:
            k = self.k
            end = self._start.t0 + (k + 1) * self.round_timeout
            await carrier.sleep(max(0.0, end - carrier.now()))
            if self.stopped or self.k != k:
                continue
            self.rounds_by_deadline += 1
            self.missing_edges += len(self._peers) - len(self._inbox[k])
            self._iterate(carrier)
            self._advance(carrier)

    def _stop(self, carrier: "Carrier") -> None:
        self.stopped = True
        self.finish_callback(self, carrier)
