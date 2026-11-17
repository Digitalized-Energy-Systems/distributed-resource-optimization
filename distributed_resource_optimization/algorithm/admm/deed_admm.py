"""DEED-ADMM (Zhu et al. 2025, Algorithm 1): peer ADMM with a distributed gradient tracker.

n participants minimise Σ_i f_i(x_i, y_i) subject to Σ_i (x_i − y_i) = 0. Participant i holds a
local copy λ_i of the multiplier and a tracker ξ_i of the mean coupling (x − y); both live in ℝ^T.
Per iteration k (line numbers of Algorithm 1):

.. math::

    \\tilde\\lambda_i = \\sum_j w_{ij} \\lambda_j,\\quad
    \\tilde\\xi_i = \\sum_j w_{ij} \\xi_j                                   \\quad (3, 4)

    \\Delta_i = \\text{local update}(\\tilde\\lambda_i, \\tilde\\xi_i)          \\quad (5\\text{–}11, 14)

    \\xi_i \\leftarrow \\tilde\\xi_i + \\Delta_i,\\quad
    \\lambda_i \\leftarrow \\tilde\\lambda_i + \\tfrac{\\gamma}{2} \\xi_i         \\quad (12, 13)

with W = I − βL of the topology (w_ij = β for a neighbour, w_ii = 1 − β·deg_i), so
λ̃_i = λ_i + β Σ_j (λ_j − λ_i). The local update (lines 5–11, 14) is the participant's
(:class:`DEEDParticipant`); it returns Δ_i = (x_i,k+1 − x_i,k) − (y_i,k+1 − y_i,k).

Rounds are synchronous: iteration k waits for every peer's (λ_j, ξ_j) of iteration k — one
message per edge per iteration. Lossy links: with ``round_timeout`` a round ends at a deadline on
the shared carrier clock; a missing (λ_j, ξ_j) drops edge (i, j) for that round, its weight goes to
self (the sum above runs over the arrived peers).

Stop: a participant is *converged* when its primal residual is small,
‖ξ_i‖₂ ≤ √T·eps_abs + eps_rel·‖x_i‖₂ (Thm 2 iii: ξ → 0), and its participant reports its dual
check converged (:meth:`DEEDParticipant.converged`). Flag flips go to the coordinator
(:class:`ConsensusCoordinator`, reused as is); its stop and ``max_iter`` end the run as for
consensus + innovations.
"""

# 2026-11-17 D-166 (Q-Z8 C): own module, the round logic copied from consensus/innovations.py —
# no shared base; consensus is unchanged.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..core import DistributedAlgorithm, OptimizationMessage
from ..firstorder.consensus.innovations import ConsensusReport, ConsensusStart

if TYPE_CHECKING:
    from ...carrier.core import Carrier


class DEEDParticipant:
    """Local problem of one participant: Algorithm 1 lines 5–11 and 14."""

    def coupling(self) -> np.ndarray:
        """x_i − y_i at the initial point (line 1: ξ_i,0)."""
        raise NotImplementedError

    def update(self, lam_tilde: np.ndarray, xi_tilde: np.ndarray) -> np.ndarray:
        """Lines 5–11 and 14 for the mixed (λ̃_i, ξ̃_i); returns Δ(x_i − y_i) of this step."""
        raise NotImplementedError

    def x_norm(self) -> float:
        """‖x_i‖₂, the scale of the primal tolerance."""
        raise NotImplementedError

    # 2026-11-17 D-175 (Q-Z9): the dual check (p̂, ν) stays with the participant, the primal one
    # (ξ) is the algorithm's.
    def converged(self, eps_abs: float, eps_rel: float) -> bool:
        """Dual check of the last update (Thm 2 i), e.g. γ‖p̂ᵏ⁺¹ − p̂ᵏ‖ ≤ √dim·eps_abs + eps_rel·‖ν‖."""
        return True


@dataclass
class DEEDMessage(OptimizationMessage):
    """(λ, ξ) of the sender after *k* iterations — one message per edge per iteration (Q-Z5 A)."""

    lam: np.ndarray
    xi: np.ndarray
    k: int


class DEEDADMMAlgorithm(DistributedAlgorithm):
    """One DEED-ADMM participant; started and stopped by a :class:`ConsensusCoordinator`
    (its λ0 is the initial λ_i,0).

    :param finish_callback: ``(algorithm, carrier)``, called once when the participant stops.
    :param participant: The local problem (:class:`DEEDParticipant`).
    :param beta: Edge weight of W = I − βL (Assumption 1 needs β·λ_max(L) < 2).
    :param gamma: ADMM penalty γ (line 13; the participant uses the same γ).
    :param eps_abs: Absolute tolerance of the stop rule.
    :param eps_rel: Relative tolerance of the stop rule.
    :param max_iter: Iteration cap.
    :param round_timeout: Round deadline (carrier clock) for lossy links; ``None`` (default) =
        wait for every peer. ``rounds_by_deadline`` / ``missing_edges`` count how often and how
        many were missing.
    """

    def __init__(
        self,
        finish_callback: Callable,
        participant: DEEDParticipant,
        beta: float,
        gamma: float,
        eps_abs: float = 1e-4,
        eps_rel: float = 1e-3,
        max_iter: int = 10_000,
        round_timeout: float | None = None,
    ) -> None:
        self.finish_callback = finish_callback
        self.participant = participant
        self.beta, self.gamma = beta, gamma
        self.eps_abs, self.eps_rel, self.max_iter = eps_abs, eps_rel, max_iter
        self.lam: np.ndarray | None = None
        self.xi: np.ndarray | None = None
        self.k = 0
        self.stopped = False
        self._start: ConsensusStart | None = None
        self._peers: list = []
        self._inbox: dict[int, list[DEEDMessage]] = defaultdict(list)
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
            self.xi = np.array(self.participant.coupling(), dtype=float)  # line 1
            self._peers = [a for a in carrier.others("") if a in message_data.participants]
            self._send(carrier)
            if self.round_timeout is not None and self._peers:
                carrier.spawn(self._deadlines(carrier))
        elif isinstance(message_data, DEEDMessage):
            if message_data.k >= self.k:  # a message of a finished (deadline) round is dropped
                self._inbox[message_data.k].append(message_data)
        elif isinstance(message_data, ConsensusReport):
            return  # a peer's report to a coordinator hosted on this agent: not for the participant
        else:  # the coordinator's end of the run (OptimizationFinishedMessage)
            self._stop(carrier)
            return
        self._advance(carrier)

    def _advance(self, carrier: "Carrier") -> None:
        """Iterate while the current round is complete (every peer's message is in)."""
        while (
            self._start is not None
            and not self.stopped
            and len(self._inbox[self.k]) == len(self._peers)
        ):
            self._iterate(carrier)

    def _iterate(self, carrier: "Carrier") -> None:
        arrived = self._inbox.pop(self.k, [])  # [] = every peer's message lost (deadline round)
        # lines 3–4, W = I − βL; a missing edge's weight stays with self (Q-Z6 A)
        lam_t = self.lam + self.beta * sum(
            (m.lam - self.lam for m in arrived), np.zeros_like(self.lam)
        )
        xi_t = self.xi + self.beta * sum((m.xi - self.xi for m in arrived), np.zeros_like(self.xi))
        self.xi = xi_t + self.participant.update(lam_t, xi_t)  # lines 5–12
        self.lam = lam_t + self.gamma / 2 * self.xi  # line 13
        self.k += 1
        primal = np.linalg.norm(self.xi) <= (
            np.sqrt(self.xi.size) * self.eps_abs + self.eps_rel * self.participant.x_norm()
        )
        # a plain bool: numpy.bool cannot be encoded by the message codecs
        converged = bool(primal and self.participant.converged(self.eps_abs, self.eps_rel))
        if self.k >= self.max_iter:
            self._report(carrier, converged, done=True)
            self._stop(carrier)
            return
        if converged != self._converged:
            self._converged = converged
            self._report(carrier, converged)
            if converged and not self._peers:
                # alone: its own check is the global one, and the coordinator's stop could not
                # arrive while this loop never yields
                self._stop(carrier)
                return
        self._send(carrier)

    def _send(self, carrier: "Carrier") -> None:
        for peer in self._peers:
            carrier.send_to_other(DEEDMessage(self.lam.copy(), self.xi.copy(), self.k), peer)

    async def _deadlines(self, carrier: "Carrier") -> None:
        """One timer per participant: round k ends at t0 + (k+1)·round_timeout for everyone; if
        the current round is still open then, iterate with the peers that arrived."""
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
