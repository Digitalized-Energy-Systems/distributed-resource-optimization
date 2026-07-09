"""Exact Diffusion algorithm (adapt-correct-combine).

Classical Diffusion (:mod:`.diffusion`) converges, for a constant step size,
to a point that is *biased* relative to the true minimiser of the network's
aggregate cost whenever participants' local costs differ -- the combine step
mixes already-adapted iterates, so the network never quite agrees on the
gradient direction it collectively descends along. Exact Diffusion (Ces
et al. 2025, Sec. 3.4, eqs. 28-30; underlying theory: Yuan, Ling & Sayed,
"Exact Diffusion for Distributed Optimization and Learning", 2018) removes
that bias by inserting a **correction** stage between adapt and combine that
cancels the first-order bias term, at the cost of one extra state variable
per participant (the previous iteration's uncorrected φ).

Per iteration, each participant:

1. **adapts**: :math:`\\varphi_i^k = \\lambda_i^{k-1} - \\varepsilon_i \\nabla J(\\lambda_i^{k-1}, \\text{data})`
   (eq. 28),
2. **corrects**: :math:`\\bar\\varphi_i^k = \\varphi_i^k + \\lambda_i^{k-1} - \\varphi_i^{k-1}`
   (eq. 30) -- skipped on the very first iteration, where
   :math:`\\bar\\varphi_i^0 := \\varphi_i^0` (no :math:`\\varphi_i^{-1}` exists yet),
3. broadcasts :math:`\\bar\\varphi_i^k` to its neighbours, and
4. **combines**: :math:`\\lambda_i^k = \\sum_{j \\in \\mathcal{N}_i} \\bar w_{ij} \\bar\\varphi_j^k`
   (eq. 29), using the left-stochastic matrix :math:`\\bar W = (I + W)/2`
   (eq. 31), where *W* is built from one of the four rules Ces et al.
   evaluate via :mod:`.._weight_rules` (``weight_rule``). The averaging with
   *I* is what keeps :math:`\\bar W` positive semi-definite -- required by
   Exact Diffusion's convergence proof -- since the raw rule matrices alone
   can have a negative eigenvalue (e.g. Mean Metropolis on a complete graph).

The per-agent feedback gain :math:`\\varepsilon_i = \\varepsilon / p_i` (eq. 31)
uses the Perron eigenvector entry :math:`p_i` of the weight matrix. Since the
weight matrix is fully determined by the (static, centrally-known) topology
at scenario-setup time, :math:`p_i` is expected to be computed once centrally
(see :func:`~.._weight_rules.perron_vector`) and passed in as *perron_scale*
-- mirroring how ``n_guess`` is already precomputed centrally for the
economic-dispatch actors. On a degree-regular graph :math:`p_i \\equiv 1`, so
*perron_scale* defaults to ``1.0`` (no scaling).

Reuses :class:`~.diffusion.DiffusionActor` (same gradient-term interface) and
:class:`~.diffusion.DiffusionMessage` (same wire shape -- the ``phi`` field
just carries the corrected :math:`\\bar\\varphi` instead of the raw one).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .._weight_rules import WeightRule, regular_graph_weights
from ..core import DistributedAlgorithm
from .diffusion import DiffusionActor, DiffusionMessage, NoDiffusionActor

if TYPE_CHECKING:
    from ...carrier.core import Carrier


class ExactDiffusionAlgorithm(DistributedAlgorithm):
    """Distributed adapt-correct-combine exact diffusion over a scheduling horizon.

    ``on_exchange_message`` shares its termination/kick-off/message-queueing
    skeleton with :class:`~.diffusion.DiffusionAlgorithm` -- a fix to one
    (e.g. the stale-message-after-termination guard below) likely needs to
    be applied to the other too.

    :param finish_callback: Called with ``(algorithm, carrier)`` when
                            :attr:`max_iter` is reached.
    :param diffusion_actor: Optional :class:`DiffusionActor` supplying the
                            gradient.  ``None`` → :class:`NoDiffusionActor`.
    :param initial_lam: Starting scalar (broadcast to all λ dimensions).
    :param epsilon: Base gradient step size (ε in eq. 31).
    :param perron_scale: This agent's Perron-eigenvector entry :math:`p_i`
                         (eq. 31); ``epsilon_i = epsilon / perron_scale``.
                         Defaults to ``1.0`` (valid on a degree-regular graph).
    :param max_iter: Maximum number of diffusion iterations (failsafe; the
                     ``tol`` criterion normally terminates the run first).
    :param tol: Convergence tolerance — a participant flags itself converged
                once ``max|λ_k − λ_{k−1}| ≤ tol`` has held for ``patience``
                consecutive rounds, and the run terminates in the first round
                where *every* participant is flagged (Ces et al. 2025 use
                1e-4 on the incremental cost).
    :param patience: Consecutive sub-``tol`` rounds required before a
                     participant flags itself converged.  Guards against the
                     turning points of oscillatory transients (the corrected
                     second-order recursion has complex eigenmodes), where λ
                     momentarily stops moving for a single round.
    :param horizon: Number of time steps in the schedule.
    :param weight_rule: Combination-weight rule for the combine step; any of
                        ``"averaging"``, ``"relative_degree"``,
                        ``"mean_metropolis"``, ``"hastings"`` (left-stochastic
                        is sufficient -- unlike classical Diffusion, Exact
                        Diffusion does not require a doubly-stochastic matrix).
    """

    def __init__(
        self,
        finish_callback: Callable,
        diffusion_actor: DiffusionActor | None = None,
        initial_lam: float = 10.0,
        epsilon: float = 0.1,
        perron_scale: float = 1.0,
        max_iter: int = 300,
        tol: float = 1e-4,
        patience: int = 3,
        horizon: int = 24,
        weight_rule: WeightRule = "mean_metropolis",
    ) -> None:
        self.finish_callback = finish_callback
        self.actor: DiffusionActor = (
            diffusion_actor if diffusion_actor is not None else NoDiffusionActor()
        )
        self.initial_lam = initial_lam
        self.epsilon = epsilon / perron_scale
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.horizon = horizon
        self.weight_rule = weight_rule

        #: How the last run ended: ``True`` if the tol criterion fired,
        #: ``False`` if it hit the max_iter failsafe (or never ran).
        self.converged: bool = False
        #: Iterations completed when the last run terminated.
        self.iterations: int = 0

        self._message_queue: dict[int, list[DiffusionMessage]] = {}
        self._first_message: bool = True
        self._started: bool = False  # True once any round has begun
        self._k: int = 0
        self._lam: np.ndarray = np.full(horizon, initial_lam)
        self._phi_prev: np.ndarray | None = None
        self._phi_bar: np.ndarray = np.full(horizon, initial_lam)
        self._converged_flag: bool = False  # own flag from the last combine
        self._stable_rounds: int = 0  # consecutive rounds with λ change ≤ tol

    def _adapt_and_correct(self, data: Any) -> np.ndarray:
        """Adapt (eq. 28) then correct (eq. 30); advances :attr:`_phi_prev`."""
        grad_J = self.actor.gradient_term(self._lam, data)
        phi = self._lam - self.epsilon * np.asarray(grad_J)
        if self._phi_prev is None:
            phi_bar = phi.copy()
        else:
            phi_bar = phi + self._lam - self._phi_prev
        self._phi_prev = phi
        return phi_bar

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: DiffusionMessage,
        meta: Any,
    ) -> None:
        """Process one incoming exact-diffusion message."""
        neighbours = carrier.others("")

        # --- Termination path (max_iter failsafe) ---
        if message_data.k >= self.max_iter:
            if self._first_message:
                return
            self.converged = False
            self.iterations = self._k
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
            self._converged_flag = False
            self._stable_rounds = 0
            self.converged = False
            self._lam = np.ones(len(message_data.phi)) * self.initial_lam
            self._phi_prev = None
            self._phi_bar = self._adapt_and_correct(message_data.data)

            phi_out = self._phi_bar.copy()
            for addr in neighbours:
                carrier.send_to_other(
                    DiffusionMessage(
                        phi=phi_out,
                        k=0,
                        data=message_data.data,
                        degree=len(neighbours),
                        weight_rule=self.weight_rule,
                    ),
                    addr,
                )

            if message_data.initial:
                return  # kick-off is not a topology neighbour; do not queue

        # --- Queue the message ---
        queue = self._message_queue.setdefault(message_data.k, [])
        queue.append(message_data)

        # --- Advance if all neighbours have reported for this iteration ---
        if len(queue) == len(neighbours):
            # regular_graph_weights() assumes every neighbour has the same
            # degree and the same weight_rule as this node -- verify that
            # rather than silently computing a wrong (non-stochastic, or
            # inconsistent-across-nodes) weight matrix if it doesn't.
            own_degree = len(neighbours)
            for m in queue:
                if m.degree != own_degree:
                    raise ValueError(
                        f"Exact Diffusion requires a degree-regular communication graph: "
                        f"this node has {own_degree} neighbours but received a message "
                        f"from a node with {m.degree} neighbours."
                    )
                if m.weight_rule != self.weight_rule:
                    raise ValueError(
                        f"All Exact Diffusion participants must share the same weight_rule: "
                        f"this node uses {self.weight_rule!r} but received a message "
                        f"from a node using {m.weight_rule!r}."
                    )

            # --- Termination path (tol criterion) ---
            # Own flag and every message in this round's queue reflect the
            # λ change of round k−1.  On the complete graphs this codebase
            # uses, every participant therefore evaluates the same set of
            # flags in the same round and terminates simultaneously.
            if self._converged_flag and all(m.converged for m in queue):
                self.converged = True
                self.iterations = self._k
                self.finish_callback(self, carrier)
                self._first_message = True
                self._message_queue.clear()
                return

            # Combination: weighted average of own φ̄ and all received φ̄'s,
            # using W̄ = (I + W)/2 (eq. 31) rather than the raw rule weights —
            # this is what makes the combine matrix satisfy the PSD condition
            # Exact Diffusion's convergence proof relies on (the raw rule
            # matrices have a negative eigenvalue on a complete graph).
            raw_self_w, raw_neighbor_w = regular_graph_weights(len(neighbours), self.weight_rule)
            self_w = (1.0 + raw_self_w) / 2.0
            neighbor_w = raw_neighbor_w / 2.0
            lam_new = self_w * self._phi_bar
            for m in queue:
                lam_new = lam_new + neighbor_w * m.phi
            if np.max(np.abs(lam_new - self._lam)) <= self.tol:
                self._stable_rounds += 1
            else:
                self._stable_rounds = 0
            self._converged_flag = self._stable_rounds >= self.patience
            self._lam = lam_new

            del self._message_queue[message_data.k]

            # Adapt + correct
            self._phi_bar = self._adapt_and_correct(message_data.data)

            self._k += 1

            phi_out = self._phi_bar.copy()
            for addr in neighbours:
                carrier.send_to_other(
                    DiffusionMessage(
                        phi=phi_out,
                        k=self._k,
                        data=message_data.data,
                        degree=len(neighbours),
                        weight_rule=self.weight_rule,
                        converged=self._converged_flag,
                    ),
                    addr,
                )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_exact_diffusion_participant(
    finish_callback: Callable,
    diffusion_actor: DiffusionActor | None = None,
    initial_lam: float = 10.0,
    epsilon: float = 0.1,
    perron_scale: float = 1.0,
    max_iter: int = 300,
    tol: float = 1e-4,
    patience: int = 3,
    horizon: int = 24,
    weight_rule: WeightRule = "mean_metropolis",
) -> ExactDiffusionAlgorithm:
    """Create an :class:`ExactDiffusionAlgorithm` participant.

    :param finish_callback: ``(algorithm, carrier) -> None`` — called when done.
    :param diffusion_actor: Optional gradient actor.  ``None`` → no gradient.
    :param initial_lam: Initial λ scalar.
    :param epsilon: Base gradient step size.
    :param perron_scale: This agent's Perron-eigenvector entry (eq. 31).
    :param max_iter: Maximum iterations (failsafe; ``tol`` normally stops first).
    :param tol: Per-round λ-change convergence tolerance.
    :param patience: Consecutive sub-``tol`` rounds required to flag convergence.
    :param horizon: Number of schedule time steps.
    :param weight_rule: Combination-weight rule for the combine step.
    """
    return ExactDiffusionAlgorithm(
        finish_callback=finish_callback,
        diffusion_actor=diffusion_actor,
        initial_lam=initial_lam,
        epsilon=epsilon,
        perron_scale=perron_scale,
        max_iter=max_iter,
        tol=tol,
        patience=patience,
        horizon=horizon,
        weight_rule=weight_rule,
    )


def create_exact_diffusion_start(
    initial_lam: float,
    data: Any = None,
    horizon: int = 1,
) -> DiffusionMessage:
    """Create the initial kick-off message for an exact-diffusion run.

    :param initial_lam: Starting scalar; broadcast to all λ dimensions.
    :param data: Auxiliary payload forwarded to each participant's
                 :meth:`DiffusionActor.gradient_term`.
    :param horizon: Number of time steps; sets the length of the φ vector that
                    recipients use to infer their own horizon.
    :returns: A :class:`DiffusionMessage` with ``initial=True``.
    """
    return DiffusionMessage(
        phi=np.full(horizon, initial_lam),
        k=0,
        data=data,
        initial=True,
    )
