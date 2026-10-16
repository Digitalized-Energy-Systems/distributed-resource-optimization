"""Exact Diffusion algorithm (adapt-correct-combine).

Classical Diffusion
(:mod:`~distributed_resource_optimization.algorithm.diffusion.diffusion`) converges, for a constant step size,
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
   evaluate via :mod:`~distributed_resource_optimization.algorithm._weight_rules` (``weight_rule``). The averaging with
   *I* is what keeps :math:`\\bar W` positive semi-definite -- required by
   Exact Diffusion's convergence proof -- since the raw rule matrices alone
   can have a negative eigenvalue (e.g. Mean Metropolis on a complete graph).

The per-agent feedback gain :math:`\\varepsilon_i = \\varepsilon / p_i` (eq. 31)
uses the Perron eigenvector entry :math:`p_i` of the weight matrix. Since the
weight matrix is fully determined by the (static, centrally-known) topology
at scenario-setup time, :math:`p_i` is expected to be computed once centrally
(the eigenvalue-1 eigenvector, normalised to sum to *N*) and passed in as *perron_scale*
-- mirroring how ``n_guess`` is already precomputed centrally for the
economic-dispatch actors. On a degree-regular graph :math:`p_i \\equiv 1`, so
*perron_scale* defaults to ``1.0`` (no scaling).

Reuses :class:`~distributed_resource_optimization.algorithm.diffusion.diffusion.DiffusionActor`
(same gradient-term interface) and
:class:`~distributed_resource_optimization.algorithm.diffusion.diffusion.DiffusionMessage` (same wire shape -- the ``phi`` field
just carries the corrected :math:`\\bar\\varphi` instead of the raw one).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .._weight_rules import WeightRule, regular_graph_weights
from .diffusion import DiffusionActor, _DiffusionRounds, create_diffusion_start

if TYPE_CHECKING:
    pass


class ExactDiffusionAlgorithm(_DiffusionRounds):
    """Distributed adapt-correct-combine exact diffusion over a scheduling horizon.

    The round handling (kick-off, termination, round deadline, last-value
    fallback) is shared with :class:`~.diffusion.DiffusionAlgorithm` via
    :class:`~.diffusion._DiffusionRounds`.  Under message loss the correction
    step no longer sees synchronous neighbour iterates, so the result is close
    to, but no longer exactly, the optimum.

    :param finish_callback: Called with ``(algorithm, carrier)`` when the run
                            terminates — via the ``tol`` criterion or the
                            ``max_iter`` failsafe (see :attr:`converged`).
    :param diffusion_actor: Optional
                            :class:`~distributed_resource_optimization.algorithm.diffusion.diffusion.DiffusionActor`
                            supplying the gradient.  ``None`` → the
                            zero-gradient base class.
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
    :param weight_rule: Combination-weight rule for the combine step; any of
                        ``"averaging"``, ``"relative_degree"``,
                        ``"mean_metropolis"``, ``"hastings"`` (left-stochastic
                        is sufficient -- unlike classical Diffusion, Exact
                        Diffusion does not require a doubly-stochastic matrix).
    :param round_timeout: Seconds (carrier clock) to wait for a round's
                          messages before advancing with the last received
                          iterates; ``None`` (default) waits for every
                          neighbour, which stalls if a message is lost.
    """

    _label = "Exact Diffusion"

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
        weight_rule: WeightRule = "mean_metropolis",
        round_timeout: float | None = None,
    ) -> None:
        self.finish_callback = finish_callback
        self.actor: DiffusionActor = (
            diffusion_actor if diffusion_actor is not None else DiffusionActor()
        )
        self.initial_lam = initial_lam
        self.epsilon = epsilon / perron_scale
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.weight_rule = weight_rule

        #: How the last run ended: ``True`` if the tol criterion fired,
        #: ``False`` if it hit the max_iter failsafe (or never ran).
        self.converged: bool = False
        #: Iterations completed when the last run terminated.
        self.iterations: int = 0

        self._init_rounds(round_timeout)
        # λ / φ state is sized from the kick-off message on (re-)initialisation.
        self._lam: np.ndarray = np.array([initial_lam])
        self._phi_prev: np.ndarray | None = None
        self._phi_bar: np.ndarray = np.array([initial_lam])

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

    def _start_state(self, horizon: int, data: Any) -> None:
        self._lam = np.ones(horizon) * self.initial_lam
        self._phi_prev = None
        self._phi_bar = self._adapt_and_correct(data)

    def _outgoing(self) -> np.ndarray:
        return self._phi_bar

    def _combine_weights(self, n_neighbours: int) -> tuple[float, float]:
        # W̄ = (I + W)/2 (eq. 31) rather than the raw rule weights -- this is
        # what makes the combine matrix satisfy the PSD condition Exact
        # Diffusion's convergence proof relies on (the raw rule matrices have
        # a negative eigenvalue on a complete graph).
        raw_self_w, raw_neighbor_w = regular_graph_weights(n_neighbours, self.weight_rule)
        return (1.0 + raw_self_w) / 2.0, raw_neighbor_w / 2.0

    def _adapt(self, data: Any) -> None:
        self._phi_bar = self._adapt_and_correct(data)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

#: Create an :class:`ExactDiffusionAlgorithm` participant (same signature).
create_exact_diffusion_participant = ExactDiffusionAlgorithm

#: Exact Diffusion reuses the classical-Diffusion kick-off message unchanged.
create_exact_diffusion_start = create_diffusion_start
