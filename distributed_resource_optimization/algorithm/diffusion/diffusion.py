"""Diffusion algorithm (adapt-then-combine).

Each participant maintains a local price estimate λ and a power iterate φ over a
scheduling horizon.  At every iteration a participant

1. **adapts** its power iterate via a local gradient step
   ``φ = λ - ε · ∇J(λ, data)``,
2. broadcasts ``φ`` to its neighbours, and
3. **combines** its own φ with all received φ's using the Mean-Metropolis
   weight matrix (Ces et al. 2025, eq. 19 -- the same doubly-stochastic
   matrix used for consensus) to form the next λ.

The update rule is:

.. math::

    \\lambda_i^{k+1} = w_{ii} \\varphi_i^k +
                    \\sum_{j \\in \\mathcal{N}_i} w_{ij} \\, \\varphi_j^k,
    \\qquad
    \\varphi_i^{k+1} = \\lambda_i^{k+1} - \\varepsilon \\, \\nabla J(\\lambda_i^{k+1}, \\text{data}),

with :math:`w_{ij} = 2 / (n_i + n_j + 1)` and :math:`w_{ii} = 1 - \\sum_j w_{ij}`.
Weights are computed via :func:`~..._weight_rules.regular_graph_weights`,
which assumes a degree-regular communication graph (true both for this
codebase's complete-graph topologies and for the paper's own ring network);
the paper notes this doubly-stochastic requirement is what allows classical
Diffusion to converge at all.

The optional :class:`DiffusionActor` plug-in supplies ``∇J``; the default
:class:`NoDiffusionActor` returns zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .._weight_rules import WeightRule, regular_graph_weights
from ..core import DistributedAlgorithm, OptimizationMessage

if TYPE_CHECKING:
    from ...carrier.core import Carrier


# ---------------------------------------------------------------------------
# DiffusionActor hierarchy
# ---------------------------------------------------------------------------


class DiffusionActor:
    """Optional plug-in that supplies the gradient term for the adapt step.

    Subclass this to bias the diffusion iterates toward a local optimum
    (e.g. economic dispatch or battery storage scheduling).
    """

    def gradient_term(self, lam: np.ndarray, data: Any) -> np.ndarray | float:
        """Return the gradient ``∇J(λ, data)`` for the current iterate *lam*.

        :param lam: Current local price/λ estimate over the schedule.
        :param data: Auxiliary data forwarded from the start message.
        :returns: Additive gradient (default: 0).
        """
        return 0


class NoDiffusionActor(DiffusionActor):
    """Neutral diffusion actor — gradient is identically zero."""


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class DiffusionMessage(OptimizationMessage):
    """Message exchanged between diffusion participants.

    :param phi: Current power iterate φ of the sender.
    :param k: Current iteration counter.
    :param data: Auxiliary payload forwarded to :meth:`DiffusionActor.gradient_term`.
    :param initial: If ``True`` this is the kick-off message; recipients
                    (re-)initialise their state.
    :param degree: Sender's own neighbour count.  Used only to verify the
                   degree-regular-graph assumption :func:`regular_graph_weights`
                   relies on -- ignored on the ``initial`` kick-off message.
    :param weight_rule: Sender's own combination-weight rule.  Used only to
                        verify every participant agrees on the same rule --
                        ignored on the ``initial`` kick-off message.
    :param converged: Whether the sender's λ has changed by at most ``tol``
                      for ``patience`` consecutive combine steps.  A round in
                      which every participant's flag is ``True`` terminates
                      the run (Ces et al. 2025 stop when the largest per-agent
                      incremental-cost change falls below their tolerance).
    """

    phi: np.ndarray
    k: int
    data: Any
    initial: bool = False
    degree: int = 0
    weight_rule: str = ""
    converged: bool = False


# ---------------------------------------------------------------------------
# DiffusionAlgorithm
# ---------------------------------------------------------------------------


class DiffusionAlgorithm(DistributedAlgorithm):
    """Distributed adapt-then-combine diffusion over a scheduling horizon.

    ``on_exchange_message`` shares its termination/kick-off/message-queueing
    skeleton with :class:`~.exact_diffusion.ExactDiffusionAlgorithm` -- a fix
    to one (e.g. the stale-message-after-termination guard below) likely
    needs to be applied to the other too.

    :param finish_callback: Called with ``(algorithm, carrier)`` when
                            :attr:`max_iter` is reached.
    :param diffusion_actor: Optional :class:`DiffusionActor` supplying the
                            gradient.  ``None`` → :class:`NoDiffusionActor`.
    :param initial_lam: Starting scalar (broadcast to all λ dimensions).
    :param epsilon: Gradient step size (ε).
    :param max_iter: Maximum number of diffusion iterations (failsafe; the
                     ``tol`` criterion normally terminates the run first).
    :param tol: Convergence tolerance — a participant flags itself converged
                once ``max|λ_k − λ_{k−1}| ≤ tol`` has held for ``patience``
                consecutive rounds, and the run terminates in the first round
                where *every* participant is flagged (Ces et al. 2025 use
                1e-4 on the incremental cost).
    :param patience: Consecutive sub-``tol`` rounds required before a
                     participant flags itself converged.  Guards against the
                     turning points of oscillatory transients, where λ
                     momentarily stops moving for a single round.
    :param horizon: Number of time steps in the schedule.
    :param weight_rule: Combination-weight rule for the combine step.  The
                        paper requires a doubly-stochastic matrix for
                        classical Diffusion to converge.  ``"mean_metropolis"``
                        (the default) and ``"hastings"`` are doubly stochastic
                        on any graph; on the degree-regular graphs this
                        codebase always uses, ``"averaging"`` and
                        ``"relative_degree"`` collapse to the same uniform
                        doubly-stochastic matrix too (see :mod:`.._weight_rules`).
    """

    def __init__(
        self,
        finish_callback: Callable,
        diffusion_actor: DiffusionActor | None = None,
        initial_lam: float = 10.0,
        epsilon: float = 0.1,
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
        self.epsilon = epsilon
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
        self._phi: np.ndarray = np.full(horizon, initial_lam)
        self._converged_flag: bool = False  # own flag from the last combine
        self._stable_rounds: int = 0  # consecutive rounds with λ change ≤ tol

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: DiffusionMessage,
        meta: Any,
    ) -> None:
        """Process one incoming diffusion message."""
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

            grad_J = self.actor.gradient_term(self._lam, message_data.data)
            self._phi = self._lam - self.epsilon * np.asarray(grad_J)

            phi_out = self._phi.copy()
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
                        f"Diffusion requires a degree-regular communication graph: "
                        f"this node has {own_degree} neighbours but received a message "
                        f"from a node with {m.degree} neighbours."
                    )
                if m.weight_rule != self.weight_rule:
                    raise ValueError(
                        f"All Diffusion participants must share the same weight_rule: "
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

            # Combination: Mean-Metropolis weighted average of own φ and all
            # received φ's (eq. 19; doubly stochastic).
            self_w, neighbor_w = regular_graph_weights(len(neighbours), self.weight_rule)
            lam_new = self_w * self._phi
            for m in queue:
                lam_new = lam_new + neighbor_w * m.phi
            if np.max(np.abs(lam_new - self._lam)) <= self.tol:
                self._stable_rounds += 1
            else:
                self._stable_rounds = 0
            self._converged_flag = self._stable_rounds >= self.patience
            self._lam = lam_new

            del self._message_queue[message_data.k]

            # Adaptation
            grad_J = self.actor.gradient_term(self._lam, message_data.data)
            self._phi = self._lam - self.epsilon * np.asarray(grad_J)

            self._k += 1

            phi_out = self._phi.copy()
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


def create_diffusion_participant(
    finish_callback: Callable,
    diffusion_actor: DiffusionActor | None = None,
    initial_lam: float = 10.0,
    epsilon: float = 0.1,
    max_iter: int = 300,
    tol: float = 1e-4,
    patience: int = 3,
    horizon: int = 24,
    weight_rule: WeightRule = "mean_metropolis",
) -> DiffusionAlgorithm:
    """Create a :class:`DiffusionAlgorithm` participant.

    :param finish_callback: ``(algorithm, carrier) -> None`` — called when done.
    :param diffusion_actor: Optional gradient actor.  ``None`` → no gradient.
    :param initial_lam: Initial λ scalar.
    :param epsilon: Gradient step size.
    :param max_iter: Maximum iterations (failsafe; ``tol`` normally stops first).
    :param tol: Per-round λ-change convergence tolerance.
    :param patience: Consecutive sub-``tol`` rounds required to flag convergence.
    :param horizon: Number of schedule time steps.
    :param weight_rule: Combination-weight rule for the combine step.
    """
    return DiffusionAlgorithm(
        finish_callback=finish_callback,
        diffusion_actor=diffusion_actor,
        initial_lam=initial_lam,
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
        patience=patience,
        horizon=horizon,
        weight_rule=weight_rule,
    )


def create_diffusion_start(
    initial_lam: float,
    data: Any = None,
    horizon: int = 1,
) -> DiffusionMessage:
    """Create the initial kick-off message for a diffusion run.

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
