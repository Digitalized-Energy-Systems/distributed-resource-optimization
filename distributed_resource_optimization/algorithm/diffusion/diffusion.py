"""Diffusion algorithm (adapt-then-combine).

Each participant maintains a local price estimate λ and a power iterate φ over a
scheduling horizon.  At every iteration a participant

1. **adapts** its power iterate via a local gradient step
   ``φ = λ - ε · ∇J(λ, data)``,
2. broadcasts ``φ`` to its neighbours, and
3. **combines** its own φ with all received φ's using the combination
   weights selected by ``weight_rule`` -- by default the Mean-Metropolis
   matrix (Ces et al. 2025, eq. 19 -- the same doubly-stochastic matrix used
   for consensus) -- to form the next λ.

The update rule is:

.. math::

    \\lambda_i^{k+1} = w_{ii} \\varphi_i^k +
                    \\sum_{j \\in \\mathcal{N}_i} w_{ij} \\, \\varphi_j^k,
    \\qquad
    \\varphi_i^{k+1} = \\lambda_i^{k+1} - \\varepsilon \\, \\nabla J(\\lambda_i^{k+1}, \\text{data}),

with (Mean-Metropolis) :math:`w_{ij} = 2 / (n_i + n_j + 1)` and
:math:`w_{ii} = 1 - \\sum_j w_{ij}`.
Weights are computed via :func:`~distributed_resource_optimization.algorithm._weight_rules.regular_graph_weights`,
which assumes a degree-regular communication graph (true both for this
codebase's complete-graph topologies and for the paper's own ring network);
the paper notes this doubly-stochastic requirement is what allows classical
Diffusion to converge at all.

The optional :class:`DiffusionActor` plug-in supplies ``∇J``; the base class
itself returns zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .._rounds import (
    EdgeFlows,
    RoundBuffer,
    RoundDeadline,
    address_of,
    convergence_radius,
    sender_key,
)
from .._weight_rules import WeightRule, combination_weights, regular_graph_weights
from ..core import DistributedAlgorithm, OptimizationMessage

if TYPE_CHECKING:
    from ...carrier.core import Carrier


# ---------------------------------------------------------------------------
# DiffusionActor hierarchy
# ---------------------------------------------------------------------------


class DiffusionActor:
    """Optional plug-in that supplies the gradient term for the adapt step.

    Subclass this to bias the diffusion iterates toward a local optimum
    (e.g. economic dispatch or battery storage scheduling).  Used as-is it is
    the neutral actor: the gradient is identically zero.
    """

    def gradient_term(self, lam: np.ndarray, data: Any) -> np.ndarray | float:
        """Return the gradient ``∇J(λ, data)`` for the current iterate *lam*.

        :param lam: Current local price/λ estimate over the schedule.
        :param data: Auxiliary data forwarded from the start message.
        :returns: Additive gradient (default: 0).
        """
        return 0


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
    :param degree: Sender's own neighbour count; recipients use it for the
                   combination weights (:func:`~distributed_resource_optimization.algorithm._weight_rules.combination_weights`)
                   -- ignored on the ``initial`` kick-off message.
    :param weight_rule: Sender's own combination-weight rule.  Used only to
                        verify every participant agrees on the same rule --
                        ignored on the ``initial`` kick-off message.
    :param converged: Whether the sender's λ has changed by at most ``tol``
                      for ``patience`` consecutive combine steps.  A round in
                      which every participant's flag is ``True`` terminates
                      the run (Ces et al. 2025 stop when the largest per-agent
                      incremental-cost change falls below their tolerance).
    :param flow_sum: If the sender owns the edge to the receiver: its
                     cumulative combine flow along that edge through round
                     ``flow_round`` (see :class:`~.._rounds.EdgeFlows`); the
                     receiver uses it to keep both ends' flows equal under
                     message loss.  ``None`` otherwise.
    :param flow_round: Last round included in ``flow_sum``.
    :param radius: Sender's convergence radius from its previous round (see
                   :func:`~.._rounds.convergence_radius`); only used on graphs
                   with a diameter above 1.
    """

    phi: np.ndarray
    k: int
    data: Any
    initial: bool = False
    degree: int = 0
    weight_rule: str = ""
    converged: bool = False
    flow_sum: np.ndarray | None = None
    flow_round: int = -1
    radius: int = -1


# ---------------------------------------------------------------------------
# Shared round handling
# ---------------------------------------------------------------------------


class _DiffusionRounds(DistributedAlgorithm):
    """Round handling shared by :class:`DiffusionAlgorithm` and
    :class:`~.exact_diffusion.ExactDiffusionAlgorithm`.

    Every round a participant broadcasts its outgoing iterate, waits for each
    neighbour's round-k iterate, combines them and adapts.  With a
    ``round_timeout`` a round also advances when its deadline passes; a
    neighbour whose round-k message is missing then contributes its last
    received iterate (this participant's own outgoing iterate if never
    heard).  The all-converged stop only fires on a complete round, so a stale
    flag can never end a run; :attr:`max_iter` stays the backstop.

    Subclasses provide :meth:`_start_state`, :meth:`_outgoing`,
    :meth:`_combine_weights` and :meth:`_adapt`.
    """

    _label = "Diffusion"

    finish_callback: Callable
    max_iter: int
    tol: float
    patience: int
    weight_rule: WeightRule
    converged: bool
    iterations: int
    _lam: np.ndarray

    def _init_rounds(self, round_timeout: float | None, diameter: int = 1) -> None:
        #: Diameter of the communication graph. Above 1 the run ends on the
        #: convergence radius (every agent converged), not on the neighbours.
        self.diameter = max(1, int(diameter))
        self._radius: int = -1
        self._degrees: dict[Any, int] = {}  # neighbour -> its degree
        self._rounds = RoundBuffer()  # (φ, converged, radius) per round and sender
        self._flows = EdgeFlows()  # combine flows, kept equal on both ends
        self._deadline = RoundDeadline(round_timeout)
        self._data: Any = None
        self._first_message: bool = True
        self._started: bool = False  # True once any round has begun
        self._k: int = 0
        self._converged_flag: bool = False  # own flag from the last combine
        self._stable_rounds: int = 0  # consecutive rounds with λ change ≤ tol

    # -- hooks ---------------------------------------------------------------

    def _start_state(self, horizon: int, data: Any) -> None:
        """(Re-)initialise λ and the outgoing iterate for a new run."""
        raise NotImplementedError

    def _outgoing(self) -> np.ndarray:
        """The iterate this participant broadcasts and combines with."""
        raise NotImplementedError

    def _combine_weights(self, n_neighbours: int) -> tuple[float, float]:
        """``(self weight, weight per neighbour)`` on a degree-regular graph."""
        raise NotImplementedError

    def _neighbour_weights(self, neighbours: list) -> tuple[float, float, dict[Any, float] | None]:
        """``(self weight, uniform neighbour weight, per-neighbour weights)``.

        On a degree-regular neighbourhood (every neighbour's degree equals this
        node's, e.g. a complete graph or a ring) the per-neighbour map is
        ``None`` and the uniform weight applies, exactly as before. Otherwise
        the weights come from each neighbour's degree (a neighbour never heard
        of counts with this node's degree).
        """
        own = len(neighbours)
        degrees = {n: self._degrees.get(n, own) for n in neighbours}
        self_w, neighbor_w = self._combine_weights(own)
        if all(d == own for d in degrees.values()):
            return self_w, neighbor_w, None
        self_w, weights = self._irregular_weights(own, degrees)
        return self_w, neighbor_w, weights

    def _irregular_weights(
        self, own_degree: int, degrees: dict[Any, int]
    ) -> tuple[float, dict[Any, float]]:
        """Combination weights from the neighbours' degrees."""
        return combination_weights(own_degree, degrees, self.weight_rule)

    def _adapt(self, data: Any) -> None:
        """Adapt step after the combine (updates the outgoing iterate)."""
        raise NotImplementedError

    # -- message handling ------------------------------------------------------

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: DiffusionMessage,
        meta: Any,
    ) -> None:
        """Process one incoming diffusion message."""
        neighbours = carrier.others()

        if message_data.k >= self.max_iter and self._first_message:
            return  # stale: this participant's run is already over
        # --- Termination path: notice of a neighbour that stopped on the
        # all-converged criterion (k > max_iter, sent only with a
        # round_timeout). A neighbour's ordinary last-round message
        # (k = max_iter) does not end this run: every participant stops by its
        # own counter and keeps that message as the neighbour's freshest value.
        if message_data.k > self.max_iter:
            if self.diameter > 1:
                # The sender certified that every agent converged: stop too and
                # pass the notice on, so no neighbour waits for this agent.
                self.converged = True
                self.iterations = self._k
                self._notify_finished(carrier, neighbours)
            else:
                self.converged = self._converged_flag
                self.iterations = self._k
            self._finish(carrier)
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
            self._radius = -1
            self.converged = False
            self._rounds.reset()
            self._flows.reset()
            self._data = message_data.data
            self._start_state(len(message_data.phi), message_data.data)
            self._broadcast(carrier, neighbours)
            if message_data.initial:
                return  # kick-off is not a topology neighbour; do not queue

        self._check_neighbour(message_data)
        sender = sender_key(carrier, meta)
        if sender is not None:
            self._degrees[sender] = int(message_data.degree)
        if sender is not None and message_data.flow_sum is not None:
            self._flows.offer(sender, message_data.flow_round, np.asarray(message_data.flow_sum))
        value = (
            np.asarray(message_data.phi),
            bool(message_data.converged),
            int(message_data.radius),
        )
        if message_data.k < self._k:
            # Too late for its round, but the freshest value of that neighbour.
            self._rounds.remember(message_data.k, sender, value)
            return
        self._rounds.add(message_data.k, sender, value)
        while not self._first_message and self._rounds.complete(self._k, len(neighbours)):
            self._advance(carrier, neighbours)

    def _check_neighbour(self, message: DiffusionMessage) -> None:
        """Every participant must use the same weight_rule, or the weight
        matrix would be inconsistent across nodes."""
        if message.weight_rule != self.weight_rule:
            raise ValueError(
                f"All {self._label} participants must share the same weight_rule: "
                f"this node uses {self.weight_rule!r} but received a message "
                f"from a node using {message.weight_rule!r}."
            )

    def _advance(self, carrier: "Carrier", neighbours: list) -> None:
        """Run round ``self._k`` with the best values available and broadcast."""
        self._deadline.cancel()
        # --- Termination path (tol criterion) ---
        # Own flag and every round-k message reflect the λ change of round
        # k−1.  Normally only a complete round counts.  Under loss a
        # neighbour may already have stopped (and its notice been lost), so a
        # round can never complete again; then every neighbour's last flag
        # must say converged and be at most `patience` rounds old.
        own = self._outgoing()
        recently_converged = self._deadline.enabled and self._rounds.all_recent(
            neighbours, self._k, self.patience, lambda value: value[1]
        )
        if self.diameter > 1:
            states = [
                (value[1], value[2])
                for value in self._rounds.values(
                    self._k, neighbours, default=lambda _n: (own, False, -1)
                )
            ]
            self._radius = convergence_radius(self._converged_flag, states, self.diameter)
            stop = self._radius >= self.diameter and (
                self._rounds.complete(self._k, len(neighbours)) or recently_converged
            )
        else:
            complete_and_converged = self._rounds.complete(self._k, len(neighbours)) and all(
                value[1] for value in self._rounds.round_values(self._k)
            )
            stop = self._converged_flag and (complete_and_converged or recently_converged)
        if stop:
            self.converged = True
            self.iterations = self._k
            if self._deadline.enabled or self.diameter > 1:
                self._notify_finished(carrier, neighbours)
            self._finish(carrier)
            return

        self_w, neighbor_w, weights = self._neighbour_weights(neighbours)
        me = address_of(carrier)
        if weights is not None and (me is None or not self._deadline.enabled):
            # Irregular neighbourhood, lossless: per-neighbour weights, values
            # in arrival order.
            values = self._rounds.value_map(
                self._k, neighbours, default=lambda _n: (own, False, -1)
            )
            lam_new = self_w * own
            for n, value in values.items():
                lam_new = lam_new + weights[n] * value[0]
        elif me is None or not self._deadline.enabled:
            # Plain weighted combine, as in the papers (Ces et al. eq. 27):
            # lossless runs (no round deadline) or a carrier that cannot name
            # this participant. The running-sum flows below only matter under
            # message loss.
            phis = [
                value[0]
                for value in self._rounds.values(
                    self._k, neighbours, default=lambda _n: (own, False, -1)
                )
            ]
            lam_new = self_w * own
            for phi in phis:
                lam_new = lam_new + neighbor_w * phi
        else:
            # Same combine written as own + Σ w·(φ_j − φ_own) (the self
            # weight is 1 − Σ w), so each pairwise flow can be kept equal on
            # both ends of its edge under message loss (EdgeFlows).
            values = self._rounds.value_map(
                self._k, neighbours, default=lambda _n: (own, False, -1)
            )
            lam_new = own.copy()
            for n in neighbours:
                w = neighbor_w if weights is None else weights[n]
                lam_new = lam_new + self._flows.flow(me, n, self._k, own, values[n][0], w)
                lam_new = lam_new + self._flows.correction(me, n)
        if np.max(np.abs(lam_new - self._lam)) <= self.tol:
            self._stable_rounds += 1
        else:
            self._stable_rounds = 0
        self._converged_flag = self._stable_rounds >= self.patience
        self._lam = lam_new

        self._adapt(self._data)
        self._k += 1
        self._rounds.discard_before(self._k)
        self._broadcast(carrier, neighbours)
        if self._k >= self.max_iter:
            self.converged = False
            self.iterations = self._k
            self._finish(carrier)

    def _broadcast(self, carrier: "Carrier", neighbours: list) -> None:
        """Send this round's iterate and arm the round deadline."""
        phi_out = self._outgoing().copy()
        me = address_of(carrier)
        for addr in neighbours:
            owned = self._flows.owner_sum(me, addr) if me is not None else None
            carrier.send_to_other(
                DiffusionMessage(
                    phi=phi_out,
                    k=self._k,
                    data=self._data,
                    degree=len(neighbours),
                    weight_rule=self.weight_rule,
                    converged=self._converged_flag,
                    flow_sum=None if owned is None else np.copy(owned[1]),
                    flow_round=-1 if owned is None else owned[0],
                    radius=self._radius,
                ),
                addr,
            )
        if self._k < self.max_iter:
            k = self._k
            self._deadline.arm(carrier, lambda: self._on_deadline(carrier, k))

    def _notify_finished(self, carrier: "Carrier", neighbours: list) -> None:
        """Tell neighbours this participant stopped on convergence (k > max_iter),
        so none keeps iterating against its frozen iterate."""
        for addr in neighbours:
            carrier.send_to_other(
                DiffusionMessage(
                    phi=self._outgoing().copy(),
                    k=self.max_iter + 1,
                    data=self._data,
                    degree=len(neighbours),
                    weight_rule=self.weight_rule,
                    converged=True,
                ),
                addr,
            )

    def _on_deadline(self, carrier: "Carrier", k: int) -> None:
        """Round *k* timed out: advance with what arrived, plus last values."""
        if self._first_message or self._k != k:
            return
        neighbours = carrier.others()
        self._advance(carrier, neighbours)
        while not self._first_message and self._rounds.complete(self._k, len(neighbours)):
            self._advance(carrier, neighbours)

    def _finish(self, carrier: "Carrier") -> None:
        self._deadline.cancel()
        self.finish_callback(self, carrier)
        self._first_message = True
        self._rounds.reset()


# ---------------------------------------------------------------------------
# DiffusionAlgorithm
# ---------------------------------------------------------------------------


class DiffusionAlgorithm(_DiffusionRounds):
    """Distributed adapt-then-combine diffusion over a scheduling horizon.

    The round handling (kick-off, termination, round deadline, last-value
    fallback) is shared with :class:`~.exact_diffusion.ExactDiffusionAlgorithm`
    via :class:`_DiffusionRounds`.

    :param finish_callback: Called with ``(algorithm, carrier)`` when the run
                            terminates — via the ``tol`` criterion or the
                            ``max_iter`` failsafe (see :attr:`converged`).
    :param diffusion_actor: Optional :class:`DiffusionActor` supplying the
                            gradient.  ``None`` → the zero-gradient base
                            :class:`DiffusionActor`.
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
    :param weight_rule: Combination-weight rule for the combine step.  The
                        paper requires a doubly-stochastic matrix for
                        classical Diffusion to converge.  ``"mean_metropolis"``
                        (the default) and ``"hastings"`` are doubly stochastic
                        on any graph; on the degree-regular graphs this
                        codebase always uses, ``"averaging"`` and
                        ``"relative_degree"`` collapse to the same uniform
                        doubly-stochastic matrix too (see
                        :mod:`~distributed_resource_optimization.algorithm._weight_rules`).
    :param round_timeout: Seconds (carrier clock) to wait for a round's
                          messages before advancing with the last received
                          iterates; ``None`` (default) waits for every
                          neighbour, which stalls if a message is lost.
    :param diameter: Diameter of the communication graph (1 = complete).
                     Above 1 a participant stops once its convergence radius
                     reaches it (every participant converged, see
                     :func:`~.._rounds.convergence_radius`) and tells its
                     neighbours, which stop as well.
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
        weight_rule: WeightRule = "mean_metropolis",
        round_timeout: float | None = None,
        diameter: int = 1,
    ) -> None:
        self.finish_callback = finish_callback
        self.actor: DiffusionActor = (
            diffusion_actor if diffusion_actor is not None else DiffusionActor()
        )
        self.initial_lam = initial_lam
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.weight_rule = weight_rule

        #: How the last run ended: ``True`` if the tol criterion fired,
        #: ``False`` if it hit the max_iter failsafe (or never ran).
        self.converged: bool = False
        #: Iterations completed when the last run terminated.
        self.iterations: int = 0

        self._init_rounds(round_timeout, diameter)
        # λ / φ state is sized from the kick-off message on (re-)initialisation.
        self._lam: np.ndarray = np.array([initial_lam])
        self._phi: np.ndarray = np.array([initial_lam])

    def _start_state(self, horizon: int, data: Any) -> None:
        self._lam = np.ones(horizon) * self.initial_lam
        self._adapt(data)

    def _outgoing(self) -> np.ndarray:
        return self._phi

    def _combine_weights(self, n_neighbours: int) -> tuple[float, float]:
        # Mean-Metropolis weighted average of own φ and all received φ's
        # (eq. 19; doubly stochastic).
        return regular_graph_weights(n_neighbours, self.weight_rule)

    def _adapt(self, data: Any) -> None:
        grad_J = self.actor.gradient_term(self._lam, data)
        self._phi = self._lam - self.epsilon * np.asarray(grad_J)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


#: Create a :class:`DiffusionAlgorithm` participant (same signature).
create_diffusion_participant = DiffusionAlgorithm


def create_diffusion_start(
    initial_lam: float,
    data: Any = None,
    horizon: int = 1,
) -> DiffusionMessage:
    """Create the initial kick-off message for a diffusion run.

    :param initial_lam: Placeholder φ value.  Recipients only use the
                        vector's *length*; each participant (re)starts from
                        its own ``initial_lam``.
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
