"""Fast distributed gradient descent method (FDGDM).

Based on: Bai et al. (2022) "Fast distributed gradient descent method for economic
dispatch of microgrids via upper bounds of second derivatives", Energy Reports 8,
1051-1060.

Each participant maintains a local power schedule P over a scheduling horizon.
At every iteration a participant:

1. Receives the gradient ``∇F_j(P_j[k])`` and the ``d·u`` product from every
   neighbour j.
2. **Updates** its power schedule via a weighted gradient step

   .. math::

       P_i[k+1] = \\operatorname{clip}\\!\\left(
           P_i[k] + \\sum_{j \\in \\mathcal{N}_i}
           w_{ij}^{\\mathrm{abs}} \\bigl(\\nabla F_j[k] - \\nabla F_i[k]\\bigr),
           0, P_{i,\\max}\\right)

   where :math:`w_{ij}^{\\mathrm{abs}} = \\min\\!\\left(
   \\tfrac{1}{d_i u_i},\\, \\tfrac{1}{d_j u_j}\\right)` and :math:`d_i`
   is the neighbour count of node *i* **plus one**, :math:`u_i` is the upper
   bound of the second derivative of :math:`F_i`.

3. Broadcasts the updated gradient ``∇F_i(P_i[k+1])`` and its own ``d·u``
   product to all neighbours.

Because the weight matrix W has zero row-sums the total power
:math:`\\sum_i P_i[k]` is conserved across iterations.  The **initial point
must therefore already satisfy the power balance constraint**.

Deliberate deviations from the paper
------------------------------------

* **d_i = |N_i| + 1, not the out-degree |N_i| of Eq. 5.**  With the paper's
  exact d_i the iteration matrix has eigenvalue −1 for two participants with
  equal curvature (period-2 oscillation that never converges); the +1 shrinks
  every off-diagonal weight, which keeps ``2U⁻¹ − W`` strictly diagonally
  dominant, so the Proposition-2 acceleration condition still holds.  Do not
  "fix" this back to ``n_neighbours`` without re-checking the n=2 case.
* **The Eq. 24 stopping criterion (‖P[k+1] − P[k]‖ < ε) is not implemented.**
  Every run executes a fixed ``max_iter`` iterations instead; converged runs
  simply stop changing until the counter runs out.
* **Termination assumes lockstep rounds** (every agent completes iteration k
  before any agent completes k+1).  This holds on the complete graphs with
  constant-delay lossless transport this codebase uses; with heterogeneous
  delays an agent could terminate one iteration early on receiving a
  neighbour's ``k ≥ max_iter`` message.

Message loss
------------

With ``round_timeout`` set, a round advances when complete or when its
deadline passes; a neighbour whose round-k gradient is missing contributes
its last received gradient (zero flow if never heard).  The pairwise terms
``w_ij (∇F_j − ∇F_i)`` are kept equal on both ends of every edge by the
running-sum protocol of :class:`~.._rounds.EdgeFlows`, so a lost message only
delays a transfer of power between two agents instead of creating or
destroying it: ``Σ_i P_i`` stays conserved (up to the projection, as in the
lossless algorithm).

The optional :class:`FDGDMActor` plug-in supplies ``∇F``, ``u``, and the
feasibility projection; the base class itself uses no cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .._rounds import EdgeFlows, RoundBuffer, RoundDeadline, address_of, sender_key
from ..core import DistributedAlgorithm, OptimizationMessage

if TYPE_CHECKING:
    from ...carrier.core import Carrier


# ---------------------------------------------------------------------------
# FDGDMActor hierarchy
# ---------------------------------------------------------------------------


class FDGDMActor:
    """Optional plug-in that supplies the gradient and projection for FDGDM.

    Subclass this to add a local cost function (e.g. economic dispatch or
    battery storage) to the FDGDM iterates.  Used as-is it is the neutral
    actor: zero gradient, unit curvature, no projection.
    """

    def gradient(self, P: np.ndarray, data: Any) -> np.ndarray:
        """Return the cost gradient ``∇F(P)`` at the current power iterate.

        :param P: Current power schedule (one value per time step).
        :param data: Auxiliary data forwarded from the start message (unused
                     by the default implementation).
        :returns: Gradient vector with the same shape as *P*.
        """
        return np.zeros_like(P)

    def curvature_bound(self) -> float:
        """Return the upper bound *u* on the second derivative of F.

        Used to compute the off-diagonal weights of the weight matrix W.
        Must be strictly positive.
        """
        return 1.0

    def project(self, P: np.ndarray) -> np.ndarray:
        """Project *P* onto the feasibility set (e.g. box constraints).

        Implementations should clip *P* to their own box constraints and cache
        the result in ``self.P`` so the scenario can read the final schedule.
        The default implementation is the identity (no constraints).
        """
        self.P = P.copy()
        return self.P


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class FDGDMMessage(OptimizationMessage):
    """Message exchanged between FDGDM participants.

    :param k: Current iteration counter.
    :param data: Auxiliary payload forwarded to :meth:`FDGDMActor.gradient`.
    :param gradient: Current cost gradient ``∇F_i(P_i[k])`` of the sender
                     (unused on the kick-off message).
    :param d_u_product: ``d_i × u_i`` — out-degree times curvature upper bound
                        of the sender.  Used by recipients to compute off-diagonal
                        weights (unused on the kick-off message).
    :param initial: If ``True`` this is the kick-off message; recipients
                    (re-)initialise their state from ``data``.
    :param flow_sum: If the sender owns the edge to the receiver: its
                     cumulative power flow along that edge through round
                     ``flow_round`` (see :class:`~.._rounds.EdgeFlows`).
    :param flow_round: Last round included in ``flow_sum``.
    """

    k: int
    data: Any
    gradient: np.ndarray | None = None
    d_u_product: float = 0.0
    initial: bool = False
    flow_sum: np.ndarray | None = None
    flow_round: int = -1


# ---------------------------------------------------------------------------
# FDGDMAlgorithm
# ---------------------------------------------------------------------------


class FDGDMAlgorithm(DistributedAlgorithm):
    """Distributed fast gradient descent over a scheduling horizon.

    :param finish_callback: Called with ``(algorithm, carrier)`` when
                            :attr:`max_iter` is reached.
    :param fdgdm_actor: Optional :class:`FDGDMActor` supplying gradient and
                        projection.  ``None`` → the neutral base :class:`FDGDMActor`.
    :param max_iter: Maximum number of FDGDM iterations.
    :param round_timeout: Seconds (carrier clock) to wait for a round's
                          gradients before advancing with the last received
                          ones; ``None`` (default) waits for every neighbour,
                          which stalls if a message is lost.
    """

    def __init__(
        self,
        finish_callback: Callable,
        fdgdm_actor: FDGDMActor | None = None,
        max_iter: int = 300,
        round_timeout: float | None = None,
    ) -> None:
        self.finish_callback = finish_callback
        self.actor: FDGDMActor = fdgdm_actor if fdgdm_actor is not None else FDGDMActor()
        self.max_iter = max_iter

        self._rounds = RoundBuffer()  # (gradient, d·u) per round and sender
        self._flows = EdgeFlows()  # power flows, kept equal on both ends
        self._deadline = RoundDeadline(round_timeout)
        self._data: Any = None
        self._first_message: bool = True
        self._started: bool = False  # True once any round has begun
        self._k: int = 0
        # Schedule state is sized from the kick-off data on (re-)initialisation.
        self._P: np.ndarray = np.zeros(1)
        self._last_grad: np.ndarray = np.zeros(1)
        self._last_du: float = 1.0

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: FDGDMMessage,
        meta: Any,
    ) -> None:
        """Process one incoming FDGDM message."""
        neighbours = carrier.others()
        n_neighbours = len(neighbours)

        # --- Termination path: a neighbour already finished its last round ---
        if message_data.k >= self.max_iter:
            if self._first_message:
                return
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
            self._rounds.reset()
            self._flows.reset()
            self._data = message_data.data

            # Bootstrap power schedule from data (initial power allocation).
            initial_p = np.asarray(message_data.data, dtype=float)
            self._P = self.actor.project(initial_p)

            # Gradient and du-product to broadcast at k=0.  Deliberate
            # deviation from Bai et al. Eq. 5: d = |N_i| + 1, not |N_i| —
            # the paper's exact d oscillates for n=2 (see module docstring).
            self._last_grad = np.asarray(
                self.actor.gradient(self._P, message_data.data), dtype=float
            )
            curvature = self.actor.curvature_bound()
            if curvature <= 0.0:
                raise ValueError(f"FDGDM curvature_bound must be > 0, got {curvature}.")
            self._last_du = float(n_neighbours + 1) * curvature

            self._broadcast(carrier, neighbours)
            if message_data.initial:
                return  # kickoff is not a real neighbour gradient; do not queue

        sender = sender_key(carrier, meta)
        if sender is not None and message_data.flow_sum is not None:
            self._flows.offer(sender, message_data.flow_round, np.asarray(message_data.flow_sum))
        value = (np.asarray(message_data.gradient, dtype=float), float(message_data.d_u_product))
        if message_data.k < self._k:
            # Too late for its round, but the freshest gradient of that neighbour.
            self._rounds.remember(message_data.k, sender, value)
            return
        self._rounds.add(message_data.k, sender, value)
        while not self._first_message and self._rounds.complete(self._k, n_neighbours):
            self._advance(carrier, neighbours)

    def _advance(self, carrier: "Carrier", neighbours: list) -> None:
        """Weighted gradient update (Eq. 9, Bai et al. 2022) for round ``self._k``."""
        self._deadline.cancel()
        own = (self._last_grad, self._last_du)
        delta_P = np.zeros_like(self._P)
        me = address_of(carrier)
        if me is None:
            # Carrier cannot name this participant: plain update.
            for grad, du in self._rounds.values(self._k, neighbours, default=lambda _n: own):
                w_abs = 1.0 / max(self._last_du, du)
                delta_P += w_abs * (grad - self._last_grad)
        else:
            values = self._rounds.value_map(self._k, neighbours, default=lambda _n: own)
            for n in neighbours:
                grad, du = values[n]
                w_abs = 1.0 / max(self._last_du, du)
                delta_P += self._flows.flow(me, n, self._k, self._last_grad, grad, w_abs)
                delta_P += self._flows.correction(me, n)

        # Apply update and project onto feasibility set (Eq. 16).
        self._P = self.actor.project(self._P + delta_P)

        # Prepare next broadcast.
        self._last_grad = np.asarray(self.actor.gradient(self._P, self._data), dtype=float)
        self._k += 1
        self._rounds.discard_before(self._k)
        self._broadcast(carrier, neighbours)
        if self._k >= self.max_iter:
            self._finish(carrier)

    def _broadcast(self, carrier: "Carrier", neighbours: list) -> None:
        """Send this round's gradient and arm the round deadline."""
        me = address_of(carrier)
        for addr in neighbours:
            owned = self._flows.owner_sum(me, addr) if me is not None else None
            carrier.send_to_other(
                FDGDMMessage(
                    gradient=self._last_grad.copy(),
                    d_u_product=self._last_du,
                    k=self._k,
                    data=self._data,
                    flow_sum=None if owned is None else np.copy(owned[1]),
                    flow_round=-1 if owned is None else owned[0],
                ),
                addr,
            )
        if self._k < self.max_iter:
            k = self._k
            self._deadline.arm(carrier, lambda: self._on_deadline(carrier, k))

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
# Factories
# ---------------------------------------------------------------------------


#: Create an :class:`FDGDMAlgorithm` participant (same signature).
create_fdgdm_participant = FDGDMAlgorithm


def create_fdgdm_start(data: Any = None) -> FDGDMMessage:
    """Create the initial kick-off message for an FDGDM run.

    :param data: Initial power allocation vector (one per time step).
                 Each participant projects this onto its own feasibility set
                 to obtain its starting power schedule.
    :returns: An :class:`FDGDMMessage` with ``initial=True``.
    """
    return FDGDMMessage(k=0, data=data, initial=True)
