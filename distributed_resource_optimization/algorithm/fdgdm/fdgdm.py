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

       P_i[k+1] = \\operatorname{proj}_i\\!\\left(
           P_i[k] + \\sum_{j \\in \\mathcal{N}_i}
           w_{ij}^{\\mathrm{abs}} \\bigl(\\nabla F_j[k] - \\nabla F_i[k]\\bigr)\\right)

   where :math:`w_{ij}^{\\mathrm{abs}} = \\min\\!\\left(
   \\tfrac{1}{d_i u_i},\\, \\tfrac{1}{d_j u_j}\\right)` and :math:`d_i`
   is the neighbour count of node *i* **plus one**, :math:`u_i` is the upper
   bound of the second derivative of :math:`F_i`, and :math:`\\operatorname{proj}_i`
   is the actor's feasibility projection (:meth:`FDGDMActor.project`, e.g. a
   box clip to :math:`[P_{i,\\min}, P_{i,\\max}]`).

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
* **Eq. 24 stopping (‖P[k+1] − P[k]‖ < ε), decentralised.**  The paper checks
  the step of the whole power vector of one time snapshot.  Here every agent
  holds a horizon vector and only knows its own P_i, so agent i is
  *converged* when ``max_t |P_i[k+1, t] − P_i[k, t]| < tol`` (the paper's
  per-snapshot scale, independent of the horizon length) and sends that flag
  with its next message.  A run stops once the agent's own flag and every
  neighbour's flag of one complete round are true — the same rule as the
  diffusion algorithms; on the complete graph this is exactly "every agent
  converged".  ``tol=None`` disables it (fixed ``max_iter`` rounds).  Under
  message loss a stale-but-recent flag also counts and a stopping agent tells
  its neighbours; agents may then stop one round apart, so the power balance
  can drift slightly (accepted for the loss extension, decision 2026-10-16).
* **Every agent stops by its own counter** at ``max_iter``; a neighbour's
  last-round message never ends a run early, so an agent that is behind
  (heterogeneous delays, message loss) still applies its remaining rounds.

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

from .._rounds import (
    EdgeFlows,
    FlowSettlement,
    RoundBuffer,
    RoundDeadline,
    ScheduledStop,
    address_of,
    sender_key,
)
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
    :param d_u_product: ``d_i × u_i`` — the sender's degree
                        ``d_i = |N_i| + 1`` times its curvature upper bound.  Used by recipients to compute off-diagonal
                        weights (unused on the kick-off message).
    :param initial: If ``True`` this is the kick-off message; recipients
                    (re-)initialise their state from ``data``.
    :param flow_sum: If the sender owns the edge to the receiver: its
                     cumulative power flow along that edge through round
                     ``flow_round`` (see :class:`~.._rounds.EdgeFlows`).
    :param flow_ack: Acknowledges the sender's final running sum (end-of-run
                     settlement under loss, see :class:`~.._rounds.FlowSettlement`).
    :param flow_round: Last round included in ``flow_sum``.
    :param converged: The sender's last step was below its Eq. 24 tolerance
                      (``max_t |ΔP_t| < tol``).  With ``k > max_iter`` the
                      message is the notice of an agent that stopped on Eq. 24
                      (sent only with a round deadline).
    :param radius: Sender's convergence radius (graphs with diameter > 1).
    :param stop_round: Earliest stop round the sender knows of (-1: none);
                       see :class:`~.._rounds.ScheduledStop`.
    """

    k: int
    data: Any
    gradient: np.ndarray | None = None
    d_u_product: float = 0.0
    initial: bool = False
    flow_sum: np.ndarray | None = None
    flow_round: int = -1
    flow_ack: bool = False
    converged: bool = False
    radius: int = -1
    stop_round: int = -1


# ---------------------------------------------------------------------------
# FDGDMAlgorithm
# ---------------------------------------------------------------------------


class FDGDMAlgorithm(DistributedAlgorithm):
    """Distributed fast gradient descent over a scheduling horizon.

    :param finish_callback: Called with ``(algorithm, carrier)`` when the run
                            ends (Eq. 24 stop or ``max_iter``).
    :param fdgdm_actor: Optional :class:`FDGDMActor` supplying gradient and
                        projection.  ``None`` → the neutral base :class:`FDGDMActor`.
    :param max_iter: Maximum number of FDGDM iterations (backstop).
    :param tol: Eq. 24 tolerance for this agent's step,
                ``max_t |P_i[k+1, t] − P_i[k, t]| < tol``; ``None`` (default)
                disables early stopping, so the run always executes
                ``max_iter`` rounds.
    :param round_timeout: Seconds (carrier clock) to wait for a round's
                          gradients before advancing with the last received
                          ones; ``None`` (default) waits for every neighbour,
                          which stalls if a message is lost.
    :param diameter: Diameter of the communication graph (1 = complete). Above
                     1 the Eq. 24 stop needs every participant converged and
                     all participants stop in the same round (see
                     :class:`~.._rounds.ScheduledStop`), so the power flows
                     stay conserved.
    """

    def __init__(
        self,
        finish_callback: Callable,
        fdgdm_actor: FDGDMActor | None = None,
        max_iter: int = 300,
        round_timeout: float | None = None,
        tol: float | None = None,
        diameter: int = 1,
    ) -> None:
        self.finish_callback = finish_callback
        self.diameter = max(1, int(diameter))
        self._stop = ScheduledStop(self.diameter)
        self.actor: FDGDMActor = fdgdm_actor if fdgdm_actor is not None else FDGDMActor()
        self.max_iter = max_iter
        self.tol = tol
        #: How the last run ended: ``True`` on the Eq. 24 stop, ``False`` at
        #: ``max_iter`` with ``tol`` set, ``None`` without ``tol``.
        self.converged: bool | None = None
        #: Rounds completed when the last run ended.
        self.iterations: int = 0
        self._converged_flag: bool = False  # own Eq. 24 flag from the last step

        self._rounds = RoundBuffer()  # (gradient, d·u) per round and sender
        self._flows = EdgeFlows()  # power flows, kept equal on both ends
        self._deadline = RoundDeadline(round_timeout)
        self._settlement = FlowSettlement(round_timeout)
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

        # --- End-of-run settlement of the running sums (loss only) ---
        sender = sender_key(carrier, meta)
        if message_data.flow_ack:
            self._settlement.acked(sender)
            return
        has_sum = sender is not None and message_data.flow_sum is not None
        if has_sum and self._settlement.ack_needed(message_data.flow_round, self.max_iter - 1):
            # The owner's final sum: confirm it (also after finishing), so the
            # owner stops resending it.
            carrier.send_to_other(
                FDGDMMessage(
                    gradient=np.zeros(1), d_u_product=0.0, k=self.max_iter, data=None, flow_ack=True
                ),
                sender,
            )
        if self._settlement.active:
            if has_sum:
                self._flows.offer(
                    sender, message_data.flow_round, np.asarray(message_data.flow_sum)
                )
                self._settlement.offer_arrived(sender, message_data.flow_round)
            return

        # Notice of a neighbour that stopped on Eq. 24 (k > max_iter, only
        # with a round deadline): stop too instead of iterating against its
        # frozen gradient.
        if message_data.k > self.max_iter:
            if not self._first_message:
                self.converged = True
                self.iterations = self._k
                self._finish(carrier)
            return

        # A neighbour's last-round message (k = max_iter) does not end this
        # run: every participant stops by its own counter, so one that is
        # behind under loss still applies its remaining flow corrections. The
        # message is kept as that neighbour's freshest value. Outside a run it
        # is stale and ignored.
        if message_data.k >= self.max_iter and self._first_message:
            return

        # After termination, ignore stale messages from the previous run. A
        # new run starts with the kick-off (initial=True) or with a
        # neighbour's round-0 message, whichever arrives first: when the
        # kick-off travels more hops than a neighbour's message (sparse
        # topologies), the neighbour may already have started. A stale round-0
        # message cannot arrive after a run ended (it would have been needed
        # to finish round 0).
        if (
            self._first_message
            and self._started
            and not message_data.initial
            and message_data.k > 0
        ):
            return
        # A kick-off for a run this participant already started from a
        # neighbour's round-0 message must not restart it.
        if message_data.initial and not self._first_message:
            return

        # --- Initialisation path ---
        if self._first_message or message_data.initial:
            self._first_message = False
            self._started = True
            self._k = 0
            self._rounds.reset()
            self._flows.reset()
            self._settlement.cancel()
            self._converged_flag = False
            self._stop.reset()
            self.converged = None
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

        if has_sum:
            self._flows.offer(sender, message_data.flow_round, np.asarray(message_data.flow_sum))
        value = (
            np.asarray(message_data.gradient, dtype=float),
            float(message_data.d_u_product),
            bool(message_data.converged),
            int(message_data.radius),
            int(message_data.stop_round),
        )
        if message_data.k < self._k:
            # Too late for its round, but the freshest gradient of that neighbour.
            self._rounds.remember(message_data.k, sender, value)
            return
        self._rounds.add(message_data.k, sender, value)
        while self._running() and self._rounds.complete(self._k, n_neighbours):
            self._advance(carrier, neighbours)

    def _running(self) -> bool:
        """A run is in progress and has rounds left (not finished or settling)."""
        return not self._first_message and self._k < self.max_iter

    def _advance(self, carrier: "Carrier", neighbours: list) -> None:
        """Weighted gradient update (Eq. 9, Bai et al. 2022) for round ``self._k``."""
        self._deadline.cancel()
        # --- Eq. 24 stop: own flag and every neighbour's flag of a complete
        # round (all describe the step of round k−1). Under loss a
        # neighbour's last flag may also be one round old.
        own = (self._last_grad, self._last_du, self._converged_flag, -1, -1)
        if self.tol is not None and self.diameter > 1:
            recently_converged = self._deadline.enabled and self._rounds.all_recent(
                neighbours, self._k, 1, lambda v: v[2]
            )
            states = [
                (v[2], v[3], v[4])
                for v in self._rounds.values(
                    self._k, neighbours, default=lambda _n: (None, 0.0, False, -1, -1)
                )
            ]
            complete = self._rounds.complete(self._k, len(neighbours))
            if self._stop.observe(
                self._k, self._converged_flag, states, complete or recently_converged
            ):
                self.converged = True
                self.iterations = self._k
                if self._deadline.enabled:
                    self._notify_finished(carrier, neighbours)
                self._finish(carrier)
                return
        elif self.tol is not None and self._converged_flag:
            complete_and_converged = self._rounds.complete(self._k, len(neighbours)) and all(
                v[2] for v in self._rounds.round_values(self._k)
            )
            recently_converged = self._deadline.enabled and self._rounds.all_recent(
                neighbours, self._k, 1, lambda v: v[2]
            )
            if complete_and_converged or recently_converged:
                self.converged = True
                self.iterations = self._k
                if self._deadline.enabled:
                    self._notify_finished(carrier, neighbours)
                self._finish(carrier)
                return
        delta_P = np.zeros_like(self._P)
        me = address_of(carrier)
        if me is None:
            # Carrier cannot name this participant: plain update.
            for grad, du, *_ in self._rounds.values(self._k, neighbours, default=lambda _n: own):
                w_abs = 1.0 / max(self._last_du, du)
                delta_P += w_abs * (grad - self._last_grad)
        else:
            values = self._rounds.value_map(self._k, neighbours, default=lambda _n: own)
            for n in neighbours:
                grad, du, *_ = values[n]
                w_abs = 1.0 / max(self._last_du, du)
                delta_P += self._flows.flow(me, n, self._k, self._last_grad, grad, w_abs)
                delta_P += self._flows.correction(me, n)

        # Apply update and project onto feasibility set (Eq. 16).
        previous_P = self._P
        self._P = self.actor.project(self._P + delta_P)
        if self.tol is not None:
            self._converged_flag = bool(np.max(np.abs(self._P - previous_P)) < self.tol)

        # Prepare next broadcast.
        self._last_grad = np.asarray(self.actor.gradient(self._P, self._data), dtype=float)
        self._k += 1
        self._rounds.discard_before(self._k)
        self._broadcast(carrier, neighbours)
        if self._k >= self.max_iter:
            self.iterations = self._k
            if self.tol is not None:
                self.converged = False
            if self._settlement.enabled and me is not None:
                self._settlement.start(
                    carrier,
                    me,
                    neighbours,
                    self._flows,
                    final_round=self.max_iter - 1,
                    resend=lambda addr: self._send_round(carrier, me, addr),
                    on_settled=lambda: self._settled(carrier, neighbours),
                )
            else:
                self._finish(carrier)

    def _settled(self, carrier: "Carrier", neighbours: list) -> None:
        """Apply the flows still owed by edge owners, then finish."""
        me = address_of(carrier)
        delta_P = np.zeros_like(self._P)
        for n in neighbours:
            delta_P += self._flows.correction(me, n)
        if np.any(delta_P):
            self._P = self.actor.project(self._P + delta_P)
        self._finish(carrier)

    def _send_round(self, carrier: "Carrier", me: Any, addr: Any) -> None:
        """Send this round's gradient (and, as edge owner, the running sum) to *addr*."""
        owned = self._flows.owner_sum(me, addr) if me is not None else None
        carrier.send_to_other(
            FDGDMMessage(
                gradient=self._last_grad.copy(),
                d_u_product=self._last_du,
                k=self._k,
                data=self._data,
                flow_sum=None if owned is None else np.copy(owned[1]),
                flow_round=-1 if owned is None else owned[0],
                converged=self._converged_flag,
                radius=self._stop.radius,
                stop_round=self._stop.carried,
            ),
            addr,
        )

    def _notify_finished(self, carrier: "Carrier", neighbours: list) -> None:
        """Tell neighbours this agent stopped on Eq. 24 (k > max_iter)."""
        for addr in neighbours:
            carrier.send_to_other(
                FDGDMMessage(
                    gradient=self._last_grad.copy(),
                    d_u_product=self._last_du,
                    k=self.max_iter + 1,
                    data=None,
                    converged=True,
                ),
                addr,
            )

    def _broadcast(self, carrier: "Carrier", neighbours: list) -> None:
        """Send this round's gradient and arm the round deadline."""
        me = address_of(carrier)
        for addr in neighbours:
            self._send_round(carrier, me, addr)
        if self._k < self.max_iter:
            k = self._k
            self._deadline.arm(carrier, lambda: self._on_deadline(carrier, k))

    def _on_deadline(self, carrier: "Carrier", k: int) -> None:
        """Round *k* timed out: advance with what arrived, plus last values."""
        if not self._running() or self._k != k:
            return
        neighbours = carrier.others()
        self._advance(carrier, neighbours)
        while self._running() and self._rounds.complete(self._k, len(neighbours)):
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
