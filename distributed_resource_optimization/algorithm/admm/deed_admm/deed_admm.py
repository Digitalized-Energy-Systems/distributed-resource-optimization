"""DEED-ADMM: Distributed Economic Dispatch via ADMM with dynamic consensus.

Zhu et al. 2025, "DEED-ADMM: A Scalable Distributed Algorithm for Economic
Dispatch in Multi-Energy Systems With Energy Storage".

Each agent i is an (electricity-only) energy hub with an optional supplier
xᵢ (generator output), the hub input yᵢ, optional storage zᵢ = [P^Sd; P^Sc]
and the local demand dᵢ, coupled by the hub balance Aᵢyᵢ + Bzᵢ = dᵢ and the
global balance Σᵢ(xᵢ − yᵢ) = 0.  Per iteration k (Algorithm 1):

  Collect from neighbours:  λ̃ᵢ = Σⱼ wᵢⱼ λⱼ,ₖ   ξ̃ᵢ = Σⱼ wᵢⱼ ξⱼ,ₖ

  Compute b-vectors:
    bₓ = γx − λ̃ + Xᵀ(γx̂ − νₓ) − (γ/2)ξ̃
    bᵧ = γy + λ̃ + (γ/2)ξ̃
    b_z = Zᵀ(γẑ − ν_z)

  Primal update (closed form):
    x ← Hₓ bₓ
    y ← Hᵧ(bᵧ + AᵀHᵢ(d − AHᵧbᵧ − BH_z b_z))
    z ← H_z(b_z + BᵀHᵢ(d − AHᵧbᵧ − BH_z b_z))
    p̂ ← clip(Mp + ν/γ, p̂_min, p̂_max)     [box projection]

  Auxiliary / dual update:
    ξ ← ξ̃ + (x_new − x_old) − (y_new − y_old)
    λ ← λ̃ + (γ/2)ξ
    ν ← ν + γ(Mp − p̂)

Electricity-only specialisation of the paper's hub model:

* Aᵢ = I (no carrier conversion) and there is no hub power limit, so the
  Y block of Mᵢ is empty and Hᵧ = (γI)⁻¹.
* X = I: the supplier box p_min ≤ x ≤ p_max is the x̂ projection.  An agent
  without a supplier (x_min = x_max) keeps x pinned to that value — the
  paper's "hub without distributed generation".
* Storage (eqs. 5, 8, 9, 11) is the z block: B = [I, −I] (discharge lowers
  the hub input, charge raises it), and Zᵢ = [I₂τ; Z₁ᵀ] where the rows of Z₁ᵀ
  give the energy level change E_t − E_0, so the SOC limits and the terminal
  reserve E_τ ≥ ω·E_max are plain box constraints on ẑ.  Without storage B
  is empty and the y-update collapses to y = dᵢ exactly.

The weight matrix is the paper's W = I − βL (Section V):
wᵢⱼ = β for each neighbour and wᵢᵢ = 1 − β·|Nᵢ|.  ``beta`` defaults to 1/n,
which on a complete graph gives the uniform average W = (1/n)·11ᵀ; on any
connected graph with β·|Nᵢ| < 1 it satisfies Assumption 1.

Deviations from the paper (by design):

* The paper's ramp-rate constraints (16)-(17) are not modelled — none of
  the benchmark networks specify ramp limits.
* The paper's Algorithm 1 runs "while not converged" without defining a
  criterion.  With ``tol`` set, an agent is *converged* when its reported
  schedule moved less than ``tol`` in every hour (``max_t |ΔP_t|``) and its
  tracker is below it (``max_t |ξ_t| < tol``; ξ → 0 at the optimum,
  Theorem 2); a run stops when the agent's own flag and every neighbour's
  flag of one complete round are true — the same rule as FDGDM's Eq. 24
  stop (decision 2026-10-16).  ``tol=None`` runs a fixed ``max_iter``.  The
  schedule is read from the box-projected p̂, so the power limits hold
  either way.  Under loss a recent flag also counts and a stopping agent
  notifies its neighbours (k > max_iter).

Communication pattern: synchronous peer-to-peer rounds.  Each agent
broadcasts its (λ, ξ) for round k and advances once it holds round-k state
from every neighbour (state from a neighbour one round ahead is buffered).
An agent stops by itself after ``max_iter`` rounds; its last broadcast is
round ``max_iter − 1``, which every neighbour consumes in its own last step,
so no protocol message outlives a lossless run.

Message loss: with ``round_timeout`` set, a round also advances when its
deadline passes.  A neighbour whose round-k state is missing contributes its
last received λ (λ needs no exact mixing: at a fixed point all λᵢ agree and
ξ = 0).  The ξ mixing ``β(ξⱼ − ξᵢ)`` is kept equal on both ends of every edge
by the running-sum protocol of :class:`~.._rounds.EdgeFlows`, which preserves
the tracking invariant ``Σᵢ ξᵢ = Σᵢ (xᵢ − yᵢ)`` that the dual update relies
on.  A finished agent ignores late messages of its run (rounds > 0); a
kick-off or a new run's round-0 message restarts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..._rounds import (
    EdgeFlows,
    FlowSettlement,
    RoundBuffer,
    RoundDeadline,
    ScheduledStop,
    address_of,
    sender_key,
)
from ...core import DistributedAlgorithm, OptimizationMessage

if TYPE_CHECKING:
    from ....carrier.core import Carrier


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


@dataclass
class DEEDADMMMessage(OptimizationMessage):
    """Message exchanged between DEED-ADMM participants.

    :param lam: Sender's local dual estimate λᵢ,ₖ (shape τ).
    :param xi: Sender's consensus tracker ξᵢ,ₖ (shape τ).
    :param k: Round the state belongs to.
    :param data: Auxiliary payload forwarded on the initial kick-off.
    :param initial: ``True`` for the external kick-off message, which carries
                    no neighbour state and (re-)starts the recipient's run.
    :param flow_sum: If the sender owns the edge to the receiver: its
                     cumulative ξ mixing flow along that edge through round
                     ``flow_round`` (see :class:`~.._rounds.EdgeFlows`).
    :param flow_ack: Acknowledges the sender's final running sum (end-of-run
                     settlement under loss, see :class:`~.._rounds.FlowSettlement`).
    :param flow_round: Last round included in ``flow_sum``.
    :param radius: Sender's convergence radius (graphs with diameter > 1).
    :param stop_round: Earliest stop round the sender knows of (-1: none);
                       see :class:`~.._rounds.ScheduledStop`.
    :param run: Sender's run counter (round messages only); a finished
                participant starts a new run from a round-0 message of a
                later run, never from a stale one.
    """

    lam: np.ndarray
    xi: np.ndarray
    k: int
    data: Any
    initial: bool = False
    flow_sum: np.ndarray | None = None
    flow_round: int = -1
    flow_ack: bool = False
    converged: bool = False
    radius: int = -1
    stop_round: int = -1
    run: int = 0


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


class DEEDADMMAlgorithm(DistributedAlgorithm):
    """Peer-to-peer DEED-ADMM participant (electricity-only energy hub).

    The base class models a hub with a supplier x and a local demand dᵢ (no
    storage); :class:`DEEDADMMStorageAlgorithm` adds the storage block.

    :param finish_callback: ``(algorithm, carrier) -> None`` called when done.
    :param cost_quad: Per-step quadratic cost coefficient aᵢ (shape τ or
        scalar); cost is c(xₜ) = aᵢ xₜ² + bᵢ xₜ.  Set to 0 for linear cost.
    :param cost_lin: Per-step linear cost coefficient bᵢ (shape τ or scalar).
    :param x_min: Lower bound on supplier output per step (shape τ or scalar).
    :param x_max: Upper bound on supplier output per step (shape τ or scalar).
    :param d_i: Demand allocated to this agent per step (shape τ).  The sum
        over all agents must equal total system demand at each step.
    :param gamma: ADMM penalty parameter γ > 0 (paper default: 0.05).
    :param max_iter: Number of DEED-ADMM rounds.
    :param n_agents: Total number of agents n; only used for the default
        ``beta = 1/n``.
    :param beta: Laplacian step of W = I − βL.  Must satisfy β·|Nᵢ| < 1.
    :param round_timeout: Seconds (carrier clock) to wait for a round's state
        before advancing with the last received values; ``None`` (default)
        waits for every neighbour, which stalls if a message is lost.
    :param tol: Convergence tolerance (MW) on this agent's per-hour schedule
        step and on its tracker ξ; ``None`` (default) runs ``max_iter`` rounds.
    :param diameter: Diameter of the communication graph (1 = complete). Above
        1 the convergence stop needs every participant converged and all stop
        in the same round (:class:`~.._rounds.ScheduledStop`), so the ξ flows
        stay conserved.
    """

    def __init__(
        self,
        finish_callback: Callable,
        cost_quad: np.ndarray | float,
        cost_lin: np.ndarray | float,
        x_min: np.ndarray | float,
        x_max: np.ndarray | float,
        d_i: np.ndarray,
        gamma: float = 0.05,
        max_iter: int = 500,
        n_agents: int = 1,
        beta: float | None = None,
        round_timeout: float | None = None,
        tol: float | None = None,
        diameter: int = 1,
    ) -> None:
        self.finish_callback = finish_callback
        self.gamma = float(gamma)
        self.max_iter = max_iter
        self.tol = tol
        self.diameter = max(1, int(diameter))
        self._stop = ScheduledStop(self.diameter)
        self._run: int = 0  # runs started so far
        #: How the last run ended: ``True`` on the convergence stop, ``False``
        #: at ``max_iter`` with ``tol`` set, ``None`` without ``tol``.
        self.converged: bool | None = None
        #: Rounds completed when the last run ended.
        self.iterations: int = 0
        self.n_agents = n_agents
        self.beta = float(beta) if beta is not None else 1.0 / max(n_agents, 1)

        tau = len(np.atleast_1d(d_i))

        def _vec(v: np.ndarray | float) -> np.ndarray:
            return np.broadcast_to(np.atleast_1d(v).astype(float), (tau,)).copy()

        self._cost_quad = _vec(cost_quad)
        self._cost_lin = _vec(cost_lin)
        self._x_min = _vec(x_min)
        self._x_max = _vec(x_max)
        if np.any(self._x_min > self._x_max):
            raise ValueError("x_min must not exceed x_max")
        self._d_i = np.asarray(d_i, dtype=float).copy()
        self._tau = tau
        # A hub without a supplier: x is pinned to its (degenerate) box.
        self._x_fixed = bool(np.all(self._x_min == self._x_max))

        # Hₓ = (∇²cᵢ + γ(I + XᵀX))⁻¹ with X = I → diag 1 / (2aᵢ + 2γ).
        self._H_x = 1.0 / (2.0 * self._cost_quad + 2.0 * self.gamma)
        # Hᵧ = (γI)⁻¹ (no hub power limit, so no Y block).
        self._H_y = 1.0 / self.gamma

        self._rounds = RoundBuffer()  # (λ, ξ) per round and sender
        self._flows = EdgeFlows()  # ξ mixing flows, kept equal on both ends
        self._deadline = RoundDeadline(round_timeout)
        self._settlement = FlowSettlement(round_timeout)
        self._data: Any = None
        self._running: bool = False
        self._has_run: bool = False  # a finished run ignores late messages
        self._reset_state()

        # Public result — written at termination.
        self.P: np.ndarray = np.zeros(tau)

    # ------------------------------------------------------------------
    # Hooks for the storage block (none in the base class)
    # ------------------------------------------------------------------

    def _reset_storage_state(self) -> None:
        pass

    def _update_y(self, b_y: np.ndarray) -> np.ndarray:
        """y-update for Aᵢ = I without storage: Hᵧ·Hᵢ = 1, so y = dᵢ exactly."""
        return self._d_i.copy()

    def _extract_schedule(self) -> np.ndarray:
        """Final schedule: the box-projected supplier output p̂ₓ."""
        return self._p_hat_x.copy()

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Initialise all iteration variables (ξᵢ,₀ = xᵢ,₀ − yᵢ,₀ = 0)."""
        tau = self._tau
        self._x = self._x_min.copy() if self._x_fixed else np.zeros(tau)
        self._p_hat_x = self._x.copy()
        self._v_x = np.zeros(tau)
        self._y = self._x.copy()
        self._xi = np.zeros(tau)
        self._lam = np.zeros(tau)
        self._k = 0
        self._rounds.reset()
        self._flows.reset()
        self._settlement.cancel()
        self._reset_storage_state()
        self._converged_flag = False
        self._stop.reset()
        self.converged = None

    def _mix(self, carrier: "Carrier", neighbours: list) -> tuple[np.ndarray, np.ndarray]:
        """W-weighted neighbour state, W = I − βL, for round ``self._k``."""
        own = (self._lam, self._xi, self._converged_flag, -1, -1)
        lam_tilde = self._lam.copy()
        xi_tilde = self._xi.copy()
        me = address_of(carrier)
        if me is None or not self._deadline.enabled:
            # Plain mixing, as in the paper (Algorithm 1): lossless runs (no
            # round deadline) or a carrier that cannot name this participant.
            # The running-sum flows below only matter under message loss.
            for lam, xi, *_ in self._rounds.values(self._k, neighbours, default=lambda _n: own):
                lam_tilde += self.beta * (lam - self._lam)
                xi_tilde += self.beta * (xi - self._xi)
            return lam_tilde, xi_tilde
        values = self._rounds.value_map(self._k, neighbours, default=lambda _n: own)
        for n in neighbours:
            lam, xi, *_ = values[n]
            lam_tilde += self.beta * (lam - self._lam)
            xi_tilde += self._flows.flow(me, n, self._k, self._xi, xi, self.beta)
            xi_tilde += self._flows.correction(me, n)
        return lam_tilde, xi_tilde

    def _step(self, lam_tilde: np.ndarray, xi_tilde: np.ndarray) -> None:
        """Run one DEED-ADMM primal/dual update (Algorithm 1, lines 3-14).

        :param lam_tilde: W-weighted neighbour λ (line 3).
        :param xi_tilde: W-weighted neighbour ξ (line 3).
        """
        g = self.gamma

        # --- 2. x-update (supplier) and its box projection ---
        if self._x_fixed:
            x_new = self._x_min.copy()
            p_hat_x_new = x_new
            v_x_new = self._v_x
        else:
            # Stationarity: (∇²c + 2γ)x = bₓ − ∇c_linear, hence the − cost_lin.
            b_x = (
                g * self._x
                - lam_tilde
                + (g * self._p_hat_x - self._v_x)
                - (g / 2.0) * xi_tilde
                - self._cost_lin
            )
            x_new = self._H_x * b_x
            p_hat_x_new = np.clip(x_new + self._v_x / g, self._x_min, self._x_max)
            v_x_new = self._v_x + g * (x_new - p_hat_x_new)

        # --- 3. y/z-update (hub balance Aᵢy + Bz = dᵢ) ---
        b_y = g * self._y + lam_tilde + (g / 2.0) * xi_tilde
        y_new = self._update_y(b_y)

        # --- 4. Auxiliary and dual updates ---
        xi_new = xi_tilde + (x_new - self._x) - (y_new - self._y)
        lam_new = lam_tilde + (g / 2.0) * xi_new

        self._x = x_new
        self._p_hat_x = p_hat_x_new
        self._v_x = v_x_new
        self._y = y_new
        self._xi = xi_new
        self._lam = lam_new

    # ------------------------------------------------------------------
    # DistributedAlgorithm interface
    # ------------------------------------------------------------------

    def _broadcast(self, carrier: "Carrier", neighbours: list) -> None:
        """Send this round's (λ, ξ) and arm the round deadline."""
        me = address_of(carrier)
        for addr in neighbours:
            owned = self._flows.owner_sum(me, addr) if me is not None else None
            carrier.send_to_other(
                DEEDADMMMessage(
                    lam=self._lam.copy(),
                    xi=self._xi.copy(),
                    k=self._k,
                    data=self._data,
                    flow_sum=None if owned is None else np.copy(owned[1]),
                    flow_round=-1 if owned is None else owned[0],
                    converged=self._converged_flag,
                    radius=self._stop.radius,
                    stop_round=self._stop.carried,
                    run=self._run,
                ),
                addr,
            )
        k = self._k
        self._deadline.arm(carrier, lambda: self._on_deadline(carrier, k))

    def _notify_finished(self, carrier: "Carrier", neighbours: list) -> None:
        """Tell neighbours this agent stopped on convergence (k > max_iter)."""
        for addr in neighbours:
            carrier.send_to_other(
                DEEDADMMMessage(
                    lam=self._lam.copy(),
                    xi=self._xi.copy(),
                    k=self.max_iter + 1,
                    data=None,
                    converged=True,
                ),
                addr,
            )

    def _finish(self, carrier: "Carrier") -> None:
        self._deadline.cancel()
        self._running = False
        self._has_run = True
        self._rounds.reset()
        self.P = self._extract_schedule()
        self.finish_callback(self, carrier)

    def _advance(self, carrier: "Carrier", neighbours: list) -> None:
        """Run round ``self._k``; broadcast the next one or finish."""
        self._deadline.cancel()
        # --- Convergence stop: own flag and every neighbour's flag of a
        # complete round (all describe step k−1); under loss a neighbour's
        # last flag may also be one round old.
        if self.tol is not None and self.diameter > 1:
            recently_converged = self._deadline.enabled and self._rounds.all_recent(
                neighbours, self._k, 1, lambda v: v[2]
            )
            states = [
                (v[2], v[3], v[4])
                for v in self._rounds.values(
                    self._k, neighbours, default=lambda _n: (None, None, False, -1, -1)
                )
            ]
            complete = self._rounds.complete(self._k, len(neighbours))
            if self._stop.observe(
                self._k, self._converged_flag, states, complete or recently_converged
            ):
                # The common stop round certifies flags of earlier rounds;
                # report this agent's own flag now (flags are not monotone).
                self.converged = self._converged_flag
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
        schedule_before = self._extract_schedule()
        self._step(*self._mix(carrier, neighbours))
        if self.tol is not None:
            step = np.max(np.abs(self._extract_schedule() - schedule_before))
            self._converged_flag = bool(step < self.tol and np.max(np.abs(self._xi)) < self.tol)
        self._k += 1
        self._rounds.discard_before(self._k)
        if self._k >= self.max_iter:
            self.iterations = self._k
            if self.tol is not None:
                self.converged = False
            me = address_of(carrier)
            if self._settlement.enabled and me is not None:
                # The last broadcast (round max_iter − 1) carried the owner sums
                # only through round max_iter − 2: send the final ones now and
                # settle the edges this participant does not own.
                for addr in neighbours:
                    if self._flows.owns(me, addr):
                        self._send_final(carrier, me, addr)
                self._settlement.start(
                    carrier,
                    me,
                    neighbours,
                    self._flows,
                    final_round=self.max_iter - 1,
                    resend=lambda addr: self._send_final(carrier, me, addr),
                    on_settled=lambda: self._settled(carrier, neighbours),
                )
            else:
                self._finish(carrier)
            return
        self._broadcast(carrier, neighbours)

    def _send_final(self, carrier: "Carrier", me: Any, addr: Any) -> None:
        """Send this edge owner's final running sum to *addr*."""
        owned = self._flows.owner_sum(me, addr)
        if owned is None:
            return
        carrier.send_to_other(
            DEEDADMMMessage(
                lam=self._lam.copy(),
                xi=self._xi.copy(),
                k=self.max_iter,
                data=None,
                flow_sum=np.copy(owned[1]),
                flow_round=owned[0],
            ),
            addr,
        )

    def _settled(self, carrier: "Carrier", neighbours: list) -> None:
        """Apply the ξ flows still owed by edge owners, then finish."""
        me = address_of(carrier)
        for n in neighbours:
            self._xi = self._xi + self._flows.correction(me, n)
        self._finish(carrier)

    def _in_rounds(self) -> bool:
        """A run is in progress and has rounds left (not finished or settling)."""
        return self._running and self._k < self.max_iter

    def _on_deadline(self, carrier: "Carrier", k: int) -> None:
        """Round *k* timed out: advance with what arrived, plus last values."""
        if not self._in_rounds() or self._k != k:
            return
        neighbours = carrier.others()
        self._advance(carrier, neighbours)
        while self._in_rounds() and self._rounds.complete(self._k, len(neighbours)):
            self._advance(carrier, neighbours)

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: DEEDADMMMessage,
        meta: Any,
    ) -> None:
        """Handle one incoming DEED-ADMM message."""
        neighbours = carrier.others()

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
                DEEDADMMMessage(
                    lam=np.zeros(1), xi=np.zeros(1), k=self.max_iter, data=None, flow_ack=True
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

        # Notice of a neighbour that stopped on convergence (k > max_iter,
        # only with a round deadline): stop too.
        if message_data.k > self.max_iter:
            if self._running:
                self.converged = True
                self.iterations = self._k
                self._finish(carrier)
            return

        # After a finished run, late messages of that run (rounds > 0) are
        # ignored; a kick-off or a new run's round-0 message restarts it.
        if (
            not self._running
            and self._has_run
            and not message_data.initial
            and not (message_data.k == 0 and message_data.run > self._run)
        ):
            return

        # A kick-off for a run already started (from a neighbour's round-0
        # message, which can arrive first on a sparse topology) is ignored.
        if message_data.initial and self._running:
            return

        # --- Start of a run: external kick-off or first neighbour message ---
        if message_data.initial or not self._running:
            if self.beta * len(neighbours) >= 1.0:
                raise ValueError(
                    f"DEED-ADMM needs a positive self-weight 1 - beta*|N_i|, got "
                    f"beta={self.beta} with {len(neighbours)} neighbours; pass "
                    f"n_agents (or beta) matching the communication graph."
                )
            self._running = True
            self._run = max(self._run + 1, int(getattr(message_data, "run", 0)))
            self._reset_state()
            self._data = message_data.data
            if not neighbours:
                # Single agent: W = [1], no one to wait for.
                for _ in range(self.max_iter):
                    self._step(self._lam.copy(), self._xi.copy())
                    self._k += 1
                self._finish(carrier)
                return
            self._broadcast(carrier, neighbours)
            if message_data.initial:
                return

        # --- Synchronous rounds: advance on complete round-k state (or at
        # the round deadline); state from rounds ahead is buffered ---
        if has_sum:
            self._flows.offer(sender, message_data.flow_round, np.asarray(message_data.flow_sum))
        if message_data.k >= self.max_iter:
            return  # an owner's final sum, ahead of this participant: offer kept
        value = (
            np.asarray(message_data.lam),
            np.asarray(message_data.xi),
            bool(message_data.converged),
            int(message_data.radius),
            int(message_data.stop_round),
        )
        if message_data.k < self._k:
            self._rounds.remember(message_data.k, sender, value)
            return
        self._rounds.add(message_data.k, sender, value)
        while self._in_rounds() and self._rounds.complete(self._k, len(neighbours)):
            self._advance(carrier, neighbours)


# ---------------------------------------------------------------------------
# Storage variant
# ---------------------------------------------------------------------------


class DEEDADMMStorageAlgorithm(DEEDADMMAlgorithm):
    """DEED-ADMM hub with electricity storage (the paper's z block).

    zᵢ = [P^Sd; P^Sc] ∈ R²τ, both non-negative.  The hub balance becomes
    y + P^Sd − P^Sc = dᵢ (B = [I, −I]), and

    * Power bounds (5):  0 ≤ P^Sd ≤ p_discharge_max, 0 ≤ P^Sc ≤ p_charge_max
    * Energy dynamics (9): E_t = E_{t−1} + Δt·(η_c P^Sc_t − P^Sd_t/η_d)
    * Energy limits (8):  soc_min·E_max ≤ E_t ≤ soc_max·E_max
    * Terminal reserve (11): E_τ ≥ e_final·E_max

    all enter through Zᵢ = [I₂τ; Z₁ᵀ] (Z₁ᵀ z = E − E_0) as box constraints on
    ẑ.  The closed-form updates use H_z = (γZᵀZ)⁻¹ and
    Hᵢ = (Hᵧ I + B H_z Bᵀ)⁻¹, both precomputed.  By Theorem 1 the convex
    relaxation is exact when η_c·η_d < 1: optimal schedules never charge and
    discharge at once.

    The reported schedule is P = P̂^Sd − P̂^Sc (positive = discharge) from the
    box-projected ẑ; :attr:`E` is the matching energy trajectory.

    :param e_max: Energy capacity in MWh.
    :param p_charge_max: Maximum charging rate (MW, positive value).
    :param p_discharge_max: Maximum discharging rate (MW, positive value).
    :param eta_charge: Charging efficiency η_c ∈ (0, 1].
    :param eta_discharge: Discharging efficiency η_d ∈ (0, 1].
    :param e_initial: Initial state of charge as a fraction of e_max.
    :param e_final: Terminal reserve ω: E_τ ≥ e_final·e_max (defaults to
        e_initial, i.e. end at least as full as it started).
    :param soc_min: Minimum SOC fraction (lower bound).
    :param soc_max: Maximum SOC fraction (upper bound).
    :param discharge_cost: Linear cost per unit of discharged power.
    :param dt: Time-step length in hours (energy = power × dt).
    """

    def __init__(
        self,
        finish_callback: Callable,
        cost_quad: np.ndarray | float,
        cost_lin: np.ndarray | float,
        x_min: np.ndarray | float,
        x_max: np.ndarray | float,
        d_i: np.ndarray,
        gamma: float = 0.05,
        max_iter: int = 500,
        n_agents: int = 1,
        beta: float | None = None,
        round_timeout: float | None = None,
        *,
        e_max: float,
        p_charge_max: float,
        p_discharge_max: float,
        eta_charge: float = 0.95,
        eta_discharge: float = 0.95,
        e_initial: float = 0.5,
        e_final: float | None = None,
        soc_min: float = 0.0,
        soc_max: float = 1.0,
        discharge_cost: float = 0.0,
        dt: float = 1.0,
        tol: float | None = None,
        diameter: int = 1,
    ) -> None:
        self._e_max = float(e_max)
        self._p_charge_max = float(p_charge_max)
        self._p_discharge_max = float(p_discharge_max)
        self._eta_c = float(eta_charge)
        self._eta_d = float(eta_discharge)
        self._e_initial_frac = float(e_initial)
        self._e_final_frac = float(e_initial if e_final is None else e_final)
        self._soc_min = float(soc_min)
        self._soc_max = float(soc_max)
        self._dt = float(dt)
        super().__init__(
            finish_callback=finish_callback,
            cost_quad=cost_quad,
            cost_lin=cost_lin,
            x_min=x_min,
            x_max=x_max,
            d_i=d_i,
            gamma=gamma,
            max_iter=max_iter,
            n_agents=n_agents,
            beta=beta,
            round_timeout=round_timeout,
            tol=tol,
            diameter=diameter,
        )
        tau = self._tau
        g = self.gamma

        # Z₁ᵀ: row t gives E_{t+1} − E_0 = Δt·Σ_{s≤t}(η_c P^Sc_s − P^Sd_s/η_d).
        lower = np.tril(np.ones((tau, tau))) * self._dt
        self._Z1T = np.hstack([-lower / self._eta_d, lower * self._eta_c])
        self._Z = np.vstack([np.eye(2 * tau), self._Z1T])
        self._B = np.hstack([np.eye(tau), -np.eye(tau)])

        e0 = self._e_initial_frac * self._e_max
        e_lo = np.full(tau, self._soc_min * self._e_max)
        e_lo[-1] = max(e_lo[-1], self._e_final_frac * self._e_max)
        e_hi = np.full(tau, self._soc_max * self._e_max)
        if e_lo[-1] > e_hi[-1] or not self._soc_min * self._e_max <= e0 <= e_hi[0]:
            raise ValueError("infeasible storage energy limits")
        self._z_min = np.concatenate([np.zeros(2 * tau), e_lo - e0])
        self._z_max = np.concatenate(
            [
                np.full(tau, self._p_discharge_max),
                np.full(tau, self._p_charge_max),
                e_hi - e0,
            ]
        )
        # Linear storage cost enters b_z as − ∇c(z).
        self._c_z = np.concatenate([np.full(tau, float(discharge_cost)), np.zeros(tau)])

        self._H_z = np.linalg.inv(g * self._Z.T @ self._Z)
        self._H_i = np.linalg.inv(self._H_y * np.eye(tau) + self._B @ self._H_z @ self._B.T)

        # Public SOC trajectory (τ+1 points), written at termination.
        self.E: np.ndarray = np.full(tau + 1, e0)
        self._reset_storage_state()

    def _reset_storage_state(self) -> None:
        if not hasattr(self, "_Z"):
            return  # called from the base __init__ before the z block exists
        self._z = np.zeros(2 * self._tau)
        self._z_hat = self._Z @ self._z
        self._v_z = np.zeros(self._Z.shape[0])

    def _update_y(self, b_y: np.ndarray) -> np.ndarray:
        """Joint y/z update (Algorithm 1, lines 7, 9-11, 14 for the z block)."""
        g = self.gamma
        b_z = self._Z.T @ (g * self._z_hat - self._v_z) - self._c_z
        h_z_b_z = self._H_z @ b_z
        mu = self._H_i @ (self._d_i - self._H_y * b_y - self._B @ h_z_b_z)
        y_new = self._H_y * (b_y + mu)
        z_new = h_z_b_z + self._H_z @ (self._B.T @ mu)

        mz = self._Z @ z_new
        z_hat_new = np.clip(mz + self._v_z / g, self._z_min, self._z_max)
        self._v_z = self._v_z + g * (mz - z_hat_new)
        self._z = z_new
        self._z_hat = z_hat_new
        return y_new

    def _extract_schedule(self) -> np.ndarray:
        """Net discharge P̂^Sd − P̂^Sc and the energy path it implies."""
        tau = self._tau
        z = self._z_hat[: 2 * tau]
        self.E = np.concatenate([[0.0], self._Z1T @ z]) + self._e_initial_frac * self._e_max
        return z[:tau] - z[tau:]
