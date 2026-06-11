"""Distributed primal-dual QP via token-passing gossip.

A fully distributed algorithm that solves the priority-weighted QP

.. math::

    \\min_{\\delta} \\;\\sum_i \\frac{1}{2 a_i}\\,\\delta_i^2
    \\quad \\text{s.t.} \\quad \\sum_i \\delta_i = T,
                             \\;\\; \\delta_i \\in [\\underline{\\delta}_i,
                                                   \\overline{\\delta}_i]

via a primal-dual gossip with a single shared dual variable
``lambda``.  At the unconstrained KKT optimum
``lambda* = T / sum(a_j)`` and ``delta_i* = a_i * lambda*``.

Mechanism
---------

A single *token* (:class:`GossipQPMessage`) circulates in a
deterministic, hash-driven order. On each receive the holder:

* **Primal (closed form)**: ``delta_i = clamp(a_i * lambda, dmin_i, dmax_i)``.
* **Ledger merge**: records its own entry; peer entries with a higher
  counter overwrite local copies (defends against double-counting in
  cyclic forwarding graphs).
* **Dual (Robbins-Monro)**:
  ``lambda += gamma_k * (T - sum(delta_j)) / sum(a_j)`` with the
  denominator restricted to *unsaturated* peers (only they can move).
* **Forward** to one next-hop chosen by SHA of ``(negotiation_id,
  counter)`` so concurrent originators route identically.

Terminates when ``|T - sum(delta_j)|`` falls below
``termination_tolerance`` or the hop counter exceeds ``max_hops``; an
optional rolling gap-window stall detector abandons early on no
movement.

Byzantine robustness: each ledger entry is clipped to
``byzantine_cap_multiple * max(|T|, 1)`` on merge (default ``5.0``,
matching the original SCARE deployment) so one bad participant cannot
poison the aggregate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ...core import DistributedAlgorithm, OptimizationMessage

if TYPE_CHECKING:
    from ....carrier.core import Carrier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deterministic_next(neighbours: list, negotiation_id: str, counter: int) -> Any:
    """SHA-driven deterministic next-hop, replacing ``random.choice`` so
    the same ``(negotiation_id, counter)`` always picks the same
    neighbour (deterministic convergence, reproducible tests)."""
    if not neighbours:
        return None
    h = hashlib.sha256(f"{negotiation_id}:{counter}".encode()).digest()
    idx = int.from_bytes(h[:4], "big") % len(neighbours)
    return neighbours[idx]


def _is_saturated(delta: float, dmin: float, dmax: float) -> bool:
    """True iff *delta* is within tolerance of either box bound. The
    tolerance scales with box magnitude so large boxes don't misclassify
    near-bound values as unsaturated on float noise."""
    sat_tol = 1e-9 + 1e-6 * max(abs(dmin), abs(dmax), 1.0)
    return delta <= dmin + sat_tol or delta >= dmax - sat_tol


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class LedgerEntry:
    """Per-participant entry in the gossip ledger.

    :param delta: Participant's committed primal value.
    :param counter: Monotone counter at which *delta* was set.
    :param weight: Per-participant responsiveness ``a_i`` (priority
        weight); used by every holder to estimate ``sum(a_j)`` for the
        dual normaliser.
    :param saturated: ``True`` when *delta* sits at one of the box
        bounds — saturated entries are excluded from the dual
        normaliser because their derivative w.r.t. ``lambda`` is zero.
    """

    delta: float
    counter: int
    weight: float
    saturated: bool


@dataclass
class GossipQPMessage(OptimizationMessage):
    """Token exchanged between participants in a single gossip round.

    :param negotiation_id: UUID-like identifier of this gossip
        instance; used to disambiguate concurrent rounds and to seed
        the deterministic next-hop hash.
    :param target: The global imbalance ``T`` to clear.  Carried in
        every message so peers joining mid-flight learn it without an
        extra round-trip.
    :param counter: Hop counter (monotone over the token's lifetime).
    :param memory: Shared ledger of per-participant entries.
    :param dual_lambda: Current shared dual variable ``lambda``.
    :param initial: ``True`` for the kick-off message sent by the
        originator; receivers (re-)initialise their local state on
        seeing it.
    """

    negotiation_id: str
    target: float
    counter: int
    memory: dict[str, LedgerEntry]
    dual_lambda: float
    initial: bool = False


@dataclass
class GossipQPFinished(OptimizationMessage):
    """Broadcast announcing the terminal of a gossip round.

    Sent by whichever participant first detects convergence,
    max-hops, or stall.  Receivers mark themselves finished and the
    originator fires its :attr:`GossipQPAlgorithm.finish_callback`.

    :param negotiation_id: Identifier of the round being terminated.
    :param terminal: One of ``"converged"``, ``"max_hops"``,
        ``"stalled"``.
    :param memory: Final ledger snapshot.
    :param dual_lambda: Final dual variable.
    """

    negotiation_id: str
    terminal: str
    memory: dict[str, LedgerEntry]
    dual_lambda: float


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


class GossipQPAlgorithm(DistributedAlgorithm):
    """Distributed primal-dual QP participant.

    :param a: Local responsiveness ``a_i`` (priority weight, ``> 0``).
        Two participants with the same ``a`` move identical ``delta``
        per unit of ``lambda``; smaller-box participants saturate
        earlier, which is the correct waterfall behaviour.
    :param dmin: Lower bound on this participant's ``delta``.
    :param dmax: Upper bound on this participant's ``delta``.
    :param convergence_rate: Robbins-Monro step ``gamma_s``.
    :param step_decay_k0: ``k0`` in the diminishing step
        ``gamma_k = gamma_s / (1 + k / k0)``.  Satisfies
        ``sum gamma_k = inf`` and ``sum gamma_k^2 < inf`` so the
        dynamics converge almost surely under bounded noise.
    :param max_hops: Cap on the token's hop count.
    :param termination_tolerance: Originator declares convergence
        once ``|T - sum(delta)| < termination_tolerance``.
    :param byzantine_cap_multiple: Per-entry magnitude cap as a
        multiple of ``max(|T|, 1)``.  Set to ``inf`` to disable.
    :param stall_window: When ``> 0``, the originator declares stall
        after ``stall_window`` consecutive residuals whose range is
        below ``stall_tolerance``.  ``0`` disables stall detection.
    :param stall_tolerance: Range threshold for stall detection.
    :param finish_callback: Invoked as ``finish_callback(algorithm,
        carrier, terminal)`` once the originator finalises the run.
        ``terminal`` is one of ``"converged"``, ``"max_hops"``, or
        ``"stalled"``.
    """

    def __init__(
        self,
        a: float,
        dmin: float,
        dmax: float,
        *,
        convergence_rate: float = 0.5,
        step_decay_k0: int = 20,
        max_hops: int = 100,
        termination_tolerance: float = 1e-5,
        byzantine_cap_multiple: float = 5.0,
        stall_window: int = 0,
        stall_tolerance: float = 1e-6,
        finish_callback: Callable[..., None] | None = None,
    ) -> None:
        if a <= 0.0:
            raise ValueError("a must be > 0 (use a higher weight for higher priority)")
        if dmin > dmax:
            raise ValueError(f"dmin ({dmin}) must be <= dmax ({dmax})")
        self.a = float(a)
        self.dmin = float(dmin)
        self.dmax = float(dmax)
        self.convergence_rate = float(convergence_rate)
        self.step_decay_k0 = max(1, int(step_decay_k0))
        self.max_hops = int(max_hops)
        self.termination_tolerance = float(termination_tolerance)
        self.byzantine_cap_multiple = float(byzantine_cap_multiple)
        self.stall_window = int(stall_window)
        self.stall_tolerance = float(stall_tolerance)
        self.finish_callback = finish_callback

        self.delta: float = 0.0
        self.negotiation_id: str | None = None
        self.is_originator: bool = False
        self.target: float = 0.0
        self.memory: dict[str, LedgerEntry] = {}
        self.dual_lambda: float = 0.0
        self.finished: bool = False
        self._gap_window: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_delta(self) -> float:
        """``sum(memory[a].delta)`` across the current ledger."""
        return sum(entry.delta for entry in self.memory.values())

    @property
    def residual(self) -> float:
        """``target - total_delta`` (zero at convergence)."""
        return self.target - self.total_delta

    # ------------------------------------------------------------------
    # Algorithm
    # ------------------------------------------------------------------

    def _step_size(self, counter: int) -> float:
        return self.convergence_rate / (1.0 + max(0, counter) / self.step_decay_k0)

    def _primal(self) -> float:
        return max(self.dmin, min(self.dmax, self.a * self.dual_lambda))

    def _self_key(self, carrier: "Carrier") -> str:
        return str(carrier.get_address())

    def _initial_lambda(self, target: float, n_neighbours: int) -> float:
        """Seed ``lambda_0 = T / ((n_neighbours + 1) * a_self)`` so the
        originator's first ``delta`` targets its fair share
        ``T / (n_neighbours + 1)``. Clamped to ``|T|`` against
        pathological weights injecting an unbounded first step."""
        n_seed = max(2, n_neighbours + 1)
        lam = target / (n_seed * self.a)
        return max(-abs(target), min(abs(target), lam))

    def _merge(self, incoming: dict[str, LedgerEntry]) -> None:
        cap_byz = self.byzantine_cap_multiple * max(abs(self.target), 1.0)
        for key, entry in incoming.items():
            local = self.memory.get(key)
            if local is None or local.counter < entry.counter:
                delta = entry.delta
                if delta > cap_byz:
                    delta = cap_byz
                elif delta < -cap_byz:
                    delta = -cap_byz
                self.memory[key] = LedgerEntry(
                    delta=delta,
                    counter=entry.counter,
                    weight=entry.weight,
                    saturated=entry.saturated,
                )

    def _update_dual(self, counter: int) -> None:
        sum_a_unsaturated = sum(
            entry.weight for entry in self.memory.values() if not entry.saturated
        )
        if sum_a_unsaturated <= 0.0:
            sum_a_unsaturated = sum(entry.weight for entry in self.memory.values()) or 1.0
        self.dual_lambda += self._step_size(counter) * self.residual / sum_a_unsaturated

    def _check_stall(self, gap: float) -> bool:
        if self.stall_window <= 0:
            return False
        self._gap_window.append(gap)
        if len(self._gap_window) > self.stall_window:
            del self._gap_window[0]
        if len(self._gap_window) < self.stall_window:
            return False
        rng = max(self._gap_window) - min(self._gap_window)
        return rng < self.stall_tolerance and abs(gap) > self.termination_tolerance

    def _broadcast_finished(self, carrier: "Carrier", terminal: str) -> None:
        """Send a :class:`GossipQPFinished` to every reachable neighbour."""
        if self.negotiation_id is None:
            return
        snapshot = GossipQPFinished(
            negotiation_id=self.negotiation_id,
            terminal=terminal,
            memory={k: LedgerEntry(**vars(v)) for k, v in self.memory.items()},
            dual_lambda=self.dual_lambda,
        )
        for addr in carrier.others(self._self_key(carrier)):
            carrier.send_to_other(snapshot, addr)

    def _finish(self, carrier: "Carrier", terminal: str, *, broadcast: bool = True) -> None:
        if self.finished:
            return
        self.finished = True
        if broadcast:
            self._broadcast_finished(carrier, terminal)
        if self.finish_callback is not None and self.is_originator:
            self.finish_callback(self, carrier, terminal)

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: Any,
        meta: Any,
    ) -> None:
        if isinstance(message_data, GossipQPFinished):
            if self.negotiation_id != message_data.negotiation_id:
                return
            self._merge(message_data.memory)
            self.dual_lambda = float(message_data.dual_lambda)
            self._finish(carrier, message_data.terminal, broadcast=False)
            return

        if self.finished and not message_data.initial:
            return

        counter = message_data.counter + 1
        if counter > self.max_hops + 1:
            return

        self_key = self._self_key(carrier)

        if message_data.initial or self.negotiation_id != message_data.negotiation_id:
            self.negotiation_id = message_data.negotiation_id
            self.target = float(message_data.target)
            self.memory = {}
            self.dual_lambda = float(message_data.dual_lambda)
            self.delta = 0.0
            self._gap_window = []
            self.finished = False
            self.is_originator = bool(message_data.initial) and self.is_originator
            if self_key not in self.memory:
                self.memory[self_key] = LedgerEntry(
                    delta=0.0, counter=0, weight=self.a, saturated=False
                )

        self.dual_lambda = float(message_data.dual_lambda)
        self._merge(message_data.memory)

        self.delta = self._primal()
        saturated = _is_saturated(self.delta, self.dmin, self.dmax)
        self.memory[self_key] = LedgerEntry(
            delta=self.delta, counter=counter, weight=self.a, saturated=saturated
        )

        self._update_dual(counter)

        gap = self.residual

        if self.is_originator and self._check_stall(gap):
            self._finish(carrier, "stalled")
            return

        if abs(gap) <= self.termination_tolerance:
            self._finish(carrier, "converged")
            return
        if counter >= self.max_hops:
            self._finish(carrier, "max_hops")
            return

        neighbours = carrier.others(self_key)
        if not neighbours:
            self._finish(carrier, "converged")
            return

        next_addr = _deterministic_next(neighbours, self.negotiation_id, counter)
        fwd = GossipQPMessage(
            negotiation_id=self.negotiation_id,
            target=self.target,
            counter=counter,
            memory={k: LedgerEntry(**vars(v)) for k, v in self.memory.items()},
            dual_lambda=self.dual_lambda,
        )
        carrier.send_to_other(fwd, next_addr)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_distributed_qp_participant(
    a: float,
    dmin: float,
    dmax: float,
    *,
    convergence_rate: float = 0.5,
    step_decay_k0: int = 20,
    max_hops: int = 100,
    termination_tolerance: float = 1e-5,
    byzantine_cap_multiple: float = 5.0,
    stall_window: int = 0,
    stall_tolerance: float = 1e-6,
    finish_callback: Callable[..., None] | None = None,
) -> GossipQPAlgorithm:
    """Create a :class:`GossipQPAlgorithm` participant.

    See :class:`GossipQPAlgorithm` for parameter semantics.
    """
    return GossipQPAlgorithm(
        a=a,
        dmin=dmin,
        dmax=dmax,
        convergence_rate=convergence_rate,
        step_decay_k0=step_decay_k0,
        max_hops=max_hops,
        termination_tolerance=termination_tolerance,
        byzantine_cap_multiple=byzantine_cap_multiple,
        stall_window=stall_window,
        stall_tolerance=stall_tolerance,
        finish_callback=finish_callback,
    )


def create_distributed_qp_start(
    originator: GossipQPAlgorithm,
    target: float,
    *,
    negotiation_id: str,
    n_neighbours: int,
) -> GossipQPMessage:
    """Create the kick-off :class:`GossipQPMessage`.

    The message must be sent from the *originator*'s carrier so the
    deterministic next-hop and the ``is_originator`` flag are seeded
    correctly.  ``n_neighbours`` is used only to derive the initial
    ``lambda`` seed (the originator's first-step share).

    :param originator: The participant that initiates the gossip; the
        kick-off message must be delivered to *one of its neighbours*
        (not back to itself).
    :param target: Imbalance ``T`` to clear.
    :param negotiation_id: Stable identifier for this gossip instance.
    :param n_neighbours: Number of neighbours the originator can reach
        — used purely to derive ``lambda_0``.
    """
    originator.is_originator = True
    originator.negotiation_id = negotiation_id
    originator.target = float(target)
    originator.dual_lambda = originator._initial_lambda(float(target), n_neighbours)
    originator.memory = {}
    originator.finished = False
    originator._gap_window = []
    return GossipQPMessage(
        negotiation_id=negotiation_id,
        target=float(target),
        counter=0,
        memory={},
        dual_lambda=originator.dual_lambda,
        initial=True,
    )
