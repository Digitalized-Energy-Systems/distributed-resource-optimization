"""Gossip-based lexicographic cascade — coordinator-free sum-sharing ADMM.

Each participant runs the same deterministic sharing-ADMM cascade as
:mod:`~distributed_resource_optimization.algorithm.admm.lexicographic.coordinator`
(reusing the closed-form ``(z, sigma)`` cell update in
:mod:`.kernel`), but rebuilds
the shared ``x_bar`` from peers' broadcast contributions
:math:`x_i = r_i\\,c_i` instead of a coordinator-side reduction. Same
answer, no coordinator.

Message flow: the initiator's host role sends ``GossipCascadeStart`` to
its own participant, which broadcasts ``GossipCascadeInit`` to every
peer and then runs the cascade locally. Each iteration every
participant broadcasts ``GossipIter(x_i)`` and waits (with timeout) for
peers' contributions before the local ``(z, sigma, u, r)`` update; on
round completion it broadcasts ``done=True`` and commits ``r`` via
``on_commit``.

Crash-fault tolerant (not Byzantine — a peer broadcasting a wrong
``x_i`` poisons the iterate):

- **Stale rounds** — messages carry their issuing ``round_id``; peers
  reject anything older than their current round.
- **Peer death mid-round** — missing peers fall back to their
  last-seen ``x_i``, so the iterate stays feasible.
- **Round timeout** — exceeding ``round_timeout_s`` force-commits the
  current iterate (feasible-suboptimal).
- **Initiator death** — after the Init broadcast the initiator is just
  a participant; survivors terminate via the round-timeout watchdog and
  the next round's initiator (chosen externally) takes over.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ...core import DistributedAlgorithm, OptimizationMessage
from ..types import SectorDemand
from .kernel import _z_sigma_cell_update

if TYPE_CHECKING:
    from ....carrier.core import Carrier

logger = logging.getLogger(__name__)


async def _wait_first(*awaitables: Any) -> None:
    """Wait until the first of *awaitables* resolves, then cancel the rest.

    Used to race a peer-ready event against a clock-domain timeout so the
    timeout obeys the carrier's clock (simulation or wall-clock).
    """
    tasks = [asyncio.ensure_future(a) for a in awaitables]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class GossipCascadeStart(OptimizationMessage):
    """In-process trigger from the initiator's host role to its own
    participant; turned into a :class:`GossipCascadeInit` broadcast.

    :param round_id: Monotonic id; peers reject older rounds.
    :param participants: Frozen round membership (reachable peers + self).
    :param demands: Per-sector demand profiles (reference-kernel shape).
    """

    round_id: int
    participants: list[str]
    demands: list[SectorDemand]
    horizon: int = 1
    rho: float = 1.0
    inner_iters_max: int = 200
    inner_abs_tol: float = 1.0e-4
    r_regularization: float = 1.0e-1
    adaptive_rho: bool = True
    rho_mu: float = 10.0
    rho_tau: float = 2.0
    minimize_usage: bool = False
    iter_timeout_s: float = 0.3
    round_timeout_s: float = 8.0


@dataclass
class GossipCascadeInit(OptimizationMessage):
    """Initiator → every peer: lock the frame for this round.

    Recipients use this to bump their own ``round_id``, build the
    canonical sector ordering, and launch their local cascade loop.
    """

    round_id: int
    initiator_cp_id: str
    participants: list[str]
    sectors: list[str]
    horizon: int
    rho: float
    r_regularization: float
    adaptive_rho: bool
    rho_mu: float
    rho_tau: float
    demands: list[SectorDemand]
    iter_timeout_s: float
    round_timeout_s: float
    inner_iters_max: int
    inner_abs_tol: float
    minimize_usage: bool = False


@dataclass
class GossipIter(OptimizationMessage):
    """Peer → every peer: contribution at iteration ``iter_k`` of tier
    ``tier_index``.

    :param x_i: Contribution ``r_i · c_i`` on the round's sector
        ordering, shape ``(n_sec, horizon)``.
    :param rho: Sender's current adaptive :math:`\\rho` (kept in sync).
    :param r_change: ``max_k |r_i^k_new - r_i^k_prev|``; diagnostics only.
    :param done: Sender has committed; stop waiting for its Iters.
    """

    round_id: int
    tier_index: int
    iter_k: int
    cp_id: str
    x_i: np.ndarray
    rho: float
    r_change: float = 0.0
    done: bool = False


# ---------------------------------------------------------------------------
# Participant — pure DistributedAlgorithm
# ---------------------------------------------------------------------------


@dataclass
class _RoundCtx:
    """Per-round mutable state — torn down and rebuilt every Init."""

    round_id: int
    initiator_id: str
    participants: list[str]
    sectors: list[str]
    sec_idx: dict[str, int]
    horizon: int
    n_sec: int
    cap_vec: np.ndarray
    cap_norm_sq: float
    rho_init: float
    alpha: float
    minimize_usage: bool
    adaptive_rho: bool
    rho_mu: float
    rho_tau: float
    iter_timeout_s: float
    round_timeout_s: float
    inner_iters_max: int
    inner_abs_tol: float
    all_tiers: list[int]
    D: np.ndarray
    base_supply: np.ndarray
    # Mutable kernel state (advances per iter / per tier).
    tier_index: int = 0
    iter_k: int = 0
    x_self: np.ndarray = field(default_factory=lambda: np.zeros(0))
    r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    x_bar: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    u: np.ndarray = field(default_factory=lambda: np.zeros(0))
    theta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rho_f: float = 1.0
    # Gossip bookkeeping.
    last_x_per_peer: dict[str, np.ndarray] = field(default_factory=dict)
    reported_this_iter: set = field(default_factory=set)
    done_peers: set = field(default_factory=set)
    iter_ready: asyncio.Event | None = None


class GossipParticipant(DistributedAlgorithm):
    """Coordinator-free coupling-point peer for the gossip cascade.

    One per CP. Holds private ``capacity_by_sector``; demands and sector
    ordering arrive in :class:`GossipCascadeInit`. Capacity is never
    broadcast — only ``x_i = r_i · c_i``.

    :param cp_id: Stable identifier; appears in every Iter and the
        round's ``participants`` list.
    :param capacity_by_sector: Signed effective capacity, load
        convention (positive consumes, negative produces). Sectors
        outside the round's frame are ignored.
    :param on_commit: Optional ``(r, converged, iterations)`` callback
        fired at round end.
    :param warm_start: Seed each round's ADMM state (r/x/z/u/rho) from the
        previous committed round instead of zeros. With a per-round budget of
        ``round_timeout_s / iter_timeout_s`` iterations a cold start cannot
        converge for realistic N, so every commit is an early partial iterate;
        carry-over lets successive rounds continue one another.
    """

    def __init__(
        self,
        cp_id: str,
        capacity_by_sector: dict[str, float],
        *,
        on_commit: Callable[[np.ndarray, bool, int], None] | None = None,
        warm_start: bool = False,
    ) -> None:
        self.cp_id = cp_id
        self._capacity_by_sector = dict(capacity_by_sector)
        self._on_commit = on_commit
        self.warm_start = bool(warm_start)
        self._warm_state: dict[str, Any] | None = None
        self._ctx: _RoundCtx | None = None  # None until first Init/Start
        self._run_task: asyncio.Task | None = None
        # Monotone token: a _begin_round suspended in its cancel-await aborts
        # when a newer Init superseded it (else the older round's ctx would
        # overwrite the newer one and leave two live cascades).
        self._begin_seq: int = 0

    def is_round_active(self) -> bool:
        return self._run_task is not None and not self._run_task.done()

    # ------------------------------------------------------------------
    # DistributedAlgorithm contract
    # ------------------------------------------------------------------

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: Any,
        meta: Any,
    ) -> None:
        if isinstance(message_data, GossipCascadeStart):
            await self._on_start(carrier, message_data)
        elif isinstance(message_data, GossipCascadeInit):
            await self._on_init(carrier, message_data)
        elif isinstance(message_data, GossipIter):
            self._on_iter(message_data)

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    async def _on_start(self, carrier: "Carrier", start: GossipCascadeStart) -> None:
        """Initiator kickoff: broadcast Init to peers, then run the
        cascade locally as if we'd received our own Init."""
        if self._ctx is not None and start.round_id <= self._ctx.round_id:
            logger.debug(
                "[%s] dropping stale Start round_id=%d (current=%d)",
                self.cp_id,
                start.round_id,
                self._ctx.round_id,
            )
            return
        sectors = sorted({d.sector for d in start.demands})
        init = GossipCascadeInit(
            round_id=start.round_id,
            initiator_cp_id=self.cp_id,
            participants=list(start.participants),
            sectors=sectors,
            horizon=start.horizon,
            rho=start.rho,
            r_regularization=start.r_regularization,
            adaptive_rho=start.adaptive_rho,
            rho_mu=start.rho_mu,
            rho_tau=start.rho_tau,
            demands=list(start.demands),
            iter_timeout_s=start.iter_timeout_s,
            round_timeout_s=start.round_timeout_s,
            inner_iters_max=start.inner_iters_max,
            inner_abs_tol=start.inner_abs_tol,
            minimize_usage=start.minimize_usage,
        )
        for addr in carrier.others(self.cp_id):
            carrier.send_to_other(init, addr)
        await self._begin_round(carrier, init)

    async def _on_init(self, carrier: "Carrier", init: GossipCascadeInit) -> None:
        """Init arrived from an initiator peer.  Join the round."""
        if self._ctx is not None:
            if init.round_id < self._ctx.round_id:
                return  # stale
            if (
                init.round_id == self._ctx.round_id
                and self._ctx.initiator_id == init.initiator_cp_id
            ):
                return  # already in this round
        if self.cp_id not in init.participants:
            logger.debug(
                "[%s] not in Init participants %s — ignoring",
                self.cp_id,
                init.participants,
            )
            return
        await self._begin_round(carrier, init)

    def _on_iter(self, msg: GossipIter) -> None:
        """Peer Iter arrived.  Stash and signal the run loop if ready."""
        if self._ctx is None:
            return
        ctx = self._ctx
        if msg.round_id != ctx.round_id:
            return  # stale (older or split-brain)
        if msg.cp_id == self.cp_id:
            return  # echo of own broadcast
        if msg.cp_id not in ctx.participants:
            return  # not a member of this round
        x_i = np.asarray(msg.x_i, dtype=float)
        if x_i.shape != (ctx.n_sec, ctx.horizon):
            return  # malformed
        ctx.last_x_per_peer[msg.cp_id] = x_i
        if msg.done:
            ctx.done_peers.add(msg.cp_id)
        # Advance the loop once every live peer has reported this iter.
        if msg.tier_index == ctx.tier_index and msg.iter_k == ctx.iter_k:
            ctx.reported_this_iter.add(msg.cp_id)
            live = (set(ctx.participants) - {self.cp_id}) - ctx.done_peers
            if ctx.reported_this_iter >= live and ctx.iter_ready is not None:
                ctx.iter_ready.set()

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    async def _begin_round(self, carrier: "Carrier", init: GossipCascadeInit) -> None:
        """Cancel any in-flight round, build a fresh context, launch
        the cascade coroutine."""
        self._begin_seq += 1
        seq = self._begin_seq
        while self._run_task is not None and not self._run_task.done():
            task = self._run_task
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            if self._begin_seq != seq:
                return  # superseded by a newer Init while we awaited the cancel
        if self._begin_seq != seq:
            return
        self._ctx = self._build_ctx(init)
        # Driver task via the carrier; under simulation it is scheduler-tracked
        # and marked idle while it awaits peer replies (see Carrier.spawn).
        self._run_task = carrier.spawn(self._run_cascade(carrier))

    def _build_ctx(self, init: GossipCascadeInit) -> _RoundCtx:
        sectors = list(init.sectors)
        sec_idx = {s: i for i, s in enumerate(sectors)}
        H = int(init.horizon)
        n_sec = len(sectors)
        cap_vec = np.array(
            [float(self._capacity_by_sector.get(s, 0.0)) for s in sectors],
            dtype=float,
        )
        all_tiers = sorted({t for d in init.demands for t in d.demand_by_tier})
        n_tier_slots = max(len(all_tiers), 1)
        D = np.zeros((n_sec, n_tier_slots, H), dtype=float)
        base_supply = np.zeros((n_sec, H), dtype=float)
        tier_idx_map = {t: i for i, t in enumerate(all_tiers)}
        for d in init.demands:
            if d.sector not in sec_idx:
                continue
            s = sec_idx[d.sector]
            base_supply[s, :] = np.asarray(d.base_supply, dtype=float)
            for tier, arr in d.demand_by_tier.items():
                if tier in tier_idx_map:
                    D[s, tier_idx_map[tier], :] = np.asarray(arr, dtype=float)
        ctx = _RoundCtx(
            round_id=int(init.round_id),
            initiator_id=str(init.initiator_cp_id),
            participants=list(init.participants),
            sectors=sectors,
            sec_idx=sec_idx,
            horizon=H,
            n_sec=n_sec,
            cap_vec=cap_vec,
            cap_norm_sq=float(cap_vec @ cap_vec),
            rho_init=float(init.rho),
            alpha=float(init.r_regularization),
            minimize_usage=bool(init.minimize_usage),
            adaptive_rho=bool(init.adaptive_rho),
            rho_mu=float(init.rho_mu),
            rho_tau=float(init.rho_tau),
            iter_timeout_s=float(init.iter_timeout_s),
            round_timeout_s=float(init.round_timeout_s),
            inner_iters_max=int(init.inner_iters_max),
            inner_abs_tol=float(init.inner_abs_tol),
            all_tiers=all_tiers,
            D=D,
            base_supply=base_supply,
            x_self=np.zeros((n_sec, H), dtype=float),
            r=np.zeros(H, dtype=float),
            x_bar=np.zeros((n_sec, H), dtype=float),
            z=np.zeros((n_sec, H), dtype=float),
            u=np.zeros((n_sec, H), dtype=float),
            theta=np.zeros((n_sec, H), dtype=float),
            rho_f=float(init.rho),
        )
        if self.warm_start:
            self._apply_warm_state(ctx)
        return ctx

    def _apply_warm_state(self, ctx: _RoundCtx) -> None:
        """Seed *ctx* from the previous committed round's ADMM state.

        Rows are remapped by sector name (the round frame can change between
        rounds); ``theta`` stays zero — it is the within-round cascade
        accumulator, not ADMM state. Skipped on horizon mismatch or non-finite
        saved state. When the participant set changed, only ``r`` is carried
        (it is scale-free against the current capacity frame): the saved z/u
        approximate a consensus over the OLD group at the OLD problem scale,
        and a stale dual can pin r at its clip bound for several ~10-iteration
        rounds — an over-commit transient landing exactly at failure time.
        """
        ws = self._warm_state
        if ws is None or int(ws["horizon"]) != ctx.horizon:
            return
        if not all(np.all(np.isfinite(ws[k])) for k in ("r", "z", "u")):
            self._warm_state = None
            return
        ctx.r = np.clip(np.asarray(ws["r"], dtype=float).copy(), 0.0, 1.0)
        # x_self consistent with the warm r under the CURRENT capacity frame.
        ctx.x_self = ctx.r[np.newaxis, :] * ctx.cap_vec[:, np.newaxis]
        if sorted(ctx.participants) != ws.get("participants"):
            return  # consensus frame changed: keep r, drop stale duals/rho
        old_idx = {s: i for i, s in enumerate(ws["sectors"])}
        for s, i_new in ctx.sec_idx.items():
            i_old = old_idx.get(s)
            if i_old is None:
                continue
            ctx.z[i_new, :] = ws["z"][i_old, :]
            ctx.u[i_new, :] = ws["u"][i_old, :]
        rho_f = float(ws.get("rho_f", ctx.rho_f))
        if rho_f > 0.0 and math.isfinite(rho_f):
            ctx.rho_f = rho_f

    def _save_warm_state(self, ctx: _RoundCtx) -> None:
        self._warm_state = {
            "sectors": list(ctx.sectors),
            "participants": sorted(ctx.participants),
            "horizon": int(ctx.horizon),
            "r": ctx.r.copy(),
            "z": ctx.z.copy(),
            "u": ctx.u.copy(),
            "rho_f": float(ctx.rho_f),
        }

    async def _run_cascade(self, carrier: "Carrier") -> None:
        """Outer lex-cascade + inner sharing-ADMM run locally, mirroring
        :func:`solve_cp_distributed_lexicographic_cascade` except
        ``x_bar`` is built from peer Iter broadcasts (held over on
        timeout) instead of per-CP local x updates."""
        ctx = self._ctx
        if ctx is None:
            return
        round_start = carrier.now()  # carrier clock: simulation or wall-clock
        N = len(ctx.participants)
        converged_total = True
        total_iters = 0

        try:
            tier_loop_count = max(len(ctx.all_tiers), 1)
            for tier_pos in range(tier_loop_count):
                ctx.tier_index = tier_pos
                if not ctx.all_tiers:
                    break  # no demand at all
                D_tau = ctx.D[:, tier_pos, :]
                slack_max = ctx.base_supply - ctx.theta
                sigma = np.zeros_like(D_tau)
                converged_round = False

                for iter_k in range(ctx.inner_iters_max):
                    if carrier.now() - round_start > ctx.round_timeout_s:
                        converged_total = False
                        logger.warning(
                            "[%s] gossip cascade round %d timed out at tier %d iter %d",
                            self.cp_id,
                            ctx.round_id,
                            tier_pos,
                            iter_k,
                        )
                        break
                    ctx.iter_k = iter_k
                    ctx.reported_this_iter = set()
                    ctx.iter_ready = asyncio.Event()

                    # No live peers left to wait for → skip the timeout.
                    live_peers_now = set(ctx.participants) - {self.cp_id} - ctx.done_peers
                    if not live_peers_now:
                        ctx.iter_ready.set()

                    # Broadcast own current x_i.
                    iter_msg = GossipIter(
                        round_id=ctx.round_id,
                        tier_index=tier_pos,
                        iter_k=iter_k,
                        cp_id=self.cp_id,
                        x_i=ctx.x_self.copy(),
                        rho=ctx.rho_f,
                        r_change=0.0,
                        done=False,
                    )
                    for addr in carrier.others(self.cp_id):
                        carrier.send_to_other(iter_msg, addr)

                    # Await peers, or fall through on timeout. The timeout
                    # runs on the carrier's clock, so under a simulation it is
                    # a discrete-step wakeup rather than real wall-clock time.
                    if N > 1:
                        await _wait_first(
                            ctx.iter_ready.wait(),
                            carrier.sleep(ctx.iter_timeout_s),
                        )

                    # x_bar = mean of own x_self + peers' last-known x.
                    x_sum = ctx.x_self.copy()
                    for cp in ctx.participants:
                        if cp == self.cp_id:
                            continue
                        last = ctx.last_x_per_peer.get(cp)
                        if last is not None:
                            x_sum = x_sum + last
                    x_bar_new = x_sum / float(N if N > 0 else 1)

                    # Shared (z, sigma) cell update.
                    z_prev = ctx.z.copy()
                    target_z = x_bar_new + ctx.u
                    for s in range(ctx.n_sec):
                        for k in range(ctx.horizon):
                            z_sk, sig_sk = _z_sigma_cell_update(
                                float(target_z[s, k]),
                                float(slack_max[s, k]),
                                float(D_tau[s, k]),
                                N if N > 0 else 1,
                                ctx.rho_f,
                            )
                            ctx.z[s, k] = z_sk
                            sigma[s, k] = sig_sk

                    # Shared scaled-dual update.
                    ctx.u = ctx.u + x_bar_new - ctx.z

                    # Own r-update (closed form). minimize_usage -> ridge
                    # toward 0 (drop the alpha r_prev term) for min-usage /
                    # no overshoot; else proximal toward r_prev. See kernel.
                    r_prev = ctx.r.copy()
                    if ctx.cap_norm_sq > 0:
                        target_x = ctx.x_self - x_bar_new + ctx.z - ctx.u
                        den = ctx.rho_f * ctx.cap_norm_sq + ctx.alpha
                        for k in range(ctx.horizon):
                            num = ctx.rho_f * float(ctx.cap_vec @ target_x[:, k])
                            if not ctx.minimize_usage:
                                num += ctx.alpha * r_prev[k]
                            r_k = num / den
                            if r_k < 0.0:
                                r_k = 0.0
                            elif r_k > 1.0:
                                r_k = 1.0
                            ctx.r[k] = r_k
                            ctx.x_self[:, k] = r_k * ctx.cap_vec
                    # else: zero-capacity CP contributes nothing.

                    # Residuals + local convergence.
                    primal_res = float(np.max(np.abs(x_bar_new - ctx.z))) if ctx.z.size else 0.0
                    dual_res = ctx.rho_f * (
                        float(np.max(np.abs(ctx.z - z_prev))) if ctx.z.size else 0.0
                    )
                    r_change = float(np.max(np.abs(ctx.r - r_prev))) if ctx.r.size else 0.0
                    ctx.x_bar = x_bar_new
                    total_iters += 1

                    if (
                        primal_res < ctx.inner_abs_tol
                        and dual_res < ctx.inner_abs_tol
                        and r_change < ctx.inner_abs_tol
                    ):
                        converged_round = True
                        break

                    # Adaptive rho (Boyd 3.4.1 residual balancing).
                    if ctx.adaptive_rho:
                        if primal_res > ctx.rho_mu * dual_res:
                            ctx.rho_f *= ctx.rho_tau
                            ctx.u /= ctx.rho_tau
                        elif dual_res > ctx.rho_mu * primal_res:
                            ctx.rho_f /= ctx.rho_tau
                            ctx.u *= ctx.rho_tau

                if not converged_round:
                    converged_total = False
                ctx.theta = ctx.theta + sigma

            # Round complete — tell peers to stop waiting on us.
            done_msg = GossipIter(
                round_id=ctx.round_id,
                tier_index=max(0, tier_loop_count - 1),
                iter_k=ctx.iter_k,
                cp_id=self.cp_id,
                x_i=ctx.x_self.copy(),
                rho=ctx.rho_f,
                r_change=0.0,
                done=True,
            )
            for addr in carrier.others(self.cp_id):
                carrier.send_to_other(done_msg, addr)

            if self.warm_start:
                self._save_warm_state(ctx)
            if self._on_commit is not None:
                self._on_commit(ctx.r.copy(), converged_total, total_iters)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] gossip cascade run failed: %s",
                self.cp_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Public read-only accessors (for tests + the SCARE role)
    # ------------------------------------------------------------------

    @property
    def r(self) -> np.ndarray:
        """Most-recent committed regulation factor (horizon-shaped)."""
        if self._ctx is None:
            return np.zeros(0)
        return self._ctx.r.copy()

    @property
    def current_round_id(self) -> int:
        return self._ctx.round_id if self._ctx is not None else -1


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_gossip_cascade_participant(
    cp_id: str,
    capacity_by_sector: dict[str, float],
    *,
    on_commit: Callable[[np.ndarray, bool, int], None] | None = None,
    warm_start: bool = False,
) -> GossipParticipant:
    """Create a :class:`GossipParticipant` for the cross-sector cascade."""
    return GossipParticipant(
        cp_id=cp_id,
        capacity_by_sector=capacity_by_sector,
        on_commit=on_commit,
        warm_start=warm_start,
    )


def create_gossip_cascade_start(
    *,
    round_id: int,
    participants: list[str],
    demands: list[SectorDemand],
    horizon: int = 1,
    rho: float = 1.0,
    inner_iters_max: int = 200,
    inner_abs_tol: float = 1.0e-4,
    r_regularization: float = 1.0e-1,
    adaptive_rho: bool = True,
    rho_mu: float = 10.0,
    rho_tau: float = 2.0,
    minimize_usage: bool = False,
    iter_timeout_s: float = 0.3,
    round_timeout_s: float = 8.0,
) -> GossipCascadeStart:
    """Build the in-process kickoff message for the initiator's participant.

    :param round_id: Monotonic round id from the initiator's host role.
    :param participants: cp_ids in this round (incl. the initiator).
    :param demands: Per-sector demand profiles + base supply.
    :param iter_timeout_s: Wait for peer Iters before using held-over values.
    :param round_timeout_s: Hard round cap; on expiry commit the current
        iterate (feasible-suboptimal).
    """
    return GossipCascadeStart(
        round_id=int(round_id),
        participants=list(participants),
        demands=list(demands),
        horizon=int(horizon),
        rho=float(rho),
        inner_iters_max=int(inner_iters_max),
        inner_abs_tol=float(inner_abs_tol),
        r_regularization=float(r_regularization),
        adaptive_rho=bool(adaptive_rho),
        rho_mu=float(rho_mu),
        rho_tau=float(rho_tau),
        minimize_usage=bool(minimize_usage),
        iter_timeout_s=float(iter_timeout_s),
        round_timeout_s=float(round_timeout_s),
    )
