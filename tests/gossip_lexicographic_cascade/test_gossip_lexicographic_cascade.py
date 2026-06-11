"""Gossip lexicographic-cascade tests.

The gossip variant runs the same sum-sharing-ADMM cascade as the
reference :func:`solve_cp_distributed_lexicographic_cascade`, but
without a coordinator: every CP broadcasts its own per-iteration
contribution and reconstructs the shared state locally.  These tests
pin parity with the reference on healthy runs and the failure-mode
contracts (stale rounds, peer death, iter timeout).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from distributed_resource_optimization import (
    ActorContainer,
    CPSpec,
    GossipCascadeInit,
    GossipCascadeStart,
    GossipIter,
    GossipParticipant,
    SectorDemand,
    SimpleCarrier,
    cid,
    create_gossip_cascade_participant,
    create_gossip_cascade_start,
    solve_cp_distributed_lexicographic_cascade,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _p2h_spec(cp_id: str, *, electricity: float, heat: float) -> CPSpec:
    return CPSpec(
        cp_id=cp_id,
        capacity_by_sector={"electricity": electricity, "heat": heat},
    )


def _heat_demand(*, demand_mw: float, base_mw: float = 0.0) -> SectorDemand:
    return SectorDemand(
        sector="heat",
        demand_by_tier={1: np.array([demand_mw])},
        base_supply=np.array([base_mw]),
    )


def _electricity_slack(base_mw: float) -> SectorDemand:
    """Electricity sector with abundant base supply (no demand)."""
    return SectorDemand(
        sector="electricity",
        demand_by_tier={1: np.zeros(1)},
        base_supply=np.array([base_mw]),
    )


async def _run_gossip(
    cps: list[CPSpec],
    demands: list[SectorDemand],
    *,
    iter_timeout_s: float = 5.0,
    round_timeout_s: float = 30.0,
    inner_iters_max: int = 500,
) -> dict[str, np.ndarray]:
    """Set up N gossip participants on a SimpleCarrier mesh, kick the
    initiator (cps[0]), wait until every participant has called its
    on_commit callback, return ``{cp_id: committed_r}``.
    """
    results: dict[str, np.ndarray] = {}
    commit_events: dict[str, asyncio.Event] = {}

    def _make_callback(cp_id: str):
        def _cb(r: np.ndarray, converged: bool, iters: int) -> None:
            results[cp_id] = r
            commit_events[cp_id].set()
        return _cb

    container = ActorContainer()
    participants: list[GossipParticipant] = []
    for spec in cps:
        commit_events[spec.cp_id] = asyncio.Event()
        p = create_gossip_cascade_participant(
            cp_id=spec.cp_id,
            capacity_by_sector=spec.capacity_by_sector,
            on_commit=_make_callback(spec.cp_id),
        )
        participants.append(p)

    carriers = [SimpleCarrier(container, p) for p in participants]
    # Initiator is cps[0]; kick it with an in-process Start by routing
    # through its own carrier (sender == receiver, so the participant
    # processes the message in its on_exchange_message).
    initiator_carrier = carriers[0]
    start = create_gossip_cascade_start(
        round_id=1,
        participants=[c.cp_id for c in cps],
        demands=demands,
        horizon=1,
        rho=1.0,
        inner_iters_max=inner_iters_max,
        inner_abs_tol=1.0e-5,
        r_regularization=0.1,
        iter_timeout_s=iter_timeout_s,
        round_timeout_s=round_timeout_s,
    )
    initiator_carrier.send_to_other(start, cid(initiator_carrier))

    # Wait for every participant to commit, with a safety timeout.
    await asyncio.wait_for(
        asyncio.gather(*[ev.wait() for ev in commit_events.values()]),
        timeout=round_timeout_s + 5.0,
    )
    return results


# ---------------------------------------------------------------------------
# Parity with the reference replicated kernel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_p2h_matches_reference_kernel():
    """One P2H, plenty of electricity, heat demand 3 — both r ≈ 1 ⇒ 3 MW heat."""
    cps = [_p2h_spec("p2h-1", electricity=1.0, heat=-1.0)]
    demands = [_heat_demand(demand_mw=3.0), _electricity_slack(base_mw=10.0)]

    reference = solve_cp_distributed_lexicographic_cascade(
        cps, demands, inner_iters_max=500
    )
    gossip = await _run_gossip(cps, demands)

    assert "p2h-1" in gossip
    assert gossip["p2h-1"][0] == pytest.approx(
        reference.factor_by_cp["p2h-1"][0], abs=5e-3
    )


@pytest.mark.asyncio
async def test_three_cps_matches_reference():
    """3 mixed CPs — gossip and reference must agree on every CP."""
    cps = [
        _p2h_spec("p2h-a", electricity=2.0, heat=-1.5),
        _p2h_spec("p2h-b", electricity=1.0, heat=-0.8),
        CPSpec(cp_id="chp-c", capacity_by_sector={"gas": 1.0, "heat": -0.6}),
    ]
    demands = [
        _heat_demand(demand_mw=2.0),
        _electricity_slack(base_mw=10.0),
        SectorDemand(
            sector="gas",
            demand_by_tier={1: np.zeros(1)},
            base_supply=np.array([5.0]),
        ),
    ]
    reference = solve_cp_distributed_lexicographic_cascade(
        cps, demands, inner_iters_max=500
    )
    gossip = await _run_gossip(cps, demands)
    for cp in cps:
        assert gossip[cp.cp_id][0] == pytest.approx(
            reference.factor_by_cp[cp.cp_id][0], abs=1e-2
        ), f"CP {cp.cp_id} disagrees with reference"


@pytest.mark.asyncio
async def test_no_demand_returns_zero_factor():
    cps = [_p2h_spec("p2h-1", electricity=1.0, heat=-1.0)]
    demands = [_electricity_slack(base_mw=10.0)]
    gossip = await _run_gossip(cps, demands)
    assert gossip["p2h-1"][0] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_round_id_iter_messages_ignored():
    """An Iter message tagged with an old round_id must not corrupt
    the current round's iterate.  We feed a stale Iter directly into
    the participant and verify ``last_x_per_peer`` stays empty."""
    p = create_gossip_cascade_participant(
        cp_id="p2h-x",
        capacity_by_sector={"electricity": 1.0, "heat": -1.0},
    )
    # Init the participant on round 5.
    init = GossipCascadeInit(
        round_id=5,
        initiator_cp_id="p2h-x",
        participants=["p2h-x", "peer-y"],
        sectors=["electricity", "heat"],
        horizon=1,
        rho=1.0,
        r_regularization=0.1,
        adaptive_rho=True,
        rho_mu=10.0,
        rho_tau=2.0,
        demands=[_heat_demand(demand_mw=1.0), _electricity_slack(base_mw=5.0)],
        iter_timeout_s=0.1,
        round_timeout_s=10.0,
        inner_iters_max=10,
        inner_abs_tol=1e-4,
    )
    # Build context without launching the cascade (so we can inspect
    # _ctx after feeding stale Iter messages).
    p._ctx = p._build_ctx(init)

    # Stale Iter (round_id=4) — must be dropped silently.
    stale = GossipIter(
        round_id=4,
        tier_index=0,
        iter_k=0,
        cp_id="peer-y",
        x_i=np.array([[10.0], [10.0]]),  # arbitrary garbage
        rho=1.0,
        r_change=0.0,
    )
    p._on_iter(stale)
    assert "peer-y" not in p._ctx.last_x_per_peer, (
        "stale Iter must be dropped before reaching last_x_per_peer"
    )

    # Current-round Iter — must be accepted.
    current = GossipIter(
        round_id=5,
        tier_index=0,
        iter_k=0,
        cp_id="peer-y",
        x_i=np.array([[1.0], [-1.0]]),
        rho=1.0,
        r_change=0.0,
    )
    p._on_iter(current)
    assert "peer-y" in p._ctx.last_x_per_peer


@pytest.mark.asyncio
async def test_non_participant_iter_dropped():
    """An Iter from a cp_id outside the round's participants list is
    rejected (defensive against rogue messages from a non-member CP)."""
    p = create_gossip_cascade_participant(
        cp_id="p2h-x",
        capacity_by_sector={"electricity": 1.0, "heat": -1.0},
    )
    init = GossipCascadeInit(
        round_id=1,
        initiator_cp_id="p2h-x",
        participants=["p2h-x", "peer-y"],
        sectors=["electricity", "heat"],
        horizon=1,
        rho=1.0,
        r_regularization=0.1,
        adaptive_rho=True,
        rho_mu=10.0,
        rho_tau=2.0,
        demands=[_heat_demand(demand_mw=1.0), _electricity_slack(base_mw=5.0)],
        iter_timeout_s=0.1,
        round_timeout_s=10.0,
        inner_iters_max=10,
        inner_abs_tol=1e-4,
    )
    p._ctx = p._build_ctx(init)

    intruder = GossipIter(
        round_id=1,
        tier_index=0,
        iter_k=0,
        cp_id="rogue-z",  # not in participants
        x_i=np.array([[1.0], [-1.0]]),
        rho=1.0,
        r_change=0.0,
    )
    p._on_iter(intruder)
    assert "rogue-z" not in p._ctx.last_x_per_peer


@pytest.mark.asyncio
async def test_iter_timeout_uses_held_over_peer_contribution():
    """When a peer never sends its first Iter, the cascade still
    converges using zero (no held-over value yet) for that peer.
    With a tight iter timeout the round must finish via the round
    timeout fallback and commit *some* factor — feasibility is
    what we pin, not optimality.
    """
    cps = [
        _p2h_spec("p2h-a", electricity=1.0, heat=-1.0),
        _p2h_spec("p2h-b", electricity=1.0, heat=-1.0),  # this one will be silent
    ]
    demands = [
        _heat_demand(demand_mw=1.0),
        _electricity_slack(base_mw=10.0),
    ]

    container = ActorContainer()
    p_a = create_gossip_cascade_participant(
        cp_id="p2h-a",
        capacity_by_sector=cps[0].capacity_by_sector,
        on_commit=lambda r, c, n: results.setdefault("p2h-a", r),
    )

    # Silent peer: a do-nothing participant that ignores every message.
    class _SilentParticipant(GossipParticipant):
        async def on_exchange_message(self, carrier, message_data, meta):
            return  # never responds

    p_b = _SilentParticipant(
        cp_id="p2h-b",
        capacity_by_sector=cps[1].capacity_by_sector,
    )
    results: dict[str, np.ndarray] = {}

    SimpleCarrier(container, p_a)
    SimpleCarrier(container, p_b)
    initiator_carrier = container.actors[0]
    start = create_gossip_cascade_start(
        round_id=1,
        participants=["p2h-a", "p2h-b"],
        demands=demands,
        iter_timeout_s=0.02,
        round_timeout_s=2.0,
        inner_iters_max=50,
    )
    initiator_carrier.send_to_other(start, cid(initiator_carrier))

    # Give the cascade time to converge through hold-over iterations
    # and the round-timeout fallback.
    await asyncio.sleep(2.5)
    assert "p2h-a" in results, "p2h-a must commit even with silent peer p2h-b"
    # Feasibility: r is in [0, 1].
    assert 0.0 <= float(results["p2h-a"][0]) <= 1.0


@pytest.mark.asyncio
async def test_initiator_re_init_with_higher_round_id_replaces_state():
    """A second Init with a strictly greater ``round_id`` replaces
    the in-flight round (e.g. when a re-initiated handover happens)."""
    p = create_gossip_cascade_participant(
        cp_id="p2h-x",
        capacity_by_sector={"electricity": 1.0, "heat": -1.0},
    )
    init1 = GossipCascadeInit(
        round_id=1, initiator_cp_id="p2h-x",
        participants=["p2h-x", "peer-y"],
        sectors=["electricity", "heat"], horizon=1,
        rho=1.0, r_regularization=0.1, adaptive_rho=True,
        rho_mu=10.0, rho_tau=2.0,
        demands=[_heat_demand(demand_mw=1.0), _electricity_slack(base_mw=5.0)],
        iter_timeout_s=0.05, round_timeout_s=10.0,
        inner_iters_max=10, inner_abs_tol=1e-4,
    )
    p._ctx = p._build_ctx(init1)
    assert p.current_round_id == 1

    # Higher round — should replace.
    init2 = GossipCascadeInit(
        round_id=7, initiator_cp_id="peer-y",
        participants=["p2h-x", "peer-y"],
        sectors=["electricity", "heat"], horizon=1,
        rho=1.0, r_regularization=0.1, adaptive_rho=True,
        rho_mu=10.0, rho_tau=2.0,
        demands=[_heat_demand(demand_mw=2.0), _electricity_slack(base_mw=5.0)],
        iter_timeout_s=0.05, round_timeout_s=10.0,
        inner_iters_max=10, inner_abs_tol=1e-4,
    )
    p._ctx = p._build_ctx(init2)
    assert p.current_round_id == 7
    assert p._ctx.initiator_id == "peer-y"


# ---------------------------------------------------------------------------
# minimize_usage flag (opt-in; default off because it is much slower)
# ---------------------------------------------------------------------------


def test_minimize_usage_flag_removes_surplus_overshoot():
    """``minimize_usage`` selects the minimum-usage point of the degenerate
    optimal face: same served demand, no surplus production.

    Two heterogeneous P2Hs cover a 4 MW heat deficit with abundant spare
    capacity. The default (proximal) update converges fast but may overshoot
    (produce > demand); the opt-in ridge update produces exactly the demand.
    """
    cps = [
        CPSpec(cp_id="a", capacity_by_sector={"electricity": 3.0, "heat": -2.5}),
        CPSpec(cp_id="b", capacity_by_sector={"electricity": 6.0, "heat": -5.0}),
    ]
    demands = [
        SectorDemand(sector="electricity", demand_by_tier={1: np.zeros(1)},
                     base_supply=np.array([10.0])),
        SectorDemand(sector="heat", demand_by_tier={1: np.array([4.0])},
                     base_supply=np.array([0.0])),
    ]

    def heat_production(res):
        return 2.5 * float(res.factor_by_cp["a"][0]) + 5.0 * float(res.factor_by_cp["b"][0])

    default = solve_cp_distributed_lexicographic_cascade(cps, demands)
    minimal = solve_cp_distributed_lexicographic_cascade(
        cps, demands, minimize_usage=True, r_regularization=1.0, inner_iters_max=2000
    )

    # Both meet the priority obligation: tier-1 heat fully served (= 4 MW).
    assert default.served_by_sector_tier["heat"][1][0] == pytest.approx(4.0, abs=1e-2)
    assert minimal.served_by_sector_tier["heat"][1][0] == pytest.approx(4.0, abs=1e-2)

    # minimize_usage produces exactly the demand (no overshoot); the default
    # overshoots here. Either way r stays in the unit box.
    assert heat_production(minimal) == pytest.approx(4.0, abs=1e-2)
    assert heat_production(default) >= heat_production(minimal) - 1e-9
    for cp in ("a", "b"):
        assert 0.0 <= float(minimal.factor_by_cp[cp][0]) <= 1.0 + 1e-9
