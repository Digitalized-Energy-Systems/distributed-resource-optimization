"""Peers must not lock one iteration out of phase under a per-hop delay.

An inbound ``GossipIter`` used to count toward the current iteration only on
an exact ``(tier, iter)`` match, and the count was wiped at the top of every
iteration. The initiator broadcasts Init and runs iteration 0 in the same
instant, so it starts one hop ahead of everyone; from then on every message
arrives off-phase, is dropped from the count, and every iteration on every
side pays the full ``iter_timeout_s``. Measured on a 4-CP world at a 0.020 s
hop with 0.2 s / 2.0 s timeouts: 160 of 199 peer waits ended on the timeout,
mean wait 0.167 s, 11-15 iterations per round, none converged.

``phase_tolerant_wait`` keeps every peer's Iter under its (tier, iter) and
lets each iteration consume exactly the peers' same-iteration values, and
tests convergence on quantities every CP holds identically, so the iterate
sequence is the synchronous one under any link timing: equal links cost one
hop per iteration, a late or slow peer costs two, never a timeout. A tier is
only ever left by convergence or its deadline, so every CP computes every
tier and theta carries each tier's served demand forward.
``rho_rescale_first`` puts the adaptive-rho rescale ahead of the r-update
(the reference kernel's order); without it the gossip iterate limit-cycles on
this problem at any budget.

The problem is the demonstrator's captured round 1 (three sectors, four
tiers), which the replicated kernel solves in [1, 77, 21, 3] iterations.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest

pytest.importorskip("mango")
pytest.importorskip("networkx")

from mango import (  # noqa: E402
    DISCRETE_EVENT,
    SimpleCommunicationSimulation,
    agent_composed_of,
    create_topology,
    run_with_simulation,
    step_simulation,
)

from distributed_resource_optimization import (  # noqa: E402
    CPSpec,
    SectorDemand,
    create_gossip_cascade_participant,
    create_gossip_cascade_start,
    solve_cp_distributed_lexicographic_cascade,
)
from distributed_resource_optimization.algorithm.admm.lexicographic.gossip import (  # noqa: E402
    GossipIter,
    GossipParticipant,
)
from distributed_resource_optimization.carrier.mango_carrier import (  # noqa: E402
    DistributedOptimizationRole,
    MangoCarrier,
)

_HOP_S = 0.02
_ITER_TIMEOUT_S = 0.2
_MAX_EVENT_STEPS = 20000
_N_TIERS = 4

_KERNEL_KWARGS = dict(
    horizon=1,
    rho=1.0,
    inner_iters_max=200,
    inner_abs_tol=1e-3,
    r_regularization=0.1,
    adaptive_rho=True,
    rho_mu=10.0,
    rho_tau=2.0,
    minimize_usage=True,
    normalize=True,
    r_regularization_relative=True,
)


def _demonstrator_round() -> tuple[list[CPSpec], list[SectorDemand]]:
    cps = [
        CPSpec("branch-11-4", {"gas": -42.444396, "electricity": 60.63485142857144}),
        CPSpec("branch-13-6", {"gas": 42.444396, "electricity": -38.1999564}),
        CPSpec("branch-8-1", {"gas": 42.444396, "electricity": -38.1999564}),
        CPSpec(
            "node-31",
            {"gas": 0.21222198, "electricity": -0.10611099, "heat": -0.10611099},
        ),
    ]
    demands = [
        SectorDemand(
            "electricity",
            {1: np.array([20.0]), 2: np.array([20.0]), 3: np.array([10.0])},
            np.array([49.790094127662265]),
        ),
        SectorDemand(
            "heat",
            {4: np.array([0.41900837768253796])},
            np.array([0.31289738768253794]),
        ),
        SectorDemand(
            "gas",
            {
                1: np.array([408.72538534800003]),
                2: np.array([63.666594]),
                3: np.array([37.35106848000001]),
                4: np.array([5.09332752]),
            },
            np.array([451.459818054]),
        ),
    ]
    return cps, demands


def _connect_complete(topo, agents):
    nodes = [topo.add_node(a) for a in agents]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            topo.add_edge(nodes[i], nodes[j])


async def _run(
    monkeypatch,
    *,
    phase_tolerant_wait: bool,
    rho_rescale_first: bool,
    round_timeout_s: float,
    cp_ids: tuple[str, ...] | None = None,
    late: tuple[str, float] | None = None,
    slow_link: tuple[str, float] | None = None,
):
    """One round in a discrete-event world with a per-hop delay.

    ``late`` holds one CP's cascade back by that many simulated seconds after
    its Init; ``slow_link`` puts every link to and from one CP at that delay.
    Returns ``{cp_id: (r, converged, iterations)}``, the simulated cost of
    every peer wait, and ``{cp_id: {tier: iterations broadcast}}``.
    """
    waits: list[float] = []
    orig_race = MangoCarrier.race

    async def race(self, *awaitables):
        t0 = self.now()
        await orig_race(self, *awaitables)
        waits.append(self.now() - t0)

    monkeypatch.setattr(MangoCarrier, "race", race)

    per_tier: dict[str, dict[int, int]] = {}
    orig_send = MangoCarrier.send_to_other

    def send_to_other(self, content, receiver, meta=None):
        if isinstance(content, GossipIter) and not content.done:
            tiers = per_tier.setdefault(content.cp_id, {})
            tiers[content.tier_index] = tiers.get(content.tier_index, 0) + 1
        return orig_send(self, content, receiver, meta)

    monkeypatch.setattr(MangoCarrier, "send_to_other", send_to_other)

    if late is not None:
        late_cp, late_by = late
        orig_run = GossipParticipant._run_cascade

        async def run_cascade(self, carrier):
            if self.cp_id == late_cp:
                await carrier.sleep(late_by)
            await orig_run(self, carrier)

        monkeypatch.setattr(GossipParticipant, "_run_cascade", run_cascade)

    cps, demands = _demonstrator_round()
    if cp_ids is not None:
        cps = [c for c in cps if c.cp_id in cp_ids]
    results: dict = {}

    def make_cb(cp_id: str):
        def _cb(r, converged, iters):
            results[cp_id] = (float(np.asarray(r).ravel()[0]), bool(converged), int(iters))

        return _cb

    participants = [
        create_gossip_cascade_participant(c.cp_id, c.capacity_by_sector, on_commit=make_cb(c.cp_id))
        for c in cps
    ]
    agents = [agent_composed_of(DistributedOptimizationRole(p)) for p in participants]
    sim = SimpleCommunicationSimulation(default_delay_s=_HOP_S)

    async with run_with_simulation(*agents, communication_sim=sim) as world:
        if slow_link is not None:
            slow_cp, delay = slow_link
            slow = agents[[c.cp_id for c in cps].index(slow_cp)]
            for other in agents:
                if other is not slow:
                    sim.delay_s_directed_edge_dict[(other.aid, slow.aid)] = delay
                    sim.delay_s_directed_edge_dict[(slow.aid, other.aid)] = delay
        with create_topology() as topo:
            _connect_complete(topo, agents)
        start = create_gossip_cascade_start(
            round_id=1,
            participants=[c.cp_id for c in cps],
            demands=demands,
            iter_timeout_s=_ITER_TIMEOUT_S,
            round_timeout_s=round_timeout_s,
            tier_fair_deadline=True,
            phase_tolerant_wait=phase_tolerant_wait,
            rho_rescale_first=rho_rescale_first,
            **_KERNEL_KWARGS,
        )
        await world.send_message(start, receiver_addr=agents[0].addr, sender_id=None)
        idle = 0
        for _ in range(_MAX_EVENT_STEPS):
            if len(results) >= len(cps):
                break
            if await step_simulation(world, step_size_s=DISCRETE_EVENT) is None:
                idle += 1
                if idle > 12:
                    raise AssertionError("gossip cascade stalled before all peers committed")
            else:
                idle = 0
    assert len(results) == len(cps), "not every peer committed within the step budget"
    return results, waits, per_tier


def _tier_timeouts(caplog) -> int:
    return sum(
        1 for rec in caplog.records if re.search(r"timed out at tier \d+ iter \d+", rec.getMessage())
    )


def _timed_out_fraction(waits: list[float]) -> float:
    return sum(1 for w in waits if w >= _ITER_TIMEOUT_S - 1e-9) / len(waits)


def _kernel_factors(cps: list[CPSpec], demands: list[SectorDemand]) -> dict[str, float]:
    ref = solve_cp_distributed_lexicographic_cascade(cps, demands, **_KERNEL_KWARGS)
    assert ref.converged
    return {cp.cp_id: float(ref.factor_by_cp[cp.cp_id][0]) for cp in cps}


@pytest.mark.asyncio
async def test_flag_off_locks_peers_one_iteration_apart(monkeypatch, caplog):
    """The defect: nearly every iteration pays the peer timeout, so a round
    gets ``round_timeout_s / iter_timeout_s`` iterations in total."""
    with caplog.at_level(logging.WARNING):
        results, waits, _ = await _run(
            monkeypatch, phase_tolerant_wait=False, rho_rescale_first=False, round_timeout_s=2.0
        )
    assert _timed_out_fraction(waits) > 0.5
    assert np.mean(waits) > 0.5 * _ITER_TIMEOUT_S
    for _cp, (_r, converged, iters) in results.items():
        assert iters <= 2.0 / _ITER_TIMEOUT_S + _N_TIERS + 2
        assert not converged
    assert _tier_timeouts(caplog) > 0


@pytest.mark.asyncio
async def test_phase_tolerant_wait_runs_at_hop_speed(monkeypatch):
    """The timing fix alone: one hop per iteration, several times the
    iterations in the same budget."""
    stock, _, _ = await _run(
        monkeypatch, phase_tolerant_wait=False, rho_rescale_first=False, round_timeout_s=2.0
    )
    fixed, waits, _ = await _run(
        monkeypatch, phase_tolerant_wait=True, rho_rescale_first=False, round_timeout_s=2.0
    )
    assert np.mean(waits) <= 2 * _HOP_S
    assert _timed_out_fraction(waits) < 0.05
    assert min(v[2] for v in fixed.values()) >= 3 * max(v[2] for v in stock.values())


@pytest.mark.asyncio
async def test_both_flags_converge_to_the_replicated_kernel(monkeypatch, caplog):
    """With the budget the problem needs (~100 hops), every peer converges,
    in the same number of iterations, to the kernel's factors."""
    cps, demands = _demonstrator_round()
    ref = _kernel_factors(cps, demands)
    with caplog.at_level(logging.WARNING):
        results, waits, _ = await _run(
            monkeypatch, phase_tolerant_wait=True, rho_rescale_first=True, round_timeout_s=4.0
        )
    assert _tier_timeouts(caplog) == 0
    assert np.mean(waits) <= 2 * _HOP_S
    assert len({v[2] for v in results.values()}) == 1, "peers left a tier out of step"
    for cp in cps:
        r, converged, _iters = results[cp.cp_id]
        assert converged, f"{cp.cp_id} did not converge"
        assert r == pytest.approx(ref[cp.cp_id], abs=5e-3)


_ASYMMETRIES = {
    "late_joiner": dict(late=("node-31", 2 * _HOP_S)),
    "slow_link": dict(slow_link=("node-31", 2 * _HOP_S)),
    "two_peers": dict(cp_ids=("branch-11-4", "branch-13-6")),
}


@pytest.mark.parametrize("asymmetry", sorted(_ASYMMETRIES))
@pytest.mark.asyncio
async def test_every_tier_is_computed_under_timing_asymmetry(monkeypatch, asymmetry):
    """A peer that starts late, sits on a slower link, or has only one
    partner changes how many hops an iteration costs, never which iterate
    it computes: every CP computes every tier (theta is the only thing that
    carries a tier's served demand into the next one) and, given the hops,
    the cascade still lands on the kernel's factors."""
    kwargs = _ASYMMETRIES[asymmetry]
    cps, demands = _demonstrator_round()
    if "cp_ids" in kwargs:
        cps = [c for c in cps if c.cp_id in kwargs["cp_ids"]]
    ref = _kernel_factors(cps, demands)

    results, waits, per_tier = await _run(
        monkeypatch, phase_tolerant_wait=True, rho_rescale_first=True, round_timeout_s=8.0, **kwargs
    )

    assert _timed_out_fraction(waits) < 0.05
    assert len({v[2] for v in results.values()}) == 1, "peers left a tier out of step"
    for cp in cps:
        computed = per_tier.get(cp.cp_id, {})
        assert set(computed) == set(range(_N_TIERS)), f"{cp.cp_id} skipped a tier: {computed}"
        r, converged, _iters = results[cp.cp_id]
        assert converged, f"{cp.cp_id} did not converge"
        # Gossip closes a tier one iteration after the kernel's own test, which
        # on the two-CP fleet moves the initiator's factor by ~1e-2.
        assert r == pytest.approx(ref[cp.cp_id], abs=2e-2)


@pytest.mark.asyncio
async def test_flag_off_is_bit_identical_run_to_run(monkeypatch):
    a, _, _ = await _run(
        monkeypatch, phase_tolerant_wait=False, rho_rescale_first=False, round_timeout_s=2.0
    )
    b, _, _ = await _run(
        monkeypatch, phase_tolerant_wait=False, rho_rescale_first=False, round_timeout_s=2.0
    )
    assert a == b
