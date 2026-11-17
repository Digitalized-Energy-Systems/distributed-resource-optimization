"""A round that follows a near-identical one should restart each tier where it
converged, not from zeros.

``warm_start`` seeds only tier 0 from the previous round's END state, which is
the bottom tier's dual and no help to the top of the next ladder; on the
demonstrator's rounds it left every round at 94-146 iterations. Each tier is
its own sharing-ADMM problem, so ``warm_start_per_tier`` snapshots every tier
at its close and seeds the same tier of the next round from it.

Both rounds run at a fixed rho, because residual balancing rides a limit
cycle on this problem (see ``rho`` in the SCARE configuration) and would make
the iteration counts meaningless.
"""

from __future__ import annotations

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
    SectorDemand,
    create_gossip_cascade_participant,
    create_gossip_cascade_start,
    solve_cp_distributed_lexicographic_cascade,
)
from distributed_resource_optimization.carrier.mango_carrier import (  # noqa: E402
    DistributedOptimizationRole,
)

from .test_phase_lock import (  # noqa: E402
    _HOP_S,
    _ITER_TIMEOUT_S,
    _KERNEL_KWARGS,
    _MAX_EVENT_STEPS,
    _connect_complete,
    _demonstrator_round,
)

_FIXED_RHO = dict(_KERNEL_KWARGS, rho=100.0, adaptive_rho=False)
_EXACT = dict(_FIXED_RHO, inner_abs_tol=1e-7, inner_iters_max=20000)


def _second_round(demands: list[SectorDemand]) -> list[SectorDemand]:
    """The demonstrator's round 2: the gas base supply has settled lower."""
    out = []
    for d in demands:
        base = d.base_supply.copy()
        if d.sector == "gas":
            base = base * (424.444 / 451.46)
        out.append(SectorDemand(d.sector, dict(d.demand_by_tier), base))
    return out


async def _drive(world, results, n_expected):
    idle = 0
    for _ in range(_MAX_EVENT_STEPS):
        if len(results) >= n_expected:
            return
        if await step_simulation(world, step_size_s=DISCRETE_EVENT) is None:
            idle += 1
            if idle > 12:
                raise AssertionError("gossip cascade stalled")
        else:
            idle = 0
    raise AssertionError("step budget exhausted")


async def _two_rounds(*, per_tier: bool, drop_peer_in_round_2: bool = False):
    cps, demands = _demonstrator_round()
    rounds: list[dict] = [{}, {}]
    current = [0]

    def make_cb(cp_id):
        def _cb(r, converged, iters):
            rounds[current[0]][cp_id] = (float(np.ravel(r)[0]), bool(converged), int(iters))

        return _cb

    participants = [
        create_gossip_cascade_participant(
            c.cp_id,
            c.capacity_by_sector,
            on_commit=make_cb(c.cp_id),
            warm_start=not per_tier,
            warm_start_per_tier=per_tier,
        )
        for c in cps
    ]
    agents = [agent_composed_of(DistributedOptimizationRole(p)) for p in participants]
    sim = SimpleCommunicationSimulation(default_delay_s=_HOP_S)
    async with run_with_simulation(*agents, communication_sim=sim) as world:
        with create_topology() as topo:
            _connect_complete(topo, agents)
        ids = [c.cp_id for c in cps]
        for round_id, round_demands in ((1, demands), (2, _second_round(demands))):
            current[0] = round_id - 1
            members = ids[:-1] if (drop_peer_in_round_2 and round_id == 2) else ids
            start = create_gossip_cascade_start(
                round_id=round_id,
                participants=members,
                demands=round_demands,
                iter_timeout_s=_ITER_TIMEOUT_S,
                round_timeout_s=8.0,
                tier_fair_deadline=True,
                phase_tolerant_wait=True,
                rho_rescale_first=True,
                **_FIXED_RHO,
            )
            await world.send_message(start, receiver_addr=agents[0].addr, sender_id=None)
            await _drive(world, rounds[current[0]], len(members))
    return cps, demands, rounds


@pytest.mark.asyncio
async def test_per_tier_seed_restarts_the_next_round_near_its_answer():
    cps, demands, (first, second) = await _two_rounds(per_tier=True)
    exact = solve_cp_distributed_lexicographic_cascade(cps, _second_round(demands), **_EXACT)
    cold = solve_cp_distributed_lexicographic_cascade(cps, _second_round(demands), **_FIXED_RHO)
    assert all(v[1] for v in second.values()), second
    for cp in cps:
        assert second[cp.cp_id][0] == pytest.approx(
            float(exact.factor_by_cp[cp.cp_id][0]), abs=1e-2
        )
    assert max(v[2] for v in second.values()) <= 0.5 * cold.iterations
    assert max(v[2] for v in second.values()) <= 0.5 * max(v[2] for v in first.values())


@pytest.mark.asyncio
async def test_end_of_round_seed_does_not_help_the_next_ladder():
    """The characterisation the per-tier seed is measured against."""
    _cps, demands, (first, second) = await _two_rounds(per_tier=False)
    assert max(v[2] for v in second.values()) > 0.7 * max(v[2] for v in first.values())


@pytest.mark.asyncio
async def test_changed_membership_falls_back_to_a_cold_dual():
    """A snapshot taken over a different group keeps only r: the saved duals
    belong to another consensus and would pin r at a bound."""
    cps, demands, (_first, second) = await _two_rounds(per_tier=True, drop_peer_in_round_2=True)
    survivors = [c for c in cps if c.cp_id in second]
    exact = solve_cp_distributed_lexicographic_cascade(survivors, _second_round(demands), **_EXACT)
    assert all(v[1] for v in second.values()), second
    for cp in survivors:
        assert second[cp.cp_id][0] == pytest.approx(
            float(exact.factor_by_cp[cp.cp_id][0]), abs=1e-2
        )
