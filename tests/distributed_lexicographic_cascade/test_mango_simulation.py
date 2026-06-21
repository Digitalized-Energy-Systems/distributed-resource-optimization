"""Mango-carrier integration tests under a discrete-time simulation.

These exercise the leader-follower lexicographic cascade end-to-end over
the :class:`~distributed_resource_optimization.carrier.mango.MangoCarrier`
inside a mango ``run_with_simulation`` world, driven by explicit
:func:`mango.step_simulation` calls (discrete time stepping).

The properties we pin:

* **Parity** — the simulated leader-follower run reaches the same
  regulation factors as the in-process kernel.
* **Clock-gated, no side track** — the protocol makes progress *only*
  when the simulation is stepped. Spinning the event loop without
  stepping advances neither the clock nor the protocol, proving no work
  runs "on a side track" off the simulation clock (the carrier routes
  every send through the agent scheduler so the step's convergence loop
  tracks it).
* **Sensible time progression** — with a per-message delay the clock
  advances by that delay on each event step, and the run terminates
  (the world goes quiescent) instead of looping forever.
"""

from __future__ import annotations

import asyncio

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
    create_distributed_lexicographic_cascade_coordinator,
    create_distributed_lexicographic_cascade_participant,
    create_distributed_lexicographic_cascade_start,
    create_gossip_cascade_participant,
    create_gossip_cascade_start,
    solve_cp_distributed_lexicographic_cascade,
)
from distributed_resource_optimization.algorithm.admm.types import (  # noqa: E402
    CPSpec,
)
from distributed_resource_optimization.carrier.mango import (  # noqa: E402
    CoordinatorRole,
    DistributedOptimizationRole,
    StartCoordinatedDistributedOptimization,
)

# Safety cap so a regression can never hang the suite.
_MAX_EVENT_STEPS = 500


def _elec(base_mw: float) -> SectorDemand:
    return SectorDemand(
        sector="electricity",
        demand_by_tier={1: np.zeros(1)},
        base_supply=np.array([base_mw]),
    )


def _heat(demand_mw: float) -> SectorDemand:
    return SectorDemand(
        sector="heat",
        demand_by_tier={1: np.array([demand_mw])},
        base_supply=np.array([0.0]),
    )


def _build_world_pieces(cps: list[CPSpec]):
    """Wire one CoordinatorRole + N follower roles into mango agents."""
    participants = [
        create_distributed_lexicographic_cascade_participant(c.cp_id, c.capacity_by_sector)
        for c in cps
    ]
    follower_roles = [DistributedOptimizationRole(p) for p in participants]
    coordinator_role = CoordinatorRole(create_distributed_lexicographic_cascade_coordinator())
    follower_agents = [agent_composed_of(r) for r in follower_roles]
    coordinator_agent = agent_composed_of(coordinator_role)
    return participants, coordinator_role, coordinator_agent, follower_agents


def _is_done(coordinator_role) -> bool:
    """The coordinator's done-future is created lazily, when it processes
    the Start message; treat 'not created yet' as 'not done'."""
    fut = coordinator_role._done_future
    return fut is not None and fut.done()


async def _step_until_done(world, coordinator_role, *, step_size_s=DISCRETE_EVENT):
    """Discrete-step the world until the coordinator finishes.

    The request/reply driver is a tracked scheduler task, so a single
    ``step_simulation`` drives it through its parked/resumed cycles within
    the step's convergence loop — no external event-loop yield is needed.
    """
    clock_trace: list[float] = []
    idle = 0
    for _ in range(_MAX_EVENT_STEPS):
        if _is_done(coordinator_role):
            break
        result = await step_simulation(world, step_size_s=step_size_s)
        if result is None:
            idle += 1
            if idle > 10:
                raise AssertionError("simulation stalled before completion")
        else:
            idle = 0
            clock_trace.append(world.clock.time)
    assert _is_done(coordinator_role), "coordinator did not finish within the step budget"
    return clock_trace


@pytest.mark.asyncio
async def test_mango_simulation_matches_kernel():
    """Stepped leader-follower run reaches the kernel's regulation factors."""
    cps = [
        CPSpec("a", {"electricity": 3.0, "heat": -2.5}),
        CPSpec("b", {"electricity": 6.0, "heat": -5.0}),
    ]
    demands = [_elec(10.0), _heat(4.0)]

    participants, coordinator_role, coord_agent, follower_agents = _build_world_pieces(cps)

    async with run_with_simulation(coord_agent, *follower_agents) as world:
        with create_topology() as topo:
            n_coord = topo.add_node(coord_agent)
            for fa in follower_agents:
                topo.add_edge(n_coord, topo.add_node(fa))

        await world.send_message(
            StartCoordinatedDistributedOptimization(
                create_distributed_lexicographic_cascade_start(demands=demands)
            ),
            receiver_addr=coord_agent.addr,
            sender_id=None,
        )
        await _step_until_done(world, coordinator_role)

    ref = solve_cp_distributed_lexicographic_cascade(cps, demands)
    for participant, cp in zip(participants, cps):
        assert participant.r.size > 0
        np.testing.assert_allclose(participant.r, ref.factor_by_cp[cp.cp_id], atol=1e-6)


@pytest.mark.asyncio
async def test_mango_simulation_is_clock_gated_no_side_track():
    """Progress happens only on a step; the clock advances by the delay.

    Pins the failure the naive carrier would exhibit: if the carrier ran
    its sends on bare ``asyncio.create_task`` they would race ahead of the
    simulation clock ("on a side track"). Here we prove the opposite —
    after the first step, spinning the event loop without stepping moves
    neither the clock nor the protocol.
    """
    cps = [
        CPSpec("a", {"electricity": 3.0, "heat": -2.5}),
        CPSpec("b", {"electricity": 6.0, "heat": -5.0}),
    ]
    demands = [_elec(10.0), _heat(4.0)]
    delay_s = 1.0

    participants, coordinator_role, coord_agent, follower_agents = _build_world_pieces(cps)
    comm = SimpleCommunicationSimulation(default_delay_s=delay_s)

    async with run_with_simulation(coord_agent, *follower_agents, communication_sim=comm) as world:
        with create_topology() as topo:
            n_coord = topo.add_node(coord_agent)
            for fa in follower_agents:
                topo.add_edge(n_coord, topo.add_node(fa))

        await world.send_message(
            StartCoordinatedDistributedOptimization(
                create_distributed_lexicographic_cascade_start(demands=demands)
            ),
            receiver_addr=coord_agent.addr,
            sender_id=None,
        )

        # ---- one step: Start delivered, clock advances by exactly one delay ----
        await step_simulation(world, step_size_s=DISCRETE_EVENT)
        assert not _is_done(coordinator_role), "multi-round protocol cannot finish in one step"
        assert world.clock.time == pytest.approx(delay_s)

        # ---- no side track: spinning the loop without stepping does nothing ----
        clock_before = world.clock.time
        for _ in range(50):
            await asyncio.sleep(0)
        assert not _is_done(coordinator_role), "protocol advanced without a simulation step"
        assert world.clock.time == clock_before, "clock advanced without a step"

        # ---- sensible time progression: each event step advances by the delay ----
        clock_trace = await _step_until_done(world, coordinator_role)
        deltas = np.diff([clock_before, *clock_trace])
        assert np.all(deltas > 0), "clock must advance monotonically"
        assert np.allclose(deltas, delay_s), (
            f"each event step should advance the clock by the message delay; got {deltas}"
        )

        # ---- the world goes quiescent: a final drain reaches 'no events' ----
        for _ in range(_MAX_EVENT_STEPS):
            if await step_simulation(world, step_size_s=DISCRETE_EVENT) is None:
                break
        else:
            raise AssertionError("world never became quiescent")

    ref = solve_cp_distributed_lexicographic_cascade(cps, demands)
    for participant, cp in zip(participants, cps):
        np.testing.assert_allclose(participant.r, ref.factor_by_cp[cp.cp_id], atol=1e-6)


# ---------------------------------------------------------------------------
# Gossip cascade — coordinator-free, formerly wall-clock-timed, now sim-aware
# ---------------------------------------------------------------------------


def _gossip_world_pieces(cps: list[CPSpec], results: dict):
    """Wire N gossip peers (complete graph) recording committed factors."""

    def make_cb(cp_id: str):
        def _cb(r, converged, iters):
            results[cp_id] = (np.asarray(r, dtype=float), bool(converged), int(iters))

        return _cb

    participants = [
        create_gossip_cascade_participant(c.cp_id, c.capacity_by_sector, on_commit=make_cb(c.cp_id))
        for c in cps
    ]
    agents = [agent_composed_of(DistributedOptimizationRole(p)) for p in participants]
    return participants, agents


def _connect_complete(topo, agents):
    nodes = [topo.add_node(a) for a in agents]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            topo.add_edge(nodes[i], nodes[j])


async def _drive_gossip(world, results, n_expected):
    """Step until every peer has committed. Each peer's cascade is a tracked
    scheduler task, so stepping alone drives it (no external loop yield)."""
    idle = 0
    for _ in range(_MAX_EVENT_STEPS):
        if len(results) >= n_expected:
            break
        r = await step_simulation(world, step_size_s=DISCRETE_EVENT)
        if r is None:
            idle += 1
            if idle > 12:
                raise AssertionError("gossip cascade stalled before all peers committed")
        else:
            idle = 0
    assert len(results) >= n_expected, "not every peer committed within the step budget"


@pytest.mark.asyncio
async def test_gossip_cascade_completes_and_is_feasible_under_simulation():
    """The (sim-aware) gossip cascade runs to a converged, feasible commit
    under discrete stepping — where its old wall-clock timers stalled."""
    cps = [
        CPSpec("a", {"electricity": 3.0, "heat": -2.5}),
        CPSpec("b", {"electricity": 6.0, "heat": -5.0}),
    ]
    demands = [_elec(10.0), _heat(4.0)]
    results: dict = {}
    participants, agents = _gossip_world_pieces(cps, results)

    async with run_with_simulation(
        *agents, communication_sim=SimpleCommunicationSimulation(default_delay_s=1.0)
    ) as world:
        with create_topology() as topo:
            _connect_complete(topo, agents)
        start = create_gossip_cascade_start(
            round_id=1,
            participants=[c.cp_id for c in cps],
            demands=demands,
            iter_timeout_s=3.0,  # > comm delay, so peers answer before timeout
            round_timeout_s=500.0,
            inner_iters_max=500,
            inner_abs_tol=1e-5,
        )
        await world.send_message(start, receiver_addr=agents[0].addr, sender_id=None)
        await _drive_gossip(world, results, len(cps))

    for c in cps:
        r, converged, _iters = results[c.cp_id]
        assert converged, f"{c.cp_id} did not converge"
        assert 0.0 <= float(r[0]) <= 1.0 + 1e-9
    # Feasibility: combined heat production covers the 4 MW deficit (the
    # default keeps the fast proximal update, which may overshoot surplus).
    produced = 2.5 * results["a"][0][0] + 5.0 * results["b"][0][0]
    assert produced >= 4.0 - 1e-2


@pytest.mark.asyncio
async def test_gossip_cascade_is_clock_gated_under_simulation():
    """The gossip round advances only on simulation steps and its per-round
    timeout obeys the simulation clock, not real wall-clock time."""
    cps = [
        CPSpec("a", {"electricity": 3.0, "heat": -2.5}),
        CPSpec("b", {"electricity": 6.0, "heat": -5.0}),
    ]
    demands = [_elec(10.0), _heat(4.0)]
    delay_s = 1.0
    results: dict = {}
    participants, agents = _gossip_world_pieces(cps, results)

    async with run_with_simulation(
        *agents, communication_sim=SimpleCommunicationSimulation(default_delay_s=delay_s)
    ) as world:
        with create_topology() as topo:
            _connect_complete(topo, agents)
        start = create_gossip_cascade_start(
            round_id=1,
            participants=[c.cp_id for c in cps],
            demands=demands,
            iter_timeout_s=3.0,
            round_timeout_s=500.0,
            inner_iters_max=500,
            inner_abs_tol=1e-5,
        )
        await world.send_message(start, receiver_addr=agents[0].addr, sender_id=None)

        # First step delivers the Start only; nobody has committed yet.
        await step_simulation(world, step_size_s=DISCRETE_EVENT)
        assert not results, "the multi-iteration cascade cannot finish in one step"

        # No side track: spinning the loop without stepping advances nothing.
        clock_before = world.clock.time
        for _ in range(50):
            await asyncio.sleep(0)
        assert not results, "cascade committed without a simulation step"
        assert world.clock.time == clock_before, "clock advanced without a step"

        await _drive_gossip(world, results, len(cps))
        # Time actually elapsed on the simulation clock (delay-gated rounds).
        assert world.clock.time >= clock_before + delay_s

    for c in cps:
        assert results[c.cp_id][1], f"{c.cp_id} did not converge"
