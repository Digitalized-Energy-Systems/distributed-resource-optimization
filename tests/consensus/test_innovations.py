"""Consensus + innovations in a mango world: exact optimum on an irregular graph, stop rules."""

from __future__ import annotations

import asyncio
import random

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("mango")

from mango import (  # noqa: E402
    DISCRETE_EVENT,
    SimpleCommunicationSimulation,
    agent_composed_of,
    create_world,
    graph_topology,
)
from mango.express.topology import AGENT_NODE_KEY  # noqa: E402
from mango.simulation.communication import PackageResult  # noqa: E402

from distributed_resource_optimization import (  # noqa: E402
    ConsensusActor,
    ConsensusCoordinator,
    ConsensusInnovationsAlgorithm,
    power_step,
)
from distributed_resource_optimization.algorithm.firstorder.consensus.innovations import (  # noqa: E402
    ConsensusMessage,
    ConsensusReport,
)
from distributed_resource_optimization.carrier.mango_carrier import (  # noqa: E402
    CoordinatorRole,
    DistributedOptimizationRole,
    StartCoordinatedDistributedOptimization,
    next_hops,
)

# x_j(λ) = clip((λ − b_j)/a_j, 0, cap_j); demand d_j per hour → λ* where Σ x_j = Σ d_j
A = [1.0, 2.0, 0.5, 1.0]
B = [1.0, 0.0, 2.0, 3.0]
CAP = [10.0, 10.0, 10.0, 1.0]
D = np.array([[2.0, 1.0], [0.0, 3.0], [3.0, 0.0], [1.0, 1.0]])


class _LoseLambda(SimpleCommunicationSimulation):
    """Loses only the λ exchange (ConsensusMessage); start, reports and stop always arrive."""

    def __init__(self, loss):
        super().__init__(default_delay_s=0.01, loss_percent=loss, rng=random.Random(0))

    def calculate_communication(self, current_time, messages):
        result = super().calculate_communication(current_time, messages)
        result.package_results[:] = [
            r if isinstance(m.content[0], ConsensusMessage) else PackageResult(True, r.delay_s)
            for m, r in zip(messages, result.package_results)
        ]
        return result


class Quadratic(ConsensusActor):
    def __init__(self, a, b, cap, d, tol=1e-6):
        self.a, self.b, self.cap, self.d, self.tol = a, b, cap, d, tol
        self.x = self.prev = None

    def innovation(self, lam):
        self.prev, self.x = self.x, np.clip((lam - self.b) / self.a, 0.0, self.cap)
        return self.x - self.d

    def converged(self):
        return self.prev is not None and np.max(np.abs(self.x - self.prev)) < self.tol


def lam_star(hour):
    total = D[:, hour].sum()
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        supply = sum(np.clip((mid - b) / a, 0, c) for a, b, c in zip(A, B, CAP))
        lo, hi = (mid, hi) if supply < total else (lo, mid)
    return lo


IRREGULAR = [("p0", "p1"), ("p1", "p2"), ("p2", "p3"), ("p1", "p3"), ("c", "p0")]


async def run(
    max_iter,
    alpha=power_step(0.3, 0.6),
    beta=power_step(0.2, 0.001),
    edges=IRREGULAR,
    demand=D,
    lam_min=-np.inf,
    settle_s=0.0,
    loss=0.0,
    round_timeout=None,
):
    # default: irregular graph p0–p1–p2–p3 plus p1–p3; coordinator "c" a leaf on p0
    graph = nx.Graph(edges)
    routes = next_hops(graph)
    finished = []
    algos = {
        f"p{j}": ConsensusInnovationsAlgorithm(
            lambda alg, carrier: finished.append(alg.k),
            Quadratic(A[j], B[j], CAP[j], demand[j]),
            alpha,
            beta,
            eps=1e-6,
            max_iter=max_iter,
            lam_min=lam_min,
            round_timeout=round_timeout,
        )
        for j in range(4)
        if f"p{j}" in graph
    }
    world = create_world(start_time=0.0, communication_sim=_LoseLambda(loss))
    agents = {
        aid: world.register(
            agent_composed_of(DistributedOptimizationRole(algo, routes=routes[aid])),
            suggested_aid=aid,
        )
        for aid, algo in algos.items()
    }
    coordinator = ConsensusCoordinator(
        [a.addr for a in agents.values()],
        np.zeros(2),
        settle_s=settle_s,
        start_latency=0.03,  # the start reaches p3 over 3 hops
    )
    agents["c"] = world.register(
        agent_composed_of(CoordinatorRole(coordinator, routes=routes["c"])), suggested_aid="c"
    )
    topo = graph_topology(graph)
    for aid, agent in agents.items():
        topo.graph.nodes[aid][AGENT_NODE_KEY].add(agent)
    topo.inject()
    async with world:
        agents["c"].handle_message(StartCoordinatedDistributedOptimization(None), {})
        while await world.step(DISCRETE_EVENT) is not None:
            pass
    return coordinator.result, algos, finished


async def test_converges_to_exact_optimum_and_stops():
    result, algos, finished = await run(max_iter=20_000)
    assert result.stop_reason == "converged"
    assert len(finished) == 4  # every participant got the stop
    # decaying steps are exact in the limit; the step-change stop (eps) ends ~1e-2 short here
    expected = [lam_star(0), lam_star(1)]
    for algo in algos.values():
        np.testing.assert_allclose(algo.lam, expected, atol=1e-2)
    total = sum(algo.actor.x for algo in algos.values())
    np.testing.assert_allclose(total, D.sum(axis=0), atol=5e-2)


async def test_max_iter_stops_everyone():
    result, algos, finished = await run(max_iter=5)
    assert result.stop_reason == "max_iter"
    assert result.iterations == 5
    assert finished == [5, 5, 5, 5]


async def test_single_participant_stops_when_converged():
    # no peers: the agent must stop at its own converged iteration, not run on to max_iter
    result, algos, finished = await run(max_iter=50_000, edges=[("c", "p0")])
    assert result.stop_reason == "converged"
    assert finished == [result.iterations]


async def test_lam_min_floor_on_surplus():
    # negative demand everywhere (supply surplus, x ≥ 0): no λ balances it, unbounded λ would
    # fall forever; with the floor every participant ends at λ = 0 with x = 0 and converges
    result, algos, finished = await run(max_iter=20_000, demand=-D, lam_min=0.0)
    assert result.stop_reason == "converged"
    for algo in algos.values():
        assert np.all(algo.lam == 0.0) and np.all(algo.actor.x == 0.0)


class _Clock:
    """Stub carrier: sleep waits until the test releases it; spawn runs the coroutine."""

    def __init__(self):
        self.release = asyncio.Event()

    async def sleep(self, seconds):
        await self.release.wait()
        self.release.clear()

    def spawn(self, coroutine):
        return asyncio.ensure_future(coroutine)


async def test_settle_ignores_a_stale_converged_snapshot():
    # far agent a flips back to "not converged", but near agent b's "converged" report arrives
    # first (multi-hop delays): the coordinator must not stop on that stale snapshot
    clock, coord = _Clock(), ConsensusCoordinator(["a", "b"], np.zeros(1), settle_s=1.0)
    coord._finished = asyncio.get_running_loop().create_future()
    report = lambda who, ok, k: coord.on_exchange_message(clock, ConsensusReport(who, ok, k), {})  # noqa: E731
    await report("a", True, 10)
    await report("b", True, 11)  # all converged → settle starts, no stop yet
    assert not coord._finished.done()
    await report("a", False, 10)  # the late flip arrives while settling
    clock.release.set()
    await asyncio.sleep(0.01)
    assert not coord._finished.done()
    await report("a", True, 12)  # converged again → settle → stop
    clock.release.set()
    await asyncio.sleep(0.01)
    assert coord.result.stop_reason == "converged" and coord.result.iterations == 12


async def test_converges_with_settle_time():
    result, algos, finished = await run(max_iter=20_000, settle_s=0.05)
    assert result.stop_reason == "converged" and len(finished) == 4


async def test_round_deadline_without_loss_changes_nothing():
    plain, *_ = await run(max_iter=20_000)
    timed, algos, _ = await run(max_iter=20_000, round_timeout=0.02)
    assert timed == plain  # same stop and iteration count
    assert sum(a.rounds_by_deadline for a in algos.values()) == 0


async def test_lossy_rounds_finish_near_the_optimum():
    # 20 % of the λ messages lost (start/reports/stop reliable): without the round deadline one
    # lost λ stalls the run; with it a round uses the peers that arrived (KH12 §IV, ω_q(t)), and
    # decaying steps stay close to the optimum (Theorem 5 setting)
    result, algos, finished = await run(max_iter=20_000, loss=0.2, round_timeout=0.02)
    assert len(finished) == 4
    expected = [lam_star(0), lam_star(1)]
    for algo in algos.values():
        np.testing.assert_allclose(algo.lam, expected, atol=2e-2)
    total = sum(algo.actor.x for algo in algos.values())
    np.testing.assert_allclose(total, D.sum(axis=0), atol=5e-2)
    assert sum(a.rounds_by_deadline for a in algos.values()) > 0
    assert sum(a.missing_edges for a in algos.values()) > 0
