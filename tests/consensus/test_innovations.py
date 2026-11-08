"""Consensus + innovations in a mango world: exact optimum on an irregular graph, stop rules."""

from __future__ import annotations

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

from distributed_resource_optimization import (  # noqa: E402
    ConsensusActor,
    ConsensusCoordinator,
    ConsensusInnovationsAlgorithm,
    power_step,
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


async def run(max_iter, alpha=power_step(0.3, 0.6), beta=power_step(0.2, 0.001)):
    # irregular graph p0–p1–p2–p3 plus p1–p3; coordinator "c" a leaf on p0
    graph = nx.Graph([("p0", "p1"), ("p1", "p2"), ("p2", "p3"), ("p1", "p3"), ("c", "p0")])
    routes = next_hops(graph)
    finished = []
    algos = {
        f"p{j}": ConsensusInnovationsAlgorithm(
            lambda alg, carrier: finished.append(alg.k),
            Quadratic(A[j], B[j], CAP[j], D[j]),
            alpha,
            beta,
            eps=1e-6,
            max_iter=max_iter,
        )
        for j in range(4)
    }
    world = create_world(
        start_time=0.0, communication_sim=SimpleCommunicationSimulation(default_delay_s=0.01)
    )
    agents = {
        aid: world.register(
            agent_composed_of(DistributedOptimizationRole(algo, routes=routes[aid])),
            suggested_aid=aid,
        )
        for aid, algo in algos.items()
    }
    coordinator = ConsensusCoordinator([a.addr for a in agents.values()], np.zeros(2))
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
