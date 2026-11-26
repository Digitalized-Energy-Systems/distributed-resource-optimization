"""Exact diffusion (Yuan et al. 2019, Alg. 1) in a mango world: exact minimiser on irregular, ring
and complete graphs; loss."""

from __future__ import annotations

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
    ConsensusCoordinator,
    ExactDiffusionActor,
    ExactDiffusionAlgorithm,
    ExactDiffusionMessage,
)
from distributed_resource_optimization.carrier.mango_carrier import (  # noqa: E402
    CoordinatorRole,
    DistributedOptimizationRole,
    StartCoordinatedDistributedOptimization,
    next_hops,
)

# J_k(w) = h_k/2 ‖w − c_k‖² over 2 periods: heterogeneous, so plain diffusion keeps an O(μ) bias
H = [1.0, 2.0, 0.5, 4.0]
C = np.array([[10.0, -3.0], [0.0, 5.0], [20.0, 1.0], [-5.0, 8.0]])
OPT = sum(h * c for h, c in zip(H, C)) / sum(H)  # minimiser of Σ_k J_k
IRREGULAR = [("p0", "p1"), ("p1", "p2"), ("p2", "p3"), ("p1", "p3")]
RING = [("p0", "p1"), ("p1", "p2"), ("p2", "p3"), ("p3", "p0")]
COMPLETE = [(f"p{a}", f"p{b}") for a in range(4) for b in range(a + 1, 4)]


class Quadratic(ExactDiffusionActor):
    def __init__(self, h, c):
        self.h, self.c = h, c

    def start(self):
        return np.zeros(2)

    def gradient(self, w):
        return self.h * (w - self.c)


class _Lose(SimpleCommunicationSimulation):
    """Loses a share of the exact-diffusion packets only."""

    def __init__(self, loss):
        super().__init__(default_delay_s=0.01, loss_percent=loss, rng=random.Random(0))

    def calculate_communication(self, current_time, messages):
        result = super().calculate_communication(current_time, messages)
        result.package_results[:] = [
            r if isinstance(m.content[0], ExactDiffusionMessage) else PackageResult(True, r.delay_s)
            for m, r in zip(messages, result.package_results)
        ]
        return result


async def run(edges, max_iter=1500, step=0.1, loss=0.0, timeout=None):
    graph = nx.Graph(edges)
    graph.add_edge("p0", "c")  # coordinator: a leaf on p0
    routes = next_hops(graph)
    world = create_world(start_time=0.0, communication_sim=_Lose(loss))
    finished, algos, agents = [], {}, {}
    for j in range(4):
        aid = f"p{j}"
        algos[aid] = ExactDiffusionAlgorithm(
            lambda alg, carrier: finished.append(alg.k),
            Quadratic(H[j], C[j]),
            step=step,
            max_iter=max_iter,
            round_timeout=timeout,
        )
        agents[aid] = world.register(
            agent_composed_of(DistributedOptimizationRole(algos[aid], routes=routes[aid])),
            suggested_aid=aid,
        )
    coordinator = ConsensusCoordinator([a.addr for a in agents.values()], (), start_latency=0.03)
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
    return coordinator, algos, finished


@pytest.mark.parametrize(
    "edges", [IRREGULAR, RING, COMPLETE], ids=["irregular", "ring", "complete"]
)
async def test_exact_minimiser(edges):
    # Yuan Part I Fig. 4: exact diffusion converges to w° itself, not an O(μ) neighbourhood
    coordinator, algos, finished = await run(edges)
    assert coordinator.result.stop_reason == "max_iter" and finished == [1500] * 4
    for algo in algos.values():
        np.testing.assert_allclose(algo.w, OPT, atol=1e-8)


async def test_lossy_rounds_end_by_deadline():
    # a missing φ's weight goes to self that round (D-240 rule): the run still ends at max_iter
    # near the minimiser (the lost rounds break local balance only while they happen)
    coordinator, algos, finished = await run(IRREGULAR, max_iter=600, loss=0.2, timeout=0.02)
    assert coordinator.result.stop_reason == "max_iter" and finished == [600] * 4
    assert sum(a.rounds_by_deadline for a in algos.values()) > 0
    assert sum(a.missing_edges for a in algos.values()) > 0
    for algo in algos.values():
        np.testing.assert_allclose(algo.w, OPT, atol=1.0)
