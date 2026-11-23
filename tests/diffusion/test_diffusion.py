"""Diffusion (de Azevedo et al. 2017) in a mango world: IS pass, optimization with nudge, loss."""

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
    DiffusionActor,
    DiffusionAlgorithm,
    DiffusionCoordinator,
    DiffusionMessage,
)
from distributed_resource_optimization.carrier.mango_carrier import (  # noqa: E402
    CoordinatorRole,
    DistributedOptimizationRole,
    StartCoordinatedDistributedOptimization,
    next_hops,
)

# J_j(w) = b_j w + c_j w² (p3 has no cost: uninformed); local demand d_j per period
B = [10.0, 20.0, 15.0, 0.0]
C = [0.5, 0.25, 1.0, 0.0]
D = np.array([[30.0, 10.0], [0.0, 40.0], [20.0, 5.0], [10.0, 25.0]])
ACTIVE = [0, 1, 2]
IRREGULAR = [("p0", "p1"), ("p1", "p2"), ("p2", "p3"), ("p1", "p3")]


class _LoseValues(SimpleCommunicationSimulation):
    """Loses only the diffusion exchange; start, reports and stop always arrive."""

    def __init__(self, loss):
        super().__init__(default_delay_s=0.01, loss_percent=loss, rng=random.Random(0))

    def calculate_communication(self, current_time, messages):
        result = super().calculate_communication(current_time, messages)
        result.package_results[:] = [
            r if isinstance(m.content[0], DiffusionMessage) else PackageResult(True, r.delay_s)
            for m, r in zip(messages, result.package_results)
        ]
        return result


class Quadratic(DiffusionActor):
    def __init__(self, b, c, d):
        self.b, self.c, self.d = b, c, d

    def measure(self, periods):
        return self.d

    def gradient(self, w_own):
        return self.b + 2 * self.c * w_own


def optimum():
    """Equal marginal cost λ with Σ w = Σ d over the active participants, per period."""
    inv = sum(1 / (2 * C[j]) for j in ACTIVE)
    lam = (D.sum(axis=0) + sum(B[j] / (2 * C[j]) for j in ACTIVE)) / inv
    return np.array([(lam - B[j]) / (2 * C[j]) for j in ACTIVE])


async def run(
    max_iter=3000, step=0.01, step_is=0.0, edges=IRREGULAR, loss=0.0, round_timeout=None, lazy=True
):
    graph = nx.Graph(edges)
    graph.add_nodes_from([f"p{j}" for j in range(4)])
    n = dict(graph.degree)
    graph.add_edge("p0", "c")  # coordinator: a leaf on p0
    routes = next_hops(graph)
    world = create_world(start_time=0.0, communication_sim=_LoseValues(loss))
    finished, algos, agents = [], {}, {}
    for j in range(4):
        aid = f"p{j}"
        algos[aid] = DiffusionAlgorithm(
            lambda alg, carrier: finished.append(alg.k),
            Quadratic(B[j], C[j], D[j]),
            {},  # filled once the addresses exist
            own=j,
            active=ACTIVE,
            entries=4,
            step=step,
            step_is=step_is,
            eps_is=1e-9,
            max_iter=max_iter,
            round_timeout=round_timeout,
        )
        agents[aid] = world.register(
            agent_composed_of(DistributedOptimizationRole(algos[aid], routes=routes[aid])),
            suggested_aid=aid,
        )
    for aid, algo in algos.items():  # de Azevedo eq. (12); lazy = (I + A)/2, eigenvalues ≥ 0
        algo.weights = {
            agents[j].addr: (0.5 if lazy else 1.0) / max(n[aid], n[j])
            for j in graph[aid]
            if j != "c"
        }
    coordinator = DiffusionCoordinator([a.addr for a in agents.values()], np.zeros(2), 0.0, 0.03)
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


def own(algos):
    return np.array([algos[f"p{j}"].value[j] for j in ACTIVE])


async def test_is_pass_then_optimum_with_one_step():
    coordinator, algos, finished = await run(max_iter=6000)
    assert coordinator.result.stop_reason == "max_iter" and finished == [6000] * 4
    assert coordinator.pass_no == 1
    for algo in algos.values():  # IS: P_N = n·x = Σ d (eq. 10)
        np.testing.assert_allclose(algo.target, D.sum(axis=0), atol=1e-6)
        np.testing.assert_allclose(algo.value.sum(axis=0), D.sum(axis=0), atol=1e-6)
        assert np.all(algo.value[3] == 0)  # the uninformed entry stays 0
    # one μ: equal marginal costs; the own entries sit O(μ) below the mean estimate
    # (μ = 0.01 here: up to 1.9 off; 0.1 / 0.01 / 0.001 gave 7.7 / 0.81 / 0.12 with Σφ)
    np.testing.assert_allclose(own(algos), optimum(), atol=2.0)


async def test_printed_nudge_diverges_with_negative_eigenvalues():
    # eq. (21) as printed (D-226): with eq. (12) weights this graph has an eigenvalue < 0, so
    # the estimate's sum error grows as (A − I)^k — a result of the paper's update
    _, algos, _ = await run(max_iter=3000, lazy=False)
    assert not all(
        np.all(np.isfinite(a.value)) and np.abs(a.value).max() < 1e6 for a in algos.values()
    )


async def test_is_step_damps_the_averaging():
    coordinator, algos, _ = await run(step_is=0.5)
    assert coordinator.pass_no == 1
    for algo in algos.values():
        np.testing.assert_allclose(algo.target, D.sum(axis=0), atol=1e-6)


async def test_lossy_rounds_finish():
    coordinator, algos, finished = await run(max_iter=1000, loss=0.2, round_timeout=0.02)
    assert len(finished) == 4
    assert sum(a.rounds_by_deadline for a in algos.values()) > 0
    assert sum(a.missing_edges for a in algos.values()) > 0
