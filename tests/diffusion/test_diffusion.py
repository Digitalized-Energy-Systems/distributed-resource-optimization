"""Diffusion (de Azevedo et al. 2017) in a mango world: information sharing with neighbour flags
and restarts, optimization with the nudge, silent neighbours, loss."""

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
    DiffusionActor,
    DiffusionAlgorithm,
    DiffusionMessage,
)
from distributed_resource_optimization.algorithm.firstorder.diffusion import (  # noqa: E402
    diffusion as module,
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


class _Lose(SimpleCommunicationSimulation):
    """Loses only diffusion packets: a share of all, or every packet of *silent*."""

    def __init__(self, loss, silent=None):
        super().__init__(default_delay_s=0.01, loss_percent=loss, rng=random.Random(0))
        self.silent = silent

    def calculate_communication(self, current_time, messages):
        result = super().calculate_communication(current_time, messages)
        out = []
        for m, r in zip(messages, result.package_results):
            packet = isinstance(m.content[0], DiffusionMessage)
            if packet and m.content[0].sender.aid == self.silent:
                out.append(PackageResult(False, r.delay_s))
            else:
                out.append(r if packet else PackageResult(True, r.delay_s))
        result.package_results[:] = out
        return result


class Quadratic(DiffusionActor):
    def __init__(self, b, c, d):
        self.b, self.c, self.d = b, c, d

    def measure(self):
        return self.d

    def gradient(self, w_own):
        return self.b + 2 * self.c * w_own


def optimum():
    """Equal marginal cost λ with Σ w = Σ d over the active participants, per period."""
    inv = sum(1 / (2 * C[j]) for j in ACTIVE)
    lam = (D.sum(axis=0) + sum(B[j] / (2 * C[j]) for j in ACTIVE)) / inv
    return np.array([(lam - B[j]) / (2 * C[j]) for j in ACTIVE])


@pytest.fixture
def lazy(monkeypatch):
    """(I + A)/2: eigenvalues ≥ 0, so the printed nudge (D-226) is stable."""
    monkeypatch.setattr(module, "metropolis", lambda n_i, n_j: 0.5 / max(n_i, n_j))


async def run(max_iter=4000, step=0.01, step_is=0.0, loss=0.0, silent=None, timeout=None):
    graph = nx.Graph(IRREGULAR)
    diameter = nx.diameter(graph)
    graph.add_edge("p0", "c")  # coordinator: a leaf on p0
    routes = next_hops(graph)
    world = create_world(start_time=0.0, communication_sim=_Lose(loss, silent))
    finished, algos, agents = [], {}, {}
    for j in range(4):
        aid = f"p{j}"
        algos[aid] = DiffusionAlgorithm(
            lambda alg, carrier: finished.append(alg.k),
            Quadratic(B[j], C[j], D[j]),
            own=j,
            active=ACTIVE,
            entries=4,
            step=step,
            step_is=step_is,
            eps_is=1e-9,
            diameter=diameter,
            max_iter=max_iter,
            round_timeout=timeout,
        )
        agents[aid] = world.register(
            agent_composed_of(DistributedOptimizationRole(algos[aid], routes=routes[aid])),
            suggested_aid=aid,
        )
    coordinator = ConsensusCoordinator([a.addr for a in agents.values()], start_latency=0.03)
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
    return np.array([algos[f"p{j}"].w[j] for j in ACTIVE])


def test_eq12_as_printed():
    assert module.metropolis(2, 3) == 1 / 3 and module.metropolis(1, 1) == 1.0


async def test_flags_restarts_and_optimum(lazy):
    coordinator, algos, finished = await run()
    assert coordinator.result.stop_reason == "max_iter" and finished == [4000] * 4
    resets = {a.resets for a in algos.values()}
    assert len(resets) == 1 and resets.pop() > 1  # one reset round for all, and it restarts
    for algo in algos.values():
        np.testing.assert_allclose(algo.target, D.sum(axis=0), atol=1e-6)  # P_N = n·x̄
        np.testing.assert_allclose(algo.w.sum(axis=0), D.sum(axis=0), atol=1e-6)
        assert np.all(algo.w[3] == 0)  # the uninformed entry stays 0
    # one μ: equal marginal costs; the own entries sit O(μ) below the mean estimate
    np.testing.assert_allclose(own(algos), optimum(), atol=2.0)


async def test_is_step_damps_the_averaging(lazy):
    _, algos, _ = await run(max_iter=500, step_is=0.5)
    for algo in algos.values():
        np.testing.assert_allclose(algo.target, D.sum(axis=0), atol=1e-6)


async def test_printed_nudge_diverges_with_negative_eigenvalues():
    # eq. (21) as printed (D-226): with eq. (12) weights this graph has an eigenvalue < 0, so
    # the estimate's sum error grows as (A − I)^k — a result of the paper's update
    _, algos, _ = await run(max_iter=3000)
    assert not all(np.all(np.isfinite(a.w)) and np.abs(a.w).max() < 1e6 for a in algos.values())


async def test_silent_neighbour_is_dropped(lazy):
    # p3 never gets a packet through: its neighbours drop it after d rounds (§III-A) and their
    # degrees (eq. 12) follow
    _, algos, finished = await run(max_iter=300, silent="p3", timeout=0.02)
    assert len(finished) == 4
    for aid in ("p1", "p2"):
        assert all(p.aid != "p3" for p in algos[aid]._peers)


async def test_lossy_rounds_finish(lazy):
    _, algos, finished = await run(max_iter=1000, loss=0.2, timeout=0.02)
    assert len(finished) == 4
    assert sum(a.rounds_by_deadline for a in algos.values()) > 0
    assert sum(a.missing_edges for a in algos.values()) > 0
