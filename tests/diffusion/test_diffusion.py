"""Diffusion (adapt-then-combine) in a mango world: balance at the fixed point, stop, passes, loss."""

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

# x_j(λ) = clip((λ − b_j)/a_j, 0, cap_j); demand d_j per hour; ∇J_j = x_j − d_j
A = [1.0, 2.0, 0.5, 1.0]
B = [1.0, 0.0, 2.0, 3.0]
CAP = [10.0, 10.0, 10.0, 1.0]
D = np.array([[2.0, 1.0], [0.0, 3.0], [3.0, 0.0], [1.0, 1.0]])
IRREGULAR = [("p0", "p1"), ("p1", "p2"), ("p2", "p3"), ("p1", "p3")]


class _LosePhi(SimpleCommunicationSimulation):
    """Loses only the φ exchange; start, reports and stop always arrive."""

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
    def __init__(self, a, b, cap, d, flag_passes=0):
        self.a, self.b, self.cap, self.d = a, b, cap, d
        self.flag_passes, self.restarts = flag_passes, 0
        self.x = None

    def gradient(self, lam):
        self.x = np.clip((lam - self.b) / self.a, 0.0, self.cap)
        return self.x - self.d

    def flags(self):
        return self.restarts < self.flag_passes

    def restart(self):
        self.restarts += 1


def lam_star(hour):
    total = D[:, hour].sum()
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        supply = sum(np.clip((mid - b) / a, 0, c) for a, b, c in zip(A, B, CAP))
        lo, hi = (mid, hi) if supply < total else (lo, mid)
    return lo


async def run(
    max_iter=20_000,
    steps=(0.3,) * 4,
    edges=IRREGULAR,
    flag_passes=(0, 0, 0, 0),
    final_stop=True,
    loss=0.0,
    round_timeout=None,
):
    graph = nx.Graph(edges)
    graph.add_edge("p0", "c")  # coordinator: a leaf on p0
    routes = next_hops(graph)
    peers = graph.copy()
    peers.remove_node("c")
    world = create_world(start_time=0.0, communication_sim=_LosePhi(loss))
    finished, algos, agents = [], {}, {}
    for j, aid in enumerate(sorted(peers)):
        algos[aid] = DiffusionAlgorithm(
            lambda alg, carrier: finished.append(alg.k),
            Quadratic(A[j], B[j], CAP[j], D[j], flag_passes[j]),
            steps[j],
            {},  # filled once the addresses exist
            eps=1e-8,
            max_iter=max_iter,
            round_timeout=round_timeout,
        )
        agents[aid] = world.register(
            agent_composed_of(DistributedOptimizationRole(algos[aid], routes=routes[aid])),
            suggested_aid=aid,
        )
    n = dict(peers.degree)
    for aid, algo in algos.items():  # Ces eq. (19)
        algo.weights = {agents[j].addr: 2 / (n[aid] + n[j] + 1) for j in peers[aid]}
    coordinator = DiffusionCoordinator(
        [a.addr for a in agents.values()], np.zeros(2), start_latency=0.03, final_stop=final_stop
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
    return coordinator, algos, finished


def mismatch(algos):
    return sum(a.actor.x - a.actor.d for a in algos.values())


async def test_equal_steps_balance_exactly_near_the_optimum():
    coordinator, algos, finished = await run()
    assert coordinator.result.stop_reason == "converged" and len(finished) == 4
    # doubly-stochastic W + equal ε: Σ_j (x_j − d_j) = 0 at the fixed point (Yuan eq. 11, q ∝ ε)
    np.testing.assert_allclose(mismatch(algos), 0.0, atol=1e-6)
    # constant steps: the λ_j stay spread by O(ε) around λ* (Yuan eq. 14)
    for algo in algos.values():
        np.testing.assert_allclose(algo.lam, [lam_star(0), lam_star(1)], atol=0.5)


async def test_unequal_steps_balance_the_weighted_sum():
    steps = (0.3, 0.1, 0.2, 0.4)
    coordinator, algos, _ = await run(steps=steps)
    assert coordinator.result.stop_reason == "converged"
    weighted = sum(s * (a.actor.x - a.actor.d) for s, a in zip(steps, algos.values()))
    np.testing.assert_allclose(weighted, 0.0, atol=1e-6)
    assert np.max(np.abs(mismatch(algos))) > 1e-3  # off the plain balance (D-204, a result)


async def test_max_iter_stops_everyone():
    coordinator, _, finished = await run(max_iter=5)
    assert coordinator.result.stop_reason == "max_iter" and coordinator.result.iterations == 5
    assert finished == [5, 5, 5, 5]


async def test_flags_start_new_passes_and_k_counts_all():
    one, *_ = await run()
    coordinator, algos, finished = await run(flag_passes=(0, 2, 0, 1))
    assert coordinator.result.stop_reason == "converged" and coordinator.pass_no == 2
    assert [a.actor.restarts for a in algos.values()] == [2, 2, 2, 2]  # every actor restarts
    assert coordinator.result.iterations > 2 * one.result.iterations  # three passes counted
    np.testing.assert_allclose(mismatch(algos), 0.0, atol=1e-6)


async def test_final_stop_off_runs_to_max_iter_but_passes_still_end():
    coordinator, algos, finished = await run(
        max_iter=3000, flag_passes=(1, 0, 0, 0), final_stop=False
    )
    assert coordinator.result.stop_reason == "max_iter" and finished == [3000] * 4
    assert coordinator.pass_no == 1


async def test_single_participant_converges_and_stops():
    coordinator, algos, finished = await run(edges=[])
    assert coordinator.result.stop_reason == "converged"
    assert finished == [coordinator.result.iterations]


async def test_round_deadline_without_loss_changes_nothing():
    plain, *_ = await run()
    timed, algos, _ = await run(round_timeout=0.02)
    assert timed.result.iterations == plain.result.iterations
    assert sum(a.rounds_by_deadline for a in algos.values()) == 0


async def test_lossy_rounds_finish():
    # 20 % of the φ lost: a missing φ_j's weight goes to self; the run still ends
    coordinator, algos, finished = await run(max_iter=3000, loss=0.2, round_timeout=0.02)
    assert len(finished) == 4
    assert sum(a.rounds_by_deadline for a in algos.values()) > 0
    assert sum(a.missing_edges for a in algos.values()) > 0
