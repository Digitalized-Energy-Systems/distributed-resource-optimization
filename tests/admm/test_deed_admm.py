"""DEED-ADMM in a mango world: exact optimum of a box-constrained exchange problem, Lemma 2
conservation, lossy rounds (S6b step 3)."""

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
    ConsensusStart,
    DEEDADMMAlgorithm,
    DEEDMessage,
    DEEDParticipant,
)
from distributed_resource_optimization.carrier.mango_carrier import (  # noqa: E402
    CoordinatorRole,
    DistributedOptimizationRole,
    StartCoordinatedDistributedOptimization,
    next_hops,
)

# min Σ a_j/2·x_j² + b_j·x_j  s.t.  Σ x_j = Σ d_j per hour, 0 ≤ x_j ≤ cap_j (y_j = d_j fixed)
A = [1.0, 2.0, 0.5, 1.0]
B = [1.0, 0.0, 2.0, 3.0]
CAP = [10.0, 10.0, 10.0, 1.0]
D = np.array([[2.0, 1.0], [0.0, 3.0], [3.0, 0.0], [1.0, 1.0]])
GAMMA = 1.0
IRREGULAR = [("p0", "p1"), ("p1", "p2"), ("p2", "p3"), ("p1", "p3"), ("c", "p0")]


class BoxQuadratic(DEEDParticipant):
    """Algorithm 1 lines 5–11, 14 for x ∈ [0, cap] (p̂ = x̂, X = I) and y = d (A = 1, no z)."""

    def __init__(self, a, b, cap, d):
        self.a, self.b, self.cap, self.d = a, b, cap, d
        self.x = np.zeros_like(d)
        self.x_hat, self.nu = np.zeros_like(d), np.zeros_like(d)
        self.dual = np.inf

    def coupling(self):
        return self.x - self.d  # y_0 = d: the y-update (lines 6, 9) keeps y = d

    def update(self, lam_t, xi_t):
        b_x = GAMMA * self.x - lam_t + (GAMMA * self.x_hat - self.nu) - GAMMA / 2 * xi_t  # line 5
        x = (b_x - self.b) / (self.a + 2 * GAMMA)  # line 8: (∇c + γ(I + XᵀX))⁻¹
        x_hat = np.clip(x + self.nu / GAMMA, 0.0, self.cap)  # line 11
        self.nu = self.nu + GAMMA * (x - x_hat)  # line 14
        self.dual = GAMMA * np.linalg.norm(x_hat - self.x_hat)
        delta, self.x, self.x_hat = x - self.x, x, x_hat
        return delta

    def x_norm(self):
        return float(np.linalg.norm(self.x))

    def converged(self, eps_abs, eps_rel):
        return self.dual <= np.sqrt(self.x_hat.size) * eps_abs + eps_rel * np.linalg.norm(self.nu)


def price_star(hour):
    """μ* with Σ clip((μ − b_j)/a_j, 0, cap_j) = Σ d_j (DEED's λ* = −μ*)."""
    total, lo, hi = D[:, hour].sum(), 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        supply = sum(np.clip((mid - b) / a, 0, c) for a, b, c in zip(A, B, CAP))
        lo, hi = (mid, hi) if supply < total else (lo, mid)
    return lo


class _LoseExchange(SimpleCommunicationSimulation):
    """Loses only the (λ, ξ) exchange (DEEDMessage); start, reports and stop always arrive."""

    def __init__(self, loss):
        super().__init__(default_delay_s=0.01, loss_percent=loss, rng=random.Random(0))

    def calculate_communication(self, current_time, messages):
        result = super().calculate_communication(current_time, messages)
        result.package_results[:] = [
            r if isinstance(m.content[0], DEEDMessage) else PackageResult(True, r.delay_s)
            for m, r in zip(messages, result.package_results)
        ]
        return result


async def run(max_iter, loss=0.0, round_timeout=None, eps_abs=1e-7, eps_rel=1e-6):
    graph = nx.Graph(IRREGULAR)
    routes = next_hops(graph)
    finished = []
    algos = {
        f"p{j}": DEEDADMMAlgorithm(
            lambda alg, carrier: finished.append(alg.k),
            BoxQuadratic(A[j], B[j], CAP[j], D[j]),
            beta=0.2,
            gamma=GAMMA,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            round_timeout=round_timeout,
        )
        for j in range(4)
    }
    world = create_world(start_time=0.0, communication_sim=_LoseExchange(loss))
    agents = {
        aid: world.register(
            agent_composed_of(DistributedOptimizationRole(algo, routes=routes[aid])),
            suggested_aid=aid,
        )
        for aid, algo in algos.items()
    }
    coordinator = ConsensusCoordinator(
        [a.addr for a in agents.values()], np.zeros(2), start_latency=0.03
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
    assert result.stop_reason == "converged" and len(finished) == 4
    mu = np.array([price_star(0), price_star(1)])
    for j, algo in enumerate(algos.values()):
        np.testing.assert_allclose(-algo.lam, mu, atol=1e-4)  # one common price
        np.testing.assert_allclose(
            algo.participant.x_hat, np.clip((mu - B[j]) / A[j], 0, CAP[j]), atol=1e-4
        )
    total = sum(algo.participant.x_hat for algo in algos.values())
    np.testing.assert_allclose(total, D.sum(axis=0), atol=1e-4)


async def test_tracker_conserves_the_coupling_sum():
    # Lemma 2: Σ ξ_i,k = Σ (x_i,k − y_i,k) every iteration (W doubly stochastic, lossless)
    result, algos, finished = await run(max_iter=30)
    assert result.stop_reason == "max_iter" and finished == [30, 30, 30, 30]
    xi = sum(algo.xi for algo in algos.values())
    coupling = sum(algo.participant.x - algo.participant.d for algo in algos.values())
    np.testing.assert_allclose(xi, coupling, atol=1e-12)


class _Carrier:
    def __init__(self):
        self.sent = []

    def get_address(self):
        return "p0"

    def send_to_other(self, content, receiver):
        self.sent.append((content, receiver))


class _Still(DEEDParticipant):
    def update(self, lam_t, xi_t):
        return np.zeros_like(xi_t)

    def x_norm(self):
        return 0.0


def test_missing_edge_weight_goes_to_self():
    # a deadline round with only peer a's message: λ̃ = λ + β(λ_a − λ), b's edge is dropped (Q-Z6 A)
    algo = DEEDADMMAlgorithm(lambda *_: None, _Still(), beta=0.25, gamma=2.0)
    algo._start = ConsensusStart(np.zeros(1), ["p0", "a", "b"], "c")
    algo._peers = ["a", "b"]
    algo.lam, algo.xi = np.array([1.0]), np.array([4.0])
    algo._inbox[0].append(DEEDMessage(np.array([5.0]), np.array([0.0]), 0))
    algo._iterate(_Carrier())
    xi = 4.0 + 0.25 * (0.0 - 4.0)
    assert algo.xi == pytest.approx([xi])
    assert algo.lam == pytest.approx([1.0 + 0.25 * (5.0 - 1.0) + 2.0 / 2 * xi])
    assert algo.k == 1


async def test_lossy_rounds_do_not_block():
    # 20 % of the (λ, ξ) messages lost: the round deadline keeps every participant iterating
    # until the end; the fixed point moves (expected, plan §4 Loss) — only progress is checked
    result, algos, finished = await run(max_iter=300, loss=0.2, round_timeout=0.02)
    assert len(finished) == 4
    assert sum(a.rounds_by_deadline for a in algos.values()) > 0
    assert sum(a.missing_edges for a in algos.values()) > 0
