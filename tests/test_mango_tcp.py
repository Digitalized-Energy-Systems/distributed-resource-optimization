"""The same consensus agents on real mango TCP containers (localhost) instead of a simulated world:
three containers, a sparse topology with routed (multi-hop) messages, wall-clock time."""

from __future__ import annotations

import asyncio
import socket

import numpy as np
import pytest

pytest.importorskip("mango")

import networkx as nx  # noqa: E402
from mango import activate, agent_composed_of, create_tcp_container, graph_topology  # noqa: E402
from mango.express.topology import AGENT_NODE_KEY  # noqa: E402

from distributed_resource_optimization import (  # noqa: E402
    ConsensusCoordinator,
    ConsensusInnovationsAlgorithm,
    ConsensusStart,
    json_codec,
    power_step,
)
from distributed_resource_optimization.carrier.mango_carrier import (  # noqa: E402
    CoordinatorRole,
    DistributedOptimizationRole,
    StartCoordinatedDistributedOptimization,
    next_hops,
)

from .consensus.test_innovations import CAP, IRREGULAR, A, B, D, Quadratic, lam_star  # noqa: E402


def _free_ports(n: int) -> list[int]:
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def test_json_codec_round_trip():
    from mango import AgentAddress

    codec = json_codec()
    a = AgentAddress(("127.0.0.1", 5555), "p0")
    start = ConsensusStart(np.array([1.5, -2.0]), [a], a, 0.25)
    back = codec.decode(codec.encode(start))
    assert back.participants[0] == a and hash(back.coordinator) == hash(a)
    np.testing.assert_array_equal(back.lam, start.lam)
    assert back.lam.dtype == np.float64 and back.t0 == 0.25


async def test_consensus_over_tcp_containers():
    graph = nx.Graph(IRREGULAR)  # p0–p1–p2–p3 + p1–p3, coordinator c a leaf on p0
    routes = next_hops(graph)
    containers = [
        create_tcp_container(("127.0.0.1", port), codec=json_codec()) for port in _free_ports(3)
    ]
    # c alone; p0, p1 together; p2, p3 together — routed messages cross containers
    home = {"c": 0, "p0": 1, "p1": 1, "p2": 2, "p3": 2}
    algos = {
        f"p{j}": ConsensusInnovationsAlgorithm(
            lambda alg, carrier: None,
            Quadratic(A[j], B[j], CAP[j], D[j]),
            power_step(0.3, 0.6),
            power_step(0.2, 0.001),
            eps=1e-6,
            max_iter=20_000,
        )
        for j in range(4)
    }
    agents = {
        aid: containers[home[aid]].register(
            agent_composed_of(DistributedOptimizationRole(algo, routes=routes[aid])),
            suggested_aid=aid,
        )
        for aid, algo in algos.items()
    }
    coordinator = ConsensusCoordinator([a.addr for a in agents.values()], np.zeros(2))
    addresses = {aid: agent.addr for aid, agent in agents.items()}
    role = CoordinatorRole(coordinator, routes=routes["c"], addresses=addresses)
    agents["c"] = containers[0].register(agent_composed_of(role), suggested_aid="c")
    topo = graph_topology(graph)
    for aid, agent in agents.items():
        topo.graph.nodes[aid][AGENT_NODE_KEY].add(agent)
    topo.inject()
    async with activate(*containers):
        agents["c"].handle_message(StartCoordinatedDistributedOptimization(None), {})
        await asyncio.wait_for(role.wait_done(), timeout=60)
    assert coordinator.result.stop_reason == "converged"
    for algo in algos.values():
        np.testing.assert_allclose(algo.lam, [lam_star(0), lam_star(1)], atol=1e-2)
