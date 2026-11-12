"""Routed (multi-hop) carrier messages: sharing ADMM on a chain gives the same result as on a star."""

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
    create_admm_flex_actor_one_to_many,
    create_admm_sharing_data,
    create_sharing_target_distance_admm_coordinator,
)
from distributed_resource_optimization.algorithm.admm.sharing_admm import (  # noqa: E402
    create_admm_start,
)
from distributed_resource_optimization.carrier.mango_carrier import (  # noqa: E402
    CoordinatorRole,
    DistributedOptimizationRole,
    StartCoordinatedDistributedOptimization,
    next_hops,
)


def test_next_hops():
    routes = next_hops(nx.path_graph(["a", "b", "c"]))
    assert routes["a"] == {"b": "b", "c": "b"}
    assert routes["c"] == {"b": "b", "a": "b"}


async def test_sharing_admm_on_a_chain():
    chain = nx.path_graph(["c", "f0", "f1", "f2"])
    routes = next_hops(chain)
    actors = [
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(15, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
    ]
    coordinator = CoordinatorRole(
        create_sharing_target_distance_admm_coordinator(), routes=routes["c"]
    )
    roles = {"c": coordinator}
    roles |= {
        f"f{i}": DistributedOptimizationRole(a, routes=routes[f"f{i}"])
        for i, a in enumerate(actors)
    }

    world = create_world(
        start_time=0.0, communication_sim=SimpleCommunicationSimulation(default_delay_s=1.0)
    )
    agents = {
        aid: world.register(agent_composed_of(role), suggested_aid=aid)
        for aid, role in roles.items()
    }
    topo = graph_topology(chain)
    for aid, agent in agents.items():
        topo.graph.nodes[aid][AGENT_NODE_KEY].add(agent)
    topo.inject()

    async with world:
        await world.send_message(
            StartCoordinatedDistributedOptimization(
                create_admm_start(create_admm_sharing_data([0.2, 1, -2]))
            ),
            receiver_addr=agents["c"].addr,
            sender_id=None,
        )
        for _ in range(5000):
            if coordinator._done_future is not None and coordinator._done_future.done():
                break
            await world.step(DISCRETE_EVENT)
        else:
            raise AssertionError("coordinator did not finish")

    assert [a.aid for a in coordinator._carrier.others("c")] == ["f0", "f1", "f2"]
    # every hop crossed a chain edge: c reached f1/f2 only via relays
    hops = [(m.sender_id, m.receiver_id) for m in world.recorded_messages if m.sender_id]
    assert hops and all(chain.has_edge(*hop) for hop in hops)
    assert ("f0", "f1") in hops and ("f1", "f2") in hops
    expected = np.array([0.06667, 0.33333, -0.66667])
    for actor in actors:
        assert np.allclose(actor.x, expected, atol=1e-2)


def test_control_messages_are_marked_and_unwrapped():
    from mango import AgentAddress

    from distributed_resource_optimization import (
        ConsensusMessage,
        ConsensusReport,
        ConsensusStart,
        ControlMessage,
        OptimizationFinishedMessage,
        StartCoordinatedDistributedOptimization,
        unwrap,
    )
    from distributed_resource_optimization.carrier.mango_carrier import _Routed

    control = [
        ConsensusStart(np.zeros(1), [], None),
        ConsensusReport(None, True, 1),
        StartCoordinatedDistributedOptimization(None),
        OptimizationFinishedMessage(None),
    ]
    assert all(isinstance(m, ControlMessage) for m in control)
    lam = ConsensusMessage(np.zeros(1), 1)  # the algorithm's own payload
    assert not isinstance(lam, ControlMessage)
    a = AgentAddress("c", "a")
    assert unwrap(_Routed(control[0], a, a)) is control[0] and unwrap(lam) is lam
