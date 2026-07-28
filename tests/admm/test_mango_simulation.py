"""End-to-end sharing-ADMM run through :class:`CoordinatorRole` under a mango
discrete-stepped simulation world.

Regression guard for the carrier park-point contract: the coordinator's
request/reply driver awaits per-iteration replies via ``carrier.gather``,
which must declare the park idle to mango's termination detection.  Without
that declaration ``step_simulation`` deadlocks on the first iteration —
exactly the construct SCARE's restoration scenario uses.
"""

from __future__ import annotations

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
)

# Safety cap so a regression can never hang the suite.
_MAX_EVENT_STEPS = 2000


@pytest.mark.asyncio
async def test_sharing_admm_coordinator_completes_under_discrete_stepping():
    """Same setup and expectations as ``test_flex_admm_sharing_partial_fulfillment``,
    but leader-follower over a simulated network with a per-hop delay, driven
    purely by discrete steps."""
    actors = [
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(15, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
    ]
    follower_agents = [agent_composed_of(DistributedOptimizationRole(a)) for a in actors]
    coordinator_role = CoordinatorRole(create_sharing_target_distance_admm_coordinator())
    coord_agent = agent_composed_of(coordinator_role)

    delay_s = 1.0
    comm = SimpleCommunicationSimulation(default_delay_s=delay_s)

    async with run_with_simulation(coord_agent, *follower_agents, communication_sim=comm) as world:
        with create_topology() as topo:
            n_coord = topo.add_node(coord_agent)
            for fa in follower_agents:
                topo.add_edge(n_coord, topo.add_node(fa))

        await world.send_message(
            StartCoordinatedDistributedOptimization(
                create_admm_start(create_admm_sharing_data([0.2, 1, -2]))
            ),
            receiver_addr=coord_agent.addr,
            sender_id=None,
        )

        idle = 0
        for _ in range(_MAX_EVENT_STEPS):
            fut = coordinator_role._done_future
            if fut is not None and fut.done():
                break
            if await step_simulation(world, step_size_s=DISCRETE_EVENT) is None:
                idle += 1
                if idle > 10:
                    raise AssertionError("simulation stalled before the coordinator finished")
            else:
                idle = 0
        else:
            raise AssertionError("coordinator did not finish within the step budget")

        results = await coordinator_role.wait_done()
        assert results is not None
        # Replies cross steps (per-hop delay), so a completed run proves the
        # driver parked and resumed across many simulated round trips.
        assert world.clock.time >= 2 * delay_s

    expected = np.array([0.06667, 0.33333, -0.66667])
    for actor in actors:
        assert np.allclose(actor.x, expected, atol=1e-2)
