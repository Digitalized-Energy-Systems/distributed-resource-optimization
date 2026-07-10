"""Integration tests for MangoCarrier, DistributedOptimizationRole, and CoordinatorRole.

These exercise the mango-agents integration path documented in
``carrier/mango.py``'s module docstring and the README's "Quick Start" —
previously untested despite being a first-class, documented usage path.
"""

from __future__ import annotations

import asyncio

import pytest
from mango import (
    RoleAgent,
    activate,
    agent_composed_of,
    auto_assign,
    complete_topology,
    create_tcp_container,
)

from distributed_resource_optimization import (
    create_admm_flex_actor_one_to_many,
    create_admm_sharing_data,
    create_sharing_target_distance_admm_coordinator,
)
from distributed_resource_optimization.algorithm.admm.sharing_admm import (
    create_sharing_admm_start,
)
from distributed_resource_optimization.algorithm.core import Coordinator
from distributed_resource_optimization.carrier.mango import (
    CoordinatorRole,
    DistributedOptimizationRole,
    StartCoordinatedDistributedOptimization,
)


async def _wait_started_and_done(role: CoordinatorRole, timeout: float = 10.0):
    """Await a CoordinatorRole run, tolerating the start handler's own latency.

    The message handler that creates ``_done_future`` runs as a separate task
    once the start message is delivered, so poll for it to appear — bounded by
    *timeout* so a hung run fails loudly instead of spinning forever.
    """

    async def _inner():
        while role._done_future is None:
            await asyncio.sleep(0.01)
        return await role.wait_done()

    return await asyncio.wait_for(_inner(), timeout)


@pytest.mark.asyncio
async def test_coordinator_wait_done_raises_before_start():
    """wait_done() should raise if the optimization hasn't been kicked off yet."""
    coordinator = create_sharing_target_distance_admm_coordinator()
    role = CoordinatorRole(coordinator)
    container = create_tcp_container(addr=("127.0.0.1", 58432))
    agent = container.register(RoleAgent())
    agent.add_role(role)

    async with activate(container):
        with pytest.raises(RuntimeError, match="not been started"):
            await role.wait_done()


@pytest.mark.asyncio
async def test_mango_carrier_get_address_matches_role_context():
    """MangoCarrier.get_address() should return the hosting role's own address."""
    actor = create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0])
    role = DistributedOptimizationRole(actor)
    container = create_tcp_container(addr=("127.0.0.1", 58433))
    agent = container.register(RoleAgent())
    agent.add_role(role)

    async with activate(container):
        assert role._carrier.get_address() == agent.addr


@pytest.mark.asyncio
async def test_admm_sharing_end_to_end_over_mango_carrier():
    """The sharing-ADMM example from the module docstring / README, run for
    real over mango Roles in a single container, must reproduce the same
    numeric result already verified for SimpleCarrier in
    tests/admm/test_sharing_admm.py::test_flex_admm_sharing_negative_efficiency_zero
    (target [-4, 0, 6], η=[0.1, 0.5, -1] x2, η=[-1, 0, 1] x1).
    """
    actors = [
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(15, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(10, [-1.0, 0.0, 1.0]),
    ]
    coordinator = create_sharing_target_distance_admm_coordinator()

    container = create_tcp_container(addr=("127.0.0.1", 58434))

    opt_roles = [DistributedOptimizationRole(actor) for actor in actors]
    opt_agents = [container.register(agent_composed_of(role)) for role in opt_roles]

    # include_self=False: the coordinator is its own (4th) node in the
    # topology, distinct from the 3 flex-actor participants it messages —
    # it must not also address itself as a participant.
    coord_role = CoordinatorRole(coordinator)
    coord_agent = container.register(agent_composed_of(coord_role))

    all_agents = [*opt_agents, coord_agent]
    auto_assign(complete_topology(len(all_agents)), all_agents)

    async with activate(container):
        start = create_sharing_admm_start(create_admm_sharing_data([-4, 0, 6]))
        await coord_agent.roles[0].context.send_message(
            StartCoordinatedDistributedOptimization(input=start), coord_agent.addr
        )
        await _wait_started_and_done(coord_role)

    assert actors[0].x == pytest.approx([0, 0, 0], abs=1e-2)
    assert actors[1].x == pytest.approx([0, 0, 0], abs=1e-2)
    assert actors[2].x == pytest.approx([-5.617, 0, 5.617], abs=1e-2)


@pytest.mark.asyncio
async def test_coordinator_failure_propagates_to_wait_done():
    """A coordinator that raises must fail wait_done() instead of hanging it.

    Regression test: the run task used to swallow the exception, leaving
    ``_done_future`` unresolved forever.
    """

    class ExplodingCoordinator(Coordinator):
        async def start_optimization(self, carrier, message_data, meta):
            raise ValueError("boom: intentionally failing coordinator")

    role = CoordinatorRole(ExplodingCoordinator())
    container = create_tcp_container(addr=("127.0.0.1", 58435))
    agent = container.register(agent_composed_of(role))

    async with activate(container):
        await agent.roles[0].context.send_message(
            StartCoordinatedDistributedOptimization(input=None), agent.addr
        )
        with pytest.raises(ValueError, match="intentionally failing"):
            await _wait_started_and_done(role, timeout=5.0)
