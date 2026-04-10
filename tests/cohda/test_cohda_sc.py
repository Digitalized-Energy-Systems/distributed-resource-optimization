"""COHDA SimpleCarrier tests."""

from __future__ import annotations

import pytest

from distributed_resource_optimization import (
    ActorContainer,
    SimpleCarrier,
    cid,
    create_cohda_participant,
    create_cohda_start_message,
    start_distributed_optimization,
)


@pytest.mark.asyncio
async def test_cohda_with_simple_carrier():
    """Low-level SimpleCarrier API: create container and carriers manually."""
    container = ActorContainer()
    algo_one = create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]])
    algo_two = create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]])
    actor_one = SimpleCarrier(container, algo_one)
    actor_two = SimpleCarrier(container, algo_two)

    initial_message = create_cohda_start_message([1.2, 2, 3])
    actor_one.send_to_other(initial_message, cid(actor_two))
    await container.done_event.wait()

    assert actor_one.actor.memory.solution_candidate is not None
    assert actor_one.actor.memory.solution_candidate.perf < 0


@pytest.mark.asyncio
async def test_cohda_with_simple_carrier_express():
    """Express API: start_distributed_optimization wraps the container."""
    actor_one = create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]])
    actor_two = create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]])

    initial_message = create_cohda_start_message([1.2, 2, 3])
    await start_distributed_optimization([actor_one, actor_two], initial_message)

    assert actor_one.memory.solution_candidate is not None
    assert actor_one.memory.solution_candidate.perf < 0
