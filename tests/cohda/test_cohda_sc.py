"""COHDA SimpleCarrier tests."""

from __future__ import annotations

import numpy as np
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
    """Low-level SimpleCarrier API: create container and carriers manually.

    With schedules {[0,1,2], [1,2,3]} per agent and target [1.2, 2, 3], the
    optimal combination is both agents picking [0,1,2] (sum [0,2,4]):
    performance −(|1.2−0| + |2−2| + |3−4|) = −2.2.  Every other combination
    scores −3.2 or worse.
    """
    container = ActorContainer()
    algo_one = create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]])
    algo_two = create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]])
    actor_one = SimpleCarrier(container, algo_one)
    actor_two = SimpleCarrier(container, algo_two)

    initial_message = create_cohda_start_message([1.2, 2, 3])
    actor_one.send_to_other(initial_message, cid(actor_two))
    await container.done_event.wait()

    for algo in (algo_one, algo_two):
        candidate = algo.memory.solution_candidate
        assert candidate is not None
        assert candidate.present == frozenset({1, 2})
        assert candidate.perf == pytest.approx(-2.2)
        np.testing.assert_allclose(candidate.schedules.sum(axis=0), [0.0, 2.0, 4.0])


@pytest.mark.asyncio
async def test_cohda_with_simple_carrier_express():
    """Express API: start_distributed_optimization wraps the container.

    Target [2, 4, 6] is exactly reachable (both agents pick [1, 2, 3]), so the
    converged performance must be 0 — the global optimum.
    """
    actor_one = create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]])
    actor_two = create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]])

    initial_message = create_cohda_start_message([2, 4, 6])
    await start_distributed_optimization([actor_one, actor_two], initial_message)

    for algo in (actor_one, actor_two):
        candidate = algo.memory.solution_candidate
        assert candidate is not None
        assert candidate.present == frozenset({1, 2})
        assert candidate.perf == pytest.approx(0.0)
        np.testing.assert_allclose(candidate.schedules.sum(axis=0), [2.0, 4.0, 6.0])
