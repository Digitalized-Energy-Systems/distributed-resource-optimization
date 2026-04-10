"""Consensus ADMM tests."""

from __future__ import annotations

import pytest
import numpy as np

from distributed_resource_optimization import (
    ADMMFlexActor,
    create_admm_flex_actor_one_to_many,
    create_consensus_target_reach_admm_coordinator,
    create_admm_start_consensus,
    start_coordinated_optimization,
)


@pytest.mark.asyncio
async def test_consensus_admm_basic():
    """All actors converge to the same x and their sum reaches the target.

    With η=[0.1, 0.5] and in_cap=10, each actor's feasible set satisfies
    x[0] = 0.2·x[1] (coupling constraint).  The target [1.5, 7.5] = 3·[0.5, 2.5]
    is reachable with x=[0.5, 2.5] per actor (0.5 = 0.2·2.5 ✓).
    """
    # Feasible target: each actor contributes x=[0.5, 2.5], sum=[1.5, 7.5]
    target = np.array([1.5, 7.5])
    actors = [
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5]),
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5]),
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5]),
    ]
    coordinator = create_consensus_target_reach_admm_coordinator()
    start = create_admm_start_consensus(target)
    results = await start_coordinated_optimization(actors, coordinator, start)

    # All actors should agree (consensus property)
    for i in range(1, len(results)):
        assert np.allclose(results[0], results[i], atol=1e-2)

    # Sum should reach the target
    total = sum(results)
    assert np.allclose(total, target, atol=0.5)


@pytest.mark.asyncio
async def test_consensus_admm_two_actors_conv_create():
    """Two identical actors converge to the same allocation.

    With η=[0.6, 0.4] both actors should reach x≈[0.816, 0.544].
    """
    actors = [
        create_admm_flex_actor_one_to_many(10, [0.6, 0.4]),
        create_admm_flex_actor_one_to_many(10, [0.6, 0.4]),
    ]
    coordinator = create_consensus_target_reach_admm_coordinator()
    start = create_admm_start_consensus([1.0, 2.0])
    results = await start_coordinated_optimization(actors, coordinator, start)

    expected = np.array([0.8163816641254231, 0.5442936838263125])
    assert np.allclose(results[0], expected, rtol=1e-2)
    assert np.allclose(results[1], expected, rtol=1e-2)


@pytest.mark.asyncio
async def test_consensus_admm_three_actors_coord_as_actor():
    """Three identical actors with coordinator-as-actor converge uniformly.

    All three actors should reach x≈[0.545, 0.364].
    """
    actors = [
        create_admm_flex_actor_one_to_many(10, [0.6, 0.4]),
        create_admm_flex_actor_one_to_many(10, [0.6, 0.4]),
        create_admm_flex_actor_one_to_many(10, [0.6, 0.4]),
    ]
    coordinator = create_consensus_target_reach_admm_coordinator()
    start = create_admm_start_consensus([1.0, 2.0])
    results = await start_coordinated_optimization(actors, coordinator, start)

    expected = np.array([0.545531954256762, 0.3637335132272603])
    for r in results:
        assert np.allclose(r, expected, rtol=1e-2)


@pytest.mark.asyncio
async def test_consensus_admm_negative_efficiency_all_zero():
    """Actors with negative efficiency η=-1 should reach zero.

    Target [2, 2, 3] is infeasible for these actors → all converge to zero.
    """
    actors = [
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
    ]
    coordinator = create_consensus_target_reach_admm_coordinator()
    start = create_admm_start_consensus([2.0, 2.0, 3.0])
    results = await start_coordinated_optimization(actors, coordinator, start)

    for r in results:
        assert np.allclose(r, [0.0, 0.0, 0.0], atol=1e-3)


@pytest.mark.asyncio
async def test_consensus_admm_complex_four_actors():
    """Four heterogeneous actors with tight box constraints.

    Each actor is at its upper bound, so the solution reproduces the box limits.
    """
    flex_actor1 = ADMMFlexActor(
        l=[0.0, 0.0, 0.0],
        u=[6.428571428571429, 0.0, 4.5],
        C=np.array([
            [1.0, 1.0, 1.0],
            [0.15555555555555556, 0.0, -0.2222222222222222],
            [-0.15555555555555556, 0.0, 0.2222222222222222],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
        d=np.array([10.928571428571429, 0.0, 0.0, 0.0, 0.0]),
        S=np.array([0.0, 0.0, 0.0]),
    )
    flex_actor2 = ADMMFlexActor(
        l=[0.0, 0.0, 0.0],
        u=[0.04000000000000001, 0.06, 0.1],
        C=np.array([
            [1.0, 1.0, 1.0],
            [24.999999999999996, 0.0, -10.0],
            [-24.999999999999996, 0.0, 10.0],
            [0.0, 16.666666666666668, -10.0],
            [0.0, -16.666666666666668, 10.0],
        ]),
        d=np.array([0.2, 0.0, 0.0, 0.0, 0.0]),
        S=np.array([0.0, 0.0, 0.0]),
    )
    flex_actor3 = ADMMFlexActor(
        l=[0.0, 0.0, 0.0],
        u=[0.3, 0.0, 0.3333333333333333],
        C=np.array([
            [1.0, 1.0, 1.0],
            [3.3333333333333335, 0.0, -3.0],
            [-3.3333333333333335, 0.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
        d=np.array([0.6333333333333333, 0.0, 0.0, 0.0, 0.0]),
        S=np.array([0.0, 0.0, 0.0]),
    )
    flex_actor4 = ADMMFlexActor(
        l=[0.0, 0.0, 0.0],
        u=[1.5, 0.0, 1.6666666666666665],
        C=np.array([
            [1.0, 1.0, 1.0],
            [0.6666666666666666, 0.0, -0.6000000000000001],
            [-0.6666666666666666, 0.0, 0.6000000000000001],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
        d=np.array([3.1666666666666665, 0.0, 0.0, 0.0, 0.0]),
        S=np.array([0.0, 0.0, 0.0]),
    )
    actors = [flex_actor1, flex_actor2, flex_actor3, flex_actor4]
    coordinator = create_consensus_target_reach_admm_coordinator()
    start = create_admm_start_consensus([22.559000761215636, -0.0, 22.559000761215636])
    results = await start_coordinated_optimization(actors, coordinator, start)

    assert np.allclose(results[0], [6.42853816370231, -1.785956854356477e-6, 4.499978435684416], atol=1e-2)
    assert np.allclose(results[1], [0.040013413013280125, 0.0600214071909219, 0.10000099993773309], atol=1e-3)
    assert np.allclose(results[2], [0.3000377811917589, -2.683958412610146e-7, 0.33336868924468077], atol=1e-3)
