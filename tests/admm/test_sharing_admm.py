"""Sharing ADMM tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    create_admm_flex_actor_one_to_many,
    create_admm_sharing_data,
    create_sharing_target_distance_admm_coordinator,
    start_coordinated_optimization,
)
from distributed_resource_optimization.algorithm.admm.sharing_admm import create_admm_start


@pytest.mark.asyncio
async def test_flex_admm_sharing_negative_efficiency_zero():
    """Target [-4, 0, 6] with η=[0.1, 0.5, -1] actors → all zero."""
    actors = [
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(15, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(10, [-1.0, 0.0, 1.0]),
    ]
    coordinator = create_sharing_target_distance_admm_coordinator()
    start = create_admm_start(create_admm_sharing_data([-4, 0, 6]))
    await start_coordinated_optimization(actors, coordinator, start)

    assert np.allclose(actors[0].x, [0, 0, 0], atol=1e-2)
    assert np.allclose(actors[1].x, [0, 0, 0], atol=1e-2)
    assert np.allclose(actors[2].x, [-5.617, 0, 5.617], atol=1e-2)


@pytest.mark.asyncio
async def test_flex_admm_sharing_partial_fulfillment():
    """Target [0.2, 1, -2] — partial fulfillment expected."""
    actors = [
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(15, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
    ]
    coordinator = create_sharing_target_distance_admm_coordinator()
    start = create_admm_start(create_admm_sharing_data([0.2, 1, -2]))
    await start_coordinated_optimization(actors, coordinator, start)

    expected = np.array([0.06667, 0.33333, -0.66667])
    for actor in actors:
        assert np.allclose(actor.x, expected, atol=1e-2)


@pytest.mark.asyncio
async def test_flex_admm_sharing_priority_third():
    """Priority on third element [1,1,5] → third actor handles most."""
    actors = [
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(15, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(10, [-1.0, 0.0, 1.0]),
    ]
    coordinator = create_sharing_target_distance_admm_coordinator()
    start = create_admm_start(create_admm_sharing_data([-4, 0, 6], [5, 1, 1]))
    await start_coordinated_optimization(actors, coordinator, start)

    # Actors 0 and 1 should be near zero; actor 2 should carry most of the load
    assert np.allclose(actors[0].x, [0, 0, 0], atol=1e-2)
    assert np.allclose(actors[1].x, [0, 0, 0], atol=1e-2)
    assert np.allclose(actors[2].x, [-3.983, 0, 3.983], atol=1e-2)


@pytest.mark.asyncio
async def test_flex_admm_sharing_heterogeneous_actors():
    """Heterogeneous actor set with different η configurations."""
    actors = [
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(15, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(10, [1.0, 0.0, -1.0]),
    ]
    coordinator = create_sharing_target_distance_admm_coordinator()
    start = create_admm_start(create_admm_sharing_data([1.2, 1, -4]))
    await start_coordinated_optimization(actors, coordinator, start)

    assert np.allclose(actors[0].x, [0.155, 0.776, -1.553], atol=1e-2)
    assert np.allclose(actors[1].x, [0.155, 0.776, -1.553], atol=1e-2)
    assert np.allclose(actors[2].x, [0.893, 0, -0.893], atol=1e-2)
