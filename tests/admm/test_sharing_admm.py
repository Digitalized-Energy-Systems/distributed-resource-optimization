"""Sharing ADMM tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    ADMMGeneratorSpec,
    create_admm_economic_dispatch_actor,
    create_admm_flex_actor_one_to_many,
    create_admm_proximal_storage_actor,
    create_admm_sharing_data,
    create_sharing_target_distance_admm_coordinator,
    start_coordinated_optimization,
)
from distributed_resource_optimization.algorithm.admm.sharing_admm import create_sharing_admm_start


@pytest.mark.asyncio
async def test_flex_admm_sharing_negative_efficiency_zero():
    """Target [-4, 0, 6] with η=[0.1, 0.5, -1] actors → all zero."""
    actors = [
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(15, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(10, [-1.0, 0.0, 1.0]),
    ]
    coordinator = create_sharing_target_distance_admm_coordinator()
    start = create_sharing_admm_start(create_admm_sharing_data([-4, 0, 6]))
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
    start = create_sharing_admm_start(create_admm_sharing_data([0.2, 1, -2]))
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
    start = create_sharing_admm_start(create_admm_sharing_data([-4, 0, 6], [5, 1, 1]))
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
    start = create_sharing_admm_start(create_admm_sharing_data([1.2, 1, -4]))
    await start_coordinated_optimization(actors, coordinator, start)

    assert np.allclose(actors[0].x, [0.155, 0.776, -1.553], atol=1e-2)
    assert np.allclose(actors[1].x, [0.155, 0.776, -1.553], atol=1e-2)
    assert np.allclose(actors[2].x, [0.893, 0, -0.893], atol=1e-2)


@pytest.mark.asyncio
async def test_single_shot_clearing_rejects_participant_count_mismatch():
    """Registering an extra actor (e.g. storage) beyond the generator specs
    must raise, not silently mis-dispatch.

    This is the historical failure mode: merit-order clearing sizes its
    price-response formula off ``len(generators)``, so a third registered
    participant not covered by a spec gets one round of a price signal
    computed for two, with no further iteration to correct it.
    """
    horizon = 2
    lb = np.zeros(horizon)
    gens = [
        create_admm_economic_dispatch_actor(
            lb=lb, u=np.full(horizon, 10.0), cost=10.0, n_participants=2, epsilon=0.1
        ),
        create_admm_economic_dispatch_actor(
            lb=lb, u=np.full(horizon, 10.0), cost=50.0, n_participants=2, epsilon=0.1
        ),
    ]
    storage = create_admm_proximal_storage_actor(
        horizon=horizon, e_max=10.0, p_charge_max=5.0, p_discharge_max=5.0
    )
    specs = [
        ADMMGeneratorSpec(cost=np.full(horizon, 10.0), lb=lb, ub=np.full(horizon, 10.0)),
        ADMMGeneratorSpec(cost=np.full(horizon, 50.0), lb=lb, ub=np.full(horizon, 10.0)),
    ]
    coordinator = create_sharing_target_distance_admm_coordinator()
    data = create_admm_sharing_data(np.array([3.0, 3.0]), generators=specs)

    with pytest.raises(ValueError, match="participant"):
        await start_coordinated_optimization(
            [*gens, storage], coordinator, create_sharing_admm_start(data)
        )
