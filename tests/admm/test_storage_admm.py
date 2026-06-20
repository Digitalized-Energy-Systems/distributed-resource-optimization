"""Reservoir storage ADMM actor tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    create_admm_economic_dispatch_actor,
    create_admm_sharing_data,
    create_admm_storage_actor,
    create_sharing_target_distance_admm_coordinator,
    start_coordinated_optimization,
)
from distributed_resource_optimization.algorithm.admm.sharing_admm import ADMMGeneratorSpec, create_admm_start


def _energy_path(
    power: np.ndarray,
    *,
    e_initial: float,
    eta_charge: float,
    eta_discharge: float,
) -> np.ndarray:
    """Simulate SOC given net discharge-positive power."""
    e = [e_initial]
    for p in power:
        if p >= 0.0:
            e.append(e[-1] - p / eta_discharge)
        else:
            e.append(e[-1] - p * eta_charge)
    return np.asarray(e[1:], dtype=float)


def _specs_for(actors_meta: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> list[ADMMGeneratorSpec]:
    return [ADMMGeneratorSpec(cost=c, lb=lb, ub=ub) for c, lb, ub in actors_meta]


@pytest.mark.asyncio
async def test_storage_actor_respects_soc_limits():
    """Storage and thermal jointly meet the target with merit-order clearing."""
    horizon = 6
    target = np.array([2.0, 4.0, 6.0, 4.0, 2.0, 0.0])
    lb = np.zeros(horizon)
    thermal_ub = np.full(horizon, 10.0)
    storage_lb = np.full(horizon, -5.0)
    storage_ub = np.full(horizon, 5.0)
    n = 2
    storage = create_admm_storage_actor(
        horizon=horizon,
        e_max=20.0,
        p_charge_max=5.0,
        p_discharge_max=5.0,
        eta_charge=0.95,
        eta_discharge=0.95,
        e_initial=0.5,
        e_final=0.5,
        n_participants=n,
    )
    thermal = create_admm_economic_dispatch_actor(
        lb, thermal_ub, cost=0.0, n_participants=n, epsilon=0.1
    )
    specs = _specs_for(
        [
            (np.zeros(horizon), storage_lb, storage_ub),
            (np.zeros(horizon), lb, thermal_ub),
        ]
    )
    coordinator = create_sharing_target_distance_admm_coordinator()
    coordinator.rho = 0.2
    start = create_admm_start(create_admm_sharing_data(target, generators=specs))
    await start_coordinated_optimization([storage, thermal], coordinator, start)

    # Power limits must be respected.
    assert np.all(storage.x <= 5.0 + 1e-6)
    assert np.all(storage.x >= -5.0 - 1e-6)

    e_path = _energy_path(
        storage.x,
        e_initial=0.5 * 20.0,
        eta_charge=0.95,
        eta_discharge=0.95,
    )
    assert np.all(e_path >= -1e-6)
    assert np.all(e_path <= 20.0 + 1e-6)

    # Terminal SOC must match e_final (the bug that previously ignored e_final).
    assert abs(e_path[-1] - 0.5 * 20.0) < 0.5


@pytest.mark.asyncio
async def test_storage_prefers_discharge_when_cheap():
    """Storage with low discharge cost responds to price; high-cost storage stays idle.

    With e_initial == e_final both actors have zero required net energy transfer, so the
    terminal constraint does not force discharge.  The cheap actor (threshold=0) follows
    the price signal and peaks during the high-demand period; the expensive actor
    (threshold=55) stays idle because the clearing price (~0.5) never crosses its
    discharge threshold.
    """
    horizon = 4
    # Varying target gives varying clearing prices so the cheap actor can exploit
    # the peak period while the expensive actor still sees prices below its threshold.
    target = np.array([1.0, 1.0, 5.0, 3.0])
    lb = np.full(horizon, -5.0)
    ub = np.full(horizon, 5.0)
    cheap_storage = create_admm_storage_actor(
        horizon=horizon,
        e_max=30.0,
        p_charge_max=5.0,
        p_discharge_max=5.0,
        e_initial=0.5,
        e_final=0.5,
        charge_cost=0.0,
        discharge_cost=0.0,
        n_participants=1,
    )
    expensive_storage = create_admm_storage_actor(
        horizon=horizon,
        e_max=30.0,
        p_charge_max=5.0,
        p_discharge_max=5.0,
        e_initial=0.5,
        e_final=0.5,
        charge_cost=0.0,
        discharge_cost=55.0,
        n_participants=1,
    )
    coordinator = create_sharing_target_distance_admm_coordinator()
    coordinator.rho = 0.2
    cheap_specs = _specs_for([(np.zeros(horizon), lb, ub)])
    expensive_specs = _specs_for([(np.full(horizon, 50.0), lb, ub)])

    await start_coordinated_optimization(
        [cheap_storage],
        coordinator,
        create_admm_start(create_admm_sharing_data(target, generators=cheap_specs)),
    )
    await start_coordinated_optimization(
        [expensive_storage],
        coordinator,
        create_admm_start(create_admm_sharing_data(target, generators=expensive_specs)),
    )

    # Cheap storage tracks the price signal: it should discharge during the peak
    # period (t=2, pi~0.5 >> discharge_threshold=0).
    assert float(np.max(cheap_storage.x)) > float(np.max(expensive_storage.x)) + 0.5
    # Expensive storage: clearing price stays well below discharge_threshold=55, so
    # with e_initial==e_final it never needs to move.
    assert np.allclose(expensive_storage.x, 0.0, atol=0.1)
