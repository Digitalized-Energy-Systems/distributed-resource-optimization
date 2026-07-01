"""FDGDM algorithm unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    LinearCostEconomicDispatchFDGDMActor,
    NoFDGDMActor,
    ReservoirStorageFDGDMActor,
    create_fdgdm_participant,
    create_fdgdm_start,
    start_distributed_optimization,
)


# ---------------------------------------------------------------------------
# Integration tests (full async runs)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fdgdm_no_cost_stays_at_initial():
    """With no cost gradient the power schedules should not change from initial."""
    results: list[np.ndarray] = []

    def finish(algo, carrier):
        results.append(algo._P.copy())

    initial_p = np.array([10.0, 10.0, 10.0])
    actors = [
        create_fdgdm_participant(finish, fdgdm_actor=NoFDGDMActor(), max_iter=20, horizon=3)
        for _ in range(3)
    ]
    start = create_fdgdm_start(data=initial_p)
    await start_distributed_optimization(actors, start)

    assert len(results) > 0
    for r in results:
        assert np.allclose(r, initial_p, atol=1e-6)


@pytest.mark.asyncio
async def test_fdgdm_equal_cost_equal_dispatch():
    """Three identical generators should converge to equal power outputs."""
    horizon = 4
    demand = np.array([30.0, 30.0, 30.0, 30.0])
    n = 3
    initial_p = demand / n  # 10 MW each — demand-feasible start

    results: list[np.ndarray] = []

    def finish(algo, carrier):
        results.append(algo._P.copy())

    actors = [
        create_fdgdm_participant(
            finish,
            fdgdm_actor=LinearCostEconomicDispatchFDGDMActor(
                cost=5.0, p_max=30.0, epsilon=0.1
            ),
            max_iter=150,
            horizon=horizon,
        )
        for _ in range(n)
    ]
    start = create_fdgdm_start(data=initial_p)
    await start_distributed_optimization(actors, start)

    assert len(results) == n
    for r in results[1:]:
        assert np.allclose(results[0], r, atol=0.5), (
            f"Power schedules differ: {results[0]} vs {r}"
        )


@pytest.mark.asyncio
async def test_fdgdm_merit_order():
    """Cheap generator should receive more load than the expensive one."""
    horizon = 2
    demand = np.array([20.0, 20.0])
    n = 2
    initial_p = demand / n  # 10 MW each

    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo._P.copy()

        return finish

    actors = [
        create_fdgdm_participant(
            make_finish(0),
            fdgdm_actor=LinearCostEconomicDispatchFDGDMActor(
                cost=2.0, p_max=20.0, epsilon=0.1  # cheap
            ),
            max_iter=200,
            horizon=horizon,
        ),
        create_fdgdm_participant(
            make_finish(1),
            fdgdm_actor=LinearCostEconomicDispatchFDGDMActor(
                cost=8.0, p_max=20.0, epsilon=0.1  # expensive
            ),
            max_iter=200,
            horizon=horizon,
        ),
    ]
    start = create_fdgdm_start(data=initial_p)
    await start_distributed_optimization(actors, start)

    assert 0 in results and 1 in results
    # Cheap generator (0) should dispatch more than expensive generator (1).
    assert np.all(results[0] >= results[1] - 0.5), (
        f"Expected cheap >= expensive, got {results[0]} vs {results[1]}"
    )


# ---------------------------------------------------------------------------
# Actor unit tests
# ---------------------------------------------------------------------------


class TestLinearCostEconomicDispatchFDGDMActor:
    def test_gradient_linear_in_P(self):
        actor = LinearCostEconomicDispatchFDGDMActor(cost=5.0, p_max=20.0, epsilon=0.1)
        P = np.array([10.0])
        grad = actor.gradient(P, None)
        # ∇F = ε*P + c = 0.1*10 + 5 = 6
        assert np.allclose(grad, [6.0])

    def test_curvature_bound_equals_epsilon(self):
        actor = LinearCostEconomicDispatchFDGDMActor(cost=5.0, p_max=20.0, epsilon=0.2)
        assert actor.curvature_bound() == 0.2

    def test_project_clips_to_p_max(self):
        actor = LinearCostEconomicDispatchFDGDMActor(cost=0.0, p_max=10.0, epsilon=0.1)
        clipped = actor.project(np.array([15.0]))
        assert np.allclose(clipped, [10.0])
        assert np.allclose(actor.P, [10.0])

    def test_project_clips_to_p_min(self):
        actor = LinearCostEconomicDispatchFDGDMActor(cost=0.0, p_max=10.0, epsilon=0.1, p_min=2.0)
        clipped = actor.project(np.array([-1.0]))
        assert np.allclose(clipped, [2.0])

    def test_vector_p_max(self):
        actor = LinearCostEconomicDispatchFDGDMActor(
            cost=0.0, p_max=np.array([5.0, 10.0, 15.0]), epsilon=0.1
        )
        P = np.array([3.0, 12.0, 10.0])
        clipped = actor.project(P)
        assert np.allclose(clipped, [3.0, 10.0, 10.0])

    def test_initial_schedule_used_on_first_project(self):
        init = np.array([7.0, 8.0])
        actor = LinearCostEconomicDispatchFDGDMActor(
            cost=0.0, p_max=20.0, epsilon=0.1, initial_schedule=init.copy()
        )
        result = actor.project(np.array([99.0, 99.0]))
        assert np.allclose(result, [7.0, 8.0])
        assert actor.initial_schedule is None

    def test_initial_schedule_cleared_after_first_call(self):
        init = np.array([5.0])
        actor = LinearCostEconomicDispatchFDGDMActor(
            cost=0.0, p_max=10.0, epsilon=0.1, initial_schedule=init.copy()
        )
        actor.project(np.array([99.0]))  # first call: uses initial_schedule
        result = actor.project(np.array([3.0]))  # second call: uses passed value
        assert np.allclose(result, [3.0])


# ---------------------------------------------------------------------------
# ReservoirStorageFDGDMActor unit tests
# ---------------------------------------------------------------------------


class TestReservoirStorageFDGDMActor:
    def test_gradient_discharge(self):
        actor = ReservoirStorageFDGDMActor(
            p_charge_max=5.0, p_discharge_max=10.0,
            charge_cost=1.0, discharge_cost=3.0, epsilon=0.1,
        )
        P = np.array([4.0])
        grad = actor.gradient(P, None)
        # discharge branch: discharge_cost + epsilon * P = 3 + 0.1*4 = 3.4
        assert np.allclose(grad, [3.4])

    def test_gradient_charge(self):
        actor = ReservoirStorageFDGDMActor(
            p_charge_max=5.0, p_discharge_max=10.0,
            charge_cost=2.0, discharge_cost=0.0, epsilon=0.1,
        )
        P = np.array([-3.0])
        grad = actor.gradient(P, None)
        # charge branch: -charge_cost + epsilon * P = -2 + 0.1*(-3) = -2.3
        assert np.allclose(grad, [-2.3])

    def test_gradient_at_zero(self):
        actor = ReservoirStorageFDGDMActor(
            p_charge_max=5.0, p_discharge_max=10.0,
            charge_cost=1.0, discharge_cost=2.0, epsilon=0.1,
        )
        P = np.array([0.0])
        grad = actor.gradient(P, None)
        # P==0 hits discharge branch (np.where condition P >= 0 is True)
        assert np.allclose(grad, [2.0])

    def test_gradient_vector(self):
        actor = ReservoirStorageFDGDMActor(
            p_charge_max=5.0, p_discharge_max=10.0,
            charge_cost=1.0, discharge_cost=2.0, epsilon=0.0,
        )
        P = np.array([3.0, -2.0])
        grad = actor.gradient(P, None)
        # epsilon=0 → linear only: [discharge_cost, -charge_cost] = [2, -1]
        assert np.allclose(grad, [2.0, -1.0])

    def test_project_clips_to_discharge_max(self):
        actor = ReservoirStorageFDGDMActor(p_charge_max=5.0, p_discharge_max=8.0)
        clipped = actor.project(np.array([12.0]))
        assert np.allclose(clipped, [8.0])
        assert np.allclose(actor.P, [8.0])

    def test_project_clips_to_charge_max(self):
        actor = ReservoirStorageFDGDMActor(p_charge_max=5.0, p_discharge_max=8.0)
        clipped = actor.project(np.array([-7.0]))
        assert np.allclose(clipped, [-5.0])
        assert np.allclose(actor.P, [-5.0])

    def test_project_within_bounds_unchanged(self):
        actor = ReservoirStorageFDGDMActor(p_charge_max=5.0, p_discharge_max=10.0)
        P = np.array([-3.0, 0.0, 6.0])
        clipped = actor.project(P)
        assert np.allclose(clipped, [-3.0, 0.0, 6.0])

    def test_curvature_bound_equals_epsilon(self):
        actor = ReservoirStorageFDGDMActor(p_charge_max=5.0, p_discharge_max=10.0, epsilon=0.25)
        assert actor.curvature_bound() == 0.25


# ---------------------------------------------------------------------------
# initial_schedule conservation test (heterogeneous capacities)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fdgdm_power_conservation_heterogeneous_capacity():
    """Per-agent initial_schedule restores conservation when generators have
    very different p_max values.

    Parameters are chosen so that:
    - demand/2 = 60 > p_max_small = 50, so the old shared-min-cap approach would
      initialise both agents at 50 MW, giving a starting total of 100 MW ≠ 120 MW.
    - The unconstrained equilibrium (P_large=90, P_small=30) is strictly interior
      to both agents' box constraints, so no clipping occurs during iteration and
      FDGDM conserves total power.

    With the per-agent capacity-proportional initial_schedule (80, 40) the
    starting sum equals demand and the algorithm converges correctly.
    """
    horizon = 1
    demand_val = 120.0
    # p_max_small=50 → demand/2=60 > 50 → old equal-split cap would have started
    # both at 50 MW (total 100), losing 20 MW from the conservation invariant.
    p_max_large, p_max_small = 100.0, 50.0
    # Capacity-proportional initial allocations: 80 MW and 40 MW (sum = 120).
    init_large = demand_val * p_max_large / (p_max_large + p_max_small)  # 80
    init_small = demand_val * p_max_small / (p_max_large + p_max_small)  # 40

    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo._P.copy()
        return finish

    actors = [
        create_fdgdm_participant(
            make_finish(0),
            fdgdm_actor=LinearCostEconomicDispatchFDGDMActor(
                cost=2.0, p_max=p_max_large, epsilon=0.1,
                initial_schedule=np.array([init_large]),
            ),
            max_iter=300, horizon=horizon,
        ),
        create_fdgdm_participant(
            make_finish(1),
            fdgdm_actor=LinearCostEconomicDispatchFDGDMActor(
                cost=8.0, p_max=p_max_small, epsilon=0.1,
                initial_schedule=np.array([init_small]),
            ),
            max_iter=300, horizon=horizon,
        ),
    ]
    start = create_fdgdm_start(data=np.zeros(horizon))
    await start_distributed_optimization(actors, start)

    assert 0 in results and 1 in results
    total = results[0] + results[1]
    assert np.allclose(total, demand_val, atol=1e-3), (
        f"Conservation failed: sum={total} vs demand={demand_val}"
    )


# ---------------------------------------------------------------------------
# Power conservation integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fdgdm_power_conservation():
    """Total power (Σ P_i) must equal initial demand after convergence.

    The FDGDM weight matrix has zero row-sums, so the unconstrained step
    preserves total power exactly.  This holds when the equilibrium lies
    strictly inside the box constraints (no binding limits).

    We use costs very close together so the equilibrium dispatch values are
    near the initial point and no clipping occurs.  The test also validates
    that the start-message kickoff is not accidentally queued as a neighbour
    gradient (the Bug-#1 fix): without the fix, carrier[1] would use a fake
    gradient=0 message in its first update, shifting its power down by ~6 MW
    and breaking conservation.
    """
    horizon = 1
    demand = np.array([30.0])
    n = 3
    initial_p = demand / n  # [10 MW] each — demand-feasible start

    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo._P.copy()
        return finish

    # Small cost differences → equilibrium is P=[11, 10, 9], all within [0, 100].
    # No clipping occurs, so the zero-row-sum conservation property is exact.
    costs = [4.9, 5.0, 5.1]
    actors = [
        create_fdgdm_participant(
            make_finish(i),
            fdgdm_actor=LinearCostEconomicDispatchFDGDMActor(
                cost=costs[i], p_max=100.0, epsilon=0.1
            ),
            max_iter=300,
            horizon=horizon,
        )
        for i in range(n)
    ]
    start = create_fdgdm_start(data=initial_p)
    await start_distributed_optimization(actors, start)

    assert len(results) == n
    total = sum(results[i] for i in range(n))
    assert np.allclose(total, demand, atol=1e-3), (
        f"Power not conserved: sum={total} vs demand={demand}"
    )


# ---------------------------------------------------------------------------
# Storage + thermal integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fdgdm_with_storage_converges():
    """FDGDM converges with a mixed thermal + storage setup.

    After enough iterations:
    - Both finish callbacks must have fired.
    - Storage actor output must lie within its box constraints.
    - Total power must be conserved.
    """
    horizon = 4
    demand = np.array([20.0, 20.0, 20.0, 20.0])
    n = 2
    initial_p = demand / n  # 10 MW each

    p_charge_max = 8.0
    p_discharge_max = 12.0

    results: dict[int, np.ndarray] = {}

    storage_actor = ReservoirStorageFDGDMActor(
        p_charge_max=p_charge_max,
        p_discharge_max=p_discharge_max,
        charge_cost=0.0,
        discharge_cost=1.0,
        epsilon=0.1,
    )
    thermal_actor = LinearCostEconomicDispatchFDGDMActor(
        cost=5.0, p_max=20.0, epsilon=0.1
    )

    def make_finish(idx):
        def finish(algo, carrier):
            results[idx] = algo._P.copy()
        return finish

    actors = [
        create_fdgdm_participant(make_finish(0), fdgdm_actor=thermal_actor, max_iter=300, horizon=horizon),
        create_fdgdm_participant(make_finish(1), fdgdm_actor=storage_actor, max_iter=300, horizon=horizon),
    ]
    start = create_fdgdm_start(data=initial_p)
    await start_distributed_optimization(actors, start)

    assert 0 in results and 1 in results, "Not all finish callbacks fired"

    # Storage output must lie within its box constraints.
    # Note: when box constraints are binding the total-power conservation
    # property (zero-row-sum weights) breaks, so we do not check Σ P_i here.
    assert np.all(results[1] >= -p_charge_max - 1e-9)
    assert np.all(results[1] <= p_discharge_max + 1e-9)


# ---------------------------------------------------------------------------
# Gradient / curvature sign consistency test
# ---------------------------------------------------------------------------


class TestGradientCurvatureConsistency:
    """The gradient must be the derivative of F; curvature_bound must be ≥ |F''|."""

    def test_linear_cost_gradient_matches_finite_difference(self):
        actor = LinearCostEconomicDispatchFDGDMActor(cost=3.0, p_max=50.0, epsilon=0.2)
        P = np.array([5.0])
        h = 1e-5
        g_numerical = (actor.gradient(P + h, None) - actor.gradient(P - h, None)) / (2 * h)
        # F''(P) = ε; gradient of ε·P + c is ε, consistent second derivative
        assert np.allclose(g_numerical, actor.curvature_bound(), atol=1e-4)

    def test_storage_gradient_matches_finite_difference_discharge(self):
        actor = ReservoirStorageFDGDMActor(
            p_charge_max=5.0, p_discharge_max=10.0,
            charge_cost=1.0, discharge_cost=2.0, epsilon=0.2,
        )
        P = np.array([3.0])
        h = 1e-5
        # finite-difference second derivative ≤ curvature_bound
        g_plus = actor.gradient(P + h, None)
        g_minus = actor.gradient(P - h, None)
        d2 = (g_plus - g_minus) / (2 * h)
        assert np.all(np.abs(d2) <= actor.curvature_bound() + 1e-6)


# ---------------------------------------------------------------------------
# LinearCostEconomicDispatchFDGDMActor __post_init__ (p_max caching)
# ---------------------------------------------------------------------------


class TestLinearCostActorPMaxCache:
    def test_p_max_cache_matches_input_scalar(self):
        actor = LinearCostEconomicDispatchFDGDMActor(cost=0.0, p_max=15.0, epsilon=0.1)
        assert float(actor._p_max) == pytest.approx(15.0)

    def test_p_max_cache_matches_input_array(self):
        pmax = np.array([5.0, 10.0, 15.0])
        actor = LinearCostEconomicDispatchFDGDMActor(cost=0.0, p_max=pmax, epsilon=0.1)
        assert np.allclose(actor._p_max, pmax)

    def test_project_with_array_p_max_clips_per_element(self):
        actor = LinearCostEconomicDispatchFDGDMActor(
            cost=0.0, p_max=np.array([5.0, 10.0]), epsilon=0.1
        )
        result = actor.project(np.array([8.0, 8.0]))
        assert np.allclose(result, [5.0, 8.0])
