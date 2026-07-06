"""Tests for ADMM economic dispatch actors and merit-order clearing.

Covers the functions in sharing_admm.py that implement merit-order price
discovery and the LinearCostEconomicDispatchADMMFlexActor that uses those
prices to set its output.
"""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    create_admm_economic_dispatch_actor,
    create_admm_sharing_data,
    create_sharing_target_distance_admm_coordinator,
    start_coordinated_optimization,
)
from distributed_resource_optimization.algorithm.admm.core import ADMMMessage
from distributed_resource_optimization.algorithm.admm.sharing_admm import (
    ADMMGeneratorSpec,
    _clearing_price,
    _supply_at_price,
    _z_from_clearing_prices,
    create_sharing_admm_start,
)


def _spec(cost: float, lb: float, ub: float, horizon: int = 1) -> ADMMGeneratorSpec:
    return ADMMGeneratorSpec(
        cost=np.full(horizon, cost),
        lb=np.full(horizon, lb),
        ub=np.full(horizon, ub),
    )


class TestSupplyAtPrice:
    def test_below_cost_returns_lower_bound(self):
        assert _supply_at_price(5.0, [_spec(10.0, 0.0, 5.0)], t=0, epsilon=1.0) == pytest.approx(
            0.0
        )

    def test_at_cost_returns_lower_bound(self):
        # (cost - cost) / epsilon = 0, clipped to lb=0
        assert _supply_at_price(10.0, [_spec(10.0, 0.0, 5.0)], t=0, epsilon=1.0) == pytest.approx(
            0.0
        )

    def test_above_cost_responds_proportionally(self):
        # (15 - 10) / 1 = 5, fits within [0, 100]
        assert _supply_at_price(15.0, [_spec(10.0, 0.0, 100.0)], t=0, epsilon=1.0) == pytest.approx(
            5.0
        )

    def test_capped_at_upper_bound(self):
        assert _supply_at_price(1000.0, [_spec(10.0, 0.0, 3.0)], t=0, epsilon=1.0) == pytest.approx(
            3.0
        )

    def test_two_generators_sum_correctly(self):
        # price=25, eps=1 → g1: (25-10)/1=15 capped at 5; g2: (25-20)/1=5
        total = _supply_at_price(
            25.0, [_spec(10.0, 0.0, 5.0), _spec(20.0, 0.0, 5.0)], t=0, epsilon=1.0
        )
        assert total == pytest.approx(10.0)

    def test_uses_correct_timestep(self):
        spec = ADMMGeneratorSpec(
            cost=np.array([10.0, 50.0]),
            lb=np.zeros(2),
            ub=np.full(2, 10.0),
        )
        supply_t0 = _supply_at_price(20.0, [spec], t=0, epsilon=1.0)  # cost=10 → 10
        supply_t1 = _supply_at_price(20.0, [spec], t=1, epsilon=1.0)  # cost=50 → 0
        assert supply_t0 == pytest.approx(10.0)
        assert supply_t1 == pytest.approx(0.0)


class TestClearingPrice:
    def test_zero_demand_returns_minimum_cost(self):
        p = _clearing_price(0.0, [_spec(30.0, 0.0, 5.0), _spec(10.0, 0.0, 5.0)], t=0, epsilon=0.1)
        assert p == pytest.approx(10.0)

    def test_single_generator_partial_load(self):
        # supply(p) = (p - 10) / 0.1 → want 3 → p = 10 + 0.3 = 10.3
        p = _clearing_price(3.0, [_spec(10.0, 0.0, 5.0)], t=0, epsilon=0.1)
        assert p == pytest.approx(10.3, abs=1e-3)

    def test_cheap_generator_serves_demand_alone(self):
        cheap = _spec(10.0, 0.0, 10.0)
        expensive = _spec(50.0, 0.0, 10.0)
        p = _clearing_price(3.0, [cheap, expensive], t=0, epsilon=0.1)
        supply_cheap = max(0.0, min(10.0, (p - 10.0) / 0.1))
        supply_expensive = max(0.0, min(10.0, (p - 50.0) / 0.1))
        assert supply_cheap + supply_expensive == pytest.approx(3.0, abs=0.01)
        assert supply_expensive < 0.01

    def test_both_generators_needed_when_cheap_is_capped(self):
        cheap = _spec(10.0, 0.0, 3.0)
        expensive = _spec(50.0, 0.0, 10.0)
        p = _clearing_price(6.0, [cheap, expensive], t=0, epsilon=0.1)
        supply_cheap = max(0.0, min(3.0, (p - 10.0) / 0.1))
        supply_expensive = max(0.0, min(10.0, (p - 50.0) / 0.1))
        assert supply_cheap + supply_expensive == pytest.approx(6.0, abs=0.01)
        assert supply_cheap == pytest.approx(3.0, abs=0.01)
        assert supply_expensive == pytest.approx(3.0, abs=0.1)

    def test_infeasible_demand_raises_value_error(self):
        with pytest.raises(ValueError, match="Infeasible"):
            _clearing_price(5.0, [_spec(10.0, 0.0, 2.0)], t=0, epsilon=0.1)

    def test_uses_correct_timestep_index(self):
        spec = ADMMGeneratorSpec(
            cost=np.array([10.0, 50.0]),
            lb=np.zeros(2),
            ub=np.full(2, 10.0),
        )
        p0 = _clearing_price(2.0, [spec], t=0, epsilon=0.1)
        p1 = _clearing_price(2.0, [spec], t=1, epsilon=0.1)
        assert p1 > p0


class TestZFromClearingPrices:
    def test_scales_by_rho_times_n(self):
        specs = [_spec(10.0, 0.0, 10.0)]
        data = create_admm_sharing_data([5.0], generators=specs)
        rho, n = 0.5, 3
        z = _z_from_clearing_prices(data, rho, n)
        expected_price = _clearing_price(5.0, specs, t=0, epsilon=0.1)
        assert z[0] == pytest.approx(expected_price / (rho * n), rel=1e-3)

    def test_raises_without_generators(self):
        data = create_admm_sharing_data([5.0])
        with pytest.raises(ValueError, match="generators"):
            _z_from_clearing_prices(data, rho=1.0, n=2)

    def test_output_length_matches_horizon(self):
        specs = [_spec(10.0, 0.0, 10.0, horizon=4)]
        data = create_admm_sharing_data([1.0, 2.0, 3.0, 4.0], generators=specs)
        z = _z_from_clearing_prices(data, rho=1.0, n=1)
        assert len(z) == 4

    def test_higher_demand_yields_higher_z(self):
        specs = [_spec(10.0, 0.0, 100.0, horizon=2)]
        data = create_admm_sharing_data([2.0, 8.0], generators=specs)
        z = _z_from_clearing_prices(data, rho=1.0, n=1)
        assert z[1] > z[0]


class TestLinearCostEconomicDispatchADMMFlexActor:
    def _actor(self, cost: float, ub: float, horizon: int = 3):
        return create_admm_economic_dispatch_actor(
            lb=np.zeros(horizon),
            u=np.full(horizon, ub),
            cost=cost,
            n_participants=1,
            epsilon=0.1,
        )

    def _msg(self, z_val: float, rho: float = 1.0, horizon: int = 3) -> ADMMMessage:
        return ADMMMessage(v=np.zeros(horizon), rho=rho, z=np.full(horizon, z_val))

    async def test_zero_output_when_price_below_cost(self):
        actor = self._actor(cost=10.0, ub=5.0)
        replies = []

        class FakeCarrier:
            def reply_to_other(self, answer, meta):
                replies.append(answer)

        # pi = rho * n_participants * z = 1 * 1 * 5 = 5 < cost=10
        await actor.on_exchange_message(FakeCarrier(), self._msg(5.0), {})
        assert np.allclose(replies[-1].x, 0.0)

    async def test_partial_dispatch_proportional_to_margin(self):
        actor = self._actor(cost=10.0, ub=5.0)
        replies = []

        class FakeCarrier:
            def reply_to_other(self, answer, meta):
                replies.append(answer)

        # pi = 1 * 1 * 10.3 = 10.3; x = (10.3 - 10) / 0.1 = 3
        await actor.on_exchange_message(FakeCarrier(), self._msg(10.3), {})
        assert np.allclose(replies[-1].x, 3.0, atol=1e-5)

    async def test_capped_at_upper_bound(self):
        actor = self._actor(cost=10.0, ub=5.0)
        replies = []

        class FakeCarrier:
            def reply_to_other(self, answer, meta):
                replies.append(answer)

        # Very high price: clipped at ub=5
        await actor.on_exchange_message(FakeCarrier(), self._msg(1000.0), {})
        assert np.allclose(replies[-1].x, 5.0, atol=1e-5)

    async def test_ignores_non_admm_message(self):
        actor = self._actor(cost=10.0, ub=5.0)
        called = []

        class FakeCarrier:
            def reply_to_other(self, answer, meta):
                called.append(answer)

        await actor.on_exchange_message(FakeCarrier(), "not a message", {})
        assert called == []

    async def test_vector_cost_per_timestep(self):
        """Per-timestep cost vector is supported (not just scalar cost)."""
        horizon = 2
        actor = create_admm_economic_dispatch_actor(
            lb=np.zeros(horizon),
            u=np.full(horizon, 10.0),
            cost=np.array([10.0, 50.0]),
            n_participants=1,
            epsilon=0.1,
        )
        replies = []

        class FakeCarrier:
            def reply_to_other(self, answer, meta):
                replies.append(answer)

        # pi = 10.3 for both timesteps; t0 cost=10 → x[0]=3; t1 cost=50 → x[1]=0
        msg = ADMMMessage(v=np.zeros(2), rho=1.0, z=np.full(2, 10.3))
        await actor.on_exchange_message(FakeCarrier(), msg, {})
        x = replies[-1].x
        assert x[0] == pytest.approx(3.0, abs=1e-5)
        assert x[1] == pytest.approx(0.0, abs=1e-5)


class TestMeritOrderViaADMM:
    """Integration: two economic dispatch actors cleared via merit-order sharing ADMM."""

    async def test_cheap_generator_serves_demand_alone(self):
        horizon = 3
        target = np.array([3.0, 3.0, 3.0])
        lb = np.zeros(horizon)

        cheap = create_admm_economic_dispatch_actor(
            lb=lb, u=np.full(horizon, 10.0), cost=10.0, n_participants=2, epsilon=0.1
        )
        expensive = create_admm_economic_dispatch_actor(
            lb=lb, u=np.full(horizon, 10.0), cost=50.0, n_participants=2, epsilon=0.1
        )
        specs = [
            ADMMGeneratorSpec(cost=np.full(horizon, 10.0), lb=lb, ub=np.full(horizon, 10.0)),
            ADMMGeneratorSpec(cost=np.full(horizon, 50.0), lb=lb, ub=np.full(horizon, 10.0)),
        ]
        coordinator = create_sharing_target_distance_admm_coordinator()
        coordinator.rho = 0.2
        data = create_admm_sharing_data(target, generators=specs)
        await start_coordinated_optimization(
            [cheap, expensive], coordinator, create_sharing_admm_start(data)
        )

        # Cheap should dispatch, expensive should be idle
        assert np.all(cheap.x >= expensive.x - 0.1)
        assert np.allclose(expensive.x, 0.0, atol=0.1)
        # Combined output matches demand
        assert np.allclose(cheap.x + expensive.x, target, atol=0.5)

    async def test_expensive_generator_fills_when_cheap_is_capped(self):
        horizon = 1
        target = np.array([8.0])
        lb = np.zeros(horizon)

        cheap = create_admm_economic_dispatch_actor(
            lb=lb, u=np.full(horizon, 5.0), cost=10.0, n_participants=2, epsilon=0.1
        )
        expensive = create_admm_economic_dispatch_actor(
            lb=lb, u=np.full(horizon, 10.0), cost=50.0, n_participants=2, epsilon=0.1
        )
        specs = [
            ADMMGeneratorSpec(cost=np.full(horizon, 10.0), lb=lb, ub=np.full(horizon, 5.0)),
            ADMMGeneratorSpec(cost=np.full(horizon, 50.0), lb=lb, ub=np.full(horizon, 10.0)),
        ]
        coordinator = create_sharing_target_distance_admm_coordinator()
        coordinator.rho = 0.2
        data = create_admm_sharing_data(target, generators=specs)
        await start_coordinated_optimization(
            [cheap, expensive], coordinator, create_sharing_admm_start(data)
        )

        # Cheap at capacity, expensive fills the gap
        assert cheap.x[0] == pytest.approx(5.0, abs=0.1)
        assert expensive.x[0] == pytest.approx(3.0, abs=0.3)

    async def test_three_generators_merit_order(self):
        horizon = 1
        target = np.array([12.0])
        lb = np.zeros(horizon)

        actors = [
            create_admm_economic_dispatch_actor(
                lb=lb, u=np.full(horizon, 5.0), cost=c, n_participants=3, epsilon=0.1
            )
            for c in [10.0, 30.0, 60.0]
        ]
        specs = [
            ADMMGeneratorSpec(cost=np.full(horizon, c), lb=lb, ub=np.full(horizon, 5.0))
            for c in [10.0, 30.0, 60.0]
        ]
        coordinator = create_sharing_target_distance_admm_coordinator()
        coordinator.rho = 0.2
        data = create_admm_sharing_data(target, generators=specs)
        await start_coordinated_optimization(actors, coordinator, create_sharing_admm_start(data))

        # Dispatch must respect merit order: cheapest ≥ second ≥ most expensive
        assert actors[0].x[0] >= actors[1].x[0] - 0.1
        assert actors[1].x[0] >= actors[2].x[0] - 0.1
        # Total must cover demand
        total = sum(a.x[0] for a in actors)
        assert total == pytest.approx(12.0, abs=0.5)
