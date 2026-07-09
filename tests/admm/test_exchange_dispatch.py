"""Iterative exchange-ADMM economic dispatch tests.

Validates that the consensus (exchange) coordinator with ``alpha=0`` plus
cost-bearing box actors converges to the merit-order optimum — the
configuration the ``esb-admm`` iterative mode uses.  This is the regression
net for the historical failure where cost-bearing actors dispatched ~0 MW
under the sharing coordinator.
"""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    create_admm_flex_actor_box_bounded,
    create_admm_proximal_storage_actor,
    create_admm_start_consensus,
    create_consensus_target_reach_admm_coordinator,
    start_coordinated_optimization,
)

HORIZON = 4


@pytest.mark.asyncio
async def test_exchange_admm_merit_order_three_generators():
    """Three generators with distinct costs: cheapest fills first, sum = target.

    Capacities 10/10/10, costs 10/30/50, target 15 → optimum is
    gen0 = 10 (full), gen1 = 5 (marginal), gen2 = 0.
    """
    target = np.full(HORIZON, 15.0)
    costs = [10.0, 30.0, 50.0]
    actors = [
        create_admm_flex_actor_box_bounded(
            lb=np.zeros(HORIZON), u=np.full(HORIZON, 10.0), S=np.full(HORIZON, c)
        )
        for c in costs
    ]
    coordinator = create_consensus_target_reach_admm_coordinator(rho=1.0, max_iters=2000)
    start = create_admm_start_consensus(target)
    results = await start_coordinated_optimization(actors, coordinator, start)

    total = sum(results)
    assert np.allclose(total, target, atol=0.05), f"balance violated: {total}"
    assert np.allclose(results[0], 10.0, atol=0.1), f"cheap gen not at max: {results[0]}"
    assert np.allclose(results[1], 5.0, atol=0.1), f"marginal gen wrong: {results[1]}"
    assert np.allclose(results[2], 0.0, atol=0.1), f"expensive gen not at 0: {results[2]}"

    # Optimal cost check against the analytic merit-order solution.
    cost = sum(float(np.sum(np.asarray(c) * r)) for c, r in zip(costs, results))
    optimal = HORIZON * (10.0 * 10.0 + 30.0 * 5.0)
    assert cost == pytest.approx(optimal, rel=0.01)


@pytest.mark.asyncio
async def test_exchange_admm_respects_p_min():
    """A must-run generator's lower bound displaces cheaper capacity."""
    target = np.full(HORIZON, 12.0)
    actors = [
        create_admm_flex_actor_box_bounded(
            lb=np.zeros(HORIZON), u=np.full(HORIZON, 10.0), S=np.full(HORIZON, 10.0)
        ),
        # Expensive but must-run at >= 4 MW.
        create_admm_flex_actor_box_bounded(
            lb=np.full(HORIZON, 4.0), u=np.full(HORIZON, 10.0), S=np.full(HORIZON, 90.0)
        ),
    ]
    coordinator = create_consensus_target_reach_admm_coordinator(rho=1.0, max_iters=2000)
    results = await start_coordinated_optimization(
        actors, coordinator, create_admm_start_consensus(target)
    )
    total = sum(results)
    assert np.allclose(total, target, atol=0.05)
    assert np.all(results[1] >= 4.0 - 1e-6), f"p_min violated: {results[1]}"
    assert np.allclose(results[1], 4.0, atol=0.1), "expensive must-run should stay at p_min"
    assert np.allclose(results[0], 8.0, atol=0.1)


@pytest.mark.asyncio
async def test_exchange_admm_with_storage_arbitrage():
    """Generator + storage over a price spread: battery charges cheap, discharges dear.

    Demand is low in the first half and exceeds the cheap generator's capacity
    in the second half, so serving it requires either the expensive peaker or
    the battery.  The optimum uses the battery (charged from the cheap unit
    early) and leaves the peaker near zero.
    """
    horizon = 6
    # Demand: 5,5,5 then 14,14,14 — cheap gen capacity is 10.
    target = np.array([5.0, 5.0, 5.0, 14.0, 14.0, 14.0])
    base = create_admm_flex_actor_box_bounded(
        lb=np.zeros(horizon), u=np.full(horizon, 10.0), S=np.full(horizon, 10.0)
    )
    peaker = create_admm_flex_actor_box_bounded(
        lb=np.zeros(horizon), u=np.full(horizon, 10.0), S=np.full(horizon, 100.0)
    )
    battery = create_admm_proximal_storage_actor(
        horizon=horizon,
        e_max=20.0,
        p_charge_max=5.0,
        p_discharge_max=5.0,
        eta_charge=1.0,
        eta_discharge=1.0,
        e_initial=0.0,
        e_final=0.0,
        charge_cost=0.0,
        discharge_cost=0.0,
    )
    coordinator = create_consensus_target_reach_admm_coordinator(rho=1.0, max_iters=3000)
    results = await start_coordinated_optimization(
        [base, peaker, battery], coordinator, create_admm_start_consensus(target)
    )
    total = sum(results)
    assert np.allclose(total, target, atol=0.1), f"balance violated: {total}"

    battery_x = results[2]
    # SOC feasibility (unit efficiencies): cumulative discharge never exceeds
    # what was charged, and the schedule returns to the initial (empty) level.
    soc = -np.cumsum(battery_x)
    assert np.all(soc >= -1e-3), f"SOC went negative: {soc}"
    assert np.all(soc <= 20.0 + 1e-3)
    assert abs(soc[-1]) < 0.1, f"terminal SOC missed: {soc[-1]}"
    assert np.all(np.abs(battery_x) <= 5.0 + 1e-6)

    # Battery should discharge in the expensive half; peaker stays near zero.
    assert float(np.sum(battery_x[3:])) > 5.0, f"battery did not discharge: {battery_x}"
    assert float(np.sum(results[1])) < 1.0, f"peaker ran despite battery: {results[1]}"


@pytest.mark.asyncio
async def test_exchange_admm_soft_alpha_biases_balance():
    """Documented behaviour: alpha > 0 relaxes the sum constraint.

    With cost-bearing actors a soft penalty leaves a balance gap of
    ~ (rho/alpha)·delta*, so the exact mode (alpha=0) must be used for
    dispatch.  This test pins the qualitative behaviour so a future default
    change back to a large alpha fails loudly.
    """
    target = np.full(HORIZON, 15.0)

    async def run(alpha: float) -> float:
        actors = [
            create_admm_flex_actor_box_bounded(
                lb=np.zeros(HORIZON), u=np.full(HORIZON, 10.0), S=np.full(HORIZON, c)
            )
            for c in (10.0, 30.0, 50.0)
        ]
        coordinator = create_consensus_target_reach_admm_coordinator(
            rho=1.0, max_iters=1500, alpha=alpha
        )
        results = await start_coordinated_optimization(
            actors, coordinator, create_admm_start_consensus(target)
        )
        return float(np.max(np.abs(sum(results) - target)))

    exact_gap = await run(alpha=0.0)
    soft_gap = await run(alpha=0.05)  # rho/alpha = 20 → visible bias
    assert exact_gap < 0.05
    assert soft_gap > 1.0, f"expected soft-penalty bias, got gap {soft_gap}"
