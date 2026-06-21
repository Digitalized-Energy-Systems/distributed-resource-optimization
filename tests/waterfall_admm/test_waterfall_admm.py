"""Waterfall ADMM smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    SectorDemand,
    create_waterfall_admm_coordinator,
    create_waterfall_admm_participant,
    create_waterfall_admm_start,
    marginal_priority,
    solve_cp_priority_admm,
    start_coordinated_optimization,
    tier_priority_weight,
    waterfall_serve,
)
from distributed_resource_optimization.algorithm.admm.types import CPSpec


def test_tier_priority_weight_monotone():
    """Lower tier index -> strictly higher weight (priority)."""
    weights = [tier_priority_weight(t, priority_tiers=4, base=10.0) for t in (1, 2, 3, 4)]
    assert weights == [10**4, 10**3, 10**2, 10**1]
    assert all(weights[i] > weights[i + 1] for i in range(len(weights) - 1))


def test_waterfall_serve_priority_order():
    """Pool covers tier-1 fully, partially tier-2, drops tier-3."""
    supply = np.array([[10.0]])
    demand = np.array([[[4.0], [8.0], [5.0]]])
    served = waterfall_serve(supply, demand)
    assert served[0, 0, 0] == pytest.approx(4.0)
    assert served[0, 1, 0] == pytest.approx(6.0)
    assert served[0, 2, 0] == pytest.approx(0.0)


def test_marginal_priority_zero_when_satisfied():
    """No marginal pressure when all demand is met."""
    demand = np.array([[[1.0], [1.0]]])
    served = np.array([[[1.0], [1.0]]])
    priorities = np.array([10.0, 1.0])
    lam = marginal_priority(served, demand, priorities)
    assert lam[0, 0] == 0.0


def test_marginal_priority_picks_highest_unserved_tier():
    demand = np.array([[[1.0], [1.0]]])
    served = np.array([[[0.5], [0.0]]])
    priorities = np.array([10.0, 1.0])
    lam = marginal_priority(served, demand, priorities)
    assert lam[0, 0] == pytest.approx(10.0)


def test_solve_cp_priority_admm_no_demand_returns_zero_factor():
    cps = [CPSpec(cp_id="cp-1", capacity_by_sector={"electricity": 5.0})]
    demands = [
        SectorDemand(
            sector="electricity",
            demand_by_tier={},
            base_supply=np.array([0.0]),
        )
    ]
    result = solve_cp_priority_admm(cps, demands)
    assert result.converged is True
    assert result.factor_by_cp["cp-1"] == pytest.approx(np.zeros(1))


def test_solve_cp_priority_admm_p2h_serves_heat_deficit():
    """A P2H bridges an electricity surplus to a heat deficit.

    With 5 MW base electricity supply, 0 MW heat supply, and 4 MW
    tier-1 heat demand, the P2H (electricity input 5 MW, heat output
    via eta = 0.9 -> -4.5 MW heat) should activate fully to cover the
    heat deficit.
    """
    cps = [
        CPSpec(
            cp_id="p2h-1",
            capacity_by_sector={"electricity": 5.0, "heat": -4.5},
        )
    ]
    demands = [
        SectorDemand(
            sector="electricity",
            demand_by_tier={1: np.array([0.0])},
            base_supply=np.array([5.0]),
        ),
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([4.0])},
            base_supply=np.array([0.0]),
        ),
    ]
    result = solve_cp_priority_admm(
        cps,
        demands,
        rho=1.0,
        max_iters=500,
        abs_tol=1e-5,
        priority_weight_base=10.0,
        r_damping=0.5,
    )
    assert result.converged is True
    assert result.factor_by_cp["p2h-1"][0] > 0.85


@pytest.mark.xfail(
    reason=(
        "Heterogeneous CPs enter a limit cycle: the sharing-ADMM template "
        "drives each x_i = r_i * cap_i toward a common consensus z, but with "
        "different cap_i no consensus is reachable, so r oscillates by ~0.05 "
        "indefinitely while the heat deficit remains. Tracks Issue 1 in the "
        "review — remove xfail once a non-consensus formulation lands."
    ),
    strict=True,
)
def test_solve_cp_priority_admm_heterogeneous_cps_prefers_efficient():
    """Two CPs with shared elec input, different heat efficiencies.

    5 MW elec base supply, 4 MW tier-1 heat demand, 0 MW heat base.
    Both CPs draw 5 MW elec each, so only one can run at r=1.  The
    efficient CP (cap_heat=-4.5) at r=1 alone exactly covers the 4 MW
    heat demand using all 5 MW elec — the optimum is r_eff=1,
    r_ineff=0.  The algorithm should converge to a mix that at least
    closes the heat deficit and prefers the efficient unit.
    """
    cps = [
        CPSpec(cp_id="eff", capacity_by_sector={"electricity": 5.0, "heat": -4.5}),
        CPSpec(cp_id="ineff", capacity_by_sector={"electricity": 5.0, "heat": -2.5}),
    ]
    demands = [
        SectorDemand(
            sector="electricity",
            demand_by_tier={1: np.array([0.0])},
            base_supply=np.array([5.0]),
        ),
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([4.0])},
            base_supply=np.array([0.0]),
        ),
    ]
    result = solve_cp_priority_admm(
        cps,
        demands,
        max_iters=5000,
        abs_tol=1e-5,
        priority_weight_base=10.0,
        r_damping=0.1,
    )
    r_eff = result.factor_by_cp["eff"][0]
    r_ineff = result.factor_by_cp["ineff"][0]
    heat_from_cps = 4.5 * r_eff + 2.5 * r_ineff

    assert result.converged
    assert heat_from_cps >= 4.0 - 1e-3
    assert r_eff > r_ineff


@pytest.mark.asyncio
async def test_waterfall_admm_coordinator_dispatches_factor_to_participant():
    """End-to-end: coordinator collects spec, runs kernel, returns factor."""
    participant = create_waterfall_admm_participant(
        cp_id="cp-1",
        capacity_by_sector={"electricity": 5.0, "heat": -4.5},
    )
    coordinator = create_waterfall_admm_coordinator()
    start = create_waterfall_admm_start(
        demands=[
            SectorDemand(
                sector="electricity",
                demand_by_tier={1: np.array([0.0])},
                base_supply=np.array([5.0]),
            ),
            SectorDemand(
                sector="heat",
                demand_by_tier={1: np.array([4.0])},
                base_supply=np.array([0.0]),
            ),
        ],
        rho=1.0,
        max_iters=500,
        abs_tol=1e-5,
        priority_weight_base=10.0,
        r_damping=0.5,
    )

    await start_coordinated_optimization([participant], coordinator, start)

    assert participant.r.shape == (1,)
    assert participant.r[0] > 0.85
