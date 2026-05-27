"""Consensus Waterfall ADMM (primal-dual variant) smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    SectorDemand,
    create_consensus_waterfall_admm_coordinator,
    create_consensus_waterfall_admm_participant,
    create_consensus_waterfall_admm_start,
    cutoff_tier_deficit,
    solve_cp_consensus_waterfall_admm,
    start_coordinated_optimization,
)
from distributed_resource_optimization.algorithm.waterfall_admm.core import CPSpec


# ---------------------------------------------------------------------------
# Helpers used by multiple scenarios
# ---------------------------------------------------------------------------


def _p2h_spec(cp_id: str, *, electricity: float, heat: float) -> CPSpec:
    return CPSpec(
        cp_id=cp_id,
        capacity_by_sector={"electricity": electricity, "heat": heat},
    )


def _electricity_slack(base_mw: float, *, horizon: int = 1) -> SectorDemand:
    """Electricity sector with surplus base supply and no demand."""
    return SectorDemand(
        sector="electricity",
        demand_by_tier={1: np.zeros(horizon)},
        base_supply=np.full(horizon, base_mw),
    )


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


def test_cutoff_tier_deficit_positive_when_cutoff_under_served():
    """Pool covers tier-1, partially tier-2 -> cum_D(t*) - pool > 0."""
    demand = np.array([[[1.0], [1.0]]])  # tier-1=1, tier-2=1, cum=2
    supply_net = np.array([[1.25]])      # covers tier-1, 0.25 into tier-2
    deficit = cutoff_tier_deficit(supply_net, demand)
    assert deficit.shape == (1, 1)
    assert deficit[0, 0] == pytest.approx(0.75)  # 2.0 - 1.25


def test_cutoff_tier_deficit_zero_when_supply_exactly_meets_total():
    demand = np.array([[[1.0], [1.0]]])
    supply_net = np.array([[2.0]])
    deficit = cutoff_tier_deficit(supply_net, demand)
    assert deficit[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_cutoff_tier_deficit_negative_on_surplus_supply():
    """Surplus over total demand -> negative deficit, drives mu downward."""
    demand = np.array([[[1.0], [1.0]]])
    supply_net = np.array([[3.0]])
    deficit = cutoff_tier_deficit(supply_net, demand)
    assert deficit[0, 0] == pytest.approx(-1.0)  # 2.0 - 3.0


def test_cutoff_tier_deficit_skips_zero_demand_tier():
    """A leading zero-demand tier doesn't shift cum_D."""
    demand = np.array([[[0.0], [2.0]]])
    supply_net = np.array([[0.5]])
    deficit = cutoff_tier_deficit(supply_net, demand)
    assert deficit[0, 0] == pytest.approx(1.5)  # 2.0 - 0.5


def test_cutoff_tier_deficit_no_demand_returns_zero():
    demand = np.array([[[0.0], [0.0]]])
    supply_net = np.array([[5.0]])
    deficit = cutoff_tier_deficit(supply_net, demand)
    assert deficit[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Kernel: degenerate inputs
# ---------------------------------------------------------------------------


def test_solve_consensus_waterfall_admm_no_demand_returns_zero_factor():
    cps = [CPSpec(cp_id="cp-1", capacity_by_sector={"electricity": 5.0})]
    demands = [
        SectorDemand(
            sector="electricity",
            demand_by_tier={},
            base_supply=np.array([0.0]),
        )
    ]
    result = solve_cp_consensus_waterfall_admm(cps, demands)
    assert result.converged is True
    assert result.factor_by_cp["cp-1"] == pytest.approx(np.zeros(1))


def test_solve_consensus_waterfall_admm_no_cps_returns_empty_result():
    """An empty CP list short-circuits to converged with no allocations."""
    demands = [
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([4.0])},
            base_supply=np.array([0.0]),
        )
    ]
    result = solve_cp_consensus_waterfall_admm([], demands)
    assert result.converged is True
    assert result.factor_by_cp == {}


def test_solve_consensus_waterfall_admm_rejects_horizon_below_one():
    cps = [_p2h_spec("p2h-1", electricity=5.0, heat=-4.5)]
    demands = [_electricity_slack(5.0)]
    with pytest.raises(ValueError):
        solve_cp_consensus_waterfall_admm(cps, demands, horizon=0)


def test_solve_consensus_waterfall_admm_rejects_demand_shape_mismatch():
    cps = [_p2h_spec("p2h-1", electricity=5.0, heat=-4.5)]
    demands = [
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([1.0, 2.0, 3.0])},
            base_supply=np.array([0.0]),
        )
    ]
    with pytest.raises(ValueError):
        solve_cp_consensus_waterfall_admm(cps, demands, horizon=1)


# ---------------------------------------------------------------------------
# Single-CP scenarios
# ---------------------------------------------------------------------------


def test_solve_consensus_waterfall_admm_p2h_serves_heat_deficit():
    """A P2H bridges an electricity surplus to a heat deficit."""
    cps = [_p2h_spec("p2h-1", electricity=5.0, heat=-4.5)]
    demands = [
        _electricity_slack(5.0),
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([4.0])},
            base_supply=np.array([0.0]),
        ),
    ]
    result = solve_cp_consensus_waterfall_admm(
        cps,
        demands,
        rho=1.0,
        outer_iters=200,
        inner_iters=10,
        gamma0=1.0,
        abs_tol=1e-5,
    )
    r = float(result.factor_by_cp["p2h-1"][0])
    assert r > 0.85
    # Demand is fully served (tier-1 is the only tier; B' over-shoots
    # but never under-serves a feasible high-priority cell).
    heat_served = result.served_by_sector_tier["heat"][1][0]
    assert heat_served == pytest.approx(4.0, abs=1e-3)


def test_factor_clipped_to_unit_box_under_infeasibility():
    """A 100 MW demand with 2 MW max production saturates r at 1.0."""
    cps = [_p2h_spec("p2h-1", electricity=2.0, heat=-2.0)]
    demands = [
        _electricity_slack(2.0),
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([100.0])},
            base_supply=np.array([0.0]),
        ),
    ]
    result = solve_cp_consensus_waterfall_admm(
        cps,
        demands,
        rho=1.0,
        outer_iters=50,
        inner_iters=5,
    )
    r = float(result.factor_by_cp["p2h-1"][0])
    assert 0.0 <= r <= 1.0 + 1e-9
    assert r == pytest.approx(1.0, abs=1e-3)


def test_priority_ordered_serve_under_scarcity():
    """When total cap can't cover tier-1 + tier-2, tier-1 still wins."""
    cps = [_p2h_spec("p2h-1", electricity=2.0, heat=-2.0)]
    demands = [
        _electricity_slack(2.0),
        SectorDemand(
            sector="heat",
            demand_by_tier={
                1: np.array([1.5]),
                2: np.array([2.0]),
            },
            base_supply=np.array([0.0]),
        ),
    ]
    result = solve_cp_consensus_waterfall_admm(
        cps,
        demands,
        rho=1.0,
        outer_iters=200,
        inner_iters=10,
        gamma0=1.0,
    )
    r = float(result.factor_by_cp["p2h-1"][0])
    assert r > 0.95  # need full activation against 3.5 MW total demand.
    heat_served = result.served_by_sector_tier["heat"]
    # The waterfall serves tier-1 fully before tier-2 gets anything.
    assert heat_served[1][0] == pytest.approx(1.5, abs=1e-2)
    assert heat_served[2][0] == pytest.approx(0.5, abs=1e-2)


def test_chp_serves_electricity_and_heat_jointly():
    """A CHP-style CP that consumes gas and produces both electricity
    and heat activates to cover deficits in both output sectors."""
    cps = [
        CPSpec(
            cp_id="chp-1",
            capacity_by_sector={
                "gas": 10.0,
                "electricity": -3.5,
                "heat": -4.5,
            },
        )
    ]
    demands = [
        SectorDemand(
            sector="gas",
            demand_by_tier={1: np.array([0.0])},
            base_supply=np.array([20.0]),
        ),
        SectorDemand(
            sector="electricity",
            demand_by_tier={1: np.array([3.0])},
            base_supply=np.array([0.0]),
        ),
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([4.0])},
            base_supply=np.array([0.0]),
        ),
    ]
    result = solve_cp_consensus_waterfall_admm(
        cps,
        demands,
        rho=1.0,
        outer_iters=200,
        inner_iters=10,
        gamma0=1.0,
    )
    r = float(result.factor_by_cp["chp-1"][0])
    assert 0.85 <= r <= 1.0 + 1e-9
    # Both output deficits are fully covered at the converged r.
    elec_served = result.served_by_sector_tier["electricity"][1][0]
    heat_served = result.served_by_sector_tier["heat"][1][0]
    assert elec_served == pytest.approx(3.0, abs=1e-2)
    assert heat_served > 3.85


# ---------------------------------------------------------------------------
# Multi-CP scenarios
# ---------------------------------------------------------------------------


def test_two_identical_cps_share_load_symmetrically():
    """Two identical P2Hs facing the same heat deficit produce equal r values.

    The shared sharing-ADMM consensus carries no objective to break the
    primal degeneracy, so B' lands at the symmetric box corner — but
    both CPs sit at the *same* corner.
    """
    cps = [
        _p2h_spec("p2h-a", electricity=3.0, heat=-2.5),
        _p2h_spec("p2h-b", electricity=3.0, heat=-2.5),
    ]
    demands = [
        _electricity_slack(6.0),
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([4.0])},
            base_supply=np.array([0.0]),
        ),
    ]
    result = solve_cp_consensus_waterfall_admm(
        cps,
        demands,
        rho=1.0,
        outer_iters=200,
        inner_iters=10,
        gamma0=1.0,
    )
    r_a = float(result.factor_by_cp["p2h-a"][0])
    r_b = float(result.factor_by_cp["p2h-b"][0])
    assert 0.0 <= r_a <= 1.0 + 1e-9
    assert 0.0 <= r_b <= 1.0 + 1e-9
    # Symmetric inputs yield equal factors.
    assert r_a == pytest.approx(r_b, abs=1e-3)
    # Combined production at least covers the 4 MW deficit (saddle is
    # feasible w.r.t. the inequality).
    assert 2.5 * (r_a + r_b) >= 4.0 - 1e-3


def test_three_identical_cps_share_load_symmetrically():
    """Three identical P2Hs facing the same heat deficit settle equal r."""
    cps = [
        _p2h_spec("p2h-a", electricity=2.0, heat=-2.0),
        _p2h_spec("p2h-b", electricity=2.0, heat=-2.0),
        _p2h_spec("p2h-c", electricity=2.0, heat=-2.0),
    ]
    demands = [
        _electricity_slack(6.0),
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([5.0])},
            base_supply=np.array([0.0]),
        ),
    ]
    result = solve_cp_consensus_waterfall_admm(
        cps,
        demands,
        rho=1.0,
        outer_iters=200,
        inner_iters=10,
        gamma0=1.0,
    )
    rs = [float(result.factor_by_cp[c][0]) for c in ("p2h-a", "p2h-b", "p2h-c")]
    assert all(0.0 <= r <= 1.0 + 1e-9 for r in rs)
    # Three-way symmetry: all factors agree.
    assert rs[0] == pytest.approx(rs[1], abs=1e-3)
    assert rs[1] == pytest.approx(rs[2], abs=1e-3)
    # Combined production at least covers the 5 MW heat deficit.
    assert 2.0 * sum(rs) >= 5.0 - 1e-3


# ---------------------------------------------------------------------------
# Multi-step horizon
# ---------------------------------------------------------------------------


def test_multi_step_horizon_returns_per_step_factor():
    """Each horizon step gets its own factor; both stay box-feasible."""
    H = 2
    cps = [_p2h_spec("p2h-1", electricity=5.0, heat=-4.5)]
    demands = [
        _electricity_slack(5.0, horizon=H),
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([4.0, 1.0])},
            base_supply=np.zeros(H),
        ),
    ]
    result = solve_cp_consensus_waterfall_admm(
        cps,
        demands,
        horizon=H,
        rho=1.0,
        outer_iters=200,
        inner_iters=10,
        gamma0=1.0,
    )
    r = result.factor_by_cp["p2h-1"]
    assert r.shape == (H,)
    assert all(0.0 <= float(r[k]) <= 1.0 + 1e-9 for k in range(H))
    # Both demands are fully met (tier-1 is the only tier each step).
    heat_served = result.served_by_sector_tier["heat"][1]
    assert heat_served[0] == pytest.approx(4.0, abs=1e-2)
    assert heat_served[1] == pytest.approx(1.0, abs=1e-2)


def test_multi_step_horizon_independent_per_step_state():
    """A high-demand step does not bleed activation into a no-demand step."""
    H = 2
    cps = [_p2h_spec("p2h-1", electricity=5.0, heat=-4.5)]
    demands = [
        _electricity_slack(5.0, horizon=H),
        SectorDemand(
            sector="heat",
            # Step 0 demands 4 MW; step 1 demands nothing.
            demand_by_tier={1: np.array([4.0, 0.0])},
            base_supply=np.zeros(H),
        ),
    ]
    result = solve_cp_consensus_waterfall_admm(
        cps,
        demands,
        horizon=H,
        rho=1.0,
        outer_iters=200,
        inner_iters=10,
        gamma0=1.0,
    )
    r = result.factor_by_cp["p2h-1"]
    assert r[0] > 0.85  # active step covers its demand.
    assert r[1] == pytest.approx(0.0, abs=1e-3)  # idle step stays at zero.


# ---------------------------------------------------------------------------
# Dual / convergence properties
# ---------------------------------------------------------------------------


def test_mu_rises_on_persistent_under_supply():
    """When demand can't be met, the dual variable accumulates upward."""
    cps = [_p2h_spec("p2h-1", electricity=2.0, heat=-2.0)]
    demands = [
        _electricity_slack(2.0),
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([100.0])},
            base_supply=np.array([0.0]),
        ),
    ]
    result = solve_cp_consensus_waterfall_admm(
        cps,
        demands,
        rho=1.0,
        outer_iters=30,
        inner_iters=4,
        gamma0=1.0,
        mu_upper_bound=1.0e6,
        record_history=True,
        abs_tol=0.0,  # never trigger early-stop.
    )
    mu_changes = result.history["mu_changes"]
    # Persistent positive deficit means mu strictly grows on at least one step.
    assert any(dm > 1e-6 for dm in mu_changes)
    # The final mu has settled above zero on the heat sector.
    mu_final = result.history["mu"]
    assert float(mu_final.max()) > 0.0


def test_converges_within_iter_budget_on_canonical_scenario():
    """B' early-stops on the canonical P2H scenario once the KKT slack
    saddle is reached.

    The harmonic ``gamma_nu = c / nu`` schedule walks mu down to the
    projection boundary in :math:`O(\\exp(\\mu_{\\text{peak}}/c))` outer
    iterations, but the KKT-aware early-stop trips as soon as ``r`` is
    stable at the upper bound and the deficit is non-positive — no need
    to wait for the harmonic tail to fully decay.
    """
    cps = [_p2h_spec("p2h-1", electricity=5.0, heat=-4.5)]
    demands = [
        _electricity_slack(5.0),
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([4.0])},
            base_supply=np.array([0.0]),
        ),
    ]
    result = solve_cp_consensus_waterfall_admm(
        cps,
        demands,
        outer_iters=5000,
        gamma0=1.0,
    )
    assert result.converged is True
    assert result.iterations < 5000
    assert float(result.factor_by_cp["p2h-1"][0]) > 0.85


def test_history_lengths_match_iteration_count():
    cps = [_p2h_spec("p2h-1", electricity=5.0, heat=-4.5)]
    demands = [
        _electricity_slack(5.0),
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([4.0])},
            base_supply=np.array([0.0]),
        ),
    ]
    result = solve_cp_consensus_waterfall_admm(
        cps,
        demands,
        outer_iters=20,
        inner_iters=4,
        record_history=True,
    )
    assert "mu" in result.history
    assert "mu_changes" in result.history
    assert "primal_residuals" in result.history
    assert "dual_residuals" in result.history
    assert "r_changes" in result.history
    assert len(result.history["mu_changes"]) == result.iterations
    assert len(result.history["primal_residuals"]) == result.iterations


# ---------------------------------------------------------------------------
# End-to-end coordinator round
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consensus_waterfall_admm_coordinator_dispatches_factor():
    """End-to-end: coordinator collects spec, runs kernel, returns factor."""
    participant = create_consensus_waterfall_admm_participant(
        cp_id="cp-1",
        capacity_by_sector={"electricity": 5.0, "heat": -4.5},
    )
    coordinator = create_consensus_waterfall_admm_coordinator()
    start = create_consensus_waterfall_admm_start(
        demands=[
            _electricity_slack(5.0),
            SectorDemand(
                sector="heat",
                demand_by_tier={1: np.array([4.0])},
                base_supply=np.array([0.0]),
            ),
        ],
        rho=1.0,
        outer_iters=200,
        inner_iters=10,
        gamma0=1.0,
        abs_tol=1e-5,
    )

    await start_coordinated_optimization([participant], coordinator, start)

    assert participant.r.shape == (1,)
    assert participant.r[0] > 0.85


@pytest.mark.asyncio
async def test_consensus_waterfall_admm_coordinator_dispatches_factor_per_cp():
    """Coordinator hands each participant its own slot of factor_by_cp."""
    p_a = create_consensus_waterfall_admm_participant(
        cp_id="p2h-a",
        capacity_by_sector={"electricity": 3.0, "heat": -2.5},
    )
    p_b = create_consensus_waterfall_admm_participant(
        cp_id="p2h-b",
        capacity_by_sector={"electricity": 3.0, "heat": -2.5},
    )
    coordinator = create_consensus_waterfall_admm_coordinator()
    start = create_consensus_waterfall_admm_start(
        demands=[
            _electricity_slack(6.0),
            SectorDemand(
                sector="heat",
                demand_by_tier={1: np.array([4.0])},
                base_supply=np.array([0.0]),
            ),
        ],
        rho=1.0,
        outer_iters=200,
        inner_iters=10,
        gamma0=1.0,
        abs_tol=1e-5,
    )

    await start_coordinated_optimization([p_a, p_b], coordinator, start)

    assert p_a.r.shape == (1,)
    assert p_b.r.shape == (1,)
    assert 0.0 <= p_a.r[0] <= 1.0 + 1e-9
    assert 0.0 <= p_b.r[0] <= 1.0 + 1e-9
    # Symmetric caps -> symmetric factors -> deficit covered.
    assert p_a.r[0] == pytest.approx(p_b.r[0], abs=1e-3)
    assert (2.5 * p_a.r[0] + 2.5 * p_b.r[0]) >= 4.0 - 1e-3
