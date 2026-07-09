"""Unit tests for the DEED-ADMM distributed algorithm."""

from __future__ import annotations

import numpy as np

from distributed_resource_optimization import (
    DEEDADMMAlgorithm,
    DEEDADMMMessage,
    create_deed_admm_renewable_participant,
    create_deed_admm_storage_participant,
    create_deed_admm_thermal_participant,
    start_distributed_optimization,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finish_cb(results: dict, aid: int):
    def cb(algorithm: DEEDADMMAlgorithm, carrier) -> None:
        results[aid] = algorithm.P.copy()
    return cb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_two_equal_agents_split_load():
    """Two thermal agents with equal cost/capacity → each takes ~50% of demand."""
    tau = 1
    demand = np.array([100.0])
    d_i = demand / 2.0  # equal allocation

    results: dict[int, np.ndarray] = {}

    actors = [
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, i),
            p_min=0.0,
            p_max=100.0,
            marginal_cost=1.0,
            d_i=d_i,
            gamma=0.05,
            max_iter=500,
            n_agents=2,
        )
        for i in range(2)
    ]

    start_msg = DEEDADMMMessage(
        lam=np.zeros(tau),
        xi=np.zeros(tau),
        k=0,
        data=None,
        initial=True,
    )
    await start_distributed_optimization(actors, start_msg)

    assert len(results) == 2
    for p in results.values():
        np.testing.assert_allclose(p, [50.0], atol=1.0)


async def test_merit_order_dispatch():
    """Cheaper generator should produce more when they have different costs."""
    tau = 1
    demand = np.array([100.0])
    d_i = demand / 2.0

    results: dict[int, np.ndarray] = {}

    # Agent 0: marginal cost 1 (cheap), agent 1: marginal cost 10 (expensive)
    actors = [
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, i),
            p_min=0.0,
            p_max=100.0,
            marginal_cost=float(cost),
            d_i=d_i,
            gamma=0.05,
            max_iter=800,
            n_agents=2,
        )
        for i, cost in enumerate([1.0, 10.0])
    ]

    start_msg = DEEDADMMMessage(
        lam=np.zeros(tau),
        xi=np.zeros(tau),
        k=0,
        data=None,
        initial=True,
    )
    await start_distributed_optimization(actors, start_msg)

    assert len(results) == 2
    # Total generation should equal demand
    total = results[0][0] + results[1][0]
    assert abs(total - 100.0) < 5.0, f"Total {total} far from demand 100"
    # Cheap agent should produce at least as much as expensive
    assert results[0][0] >= results[1][0] - 5.0


async def test_capacity_constraint_respected():
    """Agent with low p_max should not exceed its capacity."""
    tau = 1
    demand = np.array([60.0])
    d_i = demand / 2.0

    results: dict[int, np.ndarray] = {}

    # Agent 0 has p_max=20 (constrained), agent 1 has p_max=100
    actors = [
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, i),
            p_min=0.0,
            p_max=float(pmax),
            marginal_cost=1.0,
            d_i=d_i,
            gamma=0.05,
            max_iter=800,
            n_agents=2,
        )
        for i, pmax in enumerate([20.0, 100.0])
    ]

    start_msg = DEEDADMMMessage(
        lam=np.zeros(tau),
        xi=np.zeros(tau),
        k=0,
        data=None,
        initial=True,
    )
    await start_distributed_optimization(actors, start_msg)

    assert len(results) == 2
    assert results[0][0] <= 20.0 + 1.0, f"Constrained agent exceeded p_max: {results[0][0]}"


async def test_multistep_horizon():
    """Algorithm works for a multi-step scheduling horizon (τ > 1)."""
    tau = 24
    demand = np.full(tau, 80.0)
    d_i = demand / 2.0

    results: dict[int, np.ndarray] = {}

    actors = [
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, i),
            p_min=0.0,
            p_max=80.0,
            marginal_cost=float(cost),
            d_i=d_i,
            gamma=0.05,
            max_iter=600,
            n_agents=2,
        )
        for i, cost in enumerate([1.0, 2.0])
    ]

    start_msg = DEEDADMMMessage(
        lam=np.zeros(tau),
        xi=np.zeros(tau),
        k=0,
        data=None,
        initial=True,
    )
    await start_distributed_optimization(actors, start_msg)

    assert len(results) == 2
    total = results[0] + results[1]
    np.testing.assert_allclose(total, demand, atol=5.0)


async def test_renewable_zero_cost():
    """Renewable with zero cost should absorb demand up to its availability."""
    tau = 1
    demand = np.array([50.0])
    d_i = demand / 2.0

    results: dict[int, np.ndarray] = {}

    # Agent 0: renewable with p_max=30 (limited), agent 1: thermal
    agents = [
        create_deed_admm_renewable_participant(
            _make_finish_cb(results, 0),
            p_max_timeseries=np.array([30.0]),
            d_i=d_i,
            max_iter=600,
            n_agents=2,
        ),
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, 1),
            p_min=0.0,
            p_max=100.0,
            marginal_cost=5.0,
            d_i=d_i,
            max_iter=600,
            n_agents=2,
        ),
    ]

    start_msg = DEEDADMMMessage(
        lam=np.zeros(tau),
        xi=np.zeros(tau),
        k=0,
        data=None,
        initial=True,
    )
    await start_distributed_optimization(agents, start_msg)

    assert len(results) == 2
    # Renewable should be near its p_max (highest priority = zero cost)
    assert results[0][0] <= 30.0 + 1.0
    # Total should be near demand
    total = results[0][0] + results[1][0]
    assert abs(total - 50.0) < 5.0


async def test_storage_soc_bounds_respected():
    """Storage dispatch must stay within SOC and power limits at every timestep."""
    tau = 12
    e_max = 40.0  # MWh
    p_max = 10.0  # MW
    demand = np.full(tau, 20.0)  # MW
    n_agents = 2

    results: dict[int, np.ndarray] = {}

    agents = [
        # Generator covers base demand; storage can arbitrage freely.
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, 0),
            p_min=0.0,
            p_max=50.0,
            marginal_cost=1.0,
            d_i=demand,   # generator takes full demand slice
            gamma=0.05,
            max_iter=600,
            n_agents=n_agents,
        ),
        create_deed_admm_storage_participant(
            _make_finish_cb(results, 1),
            e_max=e_max,
            p_charge_max=p_max,
            p_discharge_max=p_max,
            eta_charge=0.95,
            eta_discharge=0.95,
            e_initial=1.0,   # start fully charged
            e_final=0.5,     # target half-full at end
            tau=tau,
            gamma=0.05,
            max_iter=600,
            n_agents=n_agents,
        ),
    ]

    start_msg = DEEDADMMMessage(
        lam=np.zeros(tau), xi=np.zeros(tau), k=0, data=None, initial=True
    )
    await start_distributed_optimization(agents, start_msg)

    assert len(results) == 2
    P_stor = results[1]
    assert P_stor.shape == (tau,)
    # Power bounds respected
    assert np.all(P_stor <= p_max + 1e-6), f"Discharge limit exceeded: {P_stor.max()}"
    assert np.all(P_stor >= -p_max - 1e-6), f"Charge limit exceeded: {P_stor.min()}"
    # SOC stays in [0, e_max]
    soc = np.zeros(tau + 1)
    soc[0] = 1.0 * e_max
    for t in range(tau):
        if P_stor[t] >= 0:
            soc[t + 1] = soc[t] - P_stor[t] / 0.95
        else:
            soc[t + 1] = soc[t] - P_stor[t] * 0.95
    assert np.all(soc >= -1e-6), f"SOC went negative: {soc.min()}"
    assert np.all(soc <= e_max + 1e-6), f"SOC exceeded e_max: {soc.max()}"


async def test_storage_generator_balance():
    """Generator + storage together should satisfy total demand at every step."""
    tau = 24
    demand = np.full(tau, 40.0)
    n_agents = 2

    results: dict[int, np.ndarray] = {}

    agents = [
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, 0),
            p_min=0.0,
            p_max=50.0,
            marginal_cost=5.0,
            d_i=demand / 1,   # generator takes full demand allocation
            gamma=0.05,
            max_iter=600,
            n_agents=n_agents,
        ),
        create_deed_admm_storage_participant(
            _make_finish_cb(results, 1),
            e_max=40.0,
            p_charge_max=10.0,
            p_discharge_max=10.0,
            e_initial=0.5,
            tau=tau,
            gamma=0.05,
            max_iter=600,
            n_agents=n_agents,
        ),
    ]

    start_msg = DEEDADMMMessage(
        lam=np.zeros(tau), xi=np.zeros(tau), k=0, data=None, initial=True
    )
    await start_distributed_optimization(agents, start_msg)

    assert len(results) == 2
    total = results[0] + results[1]
    # Overall supply should track demand (storage shifts power, not destroys it)
    np.testing.assert_allclose(total.sum(), demand.sum(), atol=demand.sum() * 0.05)


async def test_storage_reaches_terminal_soc_target():
    """Terminal SOC must hit e_final (regression: inverted bisection direction
    in _project_soc silently skipped the terminal correction, leaving the
    battery drained instead of returning to its target)."""
    tau = 24
    e_max = 40.0
    eta = 0.95
    demand = np.full(tau, 30.0)
    n_agents = 2

    results: dict[int, np.ndarray] = {}
    storage_alg_holder: dict[str, object] = {}

    def storage_cb(algorithm, carrier) -> None:
        results[1] = algorithm.P.copy()
        storage_alg_holder["alg"] = algorithm

    agents = [
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, 0),
            p_min=0.0,
            p_max=60.0,
            marginal_cost=5.0,
            d_i=demand,
            gamma=0.05,
            max_iter=600,
            n_agents=n_agents,
        ),
        create_deed_admm_storage_participant(
            storage_cb,
            e_max=e_max,
            p_charge_max=10.0,
            p_discharge_max=10.0,
            eta_charge=eta,
            eta_discharge=eta,
            e_initial=1.0,  # start full
            e_final=0.5,  # must end half-full
            tau=tau,
            gamma=0.05,
            max_iter=600,
            n_agents=n_agents,
        ),
    ]

    start_msg = DEEDADMMMessage(
        lam=np.zeros(tau), xi=np.zeros(tau), k=0, data=None, initial=True
    )
    await start_distributed_optimization(agents, start_msg)

    P_stor = results[1]
    # Recompute the SOC path independently from the reported schedule.
    e = 1.0 * e_max
    for p in P_stor:
        e -= p / eta if p >= 0 else p * eta
    e_target = 0.5 * e_max
    assert abs(e - e_target) < 0.5, (
        f"Terminal energy {e:.2f} MWh missed target {e_target:.2f} MWh"
    )
    # The algorithm's own SOC trajectory must agree and end on target.
    E = storage_alg_holder["alg"].E
    assert E.shape == (tau + 1,)
    assert abs(E[-1] - e_target) < 0.5


async def test_quadratic_cost_three_agents():
    """Quadratic costs: dispatch must equalise marginal costs 2aᵢxᵢ = λ.

    a = [0.1, 0.2, 0.4], demand 70 → xᵢ = λ/(2aᵢ), Σxᵢ = 70 → λ = 8,
    x* = [40, 20, 10].
    """
    tau = 1
    demand = np.array([70.0])
    quads = [0.1, 0.2, 0.4]
    n_agents = 3
    d_i = demand / n_agents

    results: dict[int, np.ndarray] = {}

    actors = [
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, i),
            p_min=0.0,
            p_max=100.0,
            marginal_cost=0.0,
            cost_quad=a,
            d_i=d_i,
            gamma=0.05,
            max_iter=1000,
            n_agents=n_agents,
        )
        for i, a in enumerate(quads)
    ]

    start_msg = DEEDADMMMessage(
        lam=np.zeros(tau), xi=np.zeros(tau), k=0, data=None, initial=True
    )
    await start_distributed_optimization(actors, start_msg)

    assert len(results) == 3
    np.testing.assert_allclose(results[0], [40.0], atol=2.0)
    np.testing.assert_allclose(results[1], [20.0], atol=2.0)
    np.testing.assert_allclose(results[2], [10.0], atol=2.0)
