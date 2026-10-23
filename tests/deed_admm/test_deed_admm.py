"""Unit tests for the DEED-ADMM distributed algorithm."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from scipy.optimize import linprog

from distributed_resource_optimization import (
    DEEDADMMAlgorithm,
    DEEDADMMMessage,
    create_deed_admm_renewable_participant,
    create_deed_admm_storage_participant,
    create_deed_admm_thermal_participant,
    start_distributed_optimization,
)
from distributed_resource_optimization.carrier.simple import ActorContainer, SimpleCarrier

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
            d_i=demand,  # generator takes full demand slice
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
            e_initial=1.0,  # start fully charged
            e_final=0.5,  # target half-full at end
            tau=tau,
            gamma=0.05,
            max_iter=600,
            n_agents=n_agents,
        ),
    ]

    start_msg = DEEDADMMMessage(lam=np.zeros(tau), xi=np.zeros(tau), k=0, data=None, initial=True)
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
            d_i=demand / 1,  # generator takes full demand allocation
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

    start_msg = DEEDADMMMessage(lam=np.zeros(tau), xi=np.zeros(tau), k=0, data=None, initial=True)
    await start_distributed_optimization(agents, start_msg)

    assert len(results) == 2
    total = results[0] + results[1]
    # Overall supply should track demand (storage shifts power, not destroys it)
    np.testing.assert_allclose(total.sum(), demand.sum(), atol=demand.sum() * 0.05)


async def test_storage_respects_terminal_reserve():
    """The terminal reserve E_τ ≥ e_final·e_max (paper eq. 11) must hold, and
    the reported SOC trajectory must match the reported schedule."""
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
            e_final=0.5,  # keep at least half at the end
            tau=tau,
            gamma=0.05,
            max_iter=600,
            n_agents=n_agents,
        ),
    ]

    start_msg = DEEDADMMMessage(lam=np.zeros(tau), xi=np.zeros(tau), k=0, data=None, initial=True)
    await start_distributed_optimization(agents, start_msg)

    P_stor = results[1]
    # Recompute the SOC path independently from the reported schedule.
    e = 1.0 * e_max
    for p in P_stor:
        e -= p / eta if p >= 0 else p * eta
    e_target = 0.5 * e_max
    assert e >= e_target - 0.5, f"Terminal energy {e:.2f} MWh below reserve {e_target:.2f} MWh"
    # Discharge is free and the generator is not, so the storage should use
    # the energy above the reserve rather than keep it.
    assert e <= e_target + 0.5
    E = storage_alg_holder["alg"].E
    assert E.shape == (tau + 1,)
    assert abs(E[-1] - e) < 0.5


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

    start_msg = DEEDADMMMessage(lam=np.zeros(tau), xi=np.zeros(tau), k=0, data=None, initial=True)
    await start_distributed_optimization(actors, start_msg)

    assert len(results) == 3
    np.testing.assert_allclose(results[0], [40.0], atol=2.0)
    np.testing.assert_allclose(results[1], [20.0], atol=2.0)
    np.testing.assert_allclose(results[2], [10.0], atol=2.0)


async def test_quadratic_cost_on_path_graph_stops_together():
    """Same problem on the path 1 - 2 - 3 (diameter 2) with the convergence
    stop: W = I − βL stays doubly stochastic, every agent stops in the same
    round (none waits for a stopped neighbour) and reaches x* = [40, 20, 10]."""
    demand = np.array([70.0])
    algos: dict[int, DEEDADMMAlgorithm] = {}
    results: dict[int, np.ndarray] = {}

    def finish(i):
        def cb(alg, carrier):
            algos[i] = alg
            results[i] = np.asarray(alg.P, dtype=float).copy()

        return cb

    actors = [
        create_deed_admm_thermal_participant(
            finish(i),
            p_min=0.0,
            p_max=100.0,
            marginal_cost=0.0,
            cost_quad=a,
            d_i=demand / 3,
            gamma=0.05,
            max_iter=20000,
            beta=1.0 / 3.0,  # 1 / (max degree + 1)
            tol=1e-4,
            diameter=2,
        )
        for i, a in enumerate((0.1, 0.2, 0.4))
    ]
    start_msg = DEEDADMMMessage(lam=np.zeros(1), xi=np.zeros(1), k=0, data=None, initial=True)
    await start_distributed_optimization(
        actors, start_msg, container=ActorContainer(neighbours={1: [2], 2: [1, 3], 3: [2]})
    )

    assert len(results) == 3
    assert all(a.converged for a in algos.values())
    assert len({a.iterations for a in algos.values()}) == 1
    np.testing.assert_allclose(results[0], [40.0], atol=2.0)
    np.testing.assert_allclose(results[1], [20.0], atol=2.0)
    np.testing.assert_allclose(results[2], [10.0], atol=2.0)


# ---------------------------------------------------------------------------
# Protocol robustness
# ---------------------------------------------------------------------------


class _FakeCarrier:
    """Records outgoing messages; neighbours are given explicitly."""

    def __init__(self, others: list) -> None:
        self._others = others
        self.sent: list[tuple[object, DEEDADMMMessage]] = []

    def others(self) -> list:
        return self._others

    def send_to_other(self, content, receiver) -> None:
        self.sent.append((receiver, content))


class _GraphCarrier(SimpleCarrier):
    """SimpleCarrier with a fixed neighbour set and optional random delays."""

    def __init__(self, container, actor, neighbours, rng) -> None:
        super().__init__(container, actor)
        self._neighbours = neighbours
        self._rng = rng

    def others(self) -> list[int]:
        return self._neighbours

    def send_to_other(self, content, receiver):
        if self._rng is None:
            return super().send_to_other(content, receiver)
        delay = float(self._rng.uniform(0.0, 0.002))
        target = self.container.actors[receiver - 1]
        meta = {"sender": self.aid}

        async def _deliver() -> None:
            await asyncio.sleep(delay)
            await self._dispatch_to(target, content, meta)

        return self._track_dispatch(_deliver())


async def _run_on_graph(actors, neighbours: dict[int, list[int]], tau: int, seed=None) -> None:
    container = ActorContainer()
    rng = None if seed is None else np.random.default_rng(seed)
    carriers = [
        _GraphCarrier(container, actor, neighbours[i + 1], rng) for i, actor in enumerate(actors)
    ]
    start = DEEDADMMMessage(lam=np.zeros(tau), xi=np.zeros(tau), k=0, data=None, initial=True)
    carriers[0].send_to_other(start, carriers[0].aid)
    await container.done_event.wait()


def _thermal_fleet(results, costs, p_max, demand, n, *, beta=None, max_iter=300):
    return [
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, i),
            p_min=0.0,
            p_max=p_max,
            marginal_cost=c,
            cost_quad=0.01,
            d_i=demand / n,
            gamma=0.05,
            max_iter=max_iter,
            n_agents=n,
            beta=beta,
        )
        for i, c in enumerate(costs)
    ]


async def test_agent_stops_after_its_own_last_round():
    """The agent finishes once, right after its own last round, and never
    sends round max_iter — so no message can outlive the run."""
    calls: list[int] = []
    alg = create_deed_admm_thermal_participant(
        lambda a, c: calls.append(1),
        p_min=0.0,
        p_max=10.0,
        marginal_cost=1.0,
        d_i=np.ones(2),
        max_iter=2,
        n_agents=2,
    )
    carrier = _FakeCarrier(["B"])
    z = np.zeros(2)
    await alg.on_exchange_message(carrier, DEEDADMMMessage(z, z, 0, None, initial=True), None)
    await alg.on_exchange_message(carrier, DEEDADMMMessage(z, z, 0, None), None)
    assert calls == []
    await alg.on_exchange_message(carrier, DEEDADMMMessage(z, z, 1, None), None)
    assert calls == [1]
    assert sorted(m.k for _, m in carrier.sent) == [0, 1]


async def test_every_agent_finishes_once_under_random_delays():
    tau, n = 4, 6
    demand = np.array([100.0, 150.0, 120.0, 80.0])
    full = {i: [j for j in range(1, n + 1) if j != i] for i in range(1, n + 1)}
    calls: list[int] = []
    actors = _thermal_fleet({}, [1, 2, 3, 4, 5, 6], 60.0, demand, n, max_iter=100)
    for i, actor in enumerate(actors):
        actor.finish_callback = lambda alg, carrier, i=i: calls.append(i)
    for seed in range(5):
        calls.clear()
        await _run_on_graph(actors, full, tau, seed=seed)
        assert sorted(calls) == list(range(n))


async def test_round_ahead_messages_are_buffered():
    """Round k+1 state arriving before round k must not be stepped early."""
    alg = create_deed_admm_thermal_participant(
        lambda a, c: None, p_min=0.0, p_max=10.0, marginal_cost=1.0, d_i=np.ones(1), n_agents=3
    )
    carrier = _FakeCarrier(["B", "C"])
    z = np.zeros(1)
    await alg.on_exchange_message(carrier, DEEDADMMMessage(z, z, 0, None, initial=True), None)
    await alg.on_exchange_message(carrier, DEEDADMMMessage(z, z, 0, None), None)
    await alg.on_exchange_message(carrier, DEEDADMMMessage(z, z, 1, None), None)
    await alg.on_exchange_message(carrier, DEEDADMMMessage(z, z, 1, None), None)
    assert alg._k == 0  # round 0 still incomplete
    await alg.on_exchange_message(carrier, DEEDADMMMessage(z, z, 0, None), None)
    assert alg._k == 2  # round 0 and the buffered round 1 both ran


async def test_random_delays_match_in_order_delivery():
    """Results must not depend on message timing."""
    tau, n = 4, 4
    demand = np.array([100.0, 150.0, 120.0, 80.0])
    full = {i: [j for j in range(1, n + 1) if j != i] for i in range(1, n + 1)}
    ref: dict[int, np.ndarray] = {}
    jit: dict[int, np.ndarray] = {}
    await _run_on_graph(_thermal_fleet(ref, [1, 2, 3, 4], 60.0, demand, n), full, tau)
    await _run_on_graph(_thermal_fleet(jit, [1, 2, 3, 4], 60.0, demand, n), full, tau, seed=7)
    for i in range(n):
        np.testing.assert_allclose(jit[i], ref[i], rtol=1e-9, atol=1e-9)


async def test_single_agent_finishes():
    results: dict[int, np.ndarray] = {}
    alg = create_deed_admm_thermal_participant(
        _make_finish_cb(results, 0),
        p_min=0.0,
        p_max=100.0,
        marginal_cost=1.0,
        d_i=np.array([30.0, 60.0]),
        max_iter=400,
        n_agents=1,
    )
    z = np.zeros(2)
    await alg.on_exchange_message(
        _FakeCarrier([]), DEEDADMMMessage(z, z, 0, None, initial=True), None
    )
    np.testing.assert_allclose(results[0], [30.0, 60.0], atol=0.5)


async def test_weights_must_match_graph():
    """n_agents too small for the neighbourhood → non-positive self-weight."""
    alg = create_deed_admm_thermal_participant(
        lambda a, c: None, p_min=0.0, p_max=1.0, marginal_cost=1.0, d_i=np.ones(1), n_agents=2
    )
    z = np.zeros(1)
    with pytest.raises(ValueError, match="self-weight"):
        await alg.on_exchange_message(
            _FakeCarrier(["B", "C"]), DEEDADMMMessage(z, z, 0, None, initial=True), None
        )


async def test_sparse_ring_topology():
    """W = I − βL on a ring (the paper's sparse setting) reaches the optimum."""
    tau, n = 1, 5
    demand = np.array([150.0])
    ring = {i: [(i - 2) % n + 1, i % n + 1] for i in range(1, n + 1)}
    results: dict[int, np.ndarray] = {}
    actors = _thermal_fleet(results, [1, 2, 3, 4, 5], 60.0, demand, n, beta=0.3, max_iter=3000)
    await _run_on_graph(actors, ring, tau)
    total = sum(results[i][0] for i in range(n))
    assert abs(total - 150.0) < 1.0
    # Merit order: the two cheapest at capacity, the third covers the rest.
    np.testing.assert_allclose([results[i][0] for i in range(n)], [60, 60, 30, 0, 0], atol=2.0)


async def test_renewable_marginal_cost_is_respected():
    """A costly 'renewable' (e.g. biomass) must not run before cheaper thermal."""
    tau = 1
    demand = np.array([50.0])
    results: dict[int, np.ndarray] = {}
    agents = [
        create_deed_admm_renewable_participant(
            _make_finish_cb(results, 0),
            p_max_timeseries=np.array([100.0]),
            marginal_cost=20.0,
            d_i=demand / 2,
            max_iter=800,
            n_agents=2,
        ),
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, 1),
            p_min=0.0,
            p_max=40.0,
            marginal_cost=5.0,
            d_i=demand / 2,
            max_iter=800,
            n_agents=2,
        ),
    ]
    start = DEEDADMMMessage(lam=np.zeros(tau), xi=np.zeros(tau), k=0, data=None, initial=True)
    await start_distributed_optimization(agents, start)
    np.testing.assert_allclose(results[1], [40.0], atol=1.0)
    np.testing.assert_allclose(results[0], [10.0], atol=1.0)


async def test_time_varying_thermal_limit():
    tau = 2
    demand = np.array([50.0, 50.0])
    results: dict[int, np.ndarray] = {}
    agents = [
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, 0),
            p_min=0.0,
            p_max=np.array([50.0, 20.0]),
            marginal_cost=1.0,
            d_i=demand / 2,
            max_iter=800,
            n_agents=2,
        ),
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, 1),
            p_min=0.0,
            p_max=100.0,
            marginal_cost=5.0,
            d_i=demand / 2,
            max_iter=800,
            n_agents=2,
        ),
    ]
    start = DEEDADMMMessage(lam=np.zeros(tau), xi=np.zeros(tau), k=0, data=None, initial=True)
    await start_distributed_optimization(agents, start)
    np.testing.assert_allclose(results[0], [50.0, 20.0], atol=1.0)


# ---------------------------------------------------------------------------
# Storage (paper z block)
# ---------------------------------------------------------------------------


def _lp_optimum(demand, cheap_cap, cheap_cost, exp_cost, e_max, p_max, eta, e0_frac):
    """Central LP: cheap + expensive generator and one storage unit."""
    tau = len(demand)
    # variables: g1[τ], g2[τ], sd[τ], sc[τ]
    c = np.concatenate([np.full(tau, cheap_cost), np.full(tau, exp_cost), np.zeros(2 * tau)])
    eye = np.eye(tau)
    a_eq = np.hstack([eye, eye, eye, -eye])
    lower = np.tril(np.ones((tau, tau)))
    soc = np.hstack([np.zeros((tau, 2 * tau)), -lower / eta, lower * eta])  # E − E0
    e0 = e0_frac * e_max
    a_ub = np.vstack([soc, -soc])
    b_ub = np.concatenate([np.full(tau, e_max - e0), np.full(tau, e0)])
    b_ub[-1] = 0.0  # E_τ ≥ E_0
    bounds = [(0, cheap_cap)] * tau + [(0, None)] * tau + [(0, p_max)] * (2 * tau)
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=demand, bounds=bounds)
    assert res.success
    return res.fun


async def test_storage_matches_central_optimum():
    """Peak shifting: DEED-ADMM with storage reaches the LP optimum, never
    charges and discharges at once (Theorem 1) and keeps the SOC limits."""
    demand = np.array([20.0, 20.0, 60.0, 60.0, 20.0, 20.0])
    tau = len(demand)
    e_max, p_max, eta = 30.0, 15.0, 0.9
    results: dict[int, np.ndarray] = {}
    holder: dict[str, object] = {}

    def storage_cb(alg, carrier) -> None:
        results[2] = alg.P.copy()
        holder["alg"] = alg

    gens = [
        create_deed_admm_thermal_participant(
            _make_finish_cb(results, i),
            p_min=0.0,
            p_max=cap,
            marginal_cost=cost,
            d_i=demand / 2,
            gamma=0.05,
            max_iter=3000,
            n_agents=3,
        )
        for i, (cap, cost) in enumerate([(40.0, 1.0), (200.0, 10.0)])
    ]
    storage = create_deed_admm_storage_participant(
        storage_cb,
        e_max=e_max,
        p_charge_max=p_max,
        p_discharge_max=p_max,
        eta_charge=eta,
        eta_discharge=eta,
        e_initial=0.5,
        tau=tau,
        gamma=0.05,
        max_iter=3000,
        n_agents=3,
    )
    start = DEEDADMMMessage(lam=np.zeros(tau), xi=np.zeros(tau), k=0, data=None, initial=True)
    await start_distributed_optimization([*gens, storage], start)

    np.testing.assert_allclose(results[0] + results[1] + results[2], demand, atol=0.5)
    cost = 1.0 * results[0].sum() + 10.0 * results[1].sum()
    opt = _lp_optimum(demand, 40.0, 1.0, 10.0, e_max, p_max, eta, 0.5)
    assert cost <= opt * 1.01 + 1.0, f"cost {cost:.2f} vs optimum {opt:.2f}"

    alg = holder["alg"]
    z = alg._z_hat[: 2 * tau]
    assert np.all(np.minimum(z[:tau], z[tau:]) < 0.5), "simultaneous charge and discharge"
    assert np.all(alg.E >= -0.5) and np.all(alg.E <= e_max + 0.5)
    assert alg.E[-1] >= 0.5 * e_max - 0.5


async def _run_four_hubs(tol=None, max_iter=300):
    from distributed_resource_optimization import (
        create_deed_admm_thermal_participant,
        start_distributed_optimization,
    )
    from distributed_resource_optimization.algorithm.admm.deed_admm.deed_admm import DEEDADMMMessage

    demand = np.array([100.0, 150.0, 120.0, 80.0])
    algos = {}

    def make_finish(idx):
        def finish(algo, carrier):
            algos[idx] = algo

        return finish

    actors = [
        create_deed_admm_thermal_participant(
            make_finish(i),
            p_min=0.0,
            p_max=60.0,
            marginal_cost=c,
            cost_quad=0.01,
            d_i=demand / 4,
            gamma=0.05,
            max_iter=max_iter,
            n_agents=4,
            tol=tol,
        )
        for i, c in enumerate([1.0, 2.0, 3.0, 4.0])
    ]
    start = DEEDADMMMessage(lam=np.zeros(4), xi=np.zeros(4), k=0, data=None, initial=True)
    await start_distributed_optimization(actors, start)
    return algos, demand


@pytest.mark.asyncio
async def test_convergence_stop_ends_early_on_the_same_dispatch():
    """The paper defines no criterion; with tol the run stops once every
    agent's step and tracker are below it, on (nearly) the full-run result."""
    full, demand = await _run_four_hubs(tol=None)
    early, _ = await _run_four_hubs(tol=1e-6)

    assert all(a.converged is None and a.iterations == 300 for a in full.values())
    iterations = {a.iterations for a in early.values()}
    assert len(iterations) == 1 and iterations.pop() < 300
    assert all(a.converged is True for a in early.values())
    np.testing.assert_allclose(sum(a.P for a in early.values()), demand, atol=1e-3)
    for i in full:
        np.testing.assert_allclose(early[i].P, full[i].P, atol=1e-3)


@pytest.mark.asyncio
async def test_convergence_stop_not_converged_at_max_iter():
    algos, _ = await _run_four_hubs(tol=1e-12, max_iter=2)
    assert all(a.converged is False and a.iterations == 2 for a in algos.values())
