"""Diffusion algorithm unit and integration tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    LinearCostEconomicDispatchDiffusionActor,
    NoDiffusionActor,
    ReservoirStorageDiffusionActor,
    create_diffusion_participant,
    create_diffusion_start,
    start_distributed_optimization,
)

# ---------------------------------------------------------------------------
# Integration tests (full async runs via SimpleCarrier)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diffusion_all_callbacks_fire():
    """All N participants must fire their finish_callback exactly once."""
    n = 3
    horizon = 4
    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo._lam.copy()

        return finish

    actors = [
        create_diffusion_participant(make_finish(i), max_iter=50, horizon=horizon)
        for i in range(n)
    ]
    start = create_diffusion_start(initial_lam=5.0, horizon=horizon)
    await start_distributed_optimization(actors, start)

    assert len(results) == n, f"Expected {n} callbacks, got {len(results)}"


@pytest.mark.asyncio
async def test_diffusion_no_actor_converges_to_consensus():
    """With NoDiffusionActor all agents must converge to the same λ value."""
    n = 3
    horizon = 1
    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo._lam.copy()

        return finish

    actors = [
        create_diffusion_participant(
            make_finish(i),
            diffusion_actor=NoDiffusionActor(),
            initial_lam=float((i + 1) * 5),
            max_iter=100,
            horizon=horizon,
        )
        for i in range(n)
    ]
    start = create_diffusion_start(initial_lam=5.0, horizon=horizon)
    await start_distributed_optimization(actors, start)

    assert len(results) == n
    for i in range(1, n):
        assert np.allclose(results[0], results[i], atol=0.5), (
            f"λ values differ: agent 0={results[0]} vs agent {i}={results[i]}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "weight_rule", ["mean_metropolis", "averaging", "relative_degree", "hastings"]
)
async def test_diffusion_converges_under_every_weight_rule(weight_rule):
    """Each weight_rule must actually be wired through the full async
    message-passing path, not just the standalone _weight_rules unit tests."""
    n = 3
    horizon = 1
    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo._lam.copy()

        return finish

    actors = [
        create_diffusion_participant(
            make_finish(i),
            diffusion_actor=NoDiffusionActor(),
            initial_lam=float((i + 1) * 5),
            max_iter=100,
            horizon=horizon,
            weight_rule=weight_rule,
        )
        for i in range(n)
    ]
    start = create_diffusion_start(initial_lam=5.0, horizon=horizon)
    await start_distributed_optimization(actors, start)

    assert len(results) == n
    for i in range(1, n):
        assert np.allclose(results[0], results[i], atol=0.5), (
            f"λ values differ: agent 0={results[0]} vs agent {i}={results[i]}"
        )


@pytest.mark.asyncio
async def test_diffusion_merit_order():
    """Cheap generator should receive more load than the expensive one."""
    horizon = 2
    demand = np.array([20.0, 20.0])
    n = 2
    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo.actor.P.copy()

        return finish

    actors = [
        create_diffusion_participant(
            make_finish(0),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=2.0, p_max=20.0, epsilon=0.1, n_guess=n
            ),
            max_iter=300,
            horizon=horizon,
            epsilon=0.2,
        ),
        create_diffusion_participant(
            make_finish(1),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=8.0, p_max=20.0, epsilon=0.1, n_guess=n
            ),
            max_iter=300,
            horizon=horizon,
            epsilon=0.2,
        ),
    ]
    start = create_diffusion_start(initial_lam=10.0, data=demand, horizon=horizon)
    await start_distributed_optimization(actors, start)

    assert 0 in results and 1 in results, "Not all finish callbacks fired"
    assert np.all(results[0] >= results[1] - 0.5), (
        f"Cheap generator should dispatch more: {results[0]} vs {results[1]}"
    )


@pytest.mark.asyncio
async def test_diffusion_equal_cost_equal_dispatch():
    """Equal-cost generators should converge to equal power outputs."""
    horizon = 3
    demand = np.array([30.0, 30.0, 30.0])
    n = 3
    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo.actor.P.copy()

        return finish

    actors = [
        create_diffusion_participant(
            make_finish(i),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=5.0, p_max=30.0, epsilon=0.1, n_guess=n
            ),
            max_iter=300,
            horizon=horizon,
            epsilon=0.2,
        )
        for i in range(n)
    ]
    start = create_diffusion_start(initial_lam=10.0, data=demand, horizon=horizon)
    await start_distributed_optimization(actors, start)

    assert len(results) == n, "Not all finish callbacks fired"
    for i in range(1, n):
        assert np.allclose(results[0], results[i], atol=1.0), (
            f"Power schedules differ: agent 0={results[0]} vs agent {i}={results[i]}"
        )


@pytest.mark.asyncio
async def test_diffusion_with_storage_respects_box_constraints():
    """Storage actor output must stay within charge/discharge power limits."""
    horizon = 4
    demand = np.full(horizon, 10.0)
    p_charge_max = 5.0
    p_discharge_max = 8.0
    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo.actor.P.copy()

        return finish

    actors = [
        create_diffusion_participant(
            make_finish(0),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=5.0, p_max=15.0, epsilon=0.1, n_guess=2
            ),
            max_iter=300,
            horizon=horizon,
            epsilon=0.2,
        ),
        create_diffusion_participant(
            make_finish(1),
            diffusion_actor=ReservoirStorageDiffusionActor(
                e_max=20.0,
                p_charge_max=p_charge_max,
                p_discharge_max=p_discharge_max,
                charge_cost=0.0,
                discharge_cost=1.0,
                epsilon=0.1,
                n_guess=2,
            ),
            max_iter=300,
            horizon=horizon,
            epsilon=0.2,
        ),
    ]
    start = create_diffusion_start(initial_lam=10.0, data=demand, horizon=horizon)
    await start_distributed_optimization(actors, start)

    assert 0 in results and 1 in results, "Not all finish callbacks fired"
    assert np.all(results[1] >= -p_charge_max - 1e-9), (
        f"Storage charges beyond limit: {results[1]}"
    )
    assert np.all(results[1] <= p_discharge_max + 1e-9), (
        f"Storage discharges beyond limit: {results[1]}"
    )


@pytest.mark.asyncio
async def test_diffusion_horizon_larger_than_one():
    """Multi-step horizon: λ and actor.P must have the correct length."""
    horizon = 6
    n = 2
    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo.actor.P.copy()

        return finish

    actors = [
        create_diffusion_participant(
            make_finish(i),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=5.0, p_max=20.0, epsilon=0.1, n_guess=n
            ),
            max_iter=100,
            horizon=horizon,
            epsilon=0.2,
        )
        for i in range(n)
    ]
    start = create_diffusion_start(
        initial_lam=10.0, data=np.full(horizon, 20.0), horizon=horizon
    )
    await start_distributed_optimization(actors, start)

    assert len(results) == n
    for idx, P in results.items():
        assert P.shape == (horizon,), f"Wrong shape for agent {idx}: {P.shape}"


# ---------------------------------------------------------------------------
# LinearCostEconomicDispatchDiffusionActor unit tests
# ---------------------------------------------------------------------------


class TestLinearCostEconomicDispatchDiffusionActor:
    def test_below_cost_clips_to_p_min(self):
        actor = LinearCostEconomicDispatchDiffusionActor(
            cost=10.0, p_max=30.0, epsilon=0.1, n_guess=3
        )
        lam = np.array([5.0])  # below cost → P = 0
        grad = actor.gradient_term(lam, np.array([30.0]))
        # P = clip((5-10)/0.1, 0, 30) = 0; gradient = 0 - 30/3 = -10
        assert np.allclose(grad, [-10.0])
        assert np.allclose(actor.P, [0.0])

    def test_above_cost_clips_to_p_max(self):
        actor = LinearCostEconomicDispatchDiffusionActor(
            cost=0.0, p_max=5.0, epsilon=0.1, n_guess=1
        )
        lam = np.array([100.0])  # far above cost → P = p_max
        grad = actor.gradient_term(lam, np.array([10.0]))
        # P = clip(1000, 0, 5) = 5; gradient = 5 - 10/1 = -5
        assert np.allclose(grad, [-5.0])
        assert np.allclose(actor.P, [5.0])

    def test_at_clearing_price_zero_residual(self):
        """At the market-clearing price the imbalance term is zero."""
        n = 3
        p_target = np.array([30.0])
        # P*(λ*) = p_target/n = 10 → λ* = cost + epsilon * 10 = 0 + 0.1*10 = 1.0
        actor = LinearCostEconomicDispatchDiffusionActor(
            cost=0.0, p_max=30.0, epsilon=0.1, n_guess=n
        )
        grad = actor.gradient_term(np.array([1.0]), p_target)
        assert np.allclose(grad, [0.0], atol=1e-10)

    def test_vectorised_lam_and_p_max(self):
        """Per-timestep p_max is applied element-wise."""
        actor = LinearCostEconomicDispatchDiffusionActor(
            cost=5.0, p_max=np.array([10.0, 20.0, 30.0]), epsilon=0.1, n_guess=1
        )
        lam = np.array([1000.0, 1000.0, 1000.0])  # all saturate at p_max
        actor.gradient_term(lam, np.array([0.0, 0.0, 0.0]))
        assert np.allclose(actor.P, [10.0, 20.0, 30.0])

    def test_none_p_target_treated_as_zero(self):
        actor = LinearCostEconomicDispatchDiffusionActor(
            cost=0.0, p_max=50.0, epsilon=0.1, n_guess=1
        )
        grad = actor.gradient_term(np.array([1.0]), None)
        # P = 10; gradient = 10 - 0 = 10
        assert np.allclose(grad, [10.0])

    def test_p_max_shape_mismatch_raises(self):
        actor = LinearCostEconomicDispatchDiffusionActor(
            cost=0.0, p_max=np.array([10.0, 20.0]), epsilon=0.1, n_guess=1
        )
        with pytest.raises(ValueError, match="p_max shape"):
            actor.gradient_term(np.array([1.0, 2.0, 3.0]), None)

    def test_updates_p_in_place(self):
        actor = LinearCostEconomicDispatchDiffusionActor(
            cost=10.0, p_max=20.0, epsilon=0.1, n_guess=1
        )
        lam = np.array([11.0])
        actor.gradient_term(lam, None)
        # P = clip((11-10)/0.1, 0, 20) = 10
        assert np.allclose(actor.P, [10.0])


# ---------------------------------------------------------------------------
# ReservoirStorageDiffusionActor unit tests
# ---------------------------------------------------------------------------


class TestReservoirStorageDiffusionActor:
    def _actor(self, **kw) -> ReservoirStorageDiffusionActor:
        defaults = dict(
            e_max=10.0,
            p_charge_max=5.0,
            p_discharge_max=8.0,
            charge_cost=1.0,
            discharge_cost=2.0,
            epsilon=0.1,
            n_guess=1,
        )
        defaults.update(kw)
        return ReservoirStorageDiffusionActor(**defaults)

    def test_high_price_leads_to_discharge(self):
        """Price >> discharge_threshold → actor must discharge (P > 0).

        e_initial=0.8, e_final=0.2 so the terminal target requires net discharge;
        the high price signal pushes the same direction.
        """
        actor = self._actor(charge_cost=1.0, discharge_cost=2.0, e_initial=0.8, e_final=0.2)
        lam = np.array([100.0])  # far above discharge_threshold = 3.0
        actor.gradient_term(lam, None)
        assert actor.P[0] > 0.0, "Should discharge when price is high"

    def test_low_price_leads_to_charge(self):
        """Price < charge_cost → actor must charge (P < 0).

        e_initial=0.2, e_final=0.8 so the terminal target requires net charge;
        the low price signal pushes the same direction.
        """
        actor = self._actor(charge_cost=5.0, discharge_cost=0.0, e_initial=0.2, e_final=0.8)
        lam = np.array([0.0])  # below charge_cost
        actor.gradient_term(lam, None)
        assert actor.P[0] < 0.0, "Should charge when price is low"

    def test_neutral_price_near_zero_power(self):
        """Price between charge_cost and discharge_threshold → P ≈ 0."""
        # discharge_threshold = charge_cost + discharge_cost = 1 + 2 = 3
        # charge_cost = 1; any price in (1, 3) → neither zone active
        actor = self._actor(charge_cost=1.0, discharge_cost=2.0, epsilon=0.1)
        lam = np.array([2.0])  # in the dead-band
        actor.gradient_term(lam, None)
        assert actor.P[0] == 0.0, "Should be idle in the dead-band"

    def test_box_constraints_respected(self):
        actor = self._actor(p_charge_max=3.0, p_discharge_max=6.0)
        lam = np.array([1000.0, 1000.0, 1000.0])  # push hard toward discharge
        actor.gradient_term(lam, None)
        assert np.all(actor.P >= -3.0 - 1e-9)
        assert np.all(actor.P <= 6.0 + 1e-9)

    def test_p_and_e_arrays_have_correct_horizon(self):
        actor = self._actor()
        horizon = 5
        lam = np.zeros(horizon)
        actor.gradient_term(lam, None)
        assert actor.P.shape == (horizon,), f"P shape wrong: {actor.P.shape}"
        assert actor.E.shape == (horizon,), f"E shape wrong: {actor.E.shape}"

    def test_terminal_energy_target_approx_met(self):
        """Bisection should bring the final SOC within 5% of e_final."""
        actor = ReservoirStorageDiffusionActor(
            e_max=10.0,
            p_charge_max=5.0,
            p_discharge_max=5.0,
            e_initial=0.5,
            e_final=0.5,
            charge_cost=0.0,
            discharge_cost=0.0,
            epsilon=0.1,
            n_guess=1,
        )
        lam = np.zeros(10)
        actor.gradient_term(lam, None)

        # Re-derive final energy from the stored schedule.
        e = actor.e_initial * actor.e_max
        for p in actor.P:
            if p >= 0.0:
                e -= p / actor.eta_discharge
            else:
                e -= p * actor.eta_charge
        e_target = actor.e_final * actor.e_max
        assert abs(e - e_target) < 0.5 * actor.e_max * 0.05, (
            f"Final energy {e:.3f} not close to target {e_target:.3f}"
        )

    def test_gradient_returns_p_minus_target_over_n(self):
        """Gradient = P - p_target/n_guess (target-tracking residual)."""
        actor = self._actor(n_guess=2)
        lam = np.array([0.0])  # idle (dead-band): P = 0
        p_target = np.array([10.0])
        grad = actor.gradient_term(lam, p_target)
        # P=0; gradient = 0 - 10/2 = -5
        assert np.allclose(grad, [-5.0])


# ---------------------------------------------------------------------------
# Regression tests for the regularity/consistency guards
#
# DiffusionAlgorithm shares its message-handling skeleton with
# ExactDiffusionAlgorithm (see the class docstrings); these mirror the exact-
# diffusion guard tests so a regression in either copy is caught.
# ---------------------------------------------------------------------------


class _MuteCarrier:
    """Fake carrier that reports two neighbours and swallows sends."""

    def others(self, _):
        return ["a", "b"]

    def send_to_other(self, content, receiver, meta=None):
        pass


@pytest.mark.asyncio
async def test_diffusion_rejects_mismatched_neighbour_degree():
    """regular_graph_weights() assumes every neighbour shares this node's
    degree; a neighbour reporting a different degree must raise rather than
    silently produce a non-stochastic weight matrix."""
    from distributed_resource_optimization.algorithm.diffusion.diffusion import (
        DiffusionAlgorithm,
        DiffusionMessage,
    )

    algo = DiffusionAlgorithm(finish_callback=lambda a, c: None, max_iter=10, horizon=1)
    start = create_diffusion_start(initial_lam=1.0, horizon=1)
    carrier = _MuteCarrier()
    await algo.on_exchange_message(carrier, start, {})  # initialise (2 neighbours)

    await algo.on_exchange_message(
        carrier,
        DiffusionMessage(
            phi=np.array([1.0]), k=0, data=None, degree=2, weight_rule="mean_metropolis"
        ),
        {},
    )
    with pytest.raises(ValueError, match="degree-regular"):
        await algo.on_exchange_message(
            carrier,
            DiffusionMessage(
                phi=np.array([1.0]), k=0, data=None, degree=3, weight_rule="mean_metropolis"
            ),
            {},
        )


@pytest.mark.asyncio
async def test_diffusion_rejects_mismatched_weight_rule():
    """All participants must agree on the same weight_rule; a neighbour
    reporting a different one must raise rather than silently mix weights
    computed from two different rules."""
    from distributed_resource_optimization.algorithm.diffusion.diffusion import (
        DiffusionAlgorithm,
        DiffusionMessage,
    )

    algo = DiffusionAlgorithm(
        finish_callback=lambda a, c: None, max_iter=10, horizon=1, weight_rule="mean_metropolis"
    )
    start = create_diffusion_start(initial_lam=1.0, horizon=1)
    carrier = _MuteCarrier()
    await algo.on_exchange_message(carrier, start, {})

    await algo.on_exchange_message(
        carrier,
        DiffusionMessage(
            phi=np.array([1.0]), k=0, data=None, degree=2, weight_rule="mean_metropolis"
        ),
        {},
    )
    with pytest.raises(ValueError, match="same weight_rule"):
        await algo.on_exchange_message(
            carrier,
            DiffusionMessage(
                phi=np.array([1.0]), k=0, data=None, degree=2, weight_rule="averaging"
            ),
            {},
        )


@pytest.mark.asyncio
async def test_diffusion_ignores_stale_message_after_termination():
    """A late/duplicate pre-termination message arriving after finish_callback
    has already fired must not silently restart the algorithm."""
    from distributed_resource_optimization.algorithm.diffusion.diffusion import (
        DiffusionAlgorithm,
        DiffusionMessage,
    )

    finished = []
    algo = DiffusionAlgorithm(
        finish_callback=lambda a, c: finished.append(a._lam.copy()), max_iter=1, horizon=1
    )
    carrier = _MuteCarrier()
    start = create_diffusion_start(initial_lam=1.0, horizon=1)
    await algo.on_exchange_message(carrier, start, {})  # k=0 initialisation

    # Both neighbours report k=1 (>= max_iter): terminates and fires the callback.
    await algo.on_exchange_message(
        carrier, DiffusionMessage(phi=np.array([1.0]), k=1, data=None, degree=2), {}
    )
    await algo.on_exchange_message(
        carrier, DiffusionMessage(phi=np.array([1.0]), k=1, data=None, degree=2), {}
    )
    assert len(finished) == 1

    # A stale, late-arriving k=0 message from before termination must be
    # ignored, not misread as a fresh kick-off that restarts the algorithm.
    await algo.on_exchange_message(
        carrier, DiffusionMessage(phi=np.array([99.0]), k=0, data=None, degree=2), {}
    )
    assert len(finished) == 1, "Stale pre-termination message must not restart the algorithm"
