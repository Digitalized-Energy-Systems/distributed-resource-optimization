"""Exact Diffusion algorithm unit and integration tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    LinearCostEconomicDispatchDiffusionActor,
    NoDiffusionActor,
    ReservoirStorageDiffusionActor,
    create_diffusion_participant,
    create_diffusion_start,
    create_exact_diffusion_participant,
    create_exact_diffusion_start,
    start_distributed_optimization,
)

# ---------------------------------------------------------------------------
# Integration tests (full async runs via SimpleCarrier)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exact_diffusion_all_callbacks_fire():
    """All N participants must fire their finish_callback exactly once."""
    n = 3
    horizon = 4
    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo._lam.copy()

        return finish

    actors = [
        create_exact_diffusion_participant(make_finish(i), max_iter=50, horizon=horizon)
        for i in range(n)
    ]
    start = create_exact_diffusion_start(initial_lam=5.0, horizon=horizon)
    await start_distributed_optimization(actors, start)

    assert len(results) == n, f"Expected {n} callbacks, got {len(results)}"


@pytest.mark.asyncio
async def test_exact_diffusion_no_actor_converges_to_consensus():
    """With NoDiffusionActor all agents must converge to the same λ value."""
    n = 3
    horizon = 1
    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo._lam.copy()

        return finish

    actors = [
        create_exact_diffusion_participant(
            make_finish(i),
            diffusion_actor=NoDiffusionActor(),
            initial_lam=float((i + 1) * 5),
            max_iter=100,
            horizon=horizon,
        )
        for i in range(n)
    ]
    start = create_exact_diffusion_start(initial_lam=5.0, horizon=horizon)
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
async def test_exact_diffusion_converges_under_every_weight_rule(weight_rule):
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
        create_exact_diffusion_participant(
            make_finish(i),
            diffusion_actor=NoDiffusionActor(),
            initial_lam=float((i + 1) * 5),
            max_iter=100,
            horizon=horizon,
            weight_rule=weight_rule,
        )
        for i in range(n)
    ]
    start = create_exact_diffusion_start(initial_lam=5.0, horizon=horizon)
    await start_distributed_optimization(actors, start)

    assert len(results) == n
    for i in range(1, n):
        assert np.allclose(results[0], results[i], atol=0.5), (
            f"λ values differ: agent 0={results[0]} vs agent {i}={results[i]}"
        )


@pytest.mark.asyncio
async def test_exact_diffusion_merit_order():
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
        create_exact_diffusion_participant(
            make_finish(0),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=2.0, p_max=20.0, epsilon=0.1, n_guess=n
            ),
            max_iter=300,
            horizon=horizon,
            epsilon=0.1,
        ),
        create_exact_diffusion_participant(
            make_finish(1),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=8.0, p_max=20.0, epsilon=0.1, n_guess=n
            ),
            max_iter=300,
            horizon=horizon,
            epsilon=0.1,
        ),
    ]
    start = create_exact_diffusion_start(initial_lam=10.0, data=demand, horizon=horizon)
    await start_distributed_optimization(actors, start)

    assert 0 in results and 1 in results, "Not all finish callbacks fired"
    assert np.all(results[0] >= results[1] - 0.5), (
        f"Cheap generator should dispatch more: {results[0]} vs {results[1]}"
    )


@pytest.mark.asyncio
async def test_exact_diffusion_with_storage_respects_box_constraints():
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
        create_exact_diffusion_participant(
            make_finish(0),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=5.0, p_max=15.0, epsilon=0.1, n_guess=2
            ),
            max_iter=300,
            horizon=horizon,
            epsilon=0.1,
        ),
        create_exact_diffusion_participant(
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
            epsilon=0.1,
        ),
    ]
    start = create_exact_diffusion_start(initial_lam=10.0, data=demand, horizon=horizon)
    await start_distributed_optimization(actors, start)

    assert 0 in results and 1 in results, "Not all finish callbacks fired"
    assert np.all(results[1] >= -p_charge_max - 1e-9), f"Storage charges beyond limit: {results[1]}"
    assert np.all(results[1] <= p_discharge_max + 1e-9), (
        f"Storage discharges beyond limit: {results[1]}"
    )


@pytest.mark.asyncio
async def test_exact_diffusion_horizon_larger_than_one():
    """Multi-step horizon: λ and actor.P must have the correct length."""
    horizon = 6
    n = 2
    results: dict[int, np.ndarray] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = algo.actor.P.copy()

        return finish

    actors = [
        create_exact_diffusion_participant(
            make_finish(i),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=5.0, p_max=20.0, epsilon=0.1, n_guess=n
            ),
            max_iter=100,
            horizon=horizon,
            epsilon=0.1,
        )
        for i in range(n)
    ]
    start = create_exact_diffusion_start(
        initial_lam=10.0, data=np.full(horizon, 20.0), horizon=horizon
    )
    await start_distributed_optimization(actors, start)

    assert len(results) == n
    for idx, P in results.items():
        assert P.shape == (horizon,), f"Wrong shape for agent {idx}: {P.shape}"


# ---------------------------------------------------------------------------
# The headline claim: Exact Diffusion removes classical Diffusion's bias
# ---------------------------------------------------------------------------


def _closed_form_lambda_star(costs: list[float], actor_epsilon: float, p_target: float) -> float:
    """Unconstrained equilibrium price for LinearCostEconomicDispatchDiffusionActor.

    At equilibrium, sum_i (lam* - cost_i)/actor_epsilon == p_target, i.e. all
    per-agent gradients (P_i(lam*) - p_target/n) average to zero.
    """
    n = len(costs)
    return (p_target + sum(c / actor_epsilon for c in costs)) / (n / actor_epsilon)


@pytest.mark.asyncio
async def test_exact_diffusion_removes_classical_diffusion_bias():
    """For heterogeneous local costs and a constant step size, classical
    (adapt-then-combine) Diffusion is known to converge to a point *biased*
    away from the true minimiser of the aggregate cost (Yuan, Ling & Sayed
    2018) -- Exact Diffusion's correction stage exists specifically to
    remove that bias. This is the paper's headline claim for adopting exact
    diffusion, so it's the property worth verifying, not just that the code
    runs.
    """
    costs = [2.0, 4.0, 6.0]
    actor_epsilon = 0.1
    p_max = 200.0  # large enough that the equilibrium isn't clipped
    p_target = 300.0
    algo_epsilon = 0.1  # large enough to expose the bias, still stable
    n = len(costs)

    lam_star = _closed_form_lambda_star(costs, actor_epsilon, p_target)

    async def run(exact: bool) -> dict[int, np.ndarray]:
        results: dict[int, np.ndarray] = {}

        def make_finish(idx: int):
            def finish(algo, carrier):
                results[idx] = algo._lam.copy()

            return finish

        actors = [
            (create_exact_diffusion_participant if exact else create_diffusion_participant)(
                make_finish(i),
                diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                    cost=c, p_max=p_max, epsilon=actor_epsilon, n_guess=n
                ),
                max_iter=300,
                horizon=1,
                epsilon=algo_epsilon,
            )
            for i, c in enumerate(costs)
        ]
        start = (create_exact_diffusion_start if exact else create_diffusion_start)(
            initial_lam=10.0, data=np.array([p_target]), horizon=1
        )
        await start_distributed_optimization(actors, start)
        return results

    diffusion_results = await run(exact=False)
    exact_results = await run(exact=True)

    diffusion_err = max(abs(v[0] - lam_star) for v in diffusion_results.values())
    exact_err = max(abs(v[0] - lam_star) for v in exact_results.values())

    assert diffusion_err > 0.2, (
        f"Expected classical Diffusion to show a clear bias, got error {diffusion_err}"
    )
    assert exact_err < 0.01, (
        f"Exact Diffusion should converge to the true equilibrium, got error {exact_err}"
    )


# ---------------------------------------------------------------------------
# Correction-stage unit test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exact_diffusion_first_iteration_has_no_correction():
    """At k=0 there is no φ^{-1}, so φ̄^0 must equal the raw adapted φ^0
    (eq. 30's correction term is skipped, per the paper's stated φ^0 = λ^0
    initialisation)."""
    horizon = 1
    captured: dict[str, np.ndarray] = {}

    class RecordingCarrier:
        def others(self, _):
            return ["neighbour"]

        def send_to_other(self, content, receiver, meta=None):
            captured["phi"] = content.phi.copy()

    from distributed_resource_optimization.algorithm.diffusion.exact_diffusion import (
        ExactDiffusionAlgorithm,
    )

    algo = ExactDiffusionAlgorithm(
        finish_callback=lambda a, c: None,
        diffusion_actor=NoDiffusionActor(),
        initial_lam=7.0,
        epsilon=0.1,
        max_iter=10,
        horizon=horizon,
    )
    start = create_exact_diffusion_start(initial_lam=7.0, horizon=horizon)
    await algo.on_exchange_message(RecordingCarrier(), start, {})

    # NoDiffusionActor's gradient is 0, so φ^0 = λ^0 = 7.0 exactly, uncorrected.
    assert np.allclose(captured["phi"], [7.0])


# ---------------------------------------------------------------------------
# Regression tests for the regularity/consistency guards
# ---------------------------------------------------------------------------


class _MuteCarrier:
    """Fake carrier that reports two neighbours and swallows sends."""

    def others(self, _):
        return ["a", "b"]

    def send_to_other(self, content, receiver, meta=None):
        pass


@pytest.mark.asyncio
async def test_exact_diffusion_rejects_mismatched_neighbour_degree():
    """regular_graph_weights() assumes every neighbour shares this node's
    degree; a neighbour reporting a different degree must raise rather than
    silently produce a non-stochastic weight matrix."""
    from distributed_resource_optimization.algorithm.diffusion.diffusion import DiffusionMessage
    from distributed_resource_optimization.algorithm.diffusion.exact_diffusion import (
        ExactDiffusionAlgorithm,
    )

    algo = ExactDiffusionAlgorithm(finish_callback=lambda a, c: None, max_iter=10, horizon=1)
    start = create_exact_diffusion_start(initial_lam=1.0, horizon=1)
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
async def test_exact_diffusion_rejects_mismatched_weight_rule():
    """All participants must agree on the same weight_rule; a neighbour
    reporting a different one must raise rather than silently mix weights
    computed from two different rules."""
    from distributed_resource_optimization.algorithm.diffusion.diffusion import DiffusionMessage
    from distributed_resource_optimization.algorithm.diffusion.exact_diffusion import (
        ExactDiffusionAlgorithm,
    )

    algo = ExactDiffusionAlgorithm(
        finish_callback=lambda a, c: None, max_iter=10, horizon=1, weight_rule="mean_metropolis"
    )
    start = create_exact_diffusion_start(initial_lam=1.0, horizon=1)
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
async def test_exact_diffusion_ignores_stale_message_after_termination():
    """A late/duplicate pre-termination message arriving after finish_callback
    has already fired must not silently restart the algorithm."""
    from distributed_resource_optimization.algorithm.diffusion.diffusion import DiffusionMessage
    from distributed_resource_optimization.algorithm.diffusion.exact_diffusion import (
        ExactDiffusionAlgorithm,
    )

    finished = []
    algo = ExactDiffusionAlgorithm(
        finish_callback=lambda a, c: finished.append(a._lam.copy()), max_iter=1, horizon=1
    )
    carrier = _MuteCarrier()
    start = create_exact_diffusion_start(initial_lam=1.0, horizon=1)
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
