"""Convergence safeguards from Ces et al. 2025 for diffusion / exact diffusion.

Covers the two safeguards the algorithms implement:

* tol-based termination — the run stops in the first round where every
  participant's λ changed by at most ``tol`` (paper: 1e-4 on the incremental
  cost), with ``max_iter`` demoted to a failsafe that reports
  ``converged=False``.
* stability-scaled gradient step — on GW-scale systems with capacity-scaled
  actor bands, the step ``ε = 0.5·n·band/Σp_nom`` (a quarter of the
  ``2n/Σ(p_nom/band)`` stability bound, mirroring the scenario code) converges
  to a power-balanced dispatch where the historical fixed ``ε = 0.2``
  oscillates without ever balancing.
"""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    LinearCostEconomicDispatchDiffusionActor,
    create_diffusion_participant,
    create_diffusion_start,
    create_exact_diffusion_participant,
    create_exact_diffusion_start,
    start_distributed_optimization,
)


def _gw_scale_case(n: int = 12):
    """A PyPSA-Eur-like case: GW capacities, capacity-scaled response bands.

    Returns ``(actors_kwargs, demand, band, total_p_nom)`` where each entry of
    *actors_kwargs* parametrises one LinearCostEconomicDispatchDiffusionActor
    the way the benchmark scenarios do (band = 10% of the cost spread, actor
    epsilon = band/p_nom).
    """
    rng = np.random.default_rng(42)
    p_noms = rng.uniform(2_000.0, 25_000.0, size=n)  # MW
    costs = np.linspace(0.0, 65.0, n)  # €/MWh
    band = max(0.1, 0.1 * (costs.max() - costs.min()))
    demand = 0.6 * p_noms.sum()
    actors_kwargs = [
        dict(cost=float(c), p_max=float(p), epsilon=band / p, n_guess=n)
        for c, p in zip(costs, p_noms)
    ]
    return actors_kwargs, demand, band, float(p_noms.sum())


# ---------------------------------------------------------------------------
# tol-based termination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diffusion_tol_terminates_before_max_iter():
    n, horizon = 3, 2
    finished: dict[int, object] = {}

    def make_finish(idx):
        def finish(algo, carrier):
            finished[idx] = algo

        return finish

    demand = np.full(horizon, 30.0)
    participants = [
        create_diffusion_participant(
            make_finish(i),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=5.0, p_max=30.0, epsilon=0.1, n_guess=n
            ),
            max_iter=1000,
            epsilon=0.02,
            tol=1e-4,
            horizon=horizon,
        )
        for i in range(n)
    ]
    start = create_diffusion_start(initial_lam=10.0, data=demand, horizon=horizon)
    await start_distributed_optimization(participants, start)

    assert len(finished) == n, "Not all finish callbacks fired"
    for algo in finished.values():
        assert algo.converged is True
        assert 0 < algo.iterations < 1000


@pytest.mark.asyncio
async def test_diffusion_max_iter_failsafe_reports_not_converged():
    """With an unreachable tol the failsafe fires and flags the run."""
    n, horizon, max_iter = 3, 1, 25
    finished: dict[int, object] = {}

    def make_finish(idx):
        def finish(algo, carrier):
            finished[idx] = algo

        return finish

    demand = np.full(horizon, 30.0)
    participants = [
        create_diffusion_participant(
            make_finish(i),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=5.0, p_max=30.0, epsilon=0.1, n_guess=n
            ),
            max_iter=max_iter,
            epsilon=0.02,
            tol=0.0,  # a moving λ can never satisfy this
            horizon=horizon,
        )
        for i in range(n)
    ]
    start = create_diffusion_start(initial_lam=10.0, data=demand, horizon=horizon)
    await start_distributed_optimization(participants, start)

    assert len(finished) == n, "Not all finish callbacks fired"
    for algo in finished.values():
        assert algo.converged is False
        assert algo.iterations == max_iter


@pytest.mark.asyncio
async def test_exact_diffusion_tol_terminates_before_max_iter():
    n, horizon = 3, 2
    finished: dict[int, object] = {}

    def make_finish(idx):
        def finish(algo, carrier):
            finished[idx] = algo

        return finish

    demand = np.full(horizon, 30.0)
    participants = [
        create_exact_diffusion_participant(
            make_finish(i),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=5.0, p_max=30.0, epsilon=0.1, n_guess=n
            ),
            max_iter=1000,
            epsilon=0.02,
            tol=1e-4,
            horizon=horizon,
        )
        for i in range(n)
    ]
    start = create_exact_diffusion_start(initial_lam=10.0, data=demand, horizon=horizon)
    await start_distributed_optimization(participants, start)

    assert len(finished) == n, "Not all finish callbacks fired"
    for algo in finished.values():
        assert algo.converged is True
        assert 0 < algo.iterations < 1000


@pytest.mark.asyncio
async def test_exact_diffusion_max_iter_failsafe_reports_not_converged():
    """With an unreachable tol the failsafe fires and flags the run."""
    n, horizon, max_iter = 3, 1, 25
    finished: dict[int, object] = {}

    def make_finish(idx):
        def finish(algo, carrier):
            finished[idx] = algo

        return finish

    demand = np.full(horizon, 30.0)
    participants = [
        create_exact_diffusion_participant(
            make_finish(i),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=5.0, p_max=30.0, epsilon=0.1, n_guess=n
            ),
            max_iter=max_iter,
            epsilon=0.02,
            tol=0.0,  # a moving λ can never satisfy this
            horizon=horizon,
        )
        for i in range(n)
    ]
    start = create_exact_diffusion_start(initial_lam=10.0, data=demand, horizon=horizon)
    await start_distributed_optimization(participants, start)

    assert len(finished) == n, "Not all finish callbacks fired"
    for algo in finished.values():
        assert algo.converged is False
        assert algo.iterations == max_iter


# ---------------------------------------------------------------------------
# stability-scaled gradient step on a GW-scale system
# ---------------------------------------------------------------------------


async def _run_gw_case(create_participant, create_start, epsilon, **participant_kwargs):
    actors_kwargs, demand, _, _ = _gw_scale_case()
    horizon = 1
    finished: dict[int, object] = {}

    def make_finish(idx):
        def finish(algo, carrier):
            finished[idx] = algo

        return finish

    participants = [
        create_participant(
            make_finish(i),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(**kwargs),
            max_iter=1000,
            epsilon=epsilon,
            tol=1e-4,
            horizon=horizon,
            **participant_kwargs,
        )
        for i, kwargs in enumerate(actors_kwargs)
    ]
    start = create_start(initial_lam=10.0, data=np.full(horizon, demand), horizon=horizon)
    await start_distributed_optimization(participants, start)

    assert len(finished) == len(actors_kwargs), "Not all finish callbacks fired"
    total_p = sum(float(np.asarray(algo.actor.P).sum()) for algo in finished.values())
    return finished, total_p, demand


@pytest.mark.asyncio
async def test_diffusion_gw_scale_balances_with_scaled_step():
    actors_kwargs, _, band, total_p_nom = _gw_scale_case()
    grad_step = 0.5 * len(actors_kwargs) * band / total_p_nom

    finished, total_p, demand = await _run_gw_case(
        create_diffusion_participant, create_diffusion_start, grad_step
    )
    assert all(algo.converged for algo in finished.values())
    assert abs(total_p - demand) / demand < 0.01, (
        f"Dispatch misses demand by {100 * abs(total_p - demand) / demand:.1f}%"
    )


@pytest.mark.asyncio
async def test_diffusion_gw_scale_fixed_step_does_not_balance():
    """Regression guard: the historical fixed ε=0.2 oscillates on GW scale."""
    finished, total_p, demand = await _run_gw_case(
        create_diffusion_participant, create_diffusion_start, 0.2
    )
    assert not any(algo.converged for algo in finished.values())
    assert abs(total_p - demand) / demand > 0.01


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "weight_rule", ["averaging", "relative_degree", "mean_metropolis", "hastings"]
)
async def test_exact_diffusion_gw_scale_balances_with_scaled_step(weight_rule):
    actors_kwargs, _, band, total_p_nom = _gw_scale_case()
    grad_step = 0.5 * len(actors_kwargs) * band / total_p_nom

    finished, total_p, demand = await _run_gw_case(
        create_exact_diffusion_participant,
        create_exact_diffusion_start,
        grad_step,
        weight_rule=weight_rule,
    )
    assert all(algo.converged for algo in finished.values())
    assert abs(total_p - demand) / demand < 0.01, (
        f"[{weight_rule}] dispatch misses demand by {100 * abs(total_p - demand) / demand:.1f}%"
    )
