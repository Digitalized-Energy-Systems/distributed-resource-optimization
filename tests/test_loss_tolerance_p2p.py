"""Loss tolerance of the peer-to-peer algorithms (round deadline + last values).

Runs on :class:`SimpleCarrier` with loss injection; the round deadline is in
real seconds there, so timeouts are kept small.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from distributed_resource_optimization import (
    LinearCostEconomicDispatchDiffusionActor,
    create_averaging_consensus_participant,
    create_averaging_consensus_start,
    create_diffusion_participant,
    create_diffusion_start,
    create_exact_diffusion_participant,
    create_exact_diffusion_start,
)
from distributed_resource_optimization.algorithm.firstorder.consensus.economic_dispatch import (
    LinearCostEconomicDispatchConsensusActor,
)
from distributed_resource_optimization.carrier.simple import (
    ActorContainer,
    start_distributed_optimization,
)

ROUND_TIMEOUT = 0.005


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


async def _run_consensus(loss_rate=0.0, seed=0, round_timeout=ROUND_TIMEOUT):
    """Leader-follower merit order (see tests/consensus/test_averaging.py)."""
    demand = 80.0
    specs = [(10.0, 50.0), (20.0, 30.0), (30.0, 30.0), (40.0, 30.0)]
    results: dict[int, tuple] = {}

    def make_finish(idx):
        def finish(algo, carrier):
            results[idx] = (algo._lam.copy(), float(algo.actor.P[0]), algo._k)

        return finish

    actors = [
        create_averaging_consensus_participant(
            make_finish(i),
            consensus_actor=LinearCostEconomicDispatchConsensusActor(
                cost=cost, p_max=p_max, epsilon=0.1
            ),
            initial_lam=10.0,
            alpha=0.3,
            max_iter=300,
            is_leader=(i == 0),
            leader_gain=0.05,
            round_timeout=round_timeout,
        )
        for i, (cost, p_max) in enumerate(specs)
    ]
    container = ActorContainer(loss_rate=loss_rate, rng=random.Random(seed))
    await start_distributed_optimization(
        actors, create_averaging_consensus_start(10.0, data=demand), container=container
    )
    return results, container


def _diffusion_actors(create, epsilon, round_timeout, results):
    def make_finish(idx):
        def finish(algo, carrier):
            results[idx] = algo.actor.P.copy()

        return finish

    return [
        create(
            make_finish(i),
            diffusion_actor=LinearCostEconomicDispatchDiffusionActor(
                cost=cost, p_max=20.0, epsilon=0.1, n_guess=2
            ),
            max_iter=300,
            epsilon=epsilon,
            round_timeout=round_timeout,
        )
        for i, cost in enumerate([2.0, 8.0])
    ]


async def _run_diffusion(exact, loss_rate=0.0, seed=0, round_timeout=ROUND_TIMEOUT):
    """Two-unit merit order (see tests/diffusion/test_*diffusion.py)."""
    results: dict[int, np.ndarray] = {}
    if exact:
        actors = _diffusion_actors(create_exact_diffusion_participant, 0.1, round_timeout, results)
        start = create_exact_diffusion_start(10.0, data=np.array([20.0, 20.0]), horizon=2)
    else:
        actors = _diffusion_actors(create_diffusion_participant, 0.2, round_timeout, results)
        start = create_diffusion_start(10.0, data=np.array([20.0, 20.0]), horizon=2)
    container = ActorContainer(loss_rate=loss_rate, rng=random.Random(seed))
    await start_distributed_optimization(actors, start, container=container)
    return results, container


# ---------------------------------------------------------------------------
# Lossless: a round deadline must not change anything
# ---------------------------------------------------------------------------


class TestLosslessUnchanged:
    async def test_consensus(self):
        with_timeout, _ = await _run_consensus(round_timeout=1.0)
        without, _ = await _run_consensus(round_timeout=None)
        assert with_timeout.keys() == without.keys()
        for i in without:
            assert np.array_equal(with_timeout[i][0], without[i][0])
            assert with_timeout[i][1] == without[i][1]

    @pytest.mark.parametrize("exact", [False, True])
    async def test_diffusion(self, exact):
        with_timeout, _ = await _run_diffusion(exact, round_timeout=1.0)
        without, _ = await _run_diffusion(exact, round_timeout=None)
        assert with_timeout.keys() == without.keys()
        for i in without:
            assert np.array_equal(with_timeout[i], without[i])


# ---------------------------------------------------------------------------
# Under loss: every participant finishes, results stay close
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss_rate", [0.05, 0.2])
@pytest.mark.parametrize("seed", [1, 2])
class TestUnderLoss:
    async def test_consensus_balances_and_keeps_merit_order(self, loss_rate, seed):
        results, container = await _run_consensus(loss_rate=loss_rate, seed=seed)

        assert container.dropped > 0
        assert sorted(results) == [0, 1, 2, 3]
        powers = [results[i][1] for i in range(4)]
        assert sum(powers) == pytest.approx(80.0, abs=2.0)
        assert powers[0] == pytest.approx(50.0, abs=2.0)
        assert powers[1] == pytest.approx(30.0, abs=2.0)
        assert powers[2] == pytest.approx(0.0, abs=2.0)
        assert powers[3] == pytest.approx(0.0, abs=2.0)

    @pytest.mark.parametrize("exact", [False, True])
    async def test_diffusion_close_to_lossless(self, loss_rate, seed, exact):
        reference, _ = await _run_diffusion(exact)
        results, container = await _run_diffusion(exact, loss_rate=loss_rate, seed=seed)

        assert container.dropped > 0
        assert sorted(results) == [0, 1]
        for i in reference:
            assert results[i] == pytest.approx(reference[i], abs=0.5)


async def test_participant_finishes_when_every_message_is_lost():
    """With nothing arriving, deadlines alone carry a participant to max_iter.

    Only the kicked-off participant ever starts (every other message is lost).
    """
    results, container = await _run_consensus(loss_rate=1.0)
    assert sorted(results) == [1]
    assert results[1][2] == 300


# ---------------------------------------------------------------------------
# FDGDM and DEED-ADMM: sums that must survive loss (running-sum flows)
# ---------------------------------------------------------------------------


async def _run_fdgdm(loss_rate=0.0, seed=0, round_timeout=ROUND_TIMEOUT, max_iter=300):
    """Three generators, interior optimum, total 150 MW (Σ P is conserved)."""
    from distributed_resource_optimization import (
        LinearCostEconomicDispatchFDGDMActor,
        create_fdgdm_participant,
        create_fdgdm_start,
    )

    specs = [(2.0, 100.0, 60.0), (5.0, 100.0, 50.0), (8.0, 100.0, 40.0)]  # cost, p_max, P0
    results: dict[int, np.ndarray] = {}

    def make_finish(idx):
        def finish(algo, carrier):
            results[idx] = algo._P.copy()

        return finish

    actors = [
        create_fdgdm_participant(
            make_finish(i),
            fdgdm_actor=LinearCostEconomicDispatchFDGDMActor(
                cost=cost, p_max=p_max, epsilon=0.1, initial_schedule=np.array([p0])
            ),
            max_iter=max_iter,
            round_timeout=round_timeout,
        )
        for i, (cost, p_max, p0) in enumerate(specs)
    ]
    container = ActorContainer(loss_rate=loss_rate, rng=random.Random(seed))
    await start_distributed_optimization(
        actors, create_fdgdm_start(data=np.zeros(1)), container=container
    )
    return results, container


async def _run_deed(loss_rate=0.0, seed=0, round_timeout=ROUND_TIMEOUT, max_iter=300, algos=None):
    """Four thermal hubs, 4-step horizon (see tests/deed_admm)."""
    from distributed_resource_optimization import create_deed_admm_thermal_participant
    from distributed_resource_optimization.algorithm.deed_admm.deed_admm import DEEDADMMMessage

    demand = np.array([100.0, 150.0, 120.0, 80.0])
    n = 4
    results: dict[int, np.ndarray] = {}

    def make_finish(idx):
        def finish(algo, carrier):
            results[idx] = algo.P.copy()
            if algos is not None:
                algos[idx] = algo

        return finish

    actors = [
        create_deed_admm_thermal_participant(
            make_finish(i),
            p_min=0.0,
            p_max=60.0,
            marginal_cost=c,
            cost_quad=0.01,
            d_i=demand / n,
            gamma=0.05,
            max_iter=max_iter,
            n_agents=n,
            round_timeout=round_timeout,
        )
        for i, c in enumerate([1.0, 2.0, 3.0, 4.0])
    ]
    container = ActorContainer(loss_rate=loss_rate, rng=random.Random(seed))
    start = DEEDADMMMessage(lam=np.zeros(4), xi=np.zeros(4), k=0, data=None, initial=True)
    await start_distributed_optimization(actors, start, container=container)
    return results, container, demand


class TestRunningSums:
    async def test_fdgdm_lossless_unchanged(self):
        with_timeout, _ = await _run_fdgdm(round_timeout=1.0)
        without, _ = await _run_fdgdm(round_timeout=None)
        for i in without:
            assert np.array_equal(with_timeout[i], without[i])

    async def test_deed_lossless_unchanged(self):
        # Without a deadline DEED-ADMM mixes ξ as in the paper; with one it
        # uses the running-sum flows. Algebraically equal, so a deadline that
        # never fires may only change rounding.
        with_timeout, _, _ = await _run_deed(round_timeout=1.0)
        without, _, _ = await _run_deed(round_timeout=None)
        for i in without:
            np.testing.assert_allclose(with_timeout[i], without[i], rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("loss_rate", [0.05, 0.2])
    @pytest.mark.parametrize("seed", [1, 2])
    async def test_fdgdm_conserves_total_power(self, loss_rate, seed):
        reference, _ = await _run_fdgdm()
        results, container = await _run_fdgdm(loss_rate=loss_rate, seed=seed)

        assert container.dropped > 0
        assert sorted(results) == [0, 1, 2]
        assert sum(results.values())[0] == pytest.approx(150.0, abs=1e-9)
        for i in reference:
            assert results[i] == pytest.approx(reference[i], abs=1.0)

    @pytest.mark.parametrize("loss_rate", [0.05, 0.2])
    @pytest.mark.parametrize("seed", [1, 2])
    async def test_deed_balances_demand(self, loss_rate, seed):
        reference, _, demand = await _run_deed()
        results, container, _ = await _run_deed(loss_rate=loss_rate, seed=seed)

        assert container.dropped > 0
        assert sorted(results) == [0, 1, 2, 3]
        np.testing.assert_allclose(sum(results.values()), demand, atol=1.0)
        for i in reference:
            np.testing.assert_allclose(results[i], reference[i], atol=2.0)

    # End-of-run settlement (FlowSettlement): a run cut off by max_iter before
    # it converged still has large flows in its last rounds; without settling,
    # a lost final running sum leaves the conserved sum off for good.
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
    async def test_fdgdm_cut_off_run_conserves_total_power(self, seed):
        results, container = await _run_fdgdm(
            loss_rate=0.2, seed=seed, round_timeout=0.05, max_iter=15
        )
        assert container.dropped > 0
        assert sum(results.values())[0] == pytest.approx(150.0, abs=1e-9)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
    async def test_deed_cut_off_run_conserves_xi(self, seed):
        algos: dict = {}
        _, container, _ = await _run_deed(
            loss_rate=0.2, seed=seed, round_timeout=0.05, max_iter=15, algos=algos
        )
        assert container.dropped > 0
        xi = sum(a._xi for a in algos.values())
        x_minus_y = sum(a._x - a._y for a in algos.values())
        np.testing.assert_allclose(xi, x_minus_y, atol=1e-9)
