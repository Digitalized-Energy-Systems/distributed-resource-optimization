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
from distributed_resource_optimization.algorithm.consensus.economic_dispatch import (
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
