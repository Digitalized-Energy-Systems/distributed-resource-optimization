"""Averaging consensus tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    create_averaging_consensus_participant,
    create_averaging_consensus_start,
    start_distributed_optimization,
)
from distributed_resource_optimization.algorithm.consensus.economic_dispatch import (
    LinearCostEconomicDispatchConsensusActor,
)


@pytest.mark.asyncio
async def test_averaging_consensus_converges():
    """All participants should reach the same λ value after consensus."""
    results: list = []

    def finish(algo, carrier):
        results.append(algo._lam.copy())

    actors = [
        create_averaging_consensus_participant(finish, initial_lam=v, max_iter=30)
        for v in [1.0, 5.0, 10.0]
    ]
    start = create_averaging_consensus_start(1.0, data=None)
    await start_distributed_optimization(actors, start)

    # All participants should have converged to similar values
    assert len(results) > 0
    for r in results[1:]:
        assert np.allclose(results[0], r, atol=0.5)


@pytest.mark.asyncio
async def test_averaging_consensus_leader_follower_merit_order():
    """End-to-end leader-follower economic dispatch (Jian et al. 2020, eqs. 20-23).

    One leader + three followers, each an economic-dispatch actor with a
    distinct marginal cost and capacity. The leader's ΔP-pinning term (eq. 22)
    is the core mechanism of the paper — this is the first test that actually
    exercises it, rather than only pure averaging (no leader, no ΔP) or the
    price→power clip in isolation (no consensus loop at all).

    Demand=80 with capacities/costs [(leader: 50 MW @ 10), (30 MW @ 20),
    (30 MW @ 30), (30 MW @ 40)] has a unique merit-order optimum: the two
    cheapest units run at full capacity (50 + 30 = 80) and the two most
    expensive stay idle.
    """
    demand = 80.0
    # (cost, p_max); index 0 is the leader.
    specs = [(10.0, 50.0), (20.0, 30.0), (30.0, 30.0), (40.0, 30.0)]

    results: dict[int, tuple] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = (algo._lam.copy(), algo.actor.P.copy())

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
        )
        for i, (cost, p_max) in enumerate(specs)
    ]
    start = create_averaging_consensus_start(10.0, data=demand)
    await start_distributed_optimization(actors, start)

    assert len(results) == len(specs)

    # All participants converge to the same clearing price λ*.
    lams = [results[i][0] for i in range(len(specs))]
    for lam in lams[1:]:
        assert np.allclose(lams[0], lam, atol=0.5)

    powers = [float(results[i][1][0]) for i in range(len(specs))]

    # Power balance: total dispatch must reach demand (leader's ΔP pinning).
    assert sum(powers) == pytest.approx(demand, abs=1.0)

    # Merit order: the two cheapest units (leader, follower 1) saturate their
    # capacity; the two most expensive (followers 2, 3) stay idle.
    assert powers[0] == pytest.approx(50.0, abs=1.0)
    assert powers[1] == pytest.approx(30.0, abs=1.0)
    assert powers[2] == pytest.approx(0.0, abs=1.0)
    assert powers[3] == pytest.approx(0.0, abs=1.0)
