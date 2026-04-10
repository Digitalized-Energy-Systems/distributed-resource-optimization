"""Averaging consensus SimpleCarrier tests."""

from __future__ import annotations

import pytest
import numpy as np

from distributed_resource_optimization import (
    AveragingConsensusMessage,
    LinearCostEconomicDispatchConsensusActor,
    create_averaging_consensus_participant,
    start_distributed_optimization,
)


@pytest.mark.asyncio
async def test_averaging_consensus_with_simple_carrier():
    """Economic-dispatch consensus actors converge and trigger finish_callback.

    Three actors with the same cost function optimise a 6-element power target.
    With identical costs, the clearing price λ* = cost + ε*(p_target/N) is
    achievable by all actors without clipping, so true consensus is guaranteed.
    After max_iter iterations the finish_callback of actor_one must have fired
    and all actors must agree on the same price signal within atol=1e-3.
    """
    finished = [False]

    def on_finish(algo, carrier):
        finished[0] = True

    # Same cost so all actors reach zero gradient at the same clearing price
    actor_one = create_averaging_consensus_participant(
        on_finish,
        LinearCostEconomicDispatchConsensusActor(cost=10, p_max=100, n_guess=3),
        max_iter=100,
    )
    actor_two = create_averaging_consensus_participant(
        lambda *_: None,
        LinearCostEconomicDispatchConsensusActor(cost=10, p_max=100, n_guess=3),
        max_iter=100,
    )
    actor_three = create_averaging_consensus_participant(
        lambda *_: None,
        LinearCostEconomicDispatchConsensusActor(cost=10, p_max=100, n_guess=3),
        max_iter=100,
    )

    p_target = [10, 30, 40, 45, 60, 10]
    initial_message = AveragingConsensusMessage(
        lam=np.ones(len(p_target)) * 10,
        k=0,
        data=p_target,
    )

    await start_distributed_optimization(
        [actor_one, actor_two, actor_three], initial_message
    )

    assert finished[0]
    assert np.allclose(actor_one._lam, actor_two._lam, atol=1e-3)
    assert np.allclose(actor_one._lam, actor_three._lam, atol=1e-3)
