"""Averaging consensus SimpleCarrier tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    AveragingConsensusMessage,
    LinearCostEconomicDispatchConsensusActor,
    create_averaging_consensus_participant,
    start_distributed_optimization,
)


@pytest.mark.asyncio
async def test_averaging_consensus_with_simple_carrier():
    """Leader-follower economic-dispatch consensus actors converge (eqs. 20-23).

    Three actors with the same cost function optimise a 6-element power target.
    One actor is the leader (pinning λ toward the real power imbalance ΔP);
    the other two are followers doing pure averaging. With identical costs,
    the price agreed on by all actors should equalise their identical
    dispatch, so all actors converge to the same λ.
    """
    finished = [False]

    def on_finish(algo, carrier):
        finished[0] = True

    actor_one = create_averaging_consensus_participant(
        on_finish,
        LinearCostEconomicDispatchConsensusActor(cost=10, p_max=100),
        max_iter=100,
        is_leader=True,
        leader_gain=0.02,
    )
    actor_two = create_averaging_consensus_participant(
        lambda *_: None,
        LinearCostEconomicDispatchConsensusActor(cost=10, p_max=100),
        max_iter=100,
    )
    actor_three = create_averaging_consensus_participant(
        lambda *_: None,
        LinearCostEconomicDispatchConsensusActor(cost=10, p_max=100),
        max_iter=100,
    )

    p_target = [10, 30, 40, 45, 60, 10]
    initial_message = AveragingConsensusMessage(
        lam=np.ones(len(p_target)) * 10,
        k=0,
        data=p_target,
    )

    await start_distributed_optimization([actor_one, actor_two, actor_three], initial_message)

    assert finished[0]
    assert np.allclose(actor_one._lam, actor_two._lam, atol=1e-3)
    assert np.allclose(actor_one._lam, actor_three._lam, atol=1e-3)


@pytest.mark.asyncio
async def test_merit_order_dispatch_with_heterogeneous_costs():
    """The cheaper generator must dispatch more power (eq. 21's optimality condition).

    One leader (cheap, cost=5) and one follower (pricier, cost=10) share a
    single λ via consensus. At convergence, Σ Pi should balance the target
    demand, and the cheaper unit must produce strictly more than the pricier
    one — the equal-marginal-cost condition, not an equal power split.
    """
    schedules: dict[str, np.ndarray] = {}

    def make_finish(name):
        def finish(algo, carrier):
            schedules[name] = algo.actor.P.copy()

        return finish

    cheap = create_averaging_consensus_participant(
        make_finish("cheap"),
        LinearCostEconomicDispatchConsensusActor(cost=5.0, p_max=50.0, epsilon=1.0),
        alpha=0.3,
        max_iter=300,
        is_leader=True,
        leader_gain=0.05,
    )
    pricier = create_averaging_consensus_participant(
        make_finish("pricier"),
        LinearCostEconomicDispatchConsensusActor(cost=10.0, p_max=50.0, epsilon=1.0),
        alpha=0.3,
        max_iter=300,
    )

    target = 20.0
    initial_message = AveragingConsensusMessage(lam=np.array([10.0]), k=0, data=np.array([target]))

    await start_distributed_optimization([cheap, pricier], initial_message)

    assert schedules["cheap"][0] > schedules["pricier"][0]
    assert np.isclose(schedules["cheap"][0] + schedules["pricier"][0], target, atol=1.0)
