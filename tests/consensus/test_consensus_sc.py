"""Averaging consensus SimpleCarrier tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    AveragingConsensusMessage,
    ConsensusActor,
    create_averaging_consensus_participant,
    start_distributed_optimization,
)


class QuadraticActor(ConsensusActor):
    """Local cost a/2·x² − λx, so x(λ) = λ/a; pushes λ until x equals its share of *data*."""

    def __init__(self, a=2.0, n=3, rho=0.05):
        self.a, self.n, self.rho = a, n, rho

    def gradient_term(self, lam, data):
        return -self.rho * (lam / self.a - np.asarray(data) / self.n)


@pytest.mark.asyncio
async def test_averaging_consensus_with_simple_carrier():
    """Gradient-carrying actors agree on λ, reach λ* = a·data/n and trigger finish_callback."""
    finished = [False]

    def on_finish(algo, carrier):
        finished[0] = True

    callbacks = [on_finish, lambda *_: None, lambda *_: None]
    actors = [
        create_averaging_consensus_participant(cb, QuadraticActor(), max_iter=300)
        for cb in callbacks
    ]

    target = np.array([10, 30, 40, 45, 60, 10], dtype=float)
    initial_message = AveragingConsensusMessage(lam=np.ones(len(target)) * 10, k=0, data=target)

    await start_distributed_optimization(actors, initial_message)

    assert finished[0]
    lams = [actor._lam for actor in actors]
    assert np.allclose(lams[0], lams[1], atol=1e-3)
    assert np.allclose(lams[0], lams[2], atol=1e-3)
    # identical actors agree trivially; the gradient must also drive λ to the optimum
    assert np.allclose(lams[0], 2.0 * target / 3, rtol=1e-2)
