"""Averaging consensus tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    create_averaging_consensus_participant,
    create_averaging_consensus_start,
    start_distributed_optimization,
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
