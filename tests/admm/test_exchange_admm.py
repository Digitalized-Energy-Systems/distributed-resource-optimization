"""Exchange ADMM (Boyd et al. 2011 §7.3.2) against closed-form optima."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    create_exchange_admm_coordinator,
    start_coordinated_optimization,
)
from distributed_resource_optimization.algorithm.admm.core import (
    ADMMAnswer,
    ADMMMessage,
    create_admm_start,
)
from distributed_resource_optimization.algorithm.core import DistributedAlgorithm

A = np.array([1.0, 2.0, 4.0])  # f_i(x) = a_i/2 ||x||² + b_iᵀx
B = np.array([[-3.0, 1.0], [2.0, -1.0], [-1.0, 0.5]])  # element 0: surplus at price 0


class Quadratic(DistributedAlgorithm):
    """Closed-form x-update: argmin a/2||x||² + bᵀx + ρ/2||x + v||²; sends no reply in the
    rounds *drop* (a lost reply)."""

    def __init__(self, a: float, b: np.ndarray, drop: tuple = ()) -> None:
        self.a, self.b, self.drop, self.x = a, b, drop, None

    async def on_exchange_message(self, carrier, message_data, meta) -> None:
        if not isinstance(message_data, ADMMMessage):
            return  # the coordinator's end of the run
        self.x = -(self.b + message_data.rho * message_data.v) / (self.a + message_data.rho)
        if message_data.k in self.drop:
            return
        carrier.reply_to_other(ADMMAnswer(x=self.x), meta)


def optimum(surplus: bool) -> np.ndarray:
    """x_i = -(y + b_i)/a_i with the price y that clears Σx = 0 (y ≤ 0 if a surplus may stay: x_i grows as y falls)."""
    y = -(B / A[:, None]).sum(axis=0) / (1 / A).sum()
    if surplus:
        y = np.minimum(y, 0.0)
    return -(y + B) / A[:, None]


def actors(drop: tuple = (), everyone: bool = False) -> list[Quadratic]:
    """*drop*: the rounds in which participant 1 (or *everyone*) sends no reply."""
    return [
        Quadratic(a, b, drop if everyone or i == 1 else ()) for i, (a, b) in enumerate(zip(A, B))
    ]


@pytest.mark.parametrize("surplus", [False, True])
@pytest.mark.parametrize("mu", [None, 10.0])
async def test_exchange_admm_reaches_closed_form_optimum(surplus, mu):
    parts = actors()
    coordinator = create_exchange_admm_coordinator(
        surplus, abs_tol=1e-6, rel_tol=1e-6, max_iters=5000, mu=mu
    )
    await start_coordinated_optimization(parts, coordinator, create_admm_start(None, 2))

    x = np.array([p.x for p in parts])
    assert np.allclose(x, optimum(surplus), atol=1e-3)
    assert coordinator.converged and coordinator.iterations < 5000  # eq. 3.12 stopped it
    if surplus:
        assert x[:, 0].sum() > 1.0 and abs(x[:, 1].sum()) < 1e-3  # surplus only in element 0


async def test_first_round_without_any_reply_is_not_convergence():
    # flat start: with no reply x, z, u stay 0 and both residuals are 0 (review 2026-11-16)
    parts = actors(drop=(1,), everyone=True)
    coordinator = create_exchange_admm_coordinator(
        abs_tol=1e-6, rel_tol=1e-6, max_iters=5000, round_timeout=0.05
    )
    await start_coordinated_optimization(parts, coordinator, create_admm_start(None, 2))

    assert coordinator.missing_replies == 3 and coordinator.iterations > 1
    assert np.allclose([p.x for p in parts], optimum(True), atol=1e-3)


async def test_lost_reply_does_not_block():
    parts = actors(drop=(1,))
    coordinator = create_exchange_admm_coordinator(
        abs_tol=1e-6, rel_tol=1e-6, max_iters=5000, round_timeout=0.05
    )
    await start_coordinated_optimization(parts, coordinator, create_admm_start(None, 2))

    assert coordinator.missing_replies == 1
    assert np.allclose([p.x for p in parts], optimum(True), atol=1e-3)
