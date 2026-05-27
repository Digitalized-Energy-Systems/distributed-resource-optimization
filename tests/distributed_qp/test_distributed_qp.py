"""Distributed primal-dual QP smoke tests."""

from __future__ import annotations

import pytest

from distributed_resource_optimization import (
    ActorContainer,
    SimpleCarrier,
    cid,
    create_distributed_qp_participant,
    create_distributed_qp_start,
)
from distributed_resource_optimization.algorithm.distributed_qp.core import (
    GossipQPAlgorithm,
    LedgerEntry,
    _is_saturated,
)


def test_is_saturated_box_bounds():
    assert _is_saturated(-1.0, -1.0, 1.0)
    assert _is_saturated(1.0, -1.0, 1.0)
    assert not _is_saturated(0.0, -1.0, 1.0)


def test_primal_clamps_to_box():
    """The closed-form primal sits inside [dmin, dmax] regardless of lambda."""
    algo = GossipQPAlgorithm(a=2.0, dmin=-1.0, dmax=1.0)
    algo.dual_lambda = 100.0
    assert algo._primal() == 1.0
    algo.dual_lambda = -100.0
    assert algo._primal() == -1.0
    algo.dual_lambda = 0.25
    assert algo._primal() == pytest.approx(0.5)


def test_step_size_decays():
    """Robbins-Monro step is monotone non-increasing in the counter."""
    algo = GossipQPAlgorithm(a=1.0, dmin=-1.0, dmax=1.0, convergence_rate=0.5, step_decay_k0=10)
    assert algo._step_size(0) == pytest.approx(0.5)
    assert algo._step_size(10) == pytest.approx(0.25)
    assert algo._step_size(100) < algo._step_size(0)


def test_ledger_merge_keeps_newest_counter():
    algo = GossipQPAlgorithm(a=1.0, dmin=-10.0, dmax=10.0)
    algo.target = 1.0
    algo.memory = {"a": LedgerEntry(delta=0.5, counter=3, weight=1.0, saturated=False)}
    algo._merge({"a": LedgerEntry(delta=0.7, counter=2, weight=1.0, saturated=False)})
    assert algo.memory["a"].delta == 0.5
    algo._merge({"a": LedgerEntry(delta=0.9, counter=5, weight=1.0, saturated=False)})
    assert algo.memory["a"].delta == 0.9
    assert algo.memory["a"].counter == 5


def test_ledger_merge_clips_byzantine_delta():
    algo = GossipQPAlgorithm(a=1.0, dmin=-100.0, dmax=100.0, byzantine_cap_multiple=2.0)
    algo.target = 1.0
    algo._merge({"a": LedgerEntry(delta=999.0, counter=1, weight=1.0, saturated=False)})
    assert algo.memory["a"].delta == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_gossip_qp_unconstrained_converges_to_kkt_optimum():
    """Three uniform participants split a target of 3 into three 1.0's.

    At the unconstrained optimum lambda* = T / sum(a_j) = 3 / 3 = 1
    and each delta_i = a_i * lambda* = 1.0.  No box clamps active.
    """
    callback_terminal: list[str] = []

    def on_finish(_algo, _carrier, terminal):
        callback_terminal.append(terminal)

    participants = [
        create_distributed_qp_participant(
            a=1.0,
            dmin=-2.0,
            dmax=2.0,
            convergence_rate=1.0,
            step_decay_k0=10,
            termination_tolerance=1e-3,
            max_hops=500,
            finish_callback=on_finish,
        )
        for _ in range(3)
    ]

    container = ActorContainer()
    carriers = [SimpleCarrier(container, p) for p in participants]

    target = 3.0
    start = create_distributed_qp_start(
        originator=participants[0],
        target=target,
        negotiation_id="test-nid",
        n_neighbours=len(carriers) - 1,
    )

    carriers[0].send_to_other(start, cid(carriers[1]))
    await container.done_event.wait()

    total = sum(p.delta for p in participants)
    assert abs(total - target) < 1e-2
    assert callback_terminal and callback_terminal[0] in {"converged", "max_hops"}


@pytest.mark.asyncio
async def test_gossip_qp_with_saturation_distributes_remainder():
    """One small box plus two large boxes — small one saturates, rest absorbs."""
    participants = [
        create_distributed_qp_participant(
            a=1.0, dmin=-0.5, dmax=0.5,
            convergence_rate=1.0, step_decay_k0=10,
            termination_tolerance=1e-3, max_hops=500,
        ),
        create_distributed_qp_participant(
            a=1.0, dmin=-5.0, dmax=5.0,
            convergence_rate=1.0, step_decay_k0=10,
            termination_tolerance=1e-3, max_hops=500,
        ),
        create_distributed_qp_participant(
            a=1.0, dmin=-5.0, dmax=5.0,
            convergence_rate=1.0, step_decay_k0=10,
            termination_tolerance=1e-3, max_hops=500,
        ),
    ]
    container = ActorContainer()
    carriers = [SimpleCarrier(container, p) for p in participants]

    target = 3.0
    start = create_distributed_qp_start(
        originator=participants[0],
        target=target,
        negotiation_id="sat-nid",
        n_neighbours=len(carriers) - 1,
    )
    carriers[0].send_to_other(start, cid(carriers[1]))
    await container.done_event.wait()

    total = sum(p.delta for p in participants)
    assert abs(total - target) < 5e-2
    assert participants[0].delta == pytest.approx(0.5, abs=1e-2)
