"""Loss tolerance of the carriers: timed requests, orphaned replies, ADMM under loss."""

from __future__ import annotations

import random

import numpy as np
import pytest
from mango import agent_composed_of, auto_assign, complete_topology, create_world
from mango.simulation.communication import SimpleCommunicationSimulation

from distributed_resource_optimization import (
    create_admm_flex_actor_one_to_many,
    create_admm_sharing_data,
    create_sharing_target_distance_admm_coordinator,
)
from distributed_resource_optimization.algorithm.admm.sharing_admm import (
    create_sharing_admm_start,
)
from distributed_resource_optimization.algorithm.core import DistributedAlgorithm
from distributed_resource_optimization.carrier.mango_carrier import (
    CoordinatorRole,
    DistributedOptimizationRole,
    StartCoordinatedDistributedOptimization,
)
from distributed_resource_optimization.carrier.simple import (
    ActorContainer,
    SimpleCarrier,
    start_coordinated_optimization,
)


class _Responder(DistributedAlgorithm):
    """Answers every request after *delay_s* (``0``: at once, ``None``: never)."""

    def __init__(self, answer, delay_s: float | None = 0.0) -> None:
        self.answer = answer
        self.delay_s = delay_s
        self.received: list = []

    async def on_exchange_message(self, carrier, message_data, meta):
        self.received.append(message_data)
        if self.delay_s is None:
            return
        if self.delay_s == 0.0:
            carrier.reply_to_other(self.answer, meta)
        else:
            carrier.schedule_after(self.delay_s, lambda: carrier.reply_to_other(self.answer, meta))


def _sharing_actors():
    return [
        create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(15, [0.1, 0.5, -1.0]),
        create_admm_flex_actor_one_to_many(10, [-1.0, 0.0, 1.0]),
    ]


_TARGET = np.array([-4.0, 0.0, 6.0])


def _assert_sharing_solution(actors):
    """Lossless result of target [-4, 0, 6] (see tests/admm/test_sharing_admm.py)."""
    assert actors[0].x == pytest.approx([0, 0, 0], abs=1e-2)
    assert actors[1].x == pytest.approx([0, 0, 0], abs=1e-2)
    assert actors[2].x == pytest.approx([-5.617, 0, 5.617], abs=1e-2)


def _assert_sharing_optimal(actors):
    """Optimal for target [-4, 0, 6], without pinning one optimum.

    The target-distance objective is L1, so the optimum is flat: actor 3 at
    ``a·[-1, 0, 1]`` is optimal (objective 2) for every a in [4, 6].  The
    lossless run lands on a ≈ 5.617; with lost answers ADMM takes another
    path and may stop at another point of that face, which is equally good.
    """
    objective = sum(a.S @ a.x for a in actors) + np.abs(sum(a.x for a in actors) - _TARGET).sum()
    assert objective == pytest.approx(2.0, abs=2e-2)
    for a in actors:
        assert np.all(a.x >= a.lb - 1e-6) and np.all(a.x <= a.u + 1e-6)
        if a.C.size:
            assert np.all(a.C @ a.x <= a.d + 1e-6)


# ---------------------------------------------------------------------------
# SimpleCarrier
# ---------------------------------------------------------------------------


class TestSimpleCarrierRequestAll:
    async def test_missing_reply_is_absent_after_timeout(self):
        container = ActorContainer()
        requester = SimpleCarrier(container, _Responder("unused"))
        SimpleCarrier(container, _Responder("fast"))
        SimpleCarrier(container, _Responder("never", delay_s=None))

        replies = await requester.request_all({2: "q", 3: "q"}, timeout=0.05)

        assert replies == {2: "fast"}

    async def test_reply_type_filters(self):
        container = ActorContainer()
        requester = SimpleCarrier(container, _Responder("unused"))
        SimpleCarrier(container, _Responder("a string"))

        assert await requester.request_all({2: "q"}, timeout=0.05, reply_type=int) == {}

    def test_invalid_loss_rate_rejected(self):
        with pytest.raises(ValueError):
            ActorContainer(loss_rate=1.5)


class TestADMMUnderLossSimpleCarrier:
    async def test_lossless_with_timeout_matches_reference(self):
        actors = _sharing_actors()
        coordinator = create_sharing_target_distance_admm_coordinator(round_timeout=1.0)
        start = create_sharing_admm_start(create_admm_sharing_data([-4, 0, 6]))
        await start_coordinated_optimization(actors, coordinator, start)

        _assert_sharing_solution(actors)
        assert coordinator.converged
        assert coordinator.missed_answers == 0

    async def test_converges_despite_lost_messages(self):
        actors = _sharing_actors()
        coordinator = create_sharing_target_distance_admm_coordinator(round_timeout=0.01)
        container = ActorContainer(loss_rate=0.2, rng=random.Random(1))
        start = create_sharing_admm_start(create_admm_sharing_data([-4, 0, 6]))
        await start_coordinated_optimization(actors, coordinator, start, container=container)

        assert container.dropped > 0
        assert coordinator.missed_answers > 0
        assert coordinator.converged
        _assert_sharing_optimal(actors)

    async def test_stops_when_a_participant_stays_silent(self):
        actors = _sharing_actors()
        coordinator = create_sharing_target_distance_admm_coordinator(
            round_timeout=0.01, max_missing_rounds=3
        )
        container = ActorContainer(loss_rate=1.0)
        start = create_sharing_admm_start(create_admm_sharing_data([-4, 0, 6]))
        await start_coordinated_optimization(actors, coordinator, start, container=container)

        assert not coordinator.converged
        assert coordinator.iterations == 3


# ---------------------------------------------------------------------------
# MangoCarrier in a stepped SimulationWorld
# ---------------------------------------------------------------------------


class TestMangoCarrierInSimulation:
    async def test_request_all_times_out_on_the_simulation_clock(self):
        requester_algo = _Responder("unused")
        requester = DistributedOptimizationRole(requester_algo)
        responders = [
            DistributedOptimizationRole(_Responder("fast")),
            DistributedOptimizationRole(_Responder("never", delay_s=None)),
            DistributedOptimizationRole(_Responder("late", delay_s=5.0)),
        ]
        world = create_world(communication_sim=SimpleCommunicationSimulation(default_delay_s=0.1))
        agents = [world.register(agent_composed_of(r)) for r in [requester, *responders]]
        auto_assign(complete_topology(len(agents)), agents)
        result: dict = {}

        async def ask():
            addrs = [a.addr for a in agents[1:]]
            result["replies"] = await requester._carrier.request_all(
                {addr: "q" for addr in addrs}, timeout=1.0
            )
            result["time"] = requester._carrier.now()

        async with world:
            requester.context.schedule_instant_task(ask())
            await world.step_until(20.0)

        assert list(result["replies"].values()) == ["fast"]
        assert result["time"] == pytest.approx(1.0)
        # The late answer (sent at t≈5.1) is an orphaned reply: dropped, not
        # handed to the requester's algorithm as a new message.
        assert requester_algo.received == []

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    async def test_admm_converges_under_simulated_loss(self, seed):
        actors = _sharing_actors()
        coordinator = create_sharing_target_distance_admm_coordinator(round_timeout=0.5)
        com_sim = SimpleCommunicationSimulation(
            default_delay_s=0.1, loss_percent=0.2, rng=random.Random(seed)
        )
        world = create_world(communication_sim=com_sim)
        opt_agents = [
            world.register(agent_composed_of(DistributedOptimizationRole(a))) for a in actors
        ]
        coord_role = CoordinatorRole(coordinator)
        coord_agent = world.register(agent_composed_of(coord_role))
        all_agents = [*opt_agents, coord_agent]
        auto_assign(complete_topology(len(all_agents)), all_agents)

        async with world:
            start = create_sharing_admm_start(create_admm_sharing_data([-4, 0, 6]))
            coord_role._handle_start(StartCoordinatedDistributedOptimization(input=start), {})
            await world.step_until(10_000.0)
            results = await coord_role.wait_done()

        assert coordinator.missed_answers > 0
        assert coordinator.converged
        assert np.allclose(results[2], actors[2].x)
        _assert_sharing_optimal(actors)
