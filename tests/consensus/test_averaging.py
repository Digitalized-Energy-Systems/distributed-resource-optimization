"""Averaging consensus tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    AveragingConsensusMessage,
    create_averaging_consensus_participant,
    create_averaging_consensus_start,
    start_distributed_optimization,
)
from distributed_resource_optimization.algorithm.consensus.economic_dispatch import (
    LinearCostEconomicDispatchConsensusActor,
)


class _RecordingCarrier:
    """Minimal carrier stub: fixed neighbour list, records every send."""

    def __init__(self, neighbours: list[str]) -> None:
        self._neighbours = neighbours
        self.sent: list[tuple[str, AveragingConsensusMessage]] = []

    def others(self) -> list[str]:
        return self._neighbours

    def send_to_other(self, content, receiver) -> None:
        self.sent.append((receiver, content))


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


@pytest.mark.asyncio
async def test_leader_follower_on_path_graph_hears_every_power():
    """Same dispatch on the path 1 - 2 - 3 - 4 with the leader at one end:
    followers 3 and 4 are not its neighbours and report their power to it,
    so ΔP is the system-wide imbalance; the leader's stop reaches everyone."""
    from distributed_resource_optimization.carrier.simple import ActorContainer, SimpleCarrier

    demand = 80.0
    specs = [(10.0, 50.0), (20.0, 30.0), (30.0, 30.0), (40.0, 30.0)]
    results: dict[int, tuple] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            results[idx] = (algo.converged, algo.actor.P.copy())

        return finish

    actors = [
        create_averaging_consensus_participant(
            make_finish(i),
            consensus_actor=LinearCostEconomicDispatchConsensusActor(
                cost=cost, p_max=p_max, epsilon=0.1
            ),
            initial_lam=10.0,
            alpha=0.3,
            max_iter=3000,
            is_leader=(i == 0),
            leader_gain=0.05,
            leader=None if i == 0 else 1,
            participants=[1, 2, 3, 4] if i == 0 else None,
        )
        for i, (cost, p_max) in enumerate(specs)
    ]
    start = create_averaging_consensus_start(10.0, data=demand)
    container = ActorContainer(neighbours={1: [2], 2: [1, 3], 3: [2, 4], 4: [3]})
    carriers = [SimpleCarrier(container, actor) for actor in actors]
    # Kick-off goes to the leader (carrier 1), like the scenarios' trigger.
    carriers[1]._send(start, 1, lossy=False)
    await container.done_event.wait()

    assert len(results) == len(specs)
    assert all(converged for converged, _ in results.values())
    powers = [float(results[i][1][0]) for i in range(len(specs))]
    assert sum(powers) == pytest.approx(demand, abs=1.0)
    assert powers[0] == pytest.approx(50.0, abs=1.0)
    assert powers[1] == pytest.approx(30.0, abs=1.0)


@pytest.mark.asyncio
async def test_stale_messages_do_not_rewind_iteration():
    """Out-of-order/lossy delivery must never step the iteration backwards.

    A straggler from a round the participant already passed must neither
    complete that old round again nor re-broadcast it.  (Lost messages are
    handled by the round deadline, see tests/test_loss_tolerance_p2p.py; a
    later-round message no longer forces a jump past an incomplete round.)
    """
    algo = create_averaging_consensus_participant(lambda *_: None, max_iter=50)
    carrier = _RecordingCarrier(["n1", "n2"])

    def neighbour_msg(k: int, lam: float = 1.0) -> AveragingConsensusMessage:
        return AveragingConsensusMessage(lam=np.array([lam]), k=k, data=None, p=np.array([0.0]))

    # External kick-off; participant initialises at k=0 and broadcasts.
    await algo.on_exchange_message(
        carrier,
        AveragingConsensusMessage(lam=np.array([1.0]), k=0, data=None, initial=True),
        None,
    )
    assert algo._k == 0

    # n1's round-0 message and a round-1 message: round 0 is incomplete,
    # so no advance yet.
    await algo.on_exchange_message(carrier, neighbour_msg(0), None)
    await algo.on_exchange_message(carrier, neighbour_msg(1), None)
    assert algo._k == 0

    # The second round-0 message completes round 0.
    await algo.on_exchange_message(carrier, neighbour_msg(0), None)
    assert algo._k == 1

    # A round-0 straggler must be dropped, not rewind the counter and
    # re-broadcast.
    ks_sent_before = [msg.k for _, msg in carrier.sent]
    await algo.on_exchange_message(carrier, neighbour_msg(0), None)
    assert algo._k == 1
    assert [msg.k for _, msg in carrier.sent] == ks_sent_before


def test_leader_requires_positive_gain():
    """A leader without a pinning gain would never correct ΔP: refuse it."""
    with pytest.raises(ValueError, match="leader_gain"):
        create_averaging_consensus_participant(lambda *_: None, is_leader=True)
    with pytest.raises(ValueError, match="leader_gain"):
        create_averaging_consensus_participant(lambda *_: None, is_leader=True, leader_gain=-0.1)
    # Followers ignore the gain, so the default is fine for them.
    create_averaging_consensus_participant(lambda *_: None, is_leader=False)


@pytest.mark.asyncio
async def test_data_only_sent_in_round_zero():
    """The constant demand payload travels once; the leader keeps using it.

    After round 0 every broadcast carries ``data=None``, and the leader's ΔP
    pinning must still see the cached demand (here: λ rises because the
    single neighbour reports zero power against a demand of 10).
    """
    algo = create_averaging_consensus_participant(
        lambda *_: None, initial_lam=1.0, is_leader=True, leader_gain=0.1
    )
    carrier = _RecordingCarrier(["n1"])
    demand = np.array([10.0])

    await algo.on_exchange_message(
        carrier, create_averaging_consensus_start(1.0, data=demand), None
    )
    for k in range(3):
        await algo.on_exchange_message(
            carrier,
            AveragingConsensusMessage(lam=np.array([1.0]), k=k, data=None, p=np.array([0.0])),
            None,
        )

    assert algo._k == 3
    sent = [msg for _, msg in carrier.sent]
    assert sent[0].k == 0 and np.array_equal(sent[0].data, demand)
    assert all(msg.data is None for msg in sent[1:])
    assert algo._lam[0] > 1.0


@pytest.mark.asyncio
async def test_leader_stops_on_power_balance():
    """Jian et al. 2020: iterate until ΔP = 0 (here: within 1 % of demand).

    With a generous max_iter the leader must stop the run early, every
    participant must report ``converged=True`` at the same iteration, and the
    final dispatch must be within the balance tolerance.
    """
    demand = 80.0
    specs = [(10.0, 50.0), (20.0, 30.0), (30.0, 30.0), (40.0, 30.0)]
    max_iter = 5000
    finished: dict[int, tuple] = {}

    def make_finish(idx: int):
        def finish(algo, carrier):
            finished[idx] = (algo.converged, algo.iterations, float(algo.actor.P[0]))

        return finish

    actors = [
        create_averaging_consensus_participant(
            make_finish(i),
            consensus_actor=LinearCostEconomicDispatchConsensusActor(
                cost=cost, p_max=p_max, epsilon=0.1
            ),
            alpha=0.3,
            max_iter=max_iter,
            is_leader=(i == 0),
            leader_gain=0.05,
            balance_tol=0.01,
        )
        for i, (cost, p_max) in enumerate(specs)
    ]
    await start_distributed_optimization(
        actors, create_averaging_consensus_start(10.0, data=demand)
    )

    assert len(finished) == len(specs)
    assert all(converged for converged, _, _ in finished.values())
    iterations = {it for _, it, _ in finished.values()}
    assert len(iterations) == 1
    assert iterations.pop() < max_iter
    total = sum(p for _, _, p in finished.values())
    assert abs(total - demand) <= 0.01 * demand


@pytest.mark.asyncio
async def test_follower_rolls_back_to_verified_round_on_stop():
    """A follower one round ahead of the leader's stop finishes on round k."""
    results: list = []
    algo = create_averaging_consensus_participant(
        lambda a, c: results.append((a.converged, a.iterations, a._lam.copy())),
        initial_lam=1.0,
        alpha=0.5,
    )
    carrier = _RecordingCarrier(["leader"])

    await algo.on_exchange_message(carrier, create_averaging_consensus_start(1.0), None)
    # Leader's round-0 message: follower advances to round 1 (λ 1 → 2).
    await algo.on_exchange_message(
        carrier,
        AveragingConsensusMessage(lam=np.array([3.0]), k=0, data=None, p=np.array([0.0])),
        None,
    )
    lam_round_1 = algo._lam.copy()
    # Leader's round-1 message: follower advances to round 2.
    await algo.on_exchange_message(
        carrier,
        AveragingConsensusMessage(lam=np.array([3.0]), k=1, data=None, p=np.array([0.0])),
        None,
    )
    assert algo._k == 2
    # Leader verified round 1's schedules and stops.
    await algo.on_exchange_message(
        carrier,
        AveragingConsensusMessage(lam=np.array([3.0]), k=1, data=None, stop=True),
        None,
    )

    assert len(results) == 1
    converged, iterations, lam = results[0]
    assert converged is True
    assert iterations == 1
    assert np.array_equal(lam, lam_round_1)


@pytest.mark.asyncio
async def test_without_leader_run_hits_max_iter_failsafe():
    """Pure averaging has no leader to check ΔP: stop at max_iter, not converged."""
    finished: list[tuple[bool, int]] = []

    def finish(algo, carrier):
        finished.append((algo.converged, algo.iterations))

    actors = [
        create_averaging_consensus_participant(finish, initial_lam=v, max_iter=30)
        for v in [1.0, 5.0, 10.0]
    ]
    await start_distributed_optimization(actors, create_averaging_consensus_start(1.0))

    assert finished
    assert all(not converged for converged, _ in finished)


@pytest.mark.asyncio
async def test_neighbours_last_round_message_does_not_end_run():
    """Every participant stops by its own counter (loss plan rule 4): a
    neighbour's round-max_iter message must not end a run that is behind."""
    results: list = []
    algo = create_averaging_consensus_participant(
        lambda a, c: results.append((a.converged, a.iterations)),
        initial_lam=1.0,
        alpha=0.5,
        max_iter=3,
    )
    carrier = _RecordingCarrier(["a", "b"])

    def msg(k: int) -> AveragingConsensusMessage:
        return AveragingConsensusMessage(lam=np.array([1.0]), k=k, data=None, p=np.array([0.0]))

    await algo.on_exchange_message(carrier, create_averaging_consensus_start(1.0), None)
    await algo.on_exchange_message(carrier, msg(3), None)  # a neighbour already finished
    assert results == []
    for k in range(3):  # both neighbours report rounds 0..2
        await algo.on_exchange_message(carrier, msg(k), None)
        await algo.on_exchange_message(carrier, msg(k), None)
    assert results == [(False, 3)]


class _Timer:
    def __init__(self, callback) -> None:
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.cancelled = True
            self.callback()


class _LossCarrier(_RecordingCarrier):
    """Recording carrier that names senders (meta["sender"]) and hands out
    timers the test fires by hand, as a carrier with round deadlines would."""

    def __init__(self, neighbours: list[str]) -> None:
        super().__init__(neighbours)
        self.timers: list[_Timer] = []

    def sender_of(self, meta):
        return meta["sender"]

    def schedule_after(self, delay_s, callback):
        timer = _Timer(callback)
        self.timers.append(timer)
        return timer


@pytest.mark.asyncio
async def test_late_stop_rolls_back_several_rounds_and_is_acknowledged():
    """Under loss a repeated stop can arrive several rounds late: the follower
    rolls back to the verified round's λ and acknowledges the stop."""
    results: list = []
    algo = create_averaging_consensus_participant(
        lambda a, c: results.append((a.converged, a.iterations, a._lam.copy())),
        initial_lam=1.0,
        alpha=0.5,
        round_timeout=1.0,
    )
    carrier = _LossCarrier(["leader"])

    def leader_msg(k: int) -> AveragingConsensusMessage:
        return AveragingConsensusMessage(lam=np.array([3.0 + k]), k=k, data=None, p=np.array([0.0]))

    await algo.on_exchange_message(carrier, create_averaging_consensus_start(1.0), None)
    await algo.on_exchange_message(carrier, leader_msg(0), {"sender": "leader"})
    lam_round_1 = algo._lam.copy()  # λ entering round 1
    for k in (1, 2):
        await algo.on_exchange_message(carrier, leader_msg(k), {"sender": "leader"})
    assert algo._k == 3  # three rounds past the start

    stop = AveragingConsensusMessage(lam=np.array([0.0]), k=1, data=None, stop=True)
    await algo.on_exchange_message(carrier, stop, {"sender": "leader"})
    assert len(results) == 1
    converged, iterations, lam = results[0]
    assert converged and iterations == 1
    np.testing.assert_array_equal(lam, lam_round_1)
    acks = [m for r, m in carrier.sent if r == "leader" and m.stop_ack]
    assert len(acks) == 1

    # A repeated copy after finishing is acknowledged again but changes nothing.
    await algo.on_exchange_message(carrier, stop, {"sender": "leader"})
    assert len(results) == 1
    assert len([m for r, m in carrier.sent if r == "leader" and m.stop_ack]) == 2


@pytest.mark.asyncio
async def test_leader_repeats_stop_until_every_neighbour_acknowledged():
    results: list = []
    leader = create_averaging_consensus_participant(
        lambda a, c: results.append(a.converged),
        initial_lam=1.0,
        is_leader=True,
        leader_gain=0.1,
        round_timeout=1.0,
        stop_retries=5,
    )
    carrier = _LossCarrier(["a", "b"])
    demand = np.zeros(1)  # zero demand, zero projected power: balanced at once

    await leader.on_exchange_message(
        carrier, create_averaging_consensus_start(1.0, data=demand), None
    )
    for sender in ("a", "b"):
        await leader.on_exchange_message(
            carrier,
            AveragingConsensusMessage(lam=np.array([1.0]), k=0, data=None, p=np.array([0.0])),
            {"sender": sender},
        )
    assert results == [True]

    def stops_to(receiver: str) -> int:
        return len([m for r, m in carrier.sent if r == receiver and m.stop])

    assert stops_to("a") == 1 and stops_to("b") == 1
    ack = AveragingConsensusMessage(lam=np.array([1.0]), k=0, data=None, stop_ack=True)
    await leader.on_exchange_message(carrier, ack, {"sender": "a"})

    carrier.timers[-1].fire()  # one round timeout later: repeat to "b" only
    assert stops_to("a") == 1 and stops_to("b") == 2

    await leader.on_exchange_message(carrier, ack, {"sender": "b"})
    assert carrier.timers[-1].cancelled  # everyone acked: no more repeats
    for timer in carrier.timers:
        timer.fire()
    assert stops_to("a") == 1 and stops_to("b") == 2


@pytest.mark.asyncio
async def test_rollback_restores_the_verified_rounds_power():
    """A follower that gets the leader's stop for an earlier round finishes on
    exactly the power it broadcast in that round, without asking its actor
    again (a proximal storage response depends on its previous schedule)."""
    calls = []

    class _Counting(LinearCostEconomicDispatchConsensusActor):
        def project_power(self, lam, data):
            calls.append(1)
            return super().project_power(lam, data)

    finished = []
    actor = _Counting(cost=10.0, p_max=50.0, epsilon=0.1)
    algo = create_averaging_consensus_participant(
        lambda a, c: finished.append(a.actor.P.copy()),
        consensus_actor=actor,
        initial_lam=5.0,
        max_iter=50,
    )
    carrier = _RecordingCarrier(["leader"])
    start = create_averaging_consensus_start(5.0, data=np.array([20.0]))
    await algo.on_exchange_message(carrier, start, {})
    rounds = []
    for k in range(3):
        rounds.append(algo._P.copy())
        await algo.on_exchange_message(
            carrier,
            AveragingConsensusMessage(lam=np.array([12.0 + k]), k=k, data=None, p=np.zeros(1)),
            {"sender": "leader"},
        )
    n_calls = len(calls)
    stop = AveragingConsensusMessage(lam=np.array([12.0]), k=1, data=None, stop=True)
    await algo.on_exchange_message(carrier, stop, {"sender": "leader"})
    assert len(calls) == n_calls  # restored, not recomputed
    np.testing.assert_array_equal(finished[-1], rounds[1])
