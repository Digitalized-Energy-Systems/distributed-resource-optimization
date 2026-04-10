"""Unit tests for SimpleCarrier and ActorContainer.

These tests exercise the SimpleCarrier machinery directly — registration,
address lookup, neighbour enumeration, message routing, done-event lifecycle,
and schedule_using — without going through a full algorithm run.
"""

from __future__ import annotations

import asyncio

import pytest

from distributed_resource_optimization import (
    ActorContainer,
    SimpleCarrier,
    cid,
)
from distributed_resource_optimization.algorithm.core import (
    DistributedAlgorithm,
)

# ---------------------------------------------------------------------------
# Minimal stub algorithm for routing tests
# ---------------------------------------------------------------------------

class _EchoAlgorithm(DistributedAlgorithm):
    """Stores every received (content, meta) pair in `received`."""

    def __init__(self) -> None:
        self.received: list[tuple] = []

    async def on_exchange_message(self, carrier, content, meta):
        self.received.append((content, meta))


# ---------------------------------------------------------------------------
# Registration and addressing
# ---------------------------------------------------------------------------

def test_container_registers_carriers():
    container = ActorContainer()
    algo1 = _EchoAlgorithm()
    algo2 = _EchoAlgorithm()
    c1 = SimpleCarrier(container, algo1)
    c2 = SimpleCarrier(container, algo2)

    assert cid(c1) == 1
    assert cid(c2) == 2
    assert len(container.actors) == 2


def test_get_address_matches_cid():
    container = ActorContainer()
    c = SimpleCarrier(container, _EchoAlgorithm())
    assert c.get_address() == cid(c)


def test_others_excludes_self():
    container = ActorContainer()
    c1 = SimpleCarrier(container, _EchoAlgorithm())
    c2 = SimpleCarrier(container, _EchoAlgorithm())
    c3 = SimpleCarrier(container, _EchoAlgorithm())

    # c1 should see c2 and c3
    assert set(c1.others("any")) == {2, 3}
    # c2 should see c1 and c3
    assert set(c2.others("any")) == {1, 3}
    # c3 should see c1 and c2
    assert set(c3.others("any")) == {1, 2}


# ---------------------------------------------------------------------------
# Done-event lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_done_event_fires_after_single_message():
    """done_event should be set once all tasks finish."""
    container = ActorContainer()
    received = []

    class _CollectAlgo(DistributedAlgorithm):
        async def on_exchange_message(self, carrier, content, meta):
            received.append(content)

    c1 = SimpleCarrier(container, _CollectAlgo())
    c2 = SimpleCarrier(container, _CollectAlgo())

    c1.send_to_other("hello", cid(c2))
    await asyncio.wait_for(container.done_event.wait(), timeout=2.0)

    assert received == ["hello"]


@pytest.mark.asyncio
async def test_done_event_resets_between_runs():
    """done_event should clear and re-fire on a second message burst."""
    container = ActorContainer()
    log: list[int] = []

    class _LogAlgo(DistributedAlgorithm):
        def __init__(self, tag: int) -> None:
            self.tag = tag

        async def on_exchange_message(self, carrier, content, meta):
            log.append(self.tag)

    c1 = SimpleCarrier(container, _LogAlgo(1))
    c2 = SimpleCarrier(container, _LogAlgo(2))

    # First round
    c1.send_to_other("ping", cid(c2))
    await asyncio.wait_for(container.done_event.wait(), timeout=2.0)
    assert log == [2]

    # Second round — done_event must reset and fire again
    c2.send_to_other("pong", cid(c1))
    await asyncio.wait_for(container.done_event.wait(), timeout=2.0)
    assert log == [2, 1]


# ---------------------------------------------------------------------------
# Meta propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_to_other_populates_sender_in_meta():
    """The 'sender' field in meta should match the sending carrier's aid."""
    container = ActorContainer()
    metas: list[dict] = []

    class _MetaCollect(DistributedAlgorithm):
        async def on_exchange_message(self, carrier, content, meta):
            metas.append(meta)

    c1 = SimpleCarrier(container, _MetaCollect())
    c2 = SimpleCarrier(container, _MetaCollect())

    c1.send_to_other("x", cid(c2))
    await asyncio.wait_for(container.done_event.wait(), timeout=2.0)

    assert metas[0]["sender"] == cid(c1)


# ---------------------------------------------------------------------------
# send_awaitable — request-reply
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_awaitable_resolves_on_reply():
    """send_awaitable should return a Future that resolves when the target replies."""
    container = ActorContainer()

    class _ReplyAlgo(DistributedAlgorithm):
        async def on_exchange_message(self, carrier, content, meta):
            await carrier.reply_to_other(content + "_reply", meta)

    c1 = SimpleCarrier(container, _ReplyAlgo())
    c2 = SimpleCarrier(container, _ReplyAlgo())

    future = c1.send_awaitable("ping", cid(c2))
    result = await asyncio.wait_for(asyncio.ensure_future(future), timeout=2.0)
    assert result == "ping_reply"


# ---------------------------------------------------------------------------
# schedule_using
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schedule_using_executes_function():
    """schedule_using should run the callable (after the given delay)."""
    container = ActorContainer()
    c = SimpleCarrier(container, _EchoAlgorithm())
    called = []

    c.schedule_using(lambda: called.append(True), 0.0)
    # Give the event loop a tick to execute the spawned task
    await asyncio.sleep(0.05)
    assert called == [True]
