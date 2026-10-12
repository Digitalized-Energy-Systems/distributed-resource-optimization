"""In-process SimpleCarrier backed by asyncio tasks.

Provides two convenience entry-points for running distributed or coordinated
optimizations without any network stack.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from .core import Cancellable, Carrier

if TYPE_CHECKING:
    from ..algorithm.core import Coordinator, DistributedAlgorithm


# ---------------------------------------------------------------------------
# ActorContainer
# ---------------------------------------------------------------------------


class ActorContainer:
    """Registry of :class:`SimpleCarrier` instances that share a lifecycle.

    The container tracks how many asyncio dispatch tasks and pending
    :meth:`SimpleCarrier.schedule_after` timers are currently in flight via
    :attr:`active_tasks`.  When that counter drops to zero the
    :attr:`done_event` is set, signalling that the distributed run has
    finished.

    :param loss_rate: Probability in ``[0, 1]`` that a message between
        participants is dropped (default ``0``), for testing loss tolerance
        without mango.  The kick-off message of
        :func:`start_distributed_optimization` is never dropped.
    :param rng: Random generator for the loss draws (default: a fresh
        ``random.Random(0)``, so runs are reproducible).
    """

    def __init__(self, loss_rate: float = 0.0, rng: random.Random | None = None) -> None:
        if not 0.0 <= loss_rate <= 1.0:
            raise ValueError(f"loss_rate must be in [0, 1], got {loss_rate!r}")
        self.actors: list[SimpleCarrier] = []
        self.active_tasks: int = 0
        self.done_event: asyncio.Event = asyncio.Event()
        self.loss_rate = loss_rate
        self._rng = rng if rng is not None else random.Random(0)
        #: Number of messages dropped so far.
        self.dropped: int = 0

    def _register(self, carrier: SimpleCarrier) -> int:
        self.actors.append(carrier)
        return len(self.actors)  # 1-indexed aid

    def _drop(self) -> bool:
        """Draw whether the next message is lost."""
        if self.loss_rate and self._rng.random() < self.loss_rate:
            self.dropped += 1
            return True
        return False

    def _begin(self) -> None:
        self.active_tasks += 1
        # Reset done_event if it was previously set
        if self.done_event.is_set():
            self.done_event.clear()

    def _end(self) -> None:
        self.active_tasks -= 1
        if self.active_tasks == 0:
            self.done_event.set()


class _TrackedTimer:
    """:meth:`SimpleCarrier.schedule_after` handle that keeps the run alive while pending."""

    def __init__(
        self, container: ActorContainer, delay_s: float, callback: Callable[[], Any]
    ) -> None:
        self._container = container
        self._callback = callback
        self._finished = False
        container._begin()
        self._handle = asyncio.get_running_loop().call_later(delay_s, self._fire)

    def _fire(self) -> None:
        try:
            self._callback()
        finally:
            self._finish()

    def cancel(self) -> None:
        self._handle.cancel()
        self._finish()

    def _finish(self) -> None:
        if not self._finished:
            self._finished = True
            self._container._end()


def _completed_task() -> asyncio.Future:
    """Stand-in for the dispatch task of a dropped message."""
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    future.set_result(None)
    return future


# ---------------------------------------------------------------------------
# SimpleCarrier
# ---------------------------------------------------------------------------


class SimpleCarrier(Carrier):
    """Lightweight in-process carrier for a single algorithm participant.

    Messages are dispatched as asyncio Tasks so that multiple participants can
    run concurrently on the same event loop.

    The carrier uses a 1-indexed addressing scheme (``aid`` 1 … N);
    ``cid(carrier)`` returns the integer aid.
    """

    def __init__(
        self,
        container: ActorContainer,
        actor: "DistributedAlgorithm | Coordinator",
    ) -> None:
        self.container = container
        self.actor = actor
        self.aid: int = container._register(self)
        # Maps message-ID → async handler for request-response pairs
        self._uuid_to_handler: dict[UUID, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _dispatch_to(
        self,
        target: "SimpleCarrier",
        content: Any,
        meta: dict,
    ) -> None:
        """Route an incoming message to the right handler."""
        msg_id = meta.get("message_id")
        if msg_id is not None and msg_id in target._uuid_to_handler:
            await target._uuid_to_handler[msg_id](target, content, meta)
        else:
            await target.actor.on_exchange_message(target, content, meta)

    def _track_dispatch(self, coro: Any) -> asyncio.Task:
        """Run *coro* as a task tracked by the container's active-task counter."""
        self.container._begin()

        async def _run() -> None:
            try:
                await coro
            finally:
                self.container._end()

        return asyncio.create_task(_run())

    # ------------------------------------------------------------------
    # Carrier interface
    # ------------------------------------------------------------------

    def _send(
        self,
        content: Any,
        receiver: int,
        extra_meta: dict | None = None,
        *,
        lossy: bool = True,
    ) -> asyncio.Task:
        """Dispatch *content* to 1-indexed *receiver* with optional extra meta."""
        if lossy and self.container._drop():
            return _completed_task()  # type: ignore[return-value]
        other = self.container.actors[receiver - 1]
        full_meta: dict = {"sender": self.aid, "message_id": uuid4()}
        full_meta.update(extra_meta or {})
        return self._track_dispatch(self._dispatch_to(other, content, full_meta))

    def send_to_other(self, content: Any, receiver: int) -> asyncio.Task:
        """Dispatch *content* to the carrier identified by 1-indexed *receiver*.

        The dispatch runs in a fresh asyncio Task so that the caller is not
        blocked.  The container's :attr:`~ActorContainer.active_tasks` counter
        is incremented before the task starts and decremented (with possible
        done-event notification) when the task finishes.
        """
        return self._send(content, receiver)

    def reply_to_other(self, content: Any, meta: dict) -> asyncio.Task:
        """Reply to the sender recorded in *meta*.

        The original ``message_id`` is preserved so that the coordinator's
        awaitable handler can match the response.
        """
        sender = meta["sender"]
        reply_meta = {**meta, "reply": True}
        return self._send(content, sender, reply_meta)

    def send_awaitable(self, content: Any, receiver: int) -> asyncio.Future:
        """Send *content* and return a Future that resolves to the reply.

        The reply is matched by the ``message_id`` stored in the outgoing meta.
        The sender registers a one-shot handler keyed on that ID; when the
        target calls :meth:`reply_to_other` the same ID travels back and
        triggers the handler, resolving the future.
        """
        other = self.container.actors[receiver - 1]
        msg_id = uuid4()
        full_meta: dict = {"sender": self.aid, "message_id": msg_id}

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        async def _handler(carrier: SimpleCarrier, reply_content: Any, _meta: dict) -> None:
            # One-shot: unregister before resolving so replies cannot leak handlers.
            self._uuid_to_handler.pop(msg_id, None)
            if not future.done():
                future.set_result(reply_content)

        self._uuid_to_handler[msg_id] = _handler

        if self.container._drop():
            return future  # request lost: the future never resolves
        self._track_dispatch(self._dispatch_to(other, content, full_meta))
        return future

    def others(self) -> list[int]:
        """Return all 1-indexed carrier IDs except *this* carrier's ID."""
        return [i + 1 for i in range(len(self.container.actors)) if i + 1 != self.aid]

    def schedule_after(self, delay_s: float, callback: Callable[[], Any]) -> Cancellable:
        """Like :meth:`Carrier.schedule_after`, but the pending timer counts as
        in-flight work, so :attr:`ActorContainer.done_event` does not fire while
        a round deadline is still pending."""
        return _TrackedTimer(self.container, delay_s, callback)


def cid(carrier: SimpleCarrier) -> int:
    """Return the 1-indexed ID of *carrier*."""
    return carrier.aid


# ---------------------------------------------------------------------------
# Express helpers
# ---------------------------------------------------------------------------


async def start_distributed_optimization(
    actors: list["DistributedAlgorithm"],
    start_message: Any,
    container: ActorContainer | None = None,
) -> None:
    """Run a fully distributed optimization (e.g. COHDA) and wait until done.

    Creates a fresh :class:`ActorContainer`, wraps each algorithm in a
    :class:`SimpleCarrier`, sends *start_message* from the first carrier to
    the second, then awaits completion.

    :param actors: List of algorithm participants.
    :param start_message: The initial message to kick-off the algorithm
                          (e.g. a :class:`~...cohda.core.WorkingMemory`).
    :param container: Optional pre-built :class:`ActorContainer`, e.g. one
                      with ``loss_rate`` set; it must not have actors yet.
    """
    container = container if container is not None else ActorContainer()
    carriers = [SimpleCarrier(container, actor) for actor in actors]
    carriers[0]._send(start_message, cid(carriers[1]), lossy=False)
    await container.done_event.wait()


async def start_coordinated_optimization(
    actors: list["DistributedAlgorithm"],
    coordinator: "Coordinator",
    start_message: Any,
    container: ActorContainer | None = None,
) -> list[Any]:
    """Run a coordinator-driven optimization (e.g. ADMM) and return results.

    Creates a shared :class:`ActorContainer`, registers all actor carriers
    and a coordinator carrier, then delegates to
    :meth:`~...algorithm.core.Coordinator.start_optimization`.

    :param actors: List of algorithm participants.
    :param coordinator: The coordinator (e.g. an
                        :class:`~...admm.core.ADMMGenericCoordinator`).
    :param start_message: The start payload (e.g.
                          :class:`~...admm.core.ADMMStart`).
    :param container: Optional pre-built :class:`ActorContainer`, e.g. one
                      with ``loss_rate`` set; it must not have actors yet.
    :returns: Whatever the coordinator's ``start_optimization`` returns.
    """
    container = container if container is not None else ActorContainer()
    for actor in actors:
        SimpleCarrier(container, actor)  # registers itself with the container
    coordinator_carrier = SimpleCarrier(container, coordinator)
    return await coordinator.start_optimization(coordinator_carrier, start_message, {})
