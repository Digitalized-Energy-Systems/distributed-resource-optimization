"""In-process SimpleCarrier backed by asyncio tasks.

Provides two convenience entry-points for running distributed or coordinated
optimizations without any network stack.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from .core import Carrier

if TYPE_CHECKING:
    from ..algorithm.core import Coordinator, DistributedAlgorithm


# ---------------------------------------------------------------------------
# ActorContainer
# ---------------------------------------------------------------------------


class ActorContainer:
    """Registry of :class:`SimpleCarrier` instances that share a lifecycle.

    The container tracks how many asyncio dispatch tasks are currently in
    flight via :attr:`active_tasks`.  When that counter drops to zero the
    :attr:`done_event` is set, signalling that the distributed run has
    finished.
    """

    def __init__(self) -> None:
        self.actors: list[SimpleCarrier] = []
        self.active_tasks: int = 0
        self.done_event: asyncio.Event = asyncio.Event()

    def _register(self, carrier: SimpleCarrier) -> int:
        self.actors.append(carrier)
        return len(self.actors)  # 1-indexed aid


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
        from ..algorithm.core import on_exchange_message

        msg_id = meta.get("message_id")
        if msg_id is not None and msg_id in target._uuid_to_handler:
            await target._uuid_to_handler[msg_id](target, content, meta)
        else:
            await on_exchange_message(target.actor, target, content, meta)

    def _task_done(self, task: asyncio.Task) -> None:
        """Callback invoked when a dispatch task finishes."""
        self.container.active_tasks -= 1
        if self.container.active_tasks == 0:
            self.container.done_event.set()

    # ------------------------------------------------------------------
    # Carrier interface
    # ------------------------------------------------------------------

    def send_to_other(
        self,
        content: Any,
        receiver: int,
        meta: dict | None = None,
    ) -> asyncio.Task:
        """Dispatch *content* to the carrier identified by 1-indexed *receiver*.

        The dispatch runs in a fresh asyncio Task so that the caller is not
        blocked.  The container's :attr:`~ActorContainer.active_tasks` counter
        is incremented before the task starts and decremented (with possible
        done-event notification) when the task finishes.
        """
        other = self.container.actors[receiver - 1]
        extra = meta or {}
        full_meta: dict = {"sender": self.aid, "message_id": uuid4()}
        full_meta.update(extra)

        self.container.active_tasks += 1
        # Reset done_event if it was previously set
        if self.container.done_event.is_set():
            self.container.done_event.clear()

        async def _run() -> None:
            try:
                await self._dispatch_to(other, content, full_meta)
            finally:
                self.container.active_tasks -= 1
                if self.container.active_tasks == 0:
                    self.container.done_event.set()

        task = asyncio.create_task(_run())
        return task

    def reply_to_other(self, content: Any, meta: dict) -> asyncio.Task:
        """Reply to the sender recorded in *meta*.

        The original ``message_id`` is preserved so that the coordinator's
        awaitable handler can match the response.
        """
        sender = meta["sender"]
        reply_meta = {**meta, "reply": True}
        return self.send_to_other(content, sender, meta=reply_meta)

    def send_awaitable(
        self,
        content: Any,
        receiver: int,
        meta: dict | None = None,
    ) -> asyncio.Future:
        """Send *content* and return a Future that resolves to the reply.

        The reply is matched by the ``message_id`` stored in the outgoing meta.
        The sender registers a one-shot handler keyed on that ID; when the
        target calls :meth:`reply_to_other` the same ID travels back and
        triggers the handler, resolving the future.
        """
        other = self.container.actors[receiver - 1]
        extra = meta or {}
        msg_id = uuid4()
        full_meta: dict = {"sender": self.aid, "message_id": msg_id}
        full_meta.update(extra)

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        async def _handler(carrier: SimpleCarrier, reply_content: Any, _meta: dict) -> None:
            if not future.done():
                future.set_result(reply_content)

        self._uuid_to_handler[msg_id] = _handler

        async def _run() -> None:
            await self._dispatch_to(other, content, full_meta)

        asyncio.create_task(_run())
        return future

    def others(self, participant_id: str) -> list[int]:
        """Return all 1-indexed carrier IDs except *this* carrier's ID."""
        return [i + 1 for i in range(len(self.container.actors)) if i + 1 != self.aid]

    def get_address(self) -> int:
        return self.aid

    def schedule_using(self, fn: Any, delay_s: float) -> asyncio.Task:
        """Schedule *fn* to run after *delay_s* seconds on the event loop."""

        async def _run() -> None:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            fn()

        return asyncio.create_task(_run())


def cid(carrier: SimpleCarrier) -> int:
    """Return the 1-indexed ID of *carrier*."""
    return carrier.aid


# ---------------------------------------------------------------------------
# Express helpers
# ---------------------------------------------------------------------------


async def start_distributed_optimization(
    actors: list["DistributedAlgorithm"],
    start_message: Any,
) -> None:
    """Run a fully distributed optimization (e.g. COHDA) and wait until done.

    Creates a fresh :class:`ActorContainer`, wraps each algorithm in a
    :class:`SimpleCarrier`, sends *start_message* from the first carrier to
    the second, then awaits completion.

    :param actors: List of algorithm participants.
    :param start_message: The initial message to kick-off the algorithm
                          (e.g. a :class:`~...cohda.core.WorkingMemory`).
    """
    container = ActorContainer()
    carriers = [SimpleCarrier(container, actor) for actor in actors]
    carriers[0].send_to_other(start_message, cid(carriers[1]))
    await container.done_event.wait()


async def start_coordinated_optimization(
    actors: list["DistributedAlgorithm"],
    coordinator: "Coordinator",
    start_message: Any,
) -> list[Any]:
    """Run a coordinator-driven optimization (e.g. ADMM) and return results.

    Creates a shared :class:`ActorContainer`, registers all actor carriers
    and a coordinator carrier, then delegates to
    :func:`~...algorithm.core.start_optimization`.

    :param actors: List of algorithm participants.
    :param coordinator: The coordinator (e.g. an
                        :class:`~...admm.core.ADMMGenericCoordinator`).
    :param start_message: The start payload (e.g.
                          :class:`~...admm.core.ADMMStart`).
    :returns: Whatever :func:`start_optimization` returns (coordinator-specific).
    """
    from ..algorithm.core import start_optimization

    container = ActorContainer()
    _carriers = [SimpleCarrier(container, actor) for actor in actors]
    coordinator_carrier = SimpleCarrier(container, coordinator)
    return await start_optimization(coordinator, coordinator_carrier, start_message, {})
