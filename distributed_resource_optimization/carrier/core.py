"""Abstract carrier interface and EventWithValue helper."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class EventWithValue:
    """Pairs an asyncio.Event with the value it will carry once set."""

    def __init__(self) -> None:
        self.event: asyncio.Event = asyncio.Event()
        self.value: Any = None

    async def wait(self) -> Any:
        await self.event.wait()
        return self.value


class Carrier(ABC):
    """Abstract communication carrier used by distributed algorithms.

    A concrete carrier handles the transport of messages between algorithm
    participants.  Two built-in implementations are provided:

    * :class:`~distributed_resource_optimization.carrier.simple.SimpleCarrier`
      — lightweight in-process carrier backed by asyncio tasks.
    * :class:`~distributed_resource_optimization.carrier.mango_carrier.MangoCarrier`
      — integrates with the *mango-agents* framework for networked deployments.

    Park-point contract: any wait that resolves only via an inbound message
    or the carrier clock — a :meth:`send_awaitable` future, :meth:`sleep`, or
    a combination of those — must be awaited through :meth:`wait_for`,
    :meth:`gather`, or :meth:`race`, never directly.  A simulation-backed
    carrier declares those parks idle to its scheduler; awaiting them
    directly deadlocks discrete stepping.  Sends and locally-completing
    awaitables must *not* be routed through the park point — they are work
    that has to finish within a simulation step.
    """

    @abstractmethod
    def send_to_other(self, content: Any, receiver: Any, meta: dict | None = None) -> asyncio.Task:
        """Send *content* to *receiver* (fire-and-forget, returns the task).

        :param content: Arbitrary message payload.
        :param receiver: Carrier-specific address of the target participant.
        :param meta: Optional extra metadata merged with transport defaults.
        :returns: The asyncio Task that performs the dispatch.
        """

    @abstractmethod
    def reply_to_other(self, content: Any, meta: dict) -> asyncio.Task:
        """Reply to the sender identified in *meta*.

        :param content: Reply payload.
        :param meta: Metadata from the incoming message (contains sender info).
        :returns: The asyncio Task that performs the dispatch.
        """

    @abstractmethod
    def send_awaitable(
        self, content: Any, receiver: Any, meta: dict | None = None
    ) -> asyncio.Future:
        """Send *content* to *receiver* and return a Future for the reply.

        The future resolves to the first reply message received in response to
        this particular send (matched via a unique message ID).

        :param content: Arbitrary message payload.
        :param receiver: Carrier-specific address of the target participant.
        :param meta: Optional extra metadata.
        :returns: A :class:`asyncio.Future` that yields the reply content.
        """

    @abstractmethod
    def others(self, participant_id: str) -> list[Any]:
        """Return all participant addresses except *participant_id*.

        :param participant_id: The string identifier of the calling participant.
        :returns: List of addresses for every other participant.
        """

    @abstractmethod
    def get_address(self) -> Any:
        """Return the address of this carrier's participant."""

    # ------------------------------------------------------------------
    # Time domain — wall-clock by default; a simulation carrier overrides
    # these so timeouts and background drivers run on the *simulation*
    # clock instead of racing ahead on real wall-clock time.
    # ------------------------------------------------------------------

    def now(self) -> float:
        """Current time in this carrier's clock domain (seconds)."""
        return asyncio.get_event_loop().time()

    def sleep(self, seconds: float) -> Any:
        """Return an awaitable that resolves after *seconds* in this
        carrier's clock domain."""
        return asyncio.sleep(seconds)

    def spawn(self, coroutine: Any) -> asyncio.Task:
        """Launch a long-running driver coroutine (e.g. a coordinator's
        request/reply loop or the gossip cascade round).

        The base implementation runs it as a plain event-loop task. A
        simulation-backed carrier overrides this to schedule it on the
        agent scheduler so the simulation clock tracks it.  The driver's
        reply/timeout waits must go through :meth:`wait_for` /
        :meth:`gather` / :meth:`race` (see the class docstring).
        """
        return asyncio.ensure_future(coroutine)

    async def wait_for(self, awaitable: asyncio.Future | EventWithValue) -> Any:
        """Await *awaitable*, unwrapping an :class:`EventWithValue` if needed.

        The carrier's park point: a simulation-backed carrier marks the
        calling driver idle for the duration of the wait.
        """
        if isinstance(awaitable, EventWithValue):
            return await awaitable.wait()
        return await awaitable

    async def gather(self, *awaitables: Any) -> list[Any]:
        """Await all *awaitables* through this carrier's park point."""
        return await self.wait_for(asyncio.gather(*awaitables))

    async def race(self, *awaitables: Any) -> None:
        """Wait until the first of *awaitables* resolves, then cancel the rest.

        Used to race a peer-ready event against a clock-domain timeout so the
        timeout obeys the carrier's clock (simulation or wall-clock).
        """
        tasks = [asyncio.ensure_future(a) for a in awaitables]
        try:
            await self.wait_for(
                asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
