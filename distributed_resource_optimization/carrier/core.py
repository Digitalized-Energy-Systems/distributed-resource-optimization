"""Abstract carrier interface."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Protocol


class Cancellable(Protocol):
    """Handle returned by :meth:`Carrier.schedule_after`."""

    def cancel(self) -> Any: ...


class Carrier(ABC):
    """Abstract communication carrier used by distributed algorithms.

    A concrete carrier handles the transport of messages between algorithm
    participants.  Two built-in implementations are provided:

    * :class:`~distributed_resource_optimization.carrier.simple.SimpleCarrier`
      — lightweight in-process carrier backed by asyncio tasks.
    * :class:`~distributed_resource_optimization.carrier.mango_carrier.MangoCarrier`
      — integrates with the *mango-agents* framework for networked deployments.

    The loss-tolerance helpers (:meth:`request_all`, :meth:`schedule_after`,
    :meth:`now`, :meth:`sender_of`) have defaults built on the abstract
    methods and the running event loop; carriers with their own clock (mango's
    simulation clock) override them so deadlines follow that clock.

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
    def send_to_other(self, content: Any, receiver: Any) -> asyncio.Task:
        """Send *content* to *receiver* (fire-and-forget, returns the task).

        :param content: Arbitrary message payload.
        :param receiver: Carrier-specific address of the target participant.
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
    def send_awaitable(self, content: Any, receiver: Any) -> asyncio.Future:
        """Send *content* to *receiver* and return a Future for the reply.

        The future resolves to the first reply message received in response to
        this particular send (matched via a unique message ID).  It never
        resolves if the request or the reply is lost; prefer
        :meth:`request_all`, which has a timeout.

        :param content: Arbitrary message payload.
        :param receiver: Carrier-specific address of the target participant.
        :returns: A :class:`asyncio.Future` that yields the reply content.
        """

    @abstractmethod
    def others(self) -> list[Any]:
        """Return the addresses of every other participant this one talks to."""

    # ------------------------------------------------------------------
    # Loss-tolerance helpers (overridable defaults)
    # ------------------------------------------------------------------

    async def request_all(
        self,
        requests: Mapping[Any, Any],
        *,
        timeout: float | None = None,
        reply_type: type | tuple[type, ...] | None = None,
    ) -> dict[Any, Any]:
        """Send ``requests[addr]`` to every *addr* and collect the replies.

        Returns once every receiver replied or *timeout* seconds passed,
        whichever is first.  Receivers whose request or reply was lost are
        simply absent from the result, so callers decide what a missing
        reply means (typically: keep the previous value).

        :param requests: ``{receiver: content}``; the content may differ per
            receiver.
        :param timeout: Seconds to wait (on the carrier's clock); ``None``
            waits for every reply.
        :param reply_type: If given, replies of any other type are ignored.
        :returns: ``{receiver: reply}`` for every receiver that answered.
        """
        futures = {addr: self.send_awaitable(content, addr) for addr, content in requests.items()}
        if not futures:
            return {}
        _, pending = await asyncio.wait(list(futures.values()), timeout=timeout)
        for future in pending:
            future.cancel()
        replies: dict[Any, Any] = {}
        for addr, future in futures.items():
            if not future.done() or future.cancelled() or future.exception() is not None:
                continue
            reply = future.result()
            if reply_type is None or isinstance(reply, reply_type):
                replies[addr] = reply
        return replies

    def schedule_after(self, delay_s: float, callback: Callable[[], Any]) -> Cancellable:
        """Call *callback* after *delay_s* seconds on the carrier's clock.

        :returns: A handle whose ``cancel()`` prevents the call.
        """
        return asyncio.get_running_loop().call_later(delay_s, callback)

    def now(self) -> float:
        """Current time on the carrier's clock, in seconds."""
        return asyncio.get_running_loop().time()

    def address(self) -> Any:
        """This participant's own address (as its neighbours see it in ``others()``).

        ``None`` if the carrier cannot tell; algorithms then skip features
        that need a global order of participants (edge ownership).
        """
        return None

    def sender_of(self, meta: Any) -> Any:
        """Address of the participant that sent the message described by *meta*.

        The default reads ``meta["sender"]``; carriers with another addressing
        scheme override it.
        """
        return meta["sender"]

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

    async def wait_for(self, awaitable: Any) -> Any:
        """Await *awaitable* through the carrier's park point.

        A simulation-backed carrier marks the calling driver idle for the
        duration of the wait.
        """
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
            await self.wait_for(asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
