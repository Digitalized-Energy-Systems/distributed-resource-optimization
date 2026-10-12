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
    * :class:`~distributed_resource_optimization.carrier.mango.MangoCarrier`
      — integrates with the *mango-agents* framework for networked deployments.

    The loss-tolerance helpers (:meth:`request_all`, :meth:`schedule_after`,
    :meth:`now`, :meth:`sender_of`) have defaults built on the abstract
    methods and the running event loop; carriers with their own clock (mango's
    simulation clock) override them so deadlines follow that clock.
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

    def sender_of(self, meta: Any) -> Any:
        """Address of the participant that sent the message described by *meta*.

        The default reads ``meta["sender"]``; carriers with another addressing
        scheme override it.
        """
        return meta["sender"]
