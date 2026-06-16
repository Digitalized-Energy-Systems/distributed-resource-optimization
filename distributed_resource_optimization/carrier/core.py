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
    * :class:`~distributed_resource_optimization.carrier.mango.MangoCarrier`
      — integrates with the *mango-agents* framework for networked deployments.
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

    async def wait_for(self, awaitable: asyncio.Future | EventWithValue) -> Any:
        """Await *awaitable*, unwrapping an :class:`EventWithValue` if needed."""
        if isinstance(awaitable, EventWithValue):
            return await awaitable.wait()
        return await awaitable
