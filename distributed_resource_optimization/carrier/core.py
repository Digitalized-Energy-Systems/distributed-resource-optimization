"""Abstract carrier interface."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


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
        this particular send (matched via a unique message ID).

        :param content: Arbitrary message payload.
        :param receiver: Carrier-specific address of the target participant.
        :returns: A :class:`asyncio.Future` that yields the reply content.
        """

    @abstractmethod
    def others(self) -> list[Any]:
        """Return the addresses of every other participant this one talks to."""
