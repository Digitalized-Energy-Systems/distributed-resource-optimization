"""TestCarrier — a no-network carrier for unit testing algorithm logic.

Messages sent via :meth:`send_to_other` are stored in
:attr:`test_neighbor_messages` so tests can inspect them without any
concurrency machinery.
"""

from __future__ import annotations

import asyncio
from typing import Any

from distributed_resource_optimization.carrier.core import Carrier


class TestCarrier(Carrier):
    """Synchronous stub carrier for unit tests.

    :param test_neighbors: Set of (integer) neighbour IDs that
                           :meth:`others` will return.
    """

    def __init__(self, test_neighbors: set[int]) -> None:
        self.test_neighbors = test_neighbors
        self.test_neighbor_messages: dict[int, list[Any]] = {}

    # ------------------------------------------------------------------
    # Carrier interface
    # ------------------------------------------------------------------

    def send_to_other(
        self,
        content: Any,
        receiver: int,
        meta: dict | None = None,
    ) -> asyncio.Task:
        """Store *content* in :attr:`test_neighbor_messages`."""
        buf = self.test_neighbor_messages.setdefault(receiver, [])
        buf.append(content)
        # Return a trivially completed future to satisfy the Task type hint
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        fut.set_result(None)
        return fut  # type: ignore[return-value]

    def reply_to_other(self, content: Any, meta: dict) -> asyncio.Task:
        sender = meta.get("sender", 1)
        return self.send_to_other(content, sender, meta)

    def send_awaitable(
        self,
        content: Any,
        receiver: int,
        meta: dict | None = None,
    ) -> asyncio.Future:
        self.send_to_other(content, receiver, meta)
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        fut.set_result(None)
        return fut

    def others(self, participant_id: str) -> list[int]:
        return list(self.test_neighbors)

    def get_address(self) -> int:
        return 0

    def schedule_using(self, fn: Any, delay_s: float) -> None:
        """Execute *fn* immediately (no real scheduling in tests)."""
        fn()
