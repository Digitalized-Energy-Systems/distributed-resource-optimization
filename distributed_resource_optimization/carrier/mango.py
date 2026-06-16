"""MangoCarrier — integrates with the *mango-agents* Python framework.

Provides:

* :class:`MangoCarrier` — a :class:`~.core.Carrier` backed by a mango-agents
  :class:`~mango.Role`.
* :class:`DistributedOptimizationRole` — a mango Role that hosts a
  :class:`~..algorithm.core.DistributedAlgorithm`.
* :class:`CoordinatorRole` — a mango Role that hosts a
  :class:`~..algorithm.core.Coordinator`.
* :class:`StartCoordinatedDistributedOptimization` /
  :class:`OptimizationFinishedMessage` — coordination message types.

Usage example::

    from mango import create_tcp_container, agent_composed_of, activate, complete_topology
    from distributed_resource_optimization.carrier.mango import (
        CoordinatorRole, DistributedOptimizationRole,
        StartCoordinatedDistributedOptimization, OptimizationFinishedMessage,
    )
    from distributed_resource_optimization.algorithm.admm.flex_actor import (
        create_admm_flex_actor_one_to_many,
    )
    from distributed_resource_optimization.algorithm.admm.sharing_admm import (
        create_sharing_target_distance_admm_coordinator, create_admm_start,
        create_admm_sharing_data,
    )

    async def main():
        container = create_tcp_container("127.0.0.1", 5555)

        flex_actor = create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0])
        coordinator = create_sharing_target_distance_admm_coordinator()

        dor = DistributedOptimizationRole(flex_actor)
        coord_role = CoordinatorRole(coordinator, include_self=True)

        a1 = container.register(agent_composed_of(dor))
        ac = container.register(agent_composed_of(coord_role))

        topo = complete_topology(2)
        topo.inject()

        async with activate(container):
            await a1.send_message(
                StartCoordinatedDistributedOptimization(
                    create_admm_start(create_admm_sharing_data([-4, 0, 6]))
                ),
                ac.addr,
            )
            await coord_role.wait_done()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mango import Role
from mango import sender_addr as mango_sender_addr

from .core import Carrier

if TYPE_CHECKING:
    from mango import AgentAddress

    from ..algorithm.core import Coordinator, DistributedAlgorithm


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class StartCoordinatedDistributedOptimization:
    """Sent to a :class:`CoordinatorRole` to kick off a coordinated run."""

    input: Any


@dataclass
class OptimizationFinishedMessage:
    """Broadcast by the coordinator to each participant when the run ends."""

    result: Any


@dataclass
class _CarrierRequest:
    """Internal wrapper that adds request-tracking to any optimization message."""

    content: Any
    request_id: str


@dataclass
class _CarrierReply:
    """Internal reply wrapper matched to a :class:`_CarrierRequest` by ID."""

    content: Any
    request_id: str


# ---------------------------------------------------------------------------
# MangoCarrier
# ---------------------------------------------------------------------------


class MangoCarrier(Carrier):
    """A :class:`~.core.Carrier` that delegates to a mango-agents Role.

    :param parent: The mango :class:`~mango.Role` whose context is used for
                   message sending, scheduling and address lookup.
    :param include_self: Whether :meth:`others` should include the agent's own
                         address (mirrors ``CoordinatorRole``'s ``include_self``
                         parameter).
    """

    def __init__(self, parent: Role, include_self: bool = False) -> None:
        self._parent = parent
        self._include_self = include_self
        # Pending request futures keyed by request_id
        self._pending: dict[str, asyncio.Future] = {}

    # ------------------------------------------------------------------
    # Carrier interface
    # ------------------------------------------------------------------

    def send_to_other(
        self,
        content: Any,
        receiver: "AgentAddress",
        meta: dict | None = None,
    ) -> asyncio.Task:
        """Send *content* to *receiver* asynchronously (fire-and-forget)."""

        async def _send() -> None:
            await self._parent.context.send_message(content, receiver)

        return asyncio.create_task(_send())

    def reply_to_other(self, content: Any, meta: dict) -> asyncio.Task:
        """Reply to the sender stored in *meta*.

        If *meta* contains a ``_request_id`` key (set when the message arrived
        via :meth:`send_awaitable`), the reply is wrapped in a
        :class:`_CarrierReply` so the originating future can be resolved.
        """
        addr = mango_sender_addr(meta)
        request_id = meta.get("_request_id")

        async def _send() -> None:
            if request_id:
                reply = _CarrierReply(content=content, request_id=request_id)
                await self._parent.context.send_message(reply, addr)
            else:
                await self._parent.context.send_message(content, addr)

        return asyncio.create_task(_send())

    def send_awaitable(
        self,
        content: Any,
        receiver: "AgentAddress",
        meta: dict | None = None,
    ) -> asyncio.Future:
        """Send *content* and return a Future that resolves when the reply arrives."""
        request_id = str(uuid4())
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future

        async def _send() -> None:
            await self._parent.context.send_message(
                _CarrierRequest(content=content, request_id=request_id),
                receiver,
            )

        asyncio.create_task(_send())
        return future

    def _resolve_reply(self, reply: _CarrierReply) -> bool:
        """Resolve the future for *reply.request_id*. Returns True on match."""
        future = self._pending.pop(reply.request_id, None)
        if future is not None and not future.done():
            future.set_result(reply.content)
            return True
        return False

    def others(self, participant_id: str) -> list["AgentAddress"]:
        """Return topology neighbours; optionally include own address."""
        neighbors = list(self._parent.context.neighbors())
        if self._include_self:
            neighbors.append(self._parent.context.addr)
        return neighbors

    def get_address(self) -> "AgentAddress":
        return self._parent.context.addr

    async def wait_for(self, awaitable: asyncio.Future) -> Any:
        return await awaitable


# ---------------------------------------------------------------------------
# DistributedOptimizationRole
# ---------------------------------------------------------------------------


class DistributedOptimizationRole(Role):
    """Mango Role hosting a :class:`~..algorithm.core.DistributedAlgorithm`.

    :param algorithm: The distributed algorithm instance.
    :param include_self: Passed to :class:`MangoCarrier`.
    """

    def __init__(
        self,
        algorithm: "DistributedAlgorithm",
        include_self: bool = False,
    ) -> None:
        super().__init__()
        self.algorithm = algorithm
        self._carrier: MangoCarrier | None = None
        self._include_self = include_self

    def setup(self) -> None:
        self._carrier = MangoCarrier(self, self._include_self)
        self.context.subscribe_message(
            self,
            self._handle_optimization,
            lambda c, m: (
                not isinstance(c, (_CarrierReply, StartCoordinatedDistributedOptimization))
            ),
        )
        self.context.subscribe_message(
            self,
            self._handle_reply,
            lambda c, m: isinstance(c, _CarrierReply),
        )

    def _handle_optimization(self, content: Any, meta: dict) -> None:
        from ..algorithm.core import on_exchange_message

        # Unwrap request if needed (from send_awaitable on other side)
        if isinstance(content, _CarrierRequest):
            actual_meta = {**meta, "_request_id": content.request_id}
            actual_content = content.content
        else:
            actual_meta = meta
            actual_content = content

        asyncio.create_task(
            on_exchange_message(self.algorithm, self._carrier, actual_content, actual_meta)
        )

    def _handle_reply(self, content: _CarrierReply, meta: dict) -> None:
        if self._carrier is not None:
            self._carrier._resolve_reply(content)


# ---------------------------------------------------------------------------
# CoordinatorRole
# ---------------------------------------------------------------------------


class CoordinatorRole(Role):
    """Mango Role hosting a :class:`~..algorithm.core.Coordinator`.

    Listens for :class:`StartCoordinatedDistributedOptimization` messages,
    runs the coordinator, then broadcasts :class:`OptimizationFinishedMessage`
    to all topology neighbours.

    :param coordinator: The coordinator object (e.g.
                        :class:`~..admm.core.ADMMGenericCoordinator`).
    :param include_self: Whether to include own address when asking for
                         *others* (needed when the coordinator is also one of
                         the ADMM participants).
    """

    def __init__(
        self,
        coordinator: "Coordinator",
        include_self: bool = False,
    ) -> None:
        super().__init__()
        self.coordinator = coordinator
        self._include_self = include_self
        self._carrier: MangoCarrier | None = None
        self._done_future: asyncio.Future | None = None

    def setup(self) -> None:
        self._carrier = MangoCarrier(self, self._include_self)
        self.context.subscribe_message(
            self,
            self._handle_start,
            lambda c, m: isinstance(c, StartCoordinatedDistributedOptimization),
        )
        self.context.subscribe_message(
            self,
            self._handle_reply,
            lambda c, m: isinstance(c, _CarrierReply),
        )

    def _handle_start(self, content: StartCoordinatedDistributedOptimization, meta: dict) -> None:
        from ..algorithm.core import start_optimization

        loop = asyncio.get_event_loop()
        self._done_future = loop.create_future()

        async def _run() -> None:
            results = await start_optimization(self.coordinator, self._carrier, content.input, meta)
            for i, addr in enumerate(self._carrier.others("coordinator")):
                await self.context.send_message(
                    OptimizationFinishedMessage(result=results[i]), addr
                )
            if not self._done_future.done():
                self._done_future.set_result(results)

        asyncio.create_task(_run())

    def _handle_reply(self, content: _CarrierReply, meta: dict) -> None:
        if self._carrier is not None:
            self._carrier._resolve_reply(content)

    async def wait_done(self) -> Any:
        """Await the completion of the optimization run."""
        if self._done_future is None:
            raise RuntimeError("Optimization has not been started yet.")
        return await self._done_future
