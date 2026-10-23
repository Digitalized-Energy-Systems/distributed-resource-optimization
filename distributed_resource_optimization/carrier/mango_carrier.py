"""MangoCarrier — integrates with the *mango-agents* Python framework.

Provides:

* :class:`MangoCarrier` — a :class:`~distributed_resource_optimization.carrier.core.Carrier`
  backed by a mango-agents ``Role``.
* :class:`DistributedOptimizationRole` — a mango Role that hosts a
  :class:`~distributed_resource_optimization.algorithm.core.DistributedAlgorithm`.
* :class:`CoordinatorRole` — a mango Role that hosts a
  :class:`~distributed_resource_optimization.algorithm.core.Coordinator`.
* :class:`StartCoordinatedDistributedOptimization` /
  :class:`OptimizationFinishedMessage` — coordination message types.

:meth:`MangoCarrier.others` returns the agent's mango topology neighbours, so a
topology must be set up before the container is activated.

All work runs as mango scheduler tasks and all timeouts run on the agent's
scheduler clock, so the carrier behaves the same in real time and in a
stepped :class:`~mango.SimulationWorld` (where the world only waits for
scheduler-tracked tasks before advancing time).  Requests are mango tracked
messages: :meth:`MangoCarrier.request_all` is mango's ``gather`` with a
different message per receiver, and participants answer with mango's
``reply_to``.

Usage example::

    import asyncio
    from mango import activate, agent_composed_of, auto_assign, complete_topology
    from mango import create_tcp_container
    from distributed_resource_optimization.carrier.mango_carrier import (
        CoordinatorRole, DistributedOptimizationRole,
        StartCoordinatedDistributedOptimization,
    )
    from distributed_resource_optimization.algorithm.admm.flex_actor import (
        create_admm_flex_actor_one_to_many,
    )
    from distributed_resource_optimization.algorithm.admm.sharing_admm import (
        create_sharing_target_distance_admm_coordinator, create_sharing_admm_start,
        create_admm_sharing_data,
    )

    async def main():
        container = create_tcp_container(("127.0.0.1", 5555))

        flex_actor = create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0])
        coordinator = create_sharing_target_distance_admm_coordinator()

        dor = DistributedOptimizationRole(flex_actor)
        coord_role = CoordinatorRole(coordinator)  # not a participant itself

        a1 = container.register(agent_composed_of(dor))
        ac = container.register(agent_composed_of(coord_role))

        auto_assign(complete_topology(2), [a1, ac])

        async with activate(container):
            await a1.send_message(
                StartCoordinatedDistributedOptimization(
                    create_sharing_admm_start(create_admm_sharing_data([-4, 0, 6]))
                ),
                ac.addr,
            )
            await asyncio.sleep(0.1)  # let the start handler run first
            await coord_role.wait_done()
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mango import Role
from mango import sender_addr as mango_sender_addr
from mango.util.scheduling import sleeping_wait

from .core import Cancellable, Carrier

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


class RoleLocalMessage:
    """Base for messages meant for *other* roles of an agent that also hosts a
    :class:`DistributedOptimizationRole` (e.g. a request for the finished
    schedule); such messages are never forwarded to the algorithm."""


def _is_orphaned_reply(meta: dict) -> bool:
    """A tracked reply that reached ordinary dispatch.

    mango consumes a reply whose ``tracking_id`` has an open collector before
    any role sees it; one that still arrives here answers a request whose
    :meth:`MangoCarrier.request_all` already timed out (or was duplicated), so
    it must not be handed to the algorithm as a fresh message.
    """
    return bool(meta.get("reply")) and bool(meta.get("tracking_id"))


async def _first_completed(*awaitables: Any) -> None:
    """Return when the first awaitable finishes; cancel the others."""
    tasks = [asyncio.ensure_future(a) for a in awaitables]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()


# ---------------------------------------------------------------------------
# MangoCarrier
# ---------------------------------------------------------------------------


class MangoCarrier(Carrier):
    """A :class:`~distributed_resource_optimization.carrier.core.Carrier` that delegates to a mango-agents Role.

    :param parent: The mango ``Role`` whose context is used for
                   message sending, scheduling and address lookup.
    :param include_self: Whether :meth:`others` should include the agent's own
                         address (mirrors ``CoordinatorRole``'s ``include_self``
                         parameter).
    """

    def __init__(self, parent: Role, include_self: bool = False) -> None:
        self._parent = parent
        self._include_self = include_self

    @property
    def _context(self) -> Any:
        return self._parent.context

    # ------------------------------------------------------------------
    # Carrier interface
    # ------------------------------------------------------------------

    def send_to_other(self, content: Any, receiver: "AgentAddress") -> asyncio.Task:
        """Send *content* to *receiver* asynchronously (fire-and-forget)."""
        return self._context.schedule_instant_message(content, receiver)

    def reply_to_other(self, content: Any, meta: dict) -> asyncio.Task:
        """Reply to the sender stored in *meta*.

        Uses mango's ``reply_to``, which echoes the request's ``tracking_id``
        so a waiting :meth:`request_all` / :meth:`send_awaitable` collects it.
        """
        return self._context.schedule_instant_task(self._context.reply_to(content, meta))

    def send_awaitable(self, content: Any, receiver: "AgentAddress") -> asyncio.Future:
        """Send *content* and return a Future that resolves when the reply arrives.

        Never resolves if the request or reply is lost; prefer
        :meth:`request_all` with a timeout.
        """
        future: asyncio.Future = asyncio.get_running_loop().create_future()

        async def _request() -> None:
            replies = await self.request_all({receiver: content})
            if not future.done() and receiver in replies:
                future.set_result(replies[receiver])

        self._context.schedule_instant_task(_request())
        return future

    def others(self) -> list["AgentAddress"]:
        """Return topology neighbours; optionally include own address."""
        neighbors = list(self._context.neighbors())
        if self._include_self:
            neighbors.append(self._context.addr)
        return neighbors

    # ------------------------------------------------------------------
    # Loss-tolerance helpers
    # ------------------------------------------------------------------

    async def request_all(
        self,
        requests: Mapping[Any, Any],
        *,
        timeout: float | None = None,
        reply_type: type | tuple[type, ...] | None = None,
    ) -> dict[Any, Any]:
        """mango ``gather`` with a different message per receiver.

        The timeout runs on the agent's scheduler clock and the wait is
        marked as sleeping, so a stepped simulation jumps to the deadline
        instead of blocking on it.  Replies that arrive after it are dropped
        by :func:`_is_orphaned_reply`.
        """
        if not requests:
            return {}
        context = self._context
        tracking_id = str(uuid4())
        collector = context.open_gather(tracking_id, expected=len(requests), reply_type=reply_type)
        try:
            for addr, content in requests.items():
                await context.send_message(content, addr, tracking_id=tracking_id)
            with sleeping_wait():
                if timeout is None:
                    await collector.wait()
                else:
                    await _first_completed(collector.wait(), context.context.clock.sleep(timeout))
        finally:
            context.close_gather(tracking_id)
        return dict(collector.responses)

    def schedule_after(self, delay_s: float, callback: Callable[[], Any]) -> Cancellable:
        """Run *callback* after *delay_s* seconds of scheduler-clock time."""

        async def _call() -> None:
            callback()

        return self._context.schedule_timestamp_task(_call(), self.now() + delay_s)

    def now(self) -> float:
        """Current scheduler-clock time (simulation time in a SimulationWorld)."""
        return self._context.current_timestamp

    def address(self) -> "AgentAddress":
        """This agent's mango address."""
        return self._context.addr

    def sender_of(self, meta: Any) -> "AgentAddress":
        """mango address of the agent that sent the message described by *meta*."""
        return mango_sender_addr(meta)

    def sleep(self, seconds: float) -> Any:
        """Sleep on the agent's scheduler clock, marked as sleeping, so a
        stepped simulation jumps to the wakeup instead of blocking on it."""
        return self._sim_sleep(seconds)

    async def _sim_sleep(self, seconds: float) -> None:
        with sleeping_wait():
            await self._context.context.clock.sleep(seconds)

    def spawn(self, coroutine: Any) -> asyncio.Task:
        """Run a long-running driver as a tracked agent-scheduler task.

        Its reply/timeout waits go through :meth:`wait_for`, which declares
        them idle so stepped termination detection does not deadlock on it.
        """
        return self._context.schedule_instant_task(coroutine)

    async def wait_for(self, awaitable: Any) -> Any:
        """The park point: the wait resolves only via an inbound message or
        the clock, so the driver is declared idle for its duration."""
        with sleeping_wait():
            return await super().wait_for(awaitable)


# ---------------------------------------------------------------------------
# Shared role base
# ---------------------------------------------------------------------------


class _CarrierRole(Role):
    """Base for roles that own a :class:`MangoCarrier`.

    :param include_self: Passed to :class:`MangoCarrier`.
    """

    def __init__(self, include_self: bool = False) -> None:
        super().__init__()
        self._include_self = include_self
        self._carrier: MangoCarrier | None = None

    def setup(self) -> None:
        """Create the carrier."""
        self._carrier = MangoCarrier(self, self._include_self)


# ---------------------------------------------------------------------------
# DistributedOptimizationRole
# ---------------------------------------------------------------------------


class DistributedOptimizationRole(_CarrierRole):
    """Mango Role hosting a :class:`~distributed_resource_optimization.algorithm.core.DistributedAlgorithm`.

    Each message is handled in its own scheduler task.  Algorithms must not
    ``await`` between reading and writing their own state, so concurrent
    handler tasks cannot interleave inside one update.

    :param algorithm: The distributed algorithm instance.
    :param include_self: Passed to :class:`MangoCarrier`.
    """

    def __init__(
        self,
        algorithm: "DistributedAlgorithm",
        include_self: bool = False,
    ) -> None:
        super().__init__(include_self)
        self.algorithm = algorithm

    def setup(self) -> None:
        """Create the carrier and subscribe the optimization handler."""
        super().setup()
        self.context.subscribe_message(
            self,
            self._handle_optimization,
            lambda c, m: (
                not isinstance(
                    c,
                    (
                        StartCoordinatedDistributedOptimization,
                        OptimizationFinishedMessage,
                        RoleLocalMessage,
                    ),
                )
                and not _is_orphaned_reply(m)
            ),
        )

    def start(self, start_message: Any) -> None:
        """Hand a kick-off message to the local algorithm without the network.

        For callers on the same agent (e.g. a role that received an
        acknowledged start request), so the kick-off cannot be lost.
        """
        self.context.schedule_instant_task(
            self.algorithm.on_exchange_message(self._carrier, start_message, {})
        )

    def _handle_optimization(self, content: Any, meta: dict) -> None:
        """Forward the payload to the algorithm in a scheduler task."""
        self.context.schedule_instant_task(
            self.algorithm.on_exchange_message(self._carrier, content, meta)
        )


# ---------------------------------------------------------------------------
# CoordinatorRole
# ---------------------------------------------------------------------------


class CoordinatorRole(_CarrierRole):
    """Mango Role hosting a :class:`~distributed_resource_optimization.algorithm.core.Coordinator`.

    Listens for :class:`StartCoordinatedDistributedOptimization` messages,
    runs the coordinator, then sends each participant (``others()``) its own
    entry of the result list in an :class:`OptimizationFinishedMessage` (or
    the whole result, if it is not one entry per participant).

    :param coordinator: The coordinator object (e.g.
                        :class:`~distributed_resource_optimization.algorithm.admm.core.ADMMGenericCoordinator`).
    :param include_self: Whether to include own address when asking for
                         *others* (needed when the coordinator is also one of
                         the ADMM participants).
    """

    def __init__(
        self,
        coordinator: "Coordinator",
        include_self: bool = False,
    ) -> None:
        super().__init__(include_self)
        self.coordinator = coordinator
        self._done_future: asyncio.Future | None = None

    def setup(self) -> None:
        """Create the carrier and subscribe the start handler."""
        super().setup()
        self.context.subscribe_message(
            self,
            self._handle_start,
            lambda c, m: isinstance(c, StartCoordinatedDistributedOptimization),
        )

    def start(self, start_input: Any) -> asyncio.Future:
        """Start a coordinated run from the same agent, without a message.

        Equivalent to receiving ``StartCoordinatedDistributedOptimization``
        with *start_input*, but cannot be lost in transit.

        :returns: The run's done future (the coordinator's result list, in
            ``others()`` order).  Use ``add_done_callback`` rather than
            awaiting it inside a mango task, which a stepped simulation
            cannot see as waiting.
        """
        self._handle_start(StartCoordinatedDistributedOptimization(input=start_input), {})
        return self._done_future

    def _handle_start(self, content: StartCoordinatedDistributedOptimization, meta: dict) -> None:
        """Kick off a coordinated run and broadcast the per-participant results.

        The result list returned by the coordinator is distributed positionally
        to ``others()`` — both sides rely on mango's topology ordering being
        stable between the run and the broadcast.  A second start message
        replaces :attr:`_done_future`; waiters of the previous run are only
        served if that run already finished.
        """
        loop = asyncio.get_event_loop()
        done_future = loop.create_future()
        self._done_future = done_future

        async def _run() -> None:
            try:
                results = await self.coordinator.start_optimization(
                    self._carrier, content.input, meta
                )
                neighbours = self._carrier.others()
                # Per-participant list (e.g. ADMMGenericCoordinator) -> fan out
                # the matching entry; a single aggregate result (e.g.
                # CPAdmmResult) -> broadcast the whole object to everyone.
                per_participant = isinstance(results, (list, tuple)) and len(results) == len(
                    neighbours
                )
                for i, addr in enumerate(neighbours):
                    payload = results[i] if per_participant else results
                    await self.context.send_message(
                        OptimizationFinishedMessage(result=payload), addr
                    )
            except Exception as exc:  # propagate to wait_done() instead of hanging it
                if not done_future.done():
                    done_future.set_exception(exc)
                return
            if not done_future.done():
                done_future.set_result(results)

        self._carrier.spawn(_run())

    async def wait_done(self) -> Any:
        """Await the completion of the optimization run.

        Routed through the carrier's park point so it is safe to call from
        inside a scheduled task under stepped simulation.
        """
        if self._done_future is None:
            raise RuntimeError("Optimization has not been started yet.")
        return await self._carrier.wait_for(self._done_future)
