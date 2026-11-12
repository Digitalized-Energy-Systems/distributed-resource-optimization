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
* :func:`next_hops` — routing tables for sparse topologies: with ``routes`` a
  carrier reaches non-neighbours hop by hop (each hop a normal, possibly lossy
  mango message; one mango conversation id end to end), and the receiver sees
  the original sender, so replies route back.

Usage example::

    from mango import create_tcp_container, agent_composed_of, activate, complete_topology
    from distributed_resource_optimization.carrier.mango_carrier import (
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
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import networkx as nx
from mango import AgentAddress, Role, on_message
from mango import sender_addr as mango_sender_addr
from mango.util.scheduling import sleeping_wait

from ..algorithm.core import ControlMessage, Coordinator, DistributedAlgorithm
from .core import Carrier

# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass
class StartCoordinatedDistributedOptimization(ControlMessage):
    """Sent to a :class:`CoordinatorRole` to kick off a coordinated run."""

    input: Any


@dataclass
class OptimizationFinishedMessage(ControlMessage):
    """Broadcast by the coordinator to each participant when the run ends."""

    result: Any


@dataclass
class _Routed:
    """Envelope for a message to a non-neighbour, passed on hop by hop."""

    content: Any
    receiver: AgentAddress
    sender: AgentAddress
    meta: dict = field(default_factory=dict)  # mango tracking_id / reply, restored at the end


def unwrap(content: Any) -> Any:
    """The message inside a routed (multi-hop) envelope, else *content* itself — e.g. for a
    communication simulation that inspects what it delivers."""
    return content.content if isinstance(content, _Routed) else content


def next_hops(graph: nx.Graph) -> dict[Any, dict[Any, Any]]:
    """Per node: destination → next hop on a shortest path (BFS, unweighted).

    Build the graph with agent ids as nodes and pass ``routes=next_hops(g)[aid]``
    to the roles of agent *aid*.
    """
    routes: dict[Any, dict[Any, Any]] = {node: {} for node in graph}
    for dest in graph:
        for parent, node in nx.bfs_edges(graph, dest):
            routes[node][dest] = parent
    return routes


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
    :param routes: destination aid → next-hop aid (see :func:`next_hops`);
                   messages to non-neighbours are relayed along it.
    :param address_all: :meth:`others` returns every routable agent, not only
                        the topology neighbours (used by the coordinator).
    """

    def __init__(
        self,
        parent: Role,
        include_self: bool = False,
        routes: dict[str, str] | None = None,
        address_all: bool = False,
    ) -> None:
        self._parent = parent
        self._include_self = include_self
        self._routes = routes or {}
        self._address_all = address_all

    # ------------------------------------------------------------------
    # Carrier interface
    # ------------------------------------------------------------------

    def _schedule(self, coroutine) -> asyncio.Task:
        """Run *coroutine* via the agent scheduler, not a bare
        :func:`asyncio.create_task`.

        Under a discrete-time simulation the world only advances tasks the
        scheduler knows about; a raw ``create_task`` would run "on a side
        track" off the simulation clock, so message sends would not be
        flushed within a :func:`~mango.step_simulation` convergence step.
        Routing every send through ``schedule_instant_task`` keeps the
        carrier in lock-step with the clock (and is equivalent to a plain
        task under a real, free-running container).
        """
        return self._parent.context.schedule_instant_task(coroutine)

    def send_to_other(
        self,
        content: Any,
        receiver: AgentAddress,
        meta: dict | None = None,
    ) -> asyncio.Task:
        """Send *content* to *receiver* asynchronously (fire-and-forget)."""
        return self._schedule(self._send(content, receiver))

    def reply_to_other(self, content: Any, meta: dict) -> asyncio.Task:
        """Reply to the sender stored in *meta*; echoes mango's ``tracking_id``
        (as ``reply_to`` does) so a :meth:`send_awaitable` future resolves."""
        tracking_id = meta.get("tracking_id")
        extra = {"reply": True, "tracking_id": tracking_id} if tracking_id else {}
        return self._schedule(self._send(content, mango_sender_addr(meta), **extra))

    def send_awaitable(
        self,
        content: Any,
        receiver: AgentAddress,
        meta: dict | None = None,
    ) -> asyncio.Future:
        """Send *content* and return a Future that resolves when the reply arrives.

        Same as mango's ``send_tracked_message`` (tracking_id in meta, handler in
        the agent's transaction table), which itself can only send directly, not
        along ``routes``.
        """
        future: asyncio.Future = asyncio.get_event_loop().create_future()

        def on_reply(reply: Any, _meta: dict) -> None:
            if not future.done():
                future.set_result(reply)

        tracking_id = str(uuid4())
        agent = self._parent.context._bound_agent("send_awaitable")
        agent._transaction_handlers[tracking_id] = (on_reply,)
        self._schedule(self._send(content, receiver, tracking_id=tracking_id))
        return future

    def _next_hop(self, receiver: AgentAddress) -> AgentAddress | None:
        """Neighbour to relay a message for *receiver* through; None = send directly."""
        hop = self._routes.get(receiver.aid)
        if hop is None or hop == receiver.aid:
            return None
        return next(a for a in self._parent.context.neighbors() if a.aid == hop)

    async def _send(self, content: Any, receiver: AgentAddress, **meta: Any) -> Any:
        context = self._parent.context
        hop = self._next_hop(receiver)
        if hop is None:
            return await context.send_message(content, receiver, **meta)
        async with context.open_conversation() as conversation:
            return await conversation.send(hop, _Routed(content, receiver, context.addr, meta))

    def _on_routed(self, envelope: _Routed, meta: dict) -> tuple[Any, dict] | None:
        """Relay *envelope* to the next hop, or, if it is for this agent, return
        ``(content, meta)`` with the original sender in *meta* (None if it was a
        tracked reply, handed to mango's reply dispatch like a direct one)."""
        # shortcut: an agent hosting both a CoordinatorRole and a DistributedOptimizationRole relays
        # a passing envelope twice; give relay duty to one role when such a setup is needed.
        context = self._parent.context
        if envelope.receiver.aid != context.addr.aid:

            async def relay() -> None:
                async with context.join_conversation(meta) as conversation:
                    hop = self._next_hop(envelope.receiver) or envelope.receiver
                    await conversation.send(hop, envelope)

            self._schedule(relay())
            return None
        sender = envelope.sender
        inner_meta = {
            **meta,
            **envelope.meta,
            "sender_addr": sender.protocol_addr,
            "sender_id": sender.aid,
        }
        if context._bound_agent("routed reply")._handle_tracked_reply(envelope.content, inner_meta):
            return None
        return envelope.content, inner_meta

    def others(self, participant_id: str) -> list[Any]:
        """Return topology neighbours (or every routable agent with
        ``address_all``); optionally include own address."""
        neighbors = list(self._parent.context.neighbors())
        if self._address_all:
            known = {a.aid for a in neighbors} | {self.get_address().aid}
            protocol_addr = self.get_address().protocol_addr
            neighbors += [
                AgentAddress(protocol_addr, aid) for aid in self._routes if aid not in known
            ]
        if self._include_self:
            neighbors.append(self._parent.context.addr)
        return neighbors

    def get_address(self) -> AgentAddress:
        return self._parent.context.addr

    # ---- simulation-clock time domain (overrides the wall-clock defaults) ----

    def now(self) -> float:
        """Current *simulation* time (or wall-clock under a real container)."""
        return self._parent.context.current_timestamp

    def sleep(self, seconds: float):
        """Sleep on the agent's clock: under ``run_with_simulation`` this
        registers a discrete-step wakeup at ``now + seconds`` instead of
        burning real wall-clock time."""
        return self._sim_sleep(seconds)

    async def _sim_sleep(self, seconds: float) -> None:
        with sleeping_wait():
            await self._parent.context.scheduler.clock.sleep(seconds)

    def spawn(self, coroutine) -> asyncio.Task:
        """Run the driver as a tracked agent-scheduler task.

        Under discrete stepping the simulation clock tracks the driver;
        its reply/timeout waits go through :meth:`wait_for`, which declares
        them idle so termination detection does not deadlock on the driver.
        Under a real container this is just a normal scheduled task on the
        event loop.
        """
        return self._parent.context.schedule_instant_task(coroutine)

    async def wait_for(self, awaitable) -> Any:
        # The park point: the wait resolves only via an inbound message or
        # the simulation clock, so declare the driver idle for its duration.
        with sleeping_wait():
            return await super().wait_for(awaitable)


# ---------------------------------------------------------------------------
# DistributedOptimizationRole
# ---------------------------------------------------------------------------


class DistributedOptimizationRole(Role):
    """Mango Role hosting a :class:`~..algorithm.core.DistributedAlgorithm`.

    :param algorithm: The distributed algorithm instance.
    :param include_self: Passed to :class:`MangoCarrier`.
    :param routes: Passed to :class:`MangoCarrier` (multi-hop on sparse topologies).
    """

    def __init__(
        self,
        algorithm: "DistributedAlgorithm",
        include_self: bool = False,
        routes: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.algorithm = algorithm
        self._carrier: MangoCarrier | None = None
        self._include_self = include_self
        self._routes = routes

    def setup(self) -> None:
        self._carrier = MangoCarrier(self, self._include_self, self._routes)

    @on_message(_Routed)
    def _handle_routed(self, content: _Routed, meta: dict) -> None:
        delivered = self._carrier._on_routed(content, meta)
        if delivered is not None:
            inner, inner_meta = delivered
            if not isinstance(inner, StartCoordinatedDistributedOptimization):
                self._handle_optimization(inner, inner_meta)

    @on_message(
        object,
        where=lambda self, c, m: (
            not isinstance(c, (_Routed, StartCoordinatedDistributedOptimization))
        ),
    )
    def _handle_optimization(self, content: Any, meta: dict) -> None:
        # Schedule via the agent scheduler so the simulation clock tracks the
        # handler (and any sends it issues); see MangoCarrier._schedule.
        self.context.schedule_instant_task(
            self.algorithm.on_exchange_message(self._carrier, content, meta)
        )


# ---------------------------------------------------------------------------
# CoordinatorRole
# ---------------------------------------------------------------------------


class CoordinatorRole(Role):
    """Mango Role hosting a :class:`~..algorithm.core.Coordinator`.

    Listens for :class:`StartCoordinatedDistributedOptimization` messages,
    runs the coordinator, then sends :class:`OptimizationFinishedMessage` to every
    participant it addressed (topology neighbours, or all routable agents with
    ``routes``): a result list with one entry per participant is split, anything
    else is sent whole to each.
    Other participant messages go to ``coordinator.on_exchange_message`` if the
    coordinator has one (e.g. convergence reports).

    :param coordinator: The coordinator object (e.g.
                        :class:`~..admm.core.ADMMGenericCoordinator`).
    :param include_self: Whether to include own address when asking for
                         *others* (needed when the coordinator is also one of
                         the ADMM participants).
    :param routes: Passed to :class:`MangoCarrier`; with routes the coordinator
                   addresses every routable agent, not only its neighbours.
    """

    def __init__(
        self,
        coordinator: "Coordinator",
        include_self: bool = False,
        routes: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.coordinator = coordinator
        self._include_self = include_self
        self._routes = routes
        self._carrier: MangoCarrier | None = None
        self._done_future: asyncio.Future | None = None

    def setup(self) -> None:
        self._carrier = MangoCarrier(self, self._include_self, self._routes, address_all=True)

    @on_message(StartCoordinatedDistributedOptimization)
    def _handle_start(self, content: StartCoordinatedDistributedOptimization, meta: dict) -> None:
        loop = asyncio.get_event_loop()
        self._done_future = loop.create_future()

        async def _run() -> None:
            results = await self.coordinator.start_optimization(self._carrier, content.input, meta)
            neighbours = self._carrier.others("coordinator")
            # Per-participant list (e.g. ADMMGenericCoordinator) -> fan out the
            # matching slice; a single aggregate result (e.g. CPAdmmResult) ->
            # broadcast the whole object to every participant.
            per_participant = isinstance(results, (list, tuple)) and len(results) == len(neighbours)
            for i, addr in enumerate(neighbours):
                payload = results[i] if per_participant else results
                await self._carrier._send(OptimizationFinishedMessage(result=payload), addr)
            if not self._done_future.done():
                self._done_future.set_result(results)

        # Run the request/reply driver via the carrier (a tracked scheduler
        # task under simulation; see Carrier.spawn). Its reply waits go
        # through carrier.wait_for, so discrete stepping advances it.
        self._carrier.spawn(_run())

    @on_message(_Routed)
    def _handle_routed(self, content: _Routed, meta: dict) -> None:
        delivered = self._carrier._on_routed(content, meta)
        if delivered is not None:
            inner, inner_meta = delivered
            if isinstance(inner, StartCoordinatedDistributedOptimization):
                self._handle_start(inner, inner_meta)
            else:
                self._handle_other(inner, inner_meta)

    @on_message(
        object,
        where=lambda self, c, m: (
            not isinstance(c, (_Routed, StartCoordinatedDistributedOptimization))
        ),
    )
    def _handle_other(self, content: Any, meta: dict) -> None:
        """Forward participant messages to a coordinator that handles them (e.g. reports)."""
        handler = getattr(self.coordinator, "on_exchange_message", None)
        if handler is not None:
            self.context.schedule_instant_task(handler(self._carrier, content, meta))

    async def wait_done(self) -> Any:
        """Await the completion of the optimization run.

        Routed through the carrier's park point so it is safe to call from
        inside a scheduled task under discrete stepping.
        """
        if self._done_future is None:
            raise RuntimeError("Optimization has not been started yet.")
        return await self._carrier.wait_for(self._done_future)
