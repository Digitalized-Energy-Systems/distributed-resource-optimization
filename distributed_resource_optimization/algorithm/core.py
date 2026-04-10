"""Abstract base types for distributed algorithms and coordinators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..carrier.core import Carrier


class OptimizationMessage(ABC):
    """Marker supertype for all optimization-protocol messages."""


class DistributedAlgorithm(ABC):
    """Base class for all distributed optimization algorithms.

    Concrete subclasses must implement :meth:`on_exchange_message`.
    """

    @abstractmethod
    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: Any,
        meta: Any,
    ) -> Any:
        """Handle an incoming message from another participant.

        :param carrier: The carrier that delivered the message.
        :param message_data: The message payload.
        :param meta: Transport-level metadata (sender address, IDs, …).
        """


class Coordinator(ABC):
    """Base class for optimization coordinators."""

    @abstractmethod
    async def start_optimization(
        self,
        carrier: "Carrier",
        message_data: Any,
        meta: Any,
    ) -> Any:
        """Initiate and run a complete coordinated optimization round.

        :param carrier: The carrier the coordinator uses to reach participants.
        :param message_data: Start payload (algorithm-specific).
        :param meta: Transport metadata from the triggering message.
        :returns: The final result (algorithm-specific).
        """


# ---------------------------------------------------------------------------
# Module-level shim functions — preserved for compatibility with carrier code
# ---------------------------------------------------------------------------

async def on_exchange_message(
    algorithm: DistributedAlgorithm,
    carrier: "Carrier",
    message_data: Any,
    meta: Any,
) -> Any:
    """Delegate to ``algorithm.on_exchange_message(carrier, message_data, meta)``."""
    return await algorithm.on_exchange_message(carrier, message_data, meta)


async def start_optimization(
    coordinator: Coordinator,
    carrier: "Carrier",
    message_data: Any,
    meta: Any,
) -> Any:
    """Delegate to ``coordinator.start_optimization(carrier, message_data, meta)``."""
    return await coordinator.start_optimization(carrier, message_data, meta)


# ---------------------------------------------------------------------------
# CoordinatedDistributedAlgorithm
# ---------------------------------------------------------------------------

class CoordinatedDistributedAlgorithm:
    """Bundle of a coordinator and its worker algorithms (informational only)."""

    def __init__(
        self,
        distributed_algo: list[DistributedAlgorithm],
        coordinator: Coordinator,
    ) -> None:
        self.distributed_algo = distributed_algo
        self.coordinator = coordinator
