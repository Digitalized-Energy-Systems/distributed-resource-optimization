"""Distributed Resource Optimization.

A Python package implementing distributed optimization algorithms for
resource coordination in energy systems and similar domains.

Algorithms
----------
* **COHDA** — Combinatorial Optimization Heuristic for Distributed Agents.
* **ADMM** — Alternating Direction Method of Multipliers (consensus and
  sharing variants).
* **Averaging Consensus** — distributed price/signal averaging with optional
  gradient correction (incl. economic dispatch).

Carriers
--------
* :class:`~.carrier.simple.SimpleCarrier` — asyncio-backed in-process carrier
  (no network stack required).
* :class:`~.carrier.mango.MangoCarrier` — integrates with *mango-agents* for
  networked multi-agent deployments.

Quick start — COHDA
-------------------
::

    import asyncio
    from distributed_resource_optimization import (
        create_cohda_participant,
        create_cohda_start_message,
        start_distributed_optimization,
    )

    async def main():
        actor1 = create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]])
        actor2 = create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]])
        start = create_cohda_start_message([1.2, 2, 3])
        await start_distributed_optimization([actor1, actor2], start)
        print(actor1.memory.solution_candidate.schedules.sum(axis=0))

    asyncio.run(main())

Quick start — ADMM (sharing)
-----------------------------
::

    import asyncio
    from distributed_resource_optimization import (
        create_admm_flex_actor_one_to_many,
        create_sharing_target_distance_admm_coordinator,
        create_admm_sharing_data,
        create_admm_start,
        start_coordinated_optimization,
    )

    async def main():
        actors = [
            create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
            create_admm_flex_actor_one_to_many(15, [0.1, 0.5, -1.0]),
        ]
        coordinator = create_sharing_target_distance_admm_coordinator()
        start = create_admm_start(create_admm_sharing_data([-4, 0, 6]))
        await start_coordinated_optimization(actors, coordinator, start)

    asyncio.run(main())
"""

# Carrier layer
from .algorithm.admm.consensus_admm import (
    ADMMConsensusGlobalActor,
    create_admm_start_consensus,
    create_consensus_target_reach_admm_coordinator,
)

# ADMM
from .algorithm.admm.core import (
    ADMMAnswer,
    ADMMGenericCoordinator,
    ADMMGlobalActor,
    ADMMGlobalObjective,
    ADMMMessage,
    ADMMStart,
    create_admm_start,
)
from .algorithm.admm.flex_actor import (
    ADMMFlexActor,
    create_admm_flex_actor_one_to_many,
)
from .algorithm.admm.flex_actor import (
    result as admm_flex_result,
)
from .algorithm.admm.sharing_admm import (
    ADMMSharingData,
    ADMMSharingGlobalActor,
    ADMMTargetDistanceObjective,
    create_admm_sharing_data,
    create_sharing_admm_coordinator,
    create_sharing_target_distance_admm_coordinator,
)
from .algorithm.admm.sharing_admm import (
    create_admm_start as create_sharing_admm_start,
)

# Consensus
from .algorithm.consensus.averaging import (
    AveragingConsensusAlgorithm,
    AveragingConsensusMessage,
    ConsensusActor,
    ConsensusFinishedMessage,
    NoConsensusActor,
    create_averaging_consensus_participant,
    create_averaging_consensus_start,
)
from .algorithm.consensus.economic_dispatch import (
    LinearCostEconomicDispatchConsensusActor,
)

# Algorithm base
from .algorithm.core import (
    CoordinatedDistributedAlgorithm,
    Coordinator,
    DistributedAlgorithm,
    OptimizationMessage,
    on_exchange_message,
    start_optimization,
)

# COHDA
from .algorithm.heuristic.cohda.core import (
    COHDAAlgorithmData,
    DefaultLocalDecider,
    LocalDecider,
    ScheduleSelection,
    SolutionCandidate,
    SystemConfig,
    TargetParams,
    WorkingMemory,
    act,
    cohda_default_performance,
    create_cohda_participant,
    create_cohda_participant_with_decider,
    create_cohda_start_message,
    create_from_updated_sysconf,
    decide,
    merge_candidates,
    merge_sysconfigs,
    perceive,
)
from .algorithm.heuristic.cohda.core import (
    result as cohda_result,
)
from .algorithm.heuristic.cohda.decider import LocalSearchDecider
from .carrier.core import Carrier, EventWithValue
from .carrier.simple import (
    ActorContainer,
    SimpleCarrier,
    cid,
    start_coordinated_optimization,
    start_distributed_optimization,
)

# Mango carrier (optional — only imported if mango-agents is available)
_MANGO_AVAILABLE = False
try:
    from .carrier.mango import (
        CoordinatorRole,
        DistributedOptimizationRole,
        MangoCarrier,
        OptimizationFinishedMessage,
        StartCoordinatedDistributedOptimization,
    )

    _MANGO_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

__all__ = [
    # Carrier
    "Carrier",
    "EventWithValue",
    "ActorContainer",
    "SimpleCarrier",
    "cid",
    "start_distributed_optimization",
    "start_coordinated_optimization",
    # Algorithm core
    "DistributedAlgorithm",
    "Coordinator",
    "CoordinatedDistributedAlgorithm",
    "OptimizationMessage",
    "on_exchange_message",
    "start_optimization",
    # COHDA
    "ScheduleSelection",
    "SystemConfig",
    "SolutionCandidate",
    "TargetParams",
    "WorkingMemory",
    "COHDAAlgorithmData",
    "LocalDecider",
    "DefaultLocalDecider",
    "LocalSearchDecider",
    "cohda_default_performance",
    "merge_sysconfigs",
    "merge_candidates",
    "perceive",
    "decide",
    "act",
    "create_from_updated_sysconf",
    "create_cohda_start_message",
    "create_cohda_participant",
    "create_cohda_participant_with_decider",
    "cohda_result",
    # ADMM
    "ADMMStart",
    "ADMMMessage",
    "ADMMAnswer",
    "ADMMGlobalActor",
    "ADMMGlobalObjective",
    "ADMMGenericCoordinator",
    "create_admm_start",
    "ADMMFlexActor",
    "create_admm_flex_actor_one_to_many",
    "admm_flex_result",
    "ADMMConsensusGlobalActor",
    "create_consensus_target_reach_admm_coordinator",
    "create_admm_start_consensus",
    "ADMMSharingData",
    "ADMMSharingGlobalActor",
    "ADMMTargetDistanceObjective",
    "create_admm_sharing_data",
    "create_sharing_admm_start",
    "create_sharing_target_distance_admm_coordinator",
    "create_sharing_admm_coordinator",
    # Consensus
    "ConsensusActor",
    "NoConsensusActor",
    "AveragingConsensusMessage",
    "ConsensusFinishedMessage",
    "AveragingConsensusAlgorithm",
    "create_averaging_consensus_participant",
    "create_averaging_consensus_start",
    "LinearCostEconomicDispatchConsensusActor",
]

if _MANGO_AVAILABLE:
    __all__ += [
        "MangoCarrier",
        "DistributedOptimizationRole",
        "CoordinatorRole",
        "StartCoordinatedDistributedOptimization",
        "OptimizationFinishedMessage",
    ]
