"""Distributed Resource Optimization.

A Python package implementing distributed optimization algorithms for
resource coordination in energy systems and similar domains.

Algorithms
----------
* **COHDA** — Combinatorial Optimization Heuristic for Distributed Agents.
* **ADMM** — Alternating Direction Method of Multipliers (consensus/exchange
  and sharing variants).
* **Averaging Consensus** — distributed price/signal averaging with optional
  gradient correction (incl. economic dispatch).
* **Diffusion** — distributed adapt-then-combine diffusion over a scheduling
  horizon (incl. economic dispatch and reservoir storage).
* **Exact Diffusion** — adapt-correct-combine diffusion (Ces et al. 2025;
  Yuan, Ying, Zhao & Sayed 2019) removing classical Diffusion's steady-state bias.
* **FDGDM** — Fast Distributed Gradient Descent Method for economic dispatch
  (Bai et al. 2022), using per-participant gradient/curvature bounds.
* **DEED-ADMM** — Distributed Economic Dispatch via ADMM with dynamic
  consensus (Zhu et al. 2025), incl. a storage-aware variant.

Carriers
--------
* :class:`~.carrier.simple.SimpleCarrier` — asyncio-backed in-process carrier
  (no network stack required).
* :class:`~.carrier.mango_carrier.MangoCarrier` — integrates with *mango-agents* for
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
        create_sharing_admm_start,
        start_coordinated_optimization,
    )

    async def main():
        actors = [
            create_admm_flex_actor_one_to_many(10, [0.1, 0.5, -1.0]),
            create_admm_flex_actor_one_to_many(15, [0.1, 0.5, -1.0]),
        ]
        coordinator = create_sharing_target_distance_admm_coordinator()
        start = create_sharing_admm_start(create_admm_sharing_data([-4, 0, 6]))
        await start_coordinated_optimization(actors, coordinator, start)

    asyncio.run(main())
"""

# ADMM
from .algorithm.admm.consensus_admm import (
    ADMMConsensusGlobalActor,
    create_admm_start_consensus,
    create_consensus_target_reach_admm_coordinator,
)
from .algorithm.admm.core import (
    ADMMAnswer,
    ADMMGenericCoordinator,
    ADMMGlobalActor,
    ADMMMessage,
    ADMMStart,
)
from .algorithm.admm.flex_actor import (
    ADMMFlexActor,
    create_admm_flex_actor_box_bounded,
    create_admm_flex_actor_one_to_many,
)

# Distributed lexicographic cascade (true-distributed sum-sharing ADMM)
from .algorithm.admm.lexicographic.coordinator import (
    DistributedLexicographicCascadeCoordinator,
    DistributedLexicographicCascadeDone,
    DistributedLexicographicCascadeDoneReply,
    DistributedLexicographicCascadeInit,
    DistributedLexicographicCascadeInitAck,
    DistributedLexicographicCascadeParticipant,
    DistributedLexicographicCascadeStart,
    LexicographicCascadeGlobalActor,
    create_distributed_lexicographic_cascade_coordinator,
    create_distributed_lexicographic_cascade_participant,
    create_distributed_lexicographic_cascade_start,
    solve_cp_distributed_lexicographic_cascade,
)

# Gossip lexicographic cascade (coordinator-free peer-to-peer variant)
from .algorithm.admm.lexicographic.gossip import (
    GossipCascadeInit,
    GossipCascadeStart,
    GossipIter,
    GossipParticipant,
    create_gossip_cascade_participant,
    create_gossip_cascade_start,
)
from .algorithm.admm.sharing_admm import (
    ADMMSharingData,
    ADMMSharingGlobalActor,
    create_admm_sharing_data,
    create_sharing_admm_start,
    create_sharing_target_distance_admm_coordinator,
)
from .algorithm.admm.storage import (
    ProximalStorageADMMActor,
    create_admm_proximal_storage_actor,
)

# Priority-cascade ADMM shared data contracts
from .algorithm.admm.types import (
    CPAdmmResult,
    CPSpec,
    SectorDemand,
)

# Waterfall ADMM (priority-cascaded sharing ADMM)
from .algorithm.admm.waterfall.core import (
    WaterfallADMMCoordinator,
    WaterfallADMMParticipant,
    WaterfallADMMResult,
    WaterfallADMMSpecReply,
    WaterfallADMMSpecRequest,
    WaterfallADMMStart,
    create_waterfall_admm_coordinator,
    create_waterfall_admm_participant,
    create_waterfall_admm_start,
    marginal_priority,
    solve_cp_priority_admm,
    tier_priority_weight,
    waterfall_serve,
)

# Algorithm base
from .algorithm.core import (
    Coordinator,
    DistributedAlgorithm,
    OptimizationMessage,
)

# DEED-ADMM
from .algorithm.deed_admm import (
    DEEDADMMAlgorithm,
    DEEDADMMMessage,
    DEEDADMMStorageAlgorithm,
    create_deed_admm_renewable_participant,
    create_deed_admm_storage_participant,
    create_deed_admm_thermal_participant,
)

# FDGDM
from .algorithm.fdgdm.economic_dispatch import (
    LinearCostEconomicDispatchFDGDMActor,
    ReservoirStorageFDGDMActor,
)
from .algorithm.fdgdm.fdgdm import (
    FDGDMActor,
    FDGDMAlgorithm,
    FDGDMMessage,
    create_fdgdm_participant,
    create_fdgdm_start,
)

# Consensus
from .algorithm.firstorder.consensus.averaging import (
    AveragingConsensusAlgorithm,
    AveragingConsensusMessage,
    ConsensusActor,
    create_averaging_consensus_participant,
    create_averaging_consensus_start,
)
from .algorithm.firstorder.consensus.economic_dispatch import (
    LinearCostEconomicDispatchConsensusActor,
    ReservoirStorageConsensusActor,
)

# Diffusion
from .algorithm.firstorder.diffusion.diffusion import (
    DiffusionActor,
    DiffusionAlgorithm,
    DiffusionMessage,
    create_diffusion_participant,
    create_diffusion_start,
)
from .algorithm.firstorder.diffusion.economic_dispatch import (
    LinearCostEconomicDispatchDiffusionActor,
    ReservoirStorageDiffusionActor,
)
from .algorithm.firstorder.diffusion.exact_diffusion import (
    ExactDiffusionAlgorithm,
    create_exact_diffusion_participant,
    create_exact_diffusion_start,
)

# Distributed QP (gossip primal-dual)
from .algorithm.firstorder.gossip_qp.core import (
    GossipQPAlgorithm,
    GossipQPFinished,
    GossipQPMessage,
    LedgerEntry,
    create_distributed_qp_participant,
    create_distributed_qp_start,
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
    publish_own_schedule,
)
from .algorithm.heuristic.cohda.decider import LocalSearchDecider

# Carrier layer
from .carrier.core import Carrier
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
    from .carrier.mango_carrier import (
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
    "ActorContainer",
    "SimpleCarrier",
    "cid",
    "start_distributed_optimization",
    "start_coordinated_optimization",
    # Algorithm core
    "DistributedAlgorithm",
    "Coordinator",
    "OptimizationMessage",
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
    "publish_own_schedule",
    "decide",
    "act",
    "create_from_updated_sysconf",
    "create_cohda_start_message",
    "create_cohda_participant",
    "create_cohda_participant_with_decider",
    # ADMM
    "ADMMStart",
    "ADMMMessage",
    "ADMMAnswer",
    "ADMMGlobalActor",
    "ADMMGenericCoordinator",
    "ADMMFlexActor",
    "create_admm_flex_actor_one_to_many",
    "create_admm_flex_actor_box_bounded",
    "ProximalStorageADMMActor",
    "create_admm_proximal_storage_actor",
    "ADMMConsensusGlobalActor",
    "create_consensus_target_reach_admm_coordinator",
    "create_admm_start_consensus",
    "ADMMSharingData",
    "ADMMSharingGlobalActor",
    "create_admm_sharing_data",
    "create_sharing_admm_start",
    "create_sharing_target_distance_admm_coordinator",
    # Consensus
    "ConsensusActor",
    "AveragingConsensusMessage",
    "AveragingConsensusAlgorithm",
    "create_averaging_consensus_participant",
    "create_averaging_consensus_start",
    "LinearCostEconomicDispatchConsensusActor",
    "ReservoirStorageConsensusActor",
    # FDGDM
    "FDGDMActor",
    "FDGDMMessage",
    "FDGDMAlgorithm",
    "create_fdgdm_participant",
    "create_fdgdm_start",
    "LinearCostEconomicDispatchFDGDMActor",
    "ReservoirStorageFDGDMActor",
    # DEED-ADMM
    "DEEDADMMAlgorithm",
    "DEEDADMMStorageAlgorithm",
    "DEEDADMMMessage",
    "create_deed_admm_thermal_participant",
    "create_deed_admm_renewable_participant",
    "create_deed_admm_storage_participant",
    # Diffusion
    "DiffusionActor",
    "DiffusionMessage",
    "DiffusionAlgorithm",
    "create_diffusion_participant",
    "create_diffusion_start",
    "ExactDiffusionAlgorithm",
    "create_exact_diffusion_participant",
    "create_exact_diffusion_start",
    "LinearCostEconomicDispatchDiffusionActor",
    "ReservoirStorageDiffusionActor",
    # Distributed QP (gossip primal-dual)
    "LedgerEntry",
    "GossipQPMessage",
    "GossipQPFinished",
    "GossipQPAlgorithm",
    "create_distributed_qp_participant",
    "create_distributed_qp_start",
    # Waterfall ADMM
    "CPSpec",
    "SectorDemand",
    "CPAdmmResult",
    "WaterfallADMMStart",
    "WaterfallADMMSpecRequest",
    "WaterfallADMMSpecReply",
    "WaterfallADMMResult",
    "WaterfallADMMParticipant",
    "WaterfallADMMCoordinator",
    "create_waterfall_admm_participant",
    "create_waterfall_admm_coordinator",
    "create_waterfall_admm_start",
    "solve_cp_priority_admm",
    "waterfall_serve",
    "marginal_priority",
    "tier_priority_weight",
    # Distributed lexicographic cascade
    "DistributedLexicographicCascadeStart",
    "DistributedLexicographicCascadeInit",
    "DistributedLexicographicCascadeInitAck",
    "DistributedLexicographicCascadeDone",
    "DistributedLexicographicCascadeDoneReply",
    "DistributedLexicographicCascadeParticipant",
    "DistributedLexicographicCascadeCoordinator",
    "LexicographicCascadeGlobalActor",
    "create_distributed_lexicographic_cascade_participant",
    "create_distributed_lexicographic_cascade_coordinator",
    "create_distributed_lexicographic_cascade_start",
    "solve_cp_distributed_lexicographic_cascade",
    # Gossip lexicographic cascade
    "GossipCascadeStart",
    "GossipCascadeInit",
    "GossipIter",
    "GossipParticipant",
    "create_gossip_cascade_participant",
    "create_gossip_cascade_start",
]

if _MANGO_AVAILABLE:
    __all__ += [
        "MangoCarrier",
        "DistributedOptimizationRole",
        "CoordinatorRole",
        "StartCoordinatedDistributedOptimization",
        "OptimizationFinishedMessage",
    ]
