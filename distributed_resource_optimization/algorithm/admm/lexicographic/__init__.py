"""Lexicographic-cascade sum-sharing ADMM — two coordination transports.

Both transports run the same deterministic cascade kernel (the
closed-form ``(z, sigma)`` cell update in :mod:`.kernel`); they differ
only in how the shared ``x_bar`` is assembled each iteration:

* :mod:`.coordinator` — a coordinator gathers each CP's contribution
  and runs the replicated kernel centrally.
* :mod:`.gossip` — coordinator-free; peers rebuild ``x_bar`` from
  broadcast contributions, crash-fault tolerant.
"""

from .coordinator import (
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
from .gossip import (
    GossipCascadeInit,
    GossipCascadeStart,
    GossipIter,
    GossipParticipant,
    create_gossip_cascade_participant,
    create_gossip_cascade_start,
)

__all__ = [
    # Coordinator transport
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
    # Gossip transport
    "GossipCascadeStart",
    "GossipCascadeInit",
    "GossipIter",
    "GossipParticipant",
    "create_gossip_cascade_participant",
    "create_gossip_cascade_start",
]
