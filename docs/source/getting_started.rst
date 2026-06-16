Getting Started
===============

This guide shows the most common usage patterns.  All examples use ``asyncio``
because the library is built on Python's async/await model.

Choosing an Algorithm
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 20

   * - Algorithm
     - Problem type
     - Coordination
   * - :doc:`ADMM Sharing <algorithms/admm>`
     - Continuous resource allocation, sum-to-target
     - Central coordinator
   * - :doc:`ADMM Consensus <algorithms/admm>`
     - Agents converge to a shared target vector
     - Central coordinator
   * - :doc:`COHDA <algorithms/cohda>`
     - Combinatorial schedule selection, L1 target
     - Fully distributed
   * - :doc:`Averaging Consensus <algorithms/consensus>`
     - Distributed averaging with gradient terms
     - Fully distributed

Use **ADMM** when participants have continuous, bounded decision variables and a convex
global objective (e.g. match a power target while minimising deviation cost).

Use **COHDA** when each participant picks one schedule from a discrete set and the goal
is to minimise the distance of the combined schedule to a target.

Use **Averaging Consensus** when you need distributed parameter averaging, optionally with
local gradient corrections (e.g. economic dispatch).

Choosing a Carrier
------------------

A *carrier* handles communication between participants.  The algorithm code is identical
regardless of which carrier you choose.

.. list-table::
   :header-rows: 1
   :widths: 25 55

   * - Carrier
     - When to use
   * - Express API (no setup)
     - Quick experiments and single-process simulation.
   * - :class:`~distributed_resource_optimization.SimpleCarrier`
     - Full control over message flow within one asyncio process.
   * - :class:`~distributed_resource_optimization.MangoCarrier`
     - Realistic distributed deployments with TCP networking via mango-agents.


Pattern 1 — Express API
------------------------

The simplest way.  Internally wraps everything in a ``SimpleCarrier``.

**Distributed algorithm (COHDA):**

.. doctest::

   >>> from distributed_resource_optimization import (
   ...     create_cohda_participant,
   ...     create_cohda_start_message,
   ...     start_distributed_optimization,
   ... )
   >>> actor1 = create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]])
   >>> actor2 = create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]])
   >>> start = create_cohda_start_message([1.2, 2.0, 3.0])
   >>> asyncio.run(start_distributed_optimization([actor1, actor2], start))
   >>> actor1.memory.solution_candidate.perf < 0
   True

**Coordinated algorithm (ADMM):**

.. doctest::

   >>> from distributed_resource_optimization import (
   ...     create_admm_flex_actor_one_to_many,
   ...     create_sharing_target_distance_admm_coordinator,
   ...     create_admm_sharing_data, create_admm_start,
   ...     start_coordinated_optimization,
   ... )
   >>> flex1 = create_admm_flex_actor_one_to_many(10, [0.1,  0.5, -1.0])
   >>> flex2 = create_admm_flex_actor_one_to_many(15, [0.1,  0.5, -1.0])
   >>> flex3 = create_admm_flex_actor_one_to_many(10, [-1.0, 0.0,  1.0])
   >>> coordinator = create_sharing_target_distance_admm_coordinator()
   >>> start = create_admm_start(create_admm_sharing_data([-4, 0, 6], [5, 1, 1]))
   >>> asyncio.run(start_coordinated_optimization([flex1, flex2, flex3], coordinator, start))
   [...]
   >>> np.allclose(flex3.x, [-3.983, 0, 3.983], atol=1e-2)
   True


Pattern 2 — SimpleCarrier
--------------------------

Use :class:`~distributed_resource_optimization.SimpleCarrier` when you need direct control:
custom message routing, result inspection, or integration with a larger system.

.. doctest::

   >>> from distributed_resource_optimization import (
   ...     ActorContainer, SimpleCarrier, cid,
   ...     create_cohda_participant, create_cohda_start_message,
   ... )
   >>> async def run_simple_carrier():
   ...     container = ActorContainer()
   ...     c1 = SimpleCarrier(container, create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]]))
   ...     c2 = SimpleCarrier(container, create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]]))
   ...     c1.send_to_other(create_cohda_start_message([1.2, 2.0, 3.0]), cid(c2))
   ...     await container.done_event.wait()
   ...     return c1.actor
   >>> actor = asyncio.run(run_simple_carrier())
   >>> actor.memory.solution_candidate.perf < 0
   True


Pattern 3 — MangoCarrier
-------------------------

For distributed deployments with TCP networking via
`mango-agents <https://github.com/OFFIS-DAI/mango>`_.
See :doc:`carrier/mango` for full details.

.. code-block:: python

   import asyncio
   import mango
   from distributed_resource_optimization import (
       DistributedOptimizationRole,
       CoordinatorRole,
       StartCoordinatedDistributedOptimization,
       create_cohda_participant,
       create_cohda_start_message,
   )

   async def main():
       c = mango.create_tcp_container(addr=("127.0.0.1", 5555))
       a1 = c.register(mango.agent_composed_of(DistributedOptimizationRole(
           create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]]))))
       a2 = c.register(mango.agent_composed_of(DistributedOptimizationRole(
           create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]]))))
       # ... topology setup and message sending omitted for brevity
       # See carrier/mango.rst for a complete example.

   asyncio.run(main())

Next Steps
----------

- :doc:`tutorials/energy_dispatch` — complete ADMM tutorial for energy resources
- :doc:`algorithms/admm` — ADMM mathematical background and parameters
- :doc:`howtos/custom_algorithm` — implement your own algorithm
- :doc:`api_ref/index` — complete API reference
