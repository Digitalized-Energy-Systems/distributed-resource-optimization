MangoCarrier
============

:class:`~distributed_resource_optimization.carrier.mango.MangoCarrier` integrates with the
`mango-agents <https://github.com/OFFIS-DAI/mango>`_ framework for networked multi-agent
deployments.  The algorithm code is identical to the SimpleCarrier version — only the
wiring changes.

.. note::

   mango-agents is an optional dependency.  Install it with::

      pip install "distributed-resource-optimization[mango]"

Roles
-----

Two mango ``Role`` classes are provided:

:class:`~distributed_resource_optimization.carrier.mango.DistributedOptimizationRole`
   Wraps a :class:`~distributed_resource_optimization.algorithm.core.DistributedAlgorithm` (e.g. a COHDA
   participant or ADMM flex actor) as a mango role.  Messages are routed to the algorithm's
   :meth:`~distributed_resource_optimization.algorithm.core.DistributedAlgorithm.on_exchange_message`.

:class:`~distributed_resource_optimization.carrier.mango.CoordinatorRole`
   Wraps a :class:`~distributed_resource_optimization.algorithm.core.Coordinator` (e.g. an ADMM
   coordinator).  Listens for a
   :class:`~distributed_resource_optimization.carrier.mango.StartCoordinatedDistributedOptimization`
   message, runs the optimisation, then sends each participant its own result in an
   :class:`~distributed_resource_optimization.carrier.mango.OptimizationFinishedMessage`.  Pass
   ``include_self=True`` only when the coordinator's agent is itself one of the
   participants.

Example — COHDA with MangoCarrier
----------------------------------

.. code-block:: python

   import asyncio
   import mango
   from distributed_resource_optimization import (
       DistributedOptimizationRole,
       create_cohda_participant,
       create_cohda_start_message,
   )

   async def main():
       c = mango.create_tcp_container(addr=("127.0.0.1", 5555))

       actor1 = create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]])
       actor2 = create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]])
       a1 = c.register(mango.agent_composed_of(DistributedOptimizationRole(actor1)))
       a2 = c.register(mango.agent_composed_of(DistributedOptimizationRole(actor2)))

       # carrier.others() returns the topology neighbours — wire a complete graph.
       mango.auto_assign(mango.complete_topology(2), [a1, a2])

       async with mango.activate(c):
           start = create_cohda_start_message([1.2, 2.0, 3.0])
           await a1.send_message(start, a2.addr)
           await asyncio.sleep(0.5)  # COHDA has no completion signal; let it settle

       print(actor1.memory.solution_candidate.schedules.sum(axis=0))

   asyncio.run(main())

Example — ADMM Consensus with MangoCarrier
-------------------------------------------

.. code-block:: python

   import asyncio
   import mango
   from distributed_resource_optimization import (
       DistributedOptimizationRole,
       CoordinatorRole,
       StartCoordinatedDistributedOptimization,
       create_admm_flex_actor_one_to_many,
       create_consensus_target_reach_admm_coordinator,
       create_admm_start_consensus,
   )

   async def main():
       c = mango.create_tcp_container(addr=("127.0.0.1", 5555))

       flex1 = create_admm_flex_actor_one_to_many(10, [0.6, 0.4])
       flex2 = create_admm_flex_actor_one_to_many(10, [0.6, 0.4])
       coordinator = create_consensus_target_reach_admm_coordinator()

       a1 = c.register(mango.agent_composed_of(DistributedOptimizationRole(flex1)))
       a2 = c.register(mango.agent_composed_of(DistributedOptimizationRole(flex2)))
       ca = c.register(mango.agent_composed_of(CoordinatorRole(coordinator)))

       # The coordinator addresses every topology neighbour as a participant.
       mango.auto_assign(mango.complete_topology(3), [a1, a2, ca])

       async with mango.activate(c):
           start = StartCoordinatedDistributedOptimization(
               create_admm_start_consensus([1.5, 1.0])
           )
           await ca.send_message(start, ca.addr)
           await asyncio.sleep(5.0)

       print("Actor 1:", flex1.x.round(3))
       print("Actor 2:", flex2.x.round(3))

   asyncio.run(main())

Topology
--------

The MangoCarrier uses mango's topology system to discover neighbours:
``carrier.others()`` returns the agent's topology neighbours, so without a topology no
messages are exchanged.  Set one up before activating the container — the examples
above use ``mango.auto_assign(mango.complete_topology(n), agents)``.
Refer to the
`mango topology documentation <https://offis-dai.github.io/mango/topology.html>`_ for
details.

See Also
--------

- :class:`~distributed_resource_optimization.carrier.mango.MangoCarrier`,
  :class:`~distributed_resource_optimization.carrier.mango.DistributedOptimizationRole`,
  :class:`~distributed_resource_optimization.carrier.mango.CoordinatorRole`
- :class:`~distributed_resource_optimization.carrier.mango.StartCoordinatedDistributedOptimization`,
  :class:`~distributed_resource_optimization.carrier.mango.OptimizationFinishedMessage`
- :doc:`simple` — lightweight in-process carrier
