MangoCarrier
============

:class:`~distributed_resource_optimization.MangoCarrier` integrates with the
`mango-agents <https://github.com/OFFIS-DAI/mango>`_ framework for networked multi-agent
deployments.  The algorithm code is identical to the SimpleCarrier version — only the
wiring changes.

.. note::

   mango-agents is an optional dependency.  Install it with::

      pip install "distributed-resource-optimization[mango]"

Roles
-----

Two mango ``Role`` classes are provided:

:class:`~distributed_resource_optimization.DistributedOptimizationRole`
   Wraps a :class:`~distributed_resource_optimization.DistributedAlgorithm` (e.g. a COHDA
   participant or ADMM flex actor) as a mango role.  Messages are routed to the algorithm's
   :meth:`~distributed_resource_optimization.DistributedAlgorithm.on_exchange_message`.

:class:`~distributed_resource_optimization.CoordinatorRole`
   Wraps a :class:`~distributed_resource_optimization.Coordinator` (e.g. an ADMM
   coordinator).  Listens for a
   :class:`~distributed_resource_optimization.StartCoordinatedDistributedOptimization`
   message, runs the optimisation, then broadcasts
   :class:`~distributed_resource_optimization.OptimizationFinishedMessage` to all peers.

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

       a1 = c.register(mango.agent_composed_of(
           DistributedOptimizationRole(
               create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]]))))
       a2 = c.register(mango.agent_composed_of(
           DistributedOptimizationRole(
               create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]]))))

       async with mango.activate(c):
           start = create_cohda_start_message([1.2, 2.0, 3.0])
           await a1.send_message(start, a2.addr)
           await asyncio.sleep(0.5)

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
       OptimizationFinishedMessage,
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

       async with mango.activate(c):
           start = StartCoordinatedDistributedOptimization(
               create_admm_start_consensus([1.0, 2.0])
           )
           await ca.send_message(start, ca.addr)
           await asyncio.sleep(5.0)

       print("Actor 1:", flex1.x.round(3))
       print("Actor 2:", flex2.x.round(3))

   asyncio.run(main())

Topology
--------

The MangoCarrier uses mango's topology system to discover neighbours.  Set up a topology
before activating the container so that each role knows which agents to communicate with.
Refer to the
`mango topology documentation <https://offis-dai.github.io/mango/topology.html>`_ for
details.

See Also
--------

- :class:`~distributed_resource_optimization.MangoCarrier`,
  :class:`~distributed_resource_optimization.DistributedOptimizationRole`,
  :class:`~distributed_resource_optimization.CoordinatorRole`
- :class:`~distributed_resource_optimization.StartCoordinatedDistributedOptimization`,
  :class:`~distributed_resource_optimization.OptimizationFinishedMessage`
- :doc:`simple` — lightweight in-process carrier
