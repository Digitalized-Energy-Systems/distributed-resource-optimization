distributed-resource-optimization
==================================

.. div:: sd-text-center sd-py-4

   **Distributed optimization algorithms for flexible resource coordination**

   Carrier-agnostic implementations of ADMM, COHDA, and averaging consensus —
   run in a single asyncio process or across a network via mango-agents.

   .. grid:: 3
      :class-container: sd-justify-content-center

      .. grid-item::

         .. button-ref:: getting_started
            :ref-type: doc
            :color: primary
            :shadow:

            Get started

      .. grid-item::

         .. button-ref:: tutorials/energy_dispatch
            :ref-type: doc
            :color: secondary
            :outline:

            Tutorials

      .. grid-item::

         .. button-link:: https://github.com/Digitalized-Energy-Systems/mango-optimization
            :color: secondary
            :outline:

            GitHub

   .. code-block:: bash

      pip install distributed-resource-optimization

----

Algorithms
----------

.. grid:: 1 2 2 2
   :gutter: 4

   .. grid-item-card::
      :shadow: sm

      **ADMM Sharing**
      ^^^
      Coordinate flexible resources so their aggregate output matches a target vector.
      Solves a convex QP locally at each agent; a central coordinator handles the global update.

      :doc:`algorithms/admm`

   .. grid-item-card::
      :shadow: sm

      **ADMM Consensus**
      ^^^
      Drive all agents to agree on a shared solution vector while satisfying individual
      box and coupling constraints.

      :doc:`algorithms/admm`

   .. grid-item-card::
      :shadow: sm

      **COHDA**
      ^^^
      Fully distributed heuristic for combinatorial schedule selection.
      Agents gossip solution candidates until the system configuration stabilises.

      :doc:`algorithms/cohda`

   .. grid-item-card::
      :shadow: sm

      **Averaging Consensus**
      ^^^
      Distributed averaging of a parameter vector, with optional gradient corrections
      for economic dispatch and price-signal coordination.

      :doc:`algorithms/consensus`

----

Quick look
----------

.. tab-set::

   .. tab-item:: ADMM Sharing

      .. code-block:: python

         import asyncio
         from distributed_resource_optimization import (
             create_admm_flex_actor_one_to_many,
             create_sharing_target_distance_admm_coordinator,
             create_admm_sharing_data, create_admm_start,
             start_coordinated_optimization,
         )

         async def main():
             flex1 = create_admm_flex_actor_one_to_many(10, [0.1,  0.5, -1.0])
             flex2 = create_admm_flex_actor_one_to_many(15, [0.1,  0.5, -1.0])
             flex3 = create_admm_flex_actor_one_to_many(10, [-1.0, 0.0,  1.0])
             coordinator = create_sharing_target_distance_admm_coordinator()
             start = create_admm_start(create_admm_sharing_data([-4, 0, 6], [5, 1, 1]))
             await start_coordinated_optimization([flex1, flex2, flex3], coordinator, start)
             print(flex1.x, flex2.x, flex3.x)

         asyncio.run(main())

   .. tab-item:: COHDA

      .. code-block:: python

         import asyncio
         from distributed_resource_optimization import (
             create_cohda_participant,
             create_cohda_start_message,
             start_distributed_optimization,
         )

         async def main():
             actor1 = create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]])
             actor2 = create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]])
             start = create_cohda_start_message([1.2, 2.0, 3.0])
             await start_distributed_optimization([actor1, actor2], start)
             print(actor1.memory.solution_candidate.schedules.sum(axis=0))

         asyncio.run(main())

----

Where to go next
----------------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc
      :text-align: center
      :shadow: sm

      Install and run your first optimization in minutes.

   .. grid-item-card:: Tutorials
      :link: tutorials/energy_dispatch
      :link-type: doc
      :text-align: center
      :shadow: sm

      End-to-end worked examples.

   .. grid-item-card:: User guide
      :link: algorithms/admm
      :link-type: doc
      :text-align: center
      :shadow: sm

      Algorithm background and parameter guidance.

   .. grid-item-card:: API Reference
      :link: api_ref/index
      :link-type: doc
      :text-align: center
      :shadow: sm

      Complete reference for every public class and function.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   installation
   getting_started

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials

   tutorials/energy_dispatch
   tutorials/schedule_coordination

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Algorithms

   algorithms/admm
   algorithms/cohda
   algorithms/consensus

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Carriers

   carrier/simple
   carrier/mango

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: How-to guides

   howtos/custom_algorithm
   howtos/custom_carrier

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   api_ref/index
   development


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
