Tutorial: Energy Dispatch with ADMM
=====================================

This tutorial walks through using ADMM to coordinate flexible energy resources — heat
pumps, batteries, and PV inverters — so that their combined power output matches a
target profile.

Problem Statement
-----------------

An aggregator controls three flexible resources. Each resource converts an input power
into outputs across three sectors (e.g. thermal, electrical, reactive power). The
aggregator wants the combined output to be as close as possible to:

.. code-block:: text

   target = [-4.0, 0.0, 6.0]   # kW per sector

with the first sector weighted five times more heavily (it is a critical load).

Step 1 — Define the Resources
-------------------------------

Each resource is an :class:`~distributed_resource_optimization.ADMMFlexActor`.  The
factory :func:`~distributed_resource_optimization.create_admm_flex_actor_one_to_many`
takes:

- ``in_capacity`` — maximum input power in kW
- ``eta`` — efficiency vector mapping input to outputs (negative = the resource *consumes*
  that output type)
- ``P`` (optional) — priority penalty vector that biases the local solution

.. code-block:: python

   from distributed_resource_optimization import create_admm_flex_actor_one_to_many

   # Resource 1: 10 kW heat pump
   #   eta[0]=0.1  → produces thermal
   #   eta[1]=0.5  → produces electrical
   #   eta[2]=-1.0 → consumes reactive power
   resource1 = create_admm_flex_actor_one_to_many(10.0, [0.1,  0.5, -1.0])

   # Resource 2: 15 kW battery — same efficiency, higher capacity
   resource2 = create_admm_flex_actor_one_to_many(15.0, [0.1,  0.5, -1.0])

   # Resource 3: 10 kW PV inverter — consumes thermal, neutral electrical, produces reactive
   resource3 = create_admm_flex_actor_one_to_many(10.0, [-1.0, 0.0,  1.0])

Step 2 — Create the Coordinator
--------------------------------

The coordinator solves the global z-update each ADMM iteration.  We use the sharing
variant, which minimises the weighted distance of the aggregate output to the target:

.. code-block:: python

   from distributed_resource_optimization import (
       create_sharing_target_distance_admm_coordinator,
   )

   coordinator = create_sharing_target_distance_admm_coordinator()

Step 3 — Set Up the Problem Data
----------------------------------

:func:`~distributed_resource_optimization.create_admm_sharing_data` bundles the target
vector and priority weights:

.. code-block:: python

   from distributed_resource_optimization import create_admm_sharing_data, create_admm_start

   # target = [-4, 0, 6], priorities = [5, 1, 1]
   problem_data = create_admm_sharing_data([-4.0, 0.0, 6.0], [5, 1, 1])
   start_msg    = create_admm_start(problem_data)

The priority weight ``5`` for the first sector means deviations there are penalised five
times more than deviations in the other two sectors.

Step 4 — Run the Optimisation
------------------------------

.. code-block:: python

   import asyncio
   from distributed_resource_optimization import start_coordinated_optimization

   async def main():
       await start_coordinated_optimization(
           [resource1, resource2, resource3],
           coordinator,
           start_msg,
       )

   asyncio.run(main())

Step 5 — Read the Results
--------------------------

After convergence, each resource's optimal output vector is stored in ``actor.x``:

.. code-block:: python

   print("Resource 1:", resource1.x.round(3))
   print("Resource 2:", resource2.x.round(3))
   print("Resource 3:", resource3.x.round(3))
   print("Aggregate: ", (resource1.x + resource2.x + resource3.x).round(3))
   print("Target:    ", [-4.0, 0.0, 6.0])

Complete Script
----------------

.. code-block:: python

   import asyncio
   from distributed_resource_optimization import (
       create_admm_flex_actor_one_to_many,
       create_sharing_target_distance_admm_coordinator,
       create_admm_sharing_data, create_admm_start,
       start_coordinated_optimization,
   )

   async def main():
       resource1 = create_admm_flex_actor_one_to_many(10.0, [0.1,  0.5, -1.0])
       resource2 = create_admm_flex_actor_one_to_many(15.0, [0.1,  0.5, -1.0])
       resource3 = create_admm_flex_actor_one_to_many(10.0, [-1.0, 0.0,  1.0])

       coordinator  = create_sharing_target_distance_admm_coordinator()
       problem_data = create_admm_sharing_data([-4.0, 0.0, 6.0], [5, 1, 1])
       start_msg    = create_admm_start(problem_data)

       await start_coordinated_optimization(
           [resource1, resource2, resource3], coordinator, start_msg
       )

       print("Resource 1:", resource1.x.round(3))
       print("Resource 2:", resource2.x.round(3))
       print("Resource 3:", resource3.x.round(3))
       print("Aggregate: ", (resource1.x + resource2.x + resource3.x).round(3))

   asyncio.run(main())

Tuning Convergence
------------------

If the result is not accurate enough or convergence is slow, build an
:class:`~distributed_resource_optimization.ADMMGenericCoordinator` directly:

.. code-block:: python

   from distributed_resource_optimization import (
       ADMMGenericCoordinator,
       ADMMSharingGlobalActor,
       ADMMTargetDistanceObjective,
   )

   coordinator = ADMMGenericCoordinator(
       global_actor=ADMMSharingGlobalActor(ADMMTargetDistanceObjective()),
       rho=5.0,       # larger rho enforces constraints more aggressively
       max_iters=500,
       abs_tol=1e-5,
       rel_tol=1e-4,
   )

.. tip::

   Setting large priority weights for a sector (e.g. ``[100, 1, 1]``) forces the
   optimizer to match that sector's target as closely as possible, potentially at the
   expense of other sectors. Use this to encode hard priorities in soft constraints.

Next Steps
----------

- Try adding a fourth resource and observe how the aggregate adapts.
- Experiment with different efficiency vectors ``eta`` to model other resource types.
- See :doc:`../algorithms/admm` for the mathematical background.
- See :doc:`../howtos/custom_algorithm` to write your own local model.
