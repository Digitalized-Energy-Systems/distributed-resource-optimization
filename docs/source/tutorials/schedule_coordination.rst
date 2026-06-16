Tutorial: Schedule Coordination with COHDA
==========================================

This tutorial shows how to use COHDA to coordinate a group of participants — each
selecting from a discrete set of schedules — so that their combined output is as close
as possible to a target.

Problem Statement
-----------------

Four participants each control one flexible resource. Depending on their local state they
can produce one of three predefined power profiles. The goal is to choose one profile per
participant such that the *sum* of the chosen profiles is as close as possible to:

.. code-block:: text

   target = [3.0, 3.0, 1.0, 2.0]

Step 1 — Define Schedule Sets
------------------------------

.. code-block:: python

   schedules_A = [
       [1.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 0.0],
   ]

   schedules_B = [
       [2.0, 0.0, 0.0, 0.0],
       [0.0, 2.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 2.0],
   ]

Step 2 — Create Participants
------------------------------

Each participant needs a unique 1-indexed ID and its list of feasible schedules:

.. code-block:: python

   from distributed_resource_optimization import create_cohda_participant

   actors = [
       create_cohda_participant(1, schedules_A),
       create_cohda_participant(2, schedules_B),
       create_cohda_participant(3, schedules_A),
       create_cohda_participant(4, schedules_B),
   ]

Step 3 — Run the Optimisation
------------------------------

.. code-block:: python

   import asyncio
   from distributed_resource_optimization import (
       create_cohda_start_message,
       start_distributed_optimization,
   )

   async def main():
       start = create_cohda_start_message([3.0, 3.0, 1.0, 2.0])
       await start_distributed_optimization(actors, start)

       candidate = actors[0].memory.solution_candidate
       print("Performance:", candidate.perf)
       print("Combined:  ", candidate.schedules.sum(axis=0))

   asyncio.run(main())

Step 4 — Inspect Results
--------------------------

After convergence, each participant's ``memory.solution_candidate`` holds the best
complete solution it knows about.  In a fully converged run all participants share the
same candidate:

.. code-block:: python

   for i, actor in enumerate(actors, 1):
       sc = actor.memory.solution_candidate
       print(f"Actor {i} chose: {sc.schedules[i-1].tolist()}")

Custom Performance Functions
-----------------------------

By default COHDA minimises the weighted L1 distance to the target. You can inject a
custom performance function at creation time:

.. code-block:: python

   import numpy as np
   from distributed_resource_optimization import create_cohda_participant

   def l2_performance(cluster_schedule, target_params):
       diff = target_params.schedule - cluster_schedule.sum(axis=0)
       return -float(np.sqrt((diff ** 2).sum()))

   actor = create_cohda_participant(1, schedules_A, l2_performance)

Next Steps
----------

- See :doc:`../algorithms/cohda` for the algorithm's mathematical background.
- Use :class:`~distributed_resource_optimization.LocalSearchDecider` for participants
  with continuous feasible corridors instead of discrete schedule lists.
- See :doc:`../howtos/custom_algorithm` to implement an entirely new algorithm.
