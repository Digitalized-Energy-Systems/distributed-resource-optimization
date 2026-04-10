COHDA
=====

**COHDA** (Combinatorial Optimization Heuristic for Distributed Agents) is a fully
distributed heuristic for the Multiple-Choice Combinatorial Optimization Problem (MC-COP).
It requires no central coordinator — participants exchange *solution candidates* with their
neighbours and converge through local search.

Problem Formulation
-------------------

Each of the :math:`N` participants independently selects exactly one schedule from a
personal menu of :math:`M` choices. The goal is to minimise the weighted L1 distance of
the combined schedule sum to a target vector :math:`T`:

.. math::

   \max_{x_{i,j}} \left(-\bigl\lVert T - \sum_{i=1}^{N}\sum_{j=1}^{M} U_{i,j}\cdot x_{i,j}\bigr\rVert_1\right)

   \text{s.t.}\quad \sum_{j=1}^{M} x_{i,j} = 1 \;\forall\, i,\quad x_{i,j}\in\{0,1\}

where :math:`U_{i,j} \in \mathbb{R}^m` is the :math:`j`-th schedule of participant :math:`i`.

How It Works
------------

COHDA is a gossip-style algorithm:

1. A *start message* carrying the target :math:`T` is sent to one participant.
2. On receiving a message, each participant updates its *working memory* — a view of the
   current best known system configuration — and applies local search to improve its schedule.
3. If the working memory changed, the participant broadcasts an updated *solution candidate*
   to all its neighbours.
4. The algorithm terminates when no participant can improve the objective any further (the
   system configuration stabilises).

Usage
-----

.. code-block:: python

   import asyncio
   from distributed_resource_optimization import (
       create_cohda_participant,
       create_cohda_start_message,
       start_distributed_optimization,
   )

   async def main():
       # Participant 1: two schedule choices, each 3-dimensional
       actor1 = create_cohda_participant(1, [[0.0, 1.0, 2.0],
                                             [1.0, 2.0, 3.0]])
       actor2 = create_cohda_participant(2, [[0.0, 1.0, 2.0],
                                             [1.0, 2.0, 3.0]])

       start = create_cohda_start_message([1.2, 2.0, 3.0])
       await start_distributed_optimization([actor1, actor2], start)

       # Inspect result
       sched = actor1.memory.solution_candidate.schedules
       print("Combined schedule:", sched.sum(axis=0))

   asyncio.run(main())

Complete Example
----------------

.. code-block:: python

   import asyncio
   from distributed_resource_optimization import (
       create_cohda_participant,
       create_cohda_start_message,
       start_distributed_optimization,
   )

   async def main():
       schedules_A = [[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]]

       schedules_B = [[2.0, 0.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 2.0]]

       actors = [
           create_cohda_participant(1, schedules_A),
           create_cohda_participant(2, schedules_B),
           create_cohda_participant(3, schedules_A),
           create_cohda_participant(4, schedules_B),
       ]

       start = create_cohda_start_message([3.0, 3.0, 1.0, 2.0])
       await start_distributed_optimization(actors, start)

   asyncio.run(main())

Working Memory and Solution Candidates
---------------------------------------

Internally COHDA uses three data structures:

- :class:`~distributed_resource_optimization.WorkingMemory` — each participant's current
  view of the world: target parameters, the best known system configuration, and the best
  known solution candidate.
- :class:`~distributed_resource_optimization.SystemConfig` — a mapping from participant IDs
  to their currently selected schedule.
- :class:`~distributed_resource_optimization.SolutionCandidate` — a proposed full system
  configuration together with its performance value (negative weighted L1 distance).

These types are exported and can be inspected after the algorithm runs.

Custom Performance Functions
-----------------------------

By default COHDA uses a weighted L1 distance metric. Provide a custom performance function
when creating a participant. The function receives a 2-D NumPy array (rows = participants,
columns = time steps) and a :class:`~distributed_resource_optimization.TargetParams`
instance, and must return a ``float``:

.. code-block:: python

   import numpy as np
   from distributed_resource_optimization import create_cohda_participant

   def my_perf(cluster_schedule, target_params):
       diff = target_params.schedule - cluster_schedule.sum(axis=0)
       return -float(np.sqrt((diff ** 2).sum()))  # L2 distance

   actor = create_cohda_participant(1, [[1.0, 0.0], [0.0, 1.0]], my_perf)

Local Search Decider
--------------------

For participants with *continuous* feasible sets (corridors) rather than discrete schedule
lists, use :class:`~distributed_resource_optimization.LocalSearchDecider`:

.. code-block:: python

   import numpy as np
   from distributed_resource_optimization import (
       LocalSearchDecider,
       create_cohda_participant_with_decider,
   )

   decider = LocalSearchDecider(
       initial_schedule=np.array([1.0, 1.0]),
       corridors=[(0.0, 5.0), (0.0, 5.0)],
       local_performance=lambda _: 0.0,
       convergence_force_factor=0.1,
       max_iterations=20,
       sample_size_per_value=20,
   )
   actor = create_cohda_participant_with_decider(1, decider)

.. note::

   COHDA is a heuristic and does not guarantee a globally optimal solution.  Quality
   typically improves with more participants and more schedule choices.  Convergence is
   detected when the system configuration stabilises across all participants.

See Also
--------

- :func:`~distributed_resource_optimization.create_cohda_participant`,
  :func:`~distributed_resource_optimization.create_cohda_start_message`
- :class:`~distributed_resource_optimization.WorkingMemory`,
  :class:`~distributed_resource_optimization.SolutionCandidate`,
  :class:`~distributed_resource_optimization.SystemConfig`
- :class:`~distributed_resource_optimization.LocalSearchDecider`
- :doc:`../tutorials/schedule_coordination`
