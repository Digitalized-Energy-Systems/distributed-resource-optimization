ADMM
====

The Alternating Direction Method of Multipliers (ADMM) is a convex optimization algorithm
that decomposes a large problem into smaller subproblems solved locally, coordinated by a
central update. Two ADMM variants are implemented: **Sharing** and **Consensus**.

Problem Forms
-------------

Sharing
~~~~~~~

The sharing form distributes a resource across :math:`N` agents whose individual
contributions must sum to a global variable :math:`z`:

.. math::

   \min_{\{x_i\},\,z}\; \sum_{i=1}^N f_i(x_i) + g(z)
   \quad\text{s.t.}\quad \sum_{i=1}^N x_i = z

The ADMM iterations with dual variable :math:`u` and penalty :math:`\rho` are:

.. math::

   x_i^{k+1} &= \arg\min_{x_i}\; f_i(x_i) + \tfrac{\rho}{2}\,\bigl\lVert x_i - (z^k - u^k) \bigr\rVert_2^2 \\
   \bar{x}^{\,k+1} &= \tfrac{1}{N}\sum_{i=1}^N x_i^{k+1} \\
   z^{k+1} &= \arg\min_{z}\; g(Nz) + \tfrac{N\rho}{2}\,\bigl\lVert z - \bar{x}^{\,k+1} - u^k \bigr\rVert_2^2 \\
   u^{k+1} &= u^k + \bar{x}^{\,k+1} - z^{k+1}

Use :func:`~distributed_resource_optimization.create_sharing_target_distance_admm_coordinator`
and :func:`~distributed_resource_optimization.create_admm_start`.

Consensus
~~~~~~~~~

The consensus form drives all agents to agree on a single global value :math:`z`:

.. math::

   \min_{\{x_i\},\,z}\; \sum_{i=1}^N f_i(x_i)
   \quad\text{s.t.}\quad x_i = z,\; i=1,\dots,N

The update iterations are:

.. math::

   x_i^{k+1} &= \arg\min_{x_i}\; f_i(x_i) + \frac{\rho}{2}\bigl\| x_i - (z^k - u_i^k) \bigr\|_2^2 \\
   z^{k+1} &= \frac{1}{N+\alpha/\rho}\Bigl(\alpha/\rho \cdot \text{target} + \sum_i (x_i^{k+1} + u_i^k)\Bigr) \\
   u_i^{k+1} &= u_i^k + x_i^{k+1} - z^{k+1}

Use :func:`~distributed_resource_optimization.create_consensus_target_reach_admm_coordinator`
and :func:`~distributed_resource_optimization.create_admm_start_consensus`.

Local Model: Flexibility Actor
-------------------------------

Each participant is a *flexibility actor* — a local resource with bounded and coupled
decision variables. At each ADMM iteration it solves the QP:

.. math::

   \min_x \;\tfrac{\rho}{2}\|x + v\|^2 + S^\top x
   \quad\text{s.t.}\quad l \le x \le u,\; Cx \le d

where :math:`v` is the correction sent by the coordinator, :math:`S` is a priority
penalty vector, and the constraints represent box and coupling feasibility.

One-to-Many Resource
~~~~~~~~~~~~~~~~~~~~

A common model converts a single input (capacity ``in_capacity``) into ``m`` outputs
with efficiency factors :math:`\eta \in \mathbb{R}^m`:

.. code-block:: python

   from distributed_resource_optimization import create_admm_flex_actor_one_to_many

   # 10 kW input, three outputs; negative eta means the resource consumes that output
   actor = create_admm_flex_actor_one_to_many(10.0, [0.1, 0.5, -1.0])

   # Optional priority vector biases the solution toward specific sectors
   actor = create_admm_flex_actor_one_to_many(10.0, [0.1, 0.5, -1.0], P=[5.0, 0.0, 0.0])

After optimization, retrieve the result via ``actor.x``.

Coordinator Parameters
----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 12 53

   * - Parameter
     - Default
     - Description
   * - ``rho``
     - ``1.0``
     - Penalty parameter — larger values enforce constraints faster but may slow convergence
   * - ``max_iters``
     - ``1000``
     - Maximum number of ADMM iterations
   * - ``abs_tol``
     - ``1e-4``
     - Absolute primal/dual residual tolerance
   * - ``rel_tol``
     - ``1e-3``
     - Relative primal/dual residual tolerance

Complete Example — ADMM Sharing
---------------------------------

.. code-block:: python

   import asyncio
   from distributed_resource_optimization import (
       create_admm_flex_actor_one_to_many,
       create_sharing_target_distance_admm_coordinator,
       create_admm_sharing_data, create_admm_start,
       start_coordinated_optimization,
   )

   async def main():
       flex1 = create_admm_flex_actor_one_to_many(10.0, [0.1,  0.5, -1.0])
       flex2 = create_admm_flex_actor_one_to_many(15.0, [0.1,  0.5, -1.0])
       flex3 = create_admm_flex_actor_one_to_many(10.0, [-1.0, 0.0,  1.0])

       coordinator = create_sharing_target_distance_admm_coordinator()
       start = create_admm_start(create_admm_sharing_data([-4.0, 0.0, 6.0], [5, 1, 1]))

       await start_coordinated_optimization([flex1, flex2, flex3], coordinator, start)

       print("Resource 1:", flex1.x.round(3))
       print("Resource 2:", flex2.x.round(3))
       print("Resource 3:", flex3.x.round(3))
       print("Aggregate: ", (flex1.x + flex2.x + flex3.x).round(3))

   asyncio.run(main())

Complete Example — ADMM Consensus
-----------------------------------

.. code-block:: python

   import asyncio
   from distributed_resource_optimization import (
       create_admm_flex_actor_one_to_many,
       create_consensus_target_reach_admm_coordinator,
       create_admm_start_consensus,
       start_coordinated_optimization,
   )

   async def main():
       actor1 = create_admm_flex_actor_one_to_many(10.0, [0.6, 0.4])
       actor2 = create_admm_flex_actor_one_to_many(10.0, [0.6, 0.4])

       coordinator = create_consensus_target_reach_admm_coordinator()
       start = create_admm_start_consensus([1.0, 2.0])

       await start_coordinated_optimization([actor1, actor2], coordinator, start)
       print("Actor 1:", actor1.x.round(3))
       print("Actor 2:", actor2.x.round(3))

   asyncio.run(main())

.. tip::

   If ADMM diverges or converges slowly:

   - Reduce ``rho`` when primal residuals dominate.
   - Increase ``rho`` when dual residuals dominate.
   - Tighten ``abs_tol`` / ``rel_tol`` for higher precision.
   - Increase ``max_iters`` if the warning "reached max iterations" appears.

See Also
--------

- :class:`~distributed_resource_optimization.ADMMFlexActor`,
  :func:`~distributed_resource_optimization.create_admm_flex_actor_one_to_many`
- :func:`~distributed_resource_optimization.create_sharing_target_distance_admm_coordinator`,
  :func:`~distributed_resource_optimization.create_admm_start`,
  :func:`~distributed_resource_optimization.create_admm_sharing_data`
- :func:`~distributed_resource_optimization.create_consensus_target_reach_admm_coordinator`,
  :func:`~distributed_resource_optimization.create_admm_start_consensus`
- :doc:`../tutorials/energy_dispatch`
