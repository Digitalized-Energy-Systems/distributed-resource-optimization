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

Exchange
~~~~~~~~

The exchange problem (Boyd et al. 2011, §7.3.2) is sharing with :math:`g` the indicator of
:math:`\sum_i x_i = 0`, or with ``surplus=True`` of :math:`\sum_i x_i \ge 0`:

.. math::

   z^{k+1} = 0 \quad\text{or}\quad z^{k+1} = \max(\bar{x}^{\,k+1} + u^k,\, 0)

Use :func:`~distributed_resource_optimization.create_exchange_admm_coordinator`
(``surplus``, plus the coordinator parameters below) and start it with
``create_admm_start(None, length)``; all iterates start at zero. Participants answer
``ADMMMessage(v, rho)`` with :math:`\arg\min f_i(x) + \tfrac{\rho}{2}\lVert x + v\rVert^2`.

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

.. doctest::

   >>> from distributed_resource_optimization import create_admm_flex_actor_one_to_many
   >>> actor = create_admm_flex_actor_one_to_many(10.0, [0.1, 0.5, -1.0])
   >>> actor = create_admm_flex_actor_one_to_many(10.0, [0.1, 0.5, -1.0], P=[5.0, 0.0, 0.0])

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
   * - ``mu``
     - ``None``
     - Residual balancing (Boyd et al. 2011, eq. 3.13): ``rho`` is multiplied by ``tau`` when
       the primal residual exceeds ``mu`` times the dual one, divided when the reverse holds
   * - ``tau``
     - ``2.0``
     - Factor of the residual balancing
   * - ``round_timeout``
     - ``None``
     - Lossy links: a round ends at the latest this long after its scatter (carrier clock); a
       participant whose reply is missing keeps its last ``x``

The stop test is Boyd et al. 2011, eq. 3.12, with stacked 2-norms of the primal and dual
residuals. After a run the coordinator holds ``iterations``, ``converged``, ``stop_time`` and
``missing_replies``.

Complete Example — ADMM Sharing
---------------------------------

.. doctest::

   >>> from distributed_resource_optimization import (
   ...     create_admm_flex_actor_one_to_many,
   ...     create_sharing_target_distance_admm_coordinator,
   ...     create_admm_sharing_data, create_admm_start,
   ...     start_coordinated_optimization,
   ... )
   >>> flex1 = create_admm_flex_actor_one_to_many(10.0, [0.1,  0.5, -1.0])
   >>> flex2 = create_admm_flex_actor_one_to_many(15.0, [0.1,  0.5, -1.0])
   >>> flex3 = create_admm_flex_actor_one_to_many(10.0, [-1.0, 0.0,  1.0])
   >>> coordinator = create_sharing_target_distance_admm_coordinator()
   >>> start = create_admm_start(create_admm_sharing_data([-4.0, 0.0, 6.0], [5, 1, 1]))
   >>> asyncio.run(start_coordinated_optimization([flex1, flex2, flex3], coordinator, start))
   [...]
   >>> np.allclose(flex1.x, [0, 0, 0], atol=1e-2)
   True
   >>> np.allclose(flex3.x, [-4.0, 0, 4.0], atol=1e-2)
   True

Complete Example — ADMM Consensus
-----------------------------------

.. doctest::

   >>> from distributed_resource_optimization import (
   ...     create_admm_flex_actor_one_to_many,
   ...     create_consensus_target_reach_admm_coordinator,
   ...     create_admm_start_consensus,
   ...     start_coordinated_optimization,
   ... )
   >>> actor1 = create_admm_flex_actor_one_to_many(10.0, [0.6, 0.4])
   >>> actor2 = create_admm_flex_actor_one_to_many(10.0, [0.6, 0.4])
   >>> coordinator = create_consensus_target_reach_admm_coordinator()
   >>> start = create_admm_start_consensus([1.0, 2.0])
   >>> asyncio.run(start_coordinated_optimization([actor1, actor2], coordinator, start))
   [...]
   >>> np.allclose(actor1.x, actor2.x, atol=1e-2)
   True

.. tip::

   If ADMM diverges or converges slowly:

   - Increase ``rho`` when primal residuals dominate.
   - Reduce ``rho`` when dual residuals dominate (or set ``mu`` to balance them automatically).
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
