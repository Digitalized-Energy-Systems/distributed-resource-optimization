ADMM
====

The Alternating Direction Method of Multipliers (ADMM) is a convex optimization algorithm
that decomposes a large problem into smaller subproblems solved locally, coordinated by a
central update. Two ADMM variants are implemented: **Sharing** and **Consensus**.

Problem Forms
-------------

Sharing
~~~~~~~

The sharing form distributes a resource across :math:`N` agents whose *aggregate*
contribution is scored by a shared objective :math:`g`:

.. math::

   \min_{\{x_i\}}\; \sum_{i=1}^N f_i(x_i) + g\Bigl(\sum_{i=1}^N x_i\Bigr)

With :math:`z` standing for the average allocation (:math:`\bar{x} = z` at convergence),
the ADMM iterations with scaled dual variable :math:`u` and penalty :math:`\rho` are:

.. math::

   x_i^{k+1} &= \arg\min_{x_i}\; f_i(x_i) + \tfrac{\rho}{2}\,\bigl\lVert x_i - (x_i^k - \bar{x}^{\,k} + z^k - u^k) \bigr\rVert_2^2 \\
   \bar{x}^{\,k+1} &= \tfrac{1}{N}\sum_{i=1}^N x_i^{k+1} \\
   z^{k+1} &= \arg\min_{z}\; g(Nz) + \tfrac{N\rho}{2}\,\bigl\lVert z - \bar{x}^{\,k+1} - u^k \bigr\rVert_2^2 \\
   u^{k+1} &= u^k + \bar{x}^{\,k+1} - z^{k+1}

(Boyd et al. 2011, §7.3).
:func:`~distributed_resource_optimization.algorithm.admm.sharing_admm.create_sharing_target_distance_admm_coordinator`
uses :math:`g(Nz) = \sum_t p_t\,\lvert N z_t - \text{target}_t\rvert` — a weighted L1
distance of the aggregate to the target, with the priorities :math:`p` from
:func:`~distributed_resource_optimization.algorithm.admm.sharing_admm.create_admm_sharing_data`.  Start a run with
:func:`~distributed_resource_optimization.algorithm.admm.sharing_admm.create_sharing_admm_start`.

Consensus (exchange)
~~~~~~~~~~~~~~~~~~~~

Despite its name, the "consensus" variant is Boyd's **exchange** ADMM (Boyd et al.
2011, §7.3.2) shifted by a target: the agents' individual solutions must *sum* to a
target vector, rather than all agree on one shared value.

.. math::

   \min_{\{x_i\}}\; \sum_{i=1}^N f_i(x_i)
   \quad\text{s.t.}\quad \sum_{i=1}^N x_i = \text{target}

Each agent :math:`i` has its own :math:`z_i` and scaled dual :math:`u_i`:

.. math::

   x_i^{k+1} &= \arg\min_{x_i}\; f_i(x_i) + \frac{\rho}{2}\bigl\| x_i - (z_i^k - u_i^k) \bigr\|_2^2 \\
   \delta^{k+1} &= \frac{\text{target} - \sum_i (x_i^{k+1} + u_i^k)}{N + \rho/\alpha} \\
   z_i^{k+1} &= x_i^{k+1} + u_i^k + \delta^{k+1} \\
   u_i^{k+1} &= u_i^k + x_i^{k+1} - z_i^{k+1}

With ``alpha = 0`` (the default) the :math:`\rho/\alpha` term is dropped and the
z-update is the exact projection onto :math:`\sum_i z_i = \text{target}`; ``alpha > 0``
relaxes the sum constraint to a soft quadratic penalty (which biases the converged sum
when local objectives carry costs).

Use :func:`~distributed_resource_optimization.algorithm.admm.consensus_admm.create_consensus_target_reach_admm_coordinator`
and :func:`~distributed_resource_optimization.algorithm.admm.consensus_admm.create_admm_start_consensus`.

Local Model: Flexibility Actor
-------------------------------

Each participant is a *flexibility actor* — a local resource with bounded and coupled
decision variables. At each ADMM iteration it solves the QP:

.. math::

   \min_x \;\tfrac{\rho}{2}\|x + v\|^2 + S^\top x
   \quad\text{s.t.}\quad l \le x \le u,\; Cx \le d

where :math:`v` is the correction sent by the coordinator, :math:`S` is a linear
cost/priority vector (positive entries penalise, negative entries reward an output), and
the constraints represent box and coupling feasibility.

One-to-Many Resource
~~~~~~~~~~~~~~~~~~~~

A common model converts a single input (capacity ``in_capacity``) into ``m`` outputs
with efficiency factors :math:`\eta \in \mathbb{R}^m`.  The optional priority vector
``P`` enters as :math:`S = -P`, so a positive entry *rewards* (pushes up) that output:

.. doctest::

   >>> from distributed_resource_optimization import create_admm_flex_actor_one_to_many
   >>> actor = create_admm_flex_actor_one_to_many(10.0, [0.1, 0.5, -1.0])
   >>> actor = create_admm_flex_actor_one_to_many(10.0, [0.1, 0.5, -1.0], P=[5.0, 0.0, 0.0])

After optimization, retrieve the result via ``actor.x``.

Coordinator Parameters
----------------------

Both factories build an
:class:`~distributed_resource_optimization.algorithm.admm.core.ADMMGenericCoordinator`, which takes:

.. list-table::
   :header-rows: 1
   :widths: 22 12 50

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
   * - ``single_shot_clearing``
     - ``True``
     - If the start data carries merit-order ``generators`` specs, skip the loop and
       dispatch with one round at the closed-form clearing prices (only valid when every
       participant is a price-responsive economic-dispatch actor)

:func:`~distributed_resource_optimization.algorithm.admm.consensus_admm.create_consensus_target_reach_admm_coordinator`
exposes ``rho``, ``max_iters`` and the sum-constraint softness ``alpha`` (default ``0``);
:func:`~distributed_resource_optimization.algorithm.admm.sharing_admm.create_sharing_target_distance_admm_coordinator`
takes no arguments and uses the defaults.  Construct the coordinator directly to change
anything else (see :doc:`../tutorials/energy_dispatch`).

Complete Example — ADMM Sharing
---------------------------------

.. doctest::

   >>> from distributed_resource_optimization import (
   ...     create_admm_flex_actor_one_to_many,
   ...     create_sharing_target_distance_admm_coordinator,
   ...     create_admm_sharing_data, create_sharing_admm_start,
   ...     start_coordinated_optimization,
   ... )
   >>> flex1 = create_admm_flex_actor_one_to_many(10.0, [0.1,  0.5, -1.0])
   >>> flex2 = create_admm_flex_actor_one_to_many(15.0, [0.1,  0.5, -1.0])
   >>> flex3 = create_admm_flex_actor_one_to_many(10.0, [-1.0, 0.0,  1.0])
   >>> coordinator = create_sharing_target_distance_admm_coordinator()
   >>> start = create_sharing_admm_start(create_admm_sharing_data([-4.0, 0.0, 6.0], [5, 1, 1]))
   >>> asyncio.run(start_coordinated_optimization([flex1, flex2, flex3], coordinator, start))
   [...]
   >>> np.allclose(flex1.x, [0, 0, 0], atol=1e-2)
   True
   >>> np.allclose(flex3.x, [-3.983, 0, 3.983], atol=1e-2)
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
   >>> # Each actor's outputs are tied to the ratio 0.6 : 0.4, so pick a reachable target.
   >>> start = create_admm_start_consensus([1.2, 0.8])
   >>> asyncio.run(start_coordinated_optimization([actor1, actor2], coordinator, start))
   [...]
   >>> np.allclose(actor1.x + actor2.x, [1.2, 0.8], atol=1e-2)
   True

.. tip::

   If ADMM diverges or converges slowly:

   - Reduce ``rho`` when primal residuals dominate.
   - Increase ``rho`` when dual residuals dominate.
   - Tighten ``abs_tol`` / ``rel_tol`` for higher precision.
   - Increase ``max_iters`` if the warning "reached max iterations" appears.

See Also
--------

- :class:`~distributed_resource_optimization.algorithm.admm.flex_actor.ADMMFlexActor`,
  :func:`~distributed_resource_optimization.algorithm.admm.flex_actor.create_admm_flex_actor_one_to_many`
- :func:`~distributed_resource_optimization.algorithm.admm.sharing_admm.create_sharing_target_distance_admm_coordinator`,
  :func:`~distributed_resource_optimization.algorithm.admm.sharing_admm.create_sharing_admm_start`,
  :func:`~distributed_resource_optimization.algorithm.admm.sharing_admm.create_admm_sharing_data`
- :func:`~distributed_resource_optimization.algorithm.admm.consensus_admm.create_consensus_target_reach_admm_coordinator`,
  :func:`~distributed_resource_optimization.algorithm.admm.consensus_admm.create_admm_start_consensus`
- :doc:`../tutorials/energy_dispatch`
