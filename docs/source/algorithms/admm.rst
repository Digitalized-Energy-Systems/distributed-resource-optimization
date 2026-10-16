ADMM
====

The Alternating Direction Method of Multipliers (ADMM) is a convex optimization algorithm
that decomposes a large problem into smaller subproblems solved locally, coordinated by a
central update. Two ADMM variants are implemented: **Sharing** and **Consensus**
(the latter is Boyd's *exchange* ADMM: the participants' solutions must sum to a target).
Both use the scaled dual form; the local actors receive a correction vector :math:`v`
and solve :math:`\min_x f_i(x) + \tfrac{\rho}{2}\lVert x + v\rVert^2`.

Problem Forms
-------------

Sharing
~~~~~~~

The sharing form (Boyd et al. 2011, §7.3) distributes a resource across :math:`N` agents
whose aggregate is priced by a global objective :math:`g`. The coordinator tracks the
participant *average* :math:`z`:

.. math::

   \min_{\{x_i\},\,z}\; \sum_{i=1}^N f_i(x_i) + g(Nz)
   \quad\text{s.t.}\quad \bar{x} = \tfrac{1}{N}\sum_{i=1}^N x_i = z

The ADMM iterations with scaled dual variable :math:`u` and penalty :math:`\rho` are:

.. math::

   x_i^{k+1} &= \arg\min_{x_i}\; f_i(x_i) + \tfrac{\rho}{2}\,\bigl\lVert x_i - (x_i^k - \bar{x}^{\,k} + z^k - u^k) \bigr\rVert_2^2 \\
   z^{k+1} &= \arg\min_{z}\; g(Nz) + \tfrac{N\rho}{2}\,\bigl\lVert z - \bar{x}^{\,k+1} - u^k \bigr\rVert_2^2 \\
   u^{k+1} &= u^k + \bar{x}^{\,k+1} - z^{k+1}

The implemented :math:`g` is a priority-weighted L1 distance to a target vector
:math:`t`: :math:`g(Nz) = \sum_j |p_j|\,|N z_j - t_j|`, so larger priorities
:math:`p_j` make deviations in element :math:`j` more expensive.

Use :func:`~distributed_resource_optimization.create_sharing_target_distance_admm_coordinator`,
:func:`~distributed_resource_optimization.create_admm_sharing_data` and
:func:`~distributed_resource_optimization.create_sharing_admm_start`.

Consensus (exchange)
~~~~~~~~~~~~~~~~~~~~

The consensus coordinator drives the participants' solutions to **sum to a target**
(Boyd et al. 2011, §7.3.2, shifted by the target). Participants do *not* agree on a
common value — e.g. in economic dispatch each generator ends at its own merit-order
output:

.. math::

   \min_{\{x_i\}}\; \sum_{i=1}^N f_i(x_i)
   \quad\text{s.t.}\quad \sum_{i=1}^N x_i = \text{target}

Each participant has its own copy :math:`z_i` and scaled dual :math:`u_i`:

.. math::

   x_i^{k+1} &= \arg\min_{x_i}\; f_i(x_i) + \frac{\rho}{2}\bigl\| x_i - (z_i^k - u_i^k) \bigr\|_2^2 \\
   \delta^{k+1} &= \frac{\text{target} - \sum_i (x_i^{k+1} + u_i^k)}{N + \rho/\alpha} \\
   z_i^{k+1} &= x_i^{k+1} + u_i^k + \delta^{k+1} \\
   u_i^{k+1} &= u_i^k + x_i^{k+1} - z_i^{k+1}

With ``alpha = 0`` (the default) the term :math:`\rho/\alpha` is dropped and the
z-update is the exact projection onto :math:`\sum_i z_i = \text{target}`; the scaled dual
converges to the (scaled) market-clearing price. ``alpha > 0`` softens the constraint to a
quadratic penalty :math:`\tfrac{\alpha}{2}\lVert\sum_i z_i - \text{target}\rVert^2`, which
biases the converged sum — keep ``alpha = 0`` for economic dispatch.

Besides the per-participant residuals, the loop only stops once the returned schedules
meet the target: :math:`\max_t \lvert\sum_i x_{i,t} - \text{target}_t\rvert \le
\texttt{abs\_tol} + \texttt{feas\_tol}\cdot\max_t\lvert\text{target}_t\rvert`.
The per-participant residuals alone let the aggregate imbalance grow with the number of
participants (each carries a similar :math:`x_i - z_i`), so without this check large
fleets could report convergence while missing the target by up to ``N`` times the
per-participant tolerance. The check is skipped for ``alpha > 0``.

The target must be reachable by the participants' feasible sets; otherwise the loop runs
until ``max_iters`` and logs a warning.

With ``alpha = 0`` (the default) the :math:`\rho/\alpha` term is dropped and the
z-update is the exact projection onto :math:`\sum_i z_i = \text{target}`; ``alpha > 0``
relaxes the sum constraint to a soft quadratic penalty (which biases the converged sum
when local objectives carry costs).

Use :func:`~distributed_resource_optimization.algorithm.admm.consensus_admm.create_consensus_target_reach_admm_coordinator`
and :func:`~distributed_resource_optimization.algorithm.admm.consensus_admm.create_admm_start_consensus`.

Local Models
------------

Flexibility Actor
~~~~~~~~~~~~~~~~~

Each participant is a *flexibility actor* — a local resource with bounded and coupled
decision variables. At each ADMM iteration it solves the QP:

.. math::

   \min_x \;\tfrac{\rho}{2}\|x + v\|^2 + S^\top x
   \quad\text{s.t.}\quad l \le x \le u,\; Cx \le d

where :math:`v` is the correction sent by the coordinator, :math:`S` is a linear cost
vector (negative entries act as rewards), and the constraints represent box and coupling
feasibility. Without coupling constraints the QP has the closed form
:math:`x = \mathrm{clip}(-v - S/\rho,\, l,\, u)`.

Box-Bounded Resource
^^^^^^^^^^^^^^^^^^^^

:func:`~distributed_resource_optimization.create_admm_flex_actor_box_bounded` creates an
actor with one independent variable per entry (e.g. per timestep) and a linear cost
``S`` — e.g. a generator with marginal cost ``S`` and power limits ``[lb, u]``:

.. doctest::

   >>> from distributed_resource_optimization import create_admm_flex_actor_box_bounded
   >>> gen = create_admm_flex_actor_box_bounded(lb=[0.0, 0.0], u=[10.0, 10.0], S=[5.0, 5.0])

One-to-Many Resource
^^^^^^^^^^^^^^^^^^^^

A common model converts a single input (capacity ``in_capacity``) into ``m`` outputs
with efficiency factors :math:`\eta \in \mathbb{R}^m`. All outputs run at the same
utilisation fraction of their capacity ``in_capacity * eta``. The optional priority
vector ``P`` enters the objective as :math:`S = -P`: positive entries reward an output,
negative entries penalise it.

.. doctest::

   >>> from distributed_resource_optimization import create_admm_flex_actor_one_to_many
   >>> actor = create_admm_flex_actor_one_to_many(10.0, [0.1, 0.5, -1.0])
   >>> actor = create_admm_flex_actor_one_to_many(10.0, [0.1, 0.5, -1.0], P=[5.0, 0.0, 0.0])

After optimization, retrieve the result via ``actor.x``.

Storage Actor
~~~~~~~~~~~~~

:func:`~distributed_resource_optimization.create_admm_proximal_storage_actor` creates a
battery participant for the consensus (exchange) coordinator. Each round it solves the
exact local subproblem over the whole horizon — charge/discharge costs plus the proximal
term, subject to SOC dynamics, power and energy limits and a terminal SOC (dropped if
unreachable). Its output is net power, positive = discharge.

.. note::

   Charging and discharging are separate non-negative variables with no
   complementarity constraint, so the QP may run both at once in the same timestep.
   Because of the round-trip losses this turns energy into heat without moving the
   SOC — e.g. a full battery can still "absorb" power. The optimizer only does this
   when surplus cannot be shed otherwise (such as must-run output above demand); a
   positive ``charge_cost`` makes it strictly less attractive than curtailment.

Coordinator Parameters
----------------------

:class:`~distributed_resource_optimization.ADMMGenericCoordinator` takes:

.. list-table::
   :header-rows: 1
   :widths: 22 12 50

   * - Parameter
     - Default
     - Description
   * - ``rho``
     - ``1.0``
     - Penalty parameter — larger values enforce the coupling constraint more strongly
       at the expense of the local objectives
   * - ``max_iters``
     - ``1000``
     - Maximum number of ADMM iterations
   * - ``abs_tol``
     - ``1e-4``
     - Absolute primal/dual residual tolerance
   * - ``rel_tol``
     - ``1e-3``
     - Relative primal/dual residual tolerance
   * - ``feas_tol``
     - ``1e-4``
     - Max aggregate coupling-constraint violation of the returned ``x``, relative to
       the constraint's scale (consensus: ``max|target|``); unused by sharing ADMM

At least one participant must be registered with the coordinator.

Both factories accept ``rho`` and ``max_iters``; the consensus factory also accepts
``alpha`` and ``feas_tol``. After a run, the coordinator's ``iterations`` and ``converged`` attributes
report how the loop ended.

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

Complete Example — ADMM Consensus (economic dispatch)
-------------------------------------------------------

Two generators with marginal costs 1 and 5 must together supply 8 units; the cheap one
is dispatched first:

.. doctest::

   >>> from distributed_resource_optimization import (
   ...     create_admm_flex_actor_box_bounded,
   ...     create_consensus_target_reach_admm_coordinator,
   ...     create_admm_start_consensus,
   ...     start_coordinated_optimization,
   ... )
   >>> cheap = create_admm_flex_actor_box_bounded([0.0], [5.0], S=[1.0])
   >>> expensive = create_admm_flex_actor_box_bounded([0.0], [10.0], S=[5.0])
   >>> coordinator = create_consensus_target_reach_admm_coordinator()
   >>> start = create_admm_start_consensus([8.0])
   >>> asyncio.run(start_coordinated_optimization([cheap, expensive], coordinator, start))
   [...]
   >>> coordinator.converged
   True
   >>> np.allclose([cheap.x[0], expensive.x[0]], [5.0, 3.0], atol=0.05)
   True

.. tip::

   If ADMM diverges or converges slowly:

   - Increase ``rho`` when primal residuals dominate (constraint violation stays large).
   - Decrease ``rho`` when dual residuals dominate (the solution keeps oscillating).
   - Tighten ``abs_tol`` / ``rel_tol`` for higher precision.
   - Increase ``max_iters`` if the warning "reached max iterations" appears — and check
     that the target is actually reachable.

See Also
--------

- :class:`~distributed_resource_optimization.ADMMFlexActor`,
  :func:`~distributed_resource_optimization.create_admm_flex_actor_one_to_many`,
  :func:`~distributed_resource_optimization.create_admm_flex_actor_box_bounded`,
  :func:`~distributed_resource_optimization.create_admm_proximal_storage_actor`
- :func:`~distributed_resource_optimization.create_sharing_target_distance_admm_coordinator`,
  :func:`~distributed_resource_optimization.create_sharing_admm_start`,
  :func:`~distributed_resource_optimization.create_admm_sharing_data`
- :func:`~distributed_resource_optimization.create_consensus_target_reach_admm_coordinator`,
  :func:`~distributed_resource_optimization.create_admm_start_consensus`
- :doc:`../tutorials/energy_dispatch`
