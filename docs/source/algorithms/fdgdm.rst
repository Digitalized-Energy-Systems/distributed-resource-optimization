FDGDM
=====

The Fast Distributed Gradient Descent Method (FDGDM) solves economic dispatch by
exchanging cost *gradients* between neighbouring agents (Bai et al. 2022, *"Fast
distributed gradient descent method for economic dispatch of microgrids via upper
bounds of second derivatives"*, Energy Reports 8). Unlike the price-consensus
algorithms, FDGDM iterates directly on the power schedules and **conserves the total
power at every iteration** — the initial allocation must therefore already satisfy
the demand balance.

Algorithm
---------

Each participant holds a power schedule :math:`P_i` over the horizon. Per iteration it

1. receives :math:`\nabla F_j(P_j^{k})` and the product :math:`d_j u_j` from every
   neighbour :math:`j`,
2. updates its schedule with curvature-bounded weights and projects it onto its
   feasible set:

   .. math::

      P_i^{k+1} = \operatorname{proj}\!\left(
          P_i^{k} + \sum_{j \in \mathcal{N}_i}
          \min\!\left(\tfrac{1}{d_i u_i},\, \tfrac{1}{d_j u_j}\right)
          \bigl(\nabla F_j^{k} - \nabla F_i^{k}\bigr)\right)

3. broadcasts its updated gradient :math:`\nabla F_i(P_i^{k+1})`.

Here :math:`u_i` is an upper bound on the second derivative of the local cost
:math:`F_i` and :math:`d_i` its (adjusted) degree. Because gradients flow
antisymmetrically between pairs, power moves from expensive to cheap units while the
sum stays constant — the network descends toward equal incremental cost.

Deliberate deviations from the paper
------------------------------------

- **Degree** :math:`d_i = |\mathcal{N}_i| + 1`, not the paper's :math:`|\mathcal{N}_i|`
  (eq. 5): with the paper's exact value the two-agent iteration matrix has eigenvalue
  −1 (a period-2 oscillation that never converges). The +1 keeps the acceleration
  condition of Proposition 2 intact.
- **No eq.-24 stopping criterion**: every run executes a fixed ``max_iter``
  iterations; converged runs simply stop changing early.

Usage
-----

.. doctest::

   >>> from distributed_resource_optimization import (
   ...     LinearCostEconomicDispatchFDGDMActor,
   ...     create_fdgdm_participant,
   ...     create_fdgdm_start,
   ...     start_distributed_optimization,
   ... )
   >>> def on_finish(algo, carrier):
   ...     pass
   >>> # 3 generators, total demand 90 -> feasible initial allocation 30 each.
   >>> actors = [
   ...     create_fdgdm_participant(
   ...         on_finish,
   ...         LinearCostEconomicDispatchFDGDMActor(cost=c, p_max=100.0),
   ...         max_iter=200,
   ...     )
   ...     for c in (1.0, 1.0, 1.0)
   ... ]
   >>> start = create_fdgdm_start(data=np.array([30.0]))
   >>> asyncio.run(start_distributed_optimization(actors, start))
   >>> bool(np.allclose(sum(a.actor.P for a in actors), 90.0, atol=1e-3))
   True

The kickoff message carries the same initial vector to every participant. To give
each agent its own demand-feasible starting point, set the actor's
``initial_schedule`` — it replaces the shared kickoff value on the first projection.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 50

   * - Parameter
     - Default
     - Description
   * - ``finish_callback``
     - —
     - Called with ``(algorithm, carrier)`` when ``max_iter`` is reached
   * - ``fdgdm_actor``
     - ``None``
     - :class:`~distributed_resource_optimization.FDGDMActor` supplying gradient,
       curvature bound, and feasibility projection
   * - ``max_iter``
     - ``300``
     - Number of iterations (no residual-based stop)
   * - ``horizon``
     - ``24``
     - Placeholder schedule length (resized from the first message)

Actors
------

- :class:`~distributed_resource_optimization.LinearCostEconomicDispatchFDGDMActor` —
  quadratic cost :math:`F(P) = (\varepsilon/2)P^2 + cP` with closed-form gradient
  :math:`\varepsilon P + c` and exact curvature bound :math:`\varepsilon`; box
  projection onto :math:`[P_{\min}, P_{\max}]`.
- :class:`~distributed_resource_optimization.ReservoirStorageFDGDMActor` —
  storage with piecewise-linear cost and quadratic regularisation. Power limits are
  exact; **SOC coupling across time steps is not modelled** (each step is treated
  independently) — use the Diffusion or ADMM storage actors when SOC dynamics matter.

.. note::

   The weight matrix has zero row-sums, so any imbalance in the *initial* allocation
   persists forever: FDGDM redistributes power, it never creates or destroys it.
   Feasibility projection (box clipping) can break exact conservation when limits
   bind; the benchmark verifies the final balance after every run.

See Also
--------

- :func:`~distributed_resource_optimization.create_fdgdm_participant`,
  :class:`~distributed_resource_optimization.FDGDMAlgorithm`
- :class:`~distributed_resource_optimization.FDGDMActor`,
  :class:`~distributed_resource_optimization.NoFDGDMActor`
