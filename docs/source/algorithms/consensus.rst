Averaging Consensus
===================

The Averaging Consensus algorithm distributes a parameter vector :math:`\lambda` across
:math:`N` agents via a gossip-style protocol. Each agent maintains its own copy of
:math:`\lambda` and iteratively averages it with values received from neighbours.

This implements the leader-follower incremental-cost consensus of Jian et al. 2020
(*"Distributed economic dispatch method for power system based on consensus"*, eqs. 20-23):
exactly one participant is the *leader*, the rest are *followers*. The leader additionally
pins :math:`\lambda` toward the real system-wide power imbalance; followers do pure averaging.

Algorithm
---------

Let :math:`\lambda_i^k` be the value held by agent :math:`i` at iteration :math:`k`, and
:math:`P_i(\lambda_i^k)` its locally projected power output (see
:meth:`~distributed_resource_optimization.ConsensusActor.project_power`). The update rule is:

.. math::

   \lambda_i^{k+1} = \lambda_i^k
                   + \alpha \bigl(\bar{\lambda}^k - \lambda_i^k\bigr)
                   + \begin{cases}
                       \varepsilon \, \Delta P^k & \text{if } i \text{ is the leader} \\
                       0 & \text{if } i \text{ is a follower}
                     \end{cases}

where

- :math:`\bar{\lambda}^k` is the average of all received values at iteration :math:`k`
- :math:`\alpha \in (0,1]` is the mixing step size — a numerically-stable, row-stochastic
  discretisation of the paper's :math:`\sum_j a_{ij}\lambda_j^k` averaging term
- :math:`\Delta P^k = P_{\text{target}} - \sum_i P_i(\lambda_i^k)` is the real system-wide
  power imbalance, recovered each round from every participant's projected power
- :math:`\varepsilon` (``leader_gain``) is the leader's pinning gain

The algorithm runs for a fixed number of iterations (``max_iter``) after which each agent
calls a user-supplied ``finish_callback``.

Usage
-----

.. doctest::

   >>> from distributed_resource_optimization import (
   ...     create_averaging_consensus_participant,
   ...     create_averaging_consensus_start,
   ...     start_distributed_optimization,
   ... )
   >>> finished = []
   >>> def on_finish(algo, carrier):
   ...     finished.append(algo._lam.copy())
   >>> actors = [
   ...     create_averaging_consensus_participant(on_finish, initial_lam=v, max_iter=200)
   ...     for v in [1.0, 5.0, 10.0]
   ... ]
   >>> start = create_averaging_consensus_start(1.0, data=None)
   >>> asyncio.run(start_distributed_optimization(actors, start))
   >>> len(finished) > 0
   True
   >>> np.allclose(actors[0]._lam, actors[1]._lam, atol=1e-2)
   True

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
   * - ``consensus_actor``
     - ``None``
     - :class:`~distributed_resource_optimization.ConsensusActor` providing the price-to-power projection
   * - ``initial_lam``
     - ``10.0``
     - Starting scalar value broadcast to all :math:`\lambda` components
   * - ``alpha``
     - ``0.3``
     - Mixing step size
   * - ``max_iter``
     - ``50``
     - Number of gossip rounds before finishing
   * - ``is_leader``
     - ``False``
     - Whether this participant is the leader (eq. 22). Exactly one participant per run
       should set this.
   * - ``leader_gain``
     - ``0.0``
     - Leader's power-imbalance pinning gain (:math:`\varepsilon` in eq. 22). Ignored for
       followers.

Local Power Projection
-----------------------

To steer the consensus toward a local optimum, subclass
:class:`~distributed_resource_optimization.ConsensusActor` and override
:meth:`~distributed_resource_optimization.ConsensusActor.project_power`:

.. doctest::

   >>> from distributed_resource_optimization import ConsensusActor
   >>> class ClipToRange(ConsensusActor):
   ...     def __init__(self, p_min, p_max):
   ...         self.p_min, self.p_max = p_min, p_max
   ...     def project_power(self, lam, data):
   ...         return np.clip(lam, self.p_min, self.p_max)

The ``data`` argument carries whatever was embedded in the initial
:class:`~distributed_resource_optimization.AveragingConsensusMessage` — typically the total
demand vector, used by the leader to compute the real power imbalance :math:`\Delta P`.

Economic Dispatch
-----------------

The built-in
:class:`~distributed_resource_optimization.LinearCostEconomicDispatchConsensusActor`
implements the price-to-power projection of eq. (23): each agent has a linear cost and power
limits; :math:`\lambda` is the shared incremental (marginal) cost, and

.. math::

   P(\lambda) = \operatorname{clip}\!\left(\frac{\lambda - c}{\epsilon},\; P_{\min},\; P_{\max}\right)

is eq. (23)'s ``(λ - bi)/(2ai)`` clip for a quadratic cost
``Fi(PGi) = ci + bi*PGi + ai*PGi**2``, with ``cost`` ↔ ``bi`` and ``epsilon`` ↔ ``2*ai``.
The power-balancing correction itself is *not* part of the actor — it's the leader's
:math:`\varepsilon\,\Delta P` term above, computed from every participant's projected power.

.. doctest::

   >>> from distributed_resource_optimization import (
   ...     LinearCostEconomicDispatchConsensusActor,
   ...     create_averaging_consensus_participant,
   ...     AveragingConsensusMessage,
   ...     start_distributed_optimization,
   ... )
   >>> actors = [
   ...     create_averaging_consensus_participant(
   ...         lambda *_: None,
   ...         LinearCostEconomicDispatchConsensusActor(cost=10, p_max=100),
   ...         max_iter=100,
   ...         is_leader=True,
   ...         leader_gain=0.02,
   ...     ),
   ...     create_averaging_consensus_participant(
   ...         lambda *_: None,
   ...         LinearCostEconomicDispatchConsensusActor(cost=10, p_max=100),
   ...         max_iter=100,
   ...     ),
   ...     create_averaging_consensus_participant(
   ...         lambda *_: None,
   ...         LinearCostEconomicDispatchConsensusActor(cost=10, p_max=100),
   ...         max_iter=100,
   ...     ),
   ... ]
   >>> p_target = [10, 30, 40, 45, 60, 10]
   >>> msg = AveragingConsensusMessage(lam=np.ones(len(p_target)) * 10, k=0, data=p_target)
   >>> asyncio.run(start_distributed_optimization(actors, msg))
   >>> np.allclose(actors[0]._lam, actors[1]._lam, atol=1e-3)
   True

With heterogeneous costs, the cheaper generator dispatches more power than the pricier one at
convergence (equal-marginal-cost / merit-order dispatch), rather than an equal power split —
see ``tests/consensus/test_consensus_sc.py::test_merit_order_dispatch_with_heterogeneous_costs``.

.. note::

   The algorithm terminates after exactly ``max_iter`` gossip rounds — there is no
   residual-based stopping criterion.  In a fully connected graph convergence is typically
   fast (10–30 rounds); increase ``max_iter`` for larger or sparser networks.

See Also
--------

- :func:`~distributed_resource_optimization.create_averaging_consensus_participant`,
  :class:`~distributed_resource_optimization.AveragingConsensusAlgorithm`
- :class:`~distributed_resource_optimization.ConsensusActor`,
  :class:`~distributed_resource_optimization.NoConsensusActor`
- :class:`~distributed_resource_optimization.LinearCostEconomicDispatchConsensusActor`
