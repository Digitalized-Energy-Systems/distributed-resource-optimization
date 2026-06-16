Averaging Consensus
===================

The Averaging Consensus algorithm distributes a parameter vector :math:`\lambda` across
:math:`N` agents via a gossip-style protocol. Each agent maintains its own copy of
:math:`\lambda` and iteratively averages it with values received from neighbours. An
optional gradient term allows each agent to steer the consensus towards locally desirable
values.

Algorithm
---------

Let :math:`\lambda_i^k` be the value held by agent :math:`i` at iteration :math:`k`.
The update rule is:

.. math::

   \lambda_i^{k+1} = \lambda_i^k
                   + \alpha \bigl(\bar{\lambda}^k - \lambda_i^k\bigr)
                   + \nabla_i^k

where

- :math:`\bar{\lambda}^k` is the average of all received values at iteration :math:`k`
- :math:`\alpha \in (0,1]` is the step size (mixing parameter)
- :math:`\nabla_i^k = \text{gradient\_term}(\text{actor}_i, \lambda_i^k, \text{data})`
  is the local gradient correction (zero by default)

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
     - :class:`~distributed_resource_optimization.ConsensusActor` providing gradient term
   * - ``initial_lam``
     - ``10.0``
     - Starting scalar value broadcast to all :math:`\lambda` components
   * - ``alpha``
     - ``0.3``
     - Mixing step size
   * - ``max_iter``
     - ``50``
     - Number of gossip rounds before finishing

Local Gradient Corrections
---------------------------

To steer the consensus, subclass :class:`~distributed_resource_optimization.ConsensusActor`
and override :meth:`~distributed_resource_optimization.ConsensusActor.gradient_term`:

.. doctest::

   >>> from distributed_resource_optimization import ConsensusActor
   >>> class PushToTarget(ConsensusActor):
   ...     def __init__(self, target, step=0.05):
   ...         self.target = np.asarray(target)
   ...         self.step = step
   ...     def gradient_term(self, lam, data):
   ...         return self.step * (self.target - lam)

The ``data`` argument carries whatever was embedded in the initial
:class:`~distributed_resource_optimization.AveragingConsensusMessage` — useful for
passing problem data alongside the consensus.

Economic Dispatch
-----------------

The built-in
:class:`~distributed_resource_optimization.LinearCostEconomicDispatchConsensusActor`
implements consensus-based economic dispatch. Each agent has a linear cost and power
limits; the gradient term pushes :math:`\lambda` toward the clearing price at which
supply equals demand:

.. math::

   P(\lambda) = \operatorname{clip}\!\left(\frac{\lambda - c}{\epsilon},\; P_{\min},\; P_{\max}\right)

   \nabla = -\rho \Bigl(P(\lambda) - \frac{P_{\text{target}}}{N}\Bigr)

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
   ...         LinearCostEconomicDispatchConsensusActor(cost=10, p_max=100, n_guess=3),
   ...         max_iter=100,
   ...     ),
   ...     create_averaging_consensus_participant(
   ...         lambda *_: None,
   ...         LinearCostEconomicDispatchConsensusActor(cost=10, p_max=100, n_guess=3),
   ...         max_iter=100,
   ...     ),
   ...     create_averaging_consensus_participant(
   ...         lambda *_: None,
   ...         LinearCostEconomicDispatchConsensusActor(cost=10, p_max=100, n_guess=3),
   ...         max_iter=100,
   ...     ),
   ... ]
   >>> p_target = [10, 30, 40, 45, 60, 10]
   >>> msg = AveragingConsensusMessage(lam=np.ones(len(p_target)) * 10, k=0, data=p_target)
   >>> asyncio.run(start_distributed_optimization(actors, msg))
   >>> np.allclose(actors[0]._lam, actors[1]._lam, atol=1e-3)
   True

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
