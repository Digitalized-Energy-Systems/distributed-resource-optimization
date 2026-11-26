Exact diffusion
===============

Adapt-correct-combine diffusion (Yuan, Ying, Zhao, Sayed 2019, Part I, Algorithm 1): every
participant :math:`k` holds an estimate :math:`w_k` of a common vector and the network minimises
:math:`\sum_k J_k(w)`. A correction step between adaptation and combination removes plain
diffusion's :math:`O(\mu)` bias, so the estimates reach the exact minimiser.

Algorithm
---------

Weights: Hastings rule with equal :math:`\mu_k/q_k` (Policy 1, eq. 18),
:math:`a_{\ell k} = 1/\max(n_k, n_\ell)` with :math:`n` the neighbourhood size including the node
(the same rule as :doc:`diffusion`), from the neighbours' degrees carried in the packets;
:math:`\bar A = (I + A)/2`. The rule is symmetric and doubly stochastic (locally balanced, uniform
Perron vector), so one step :math:`\mu` serves all participants (eq. 13). Per round :math:`i`:

.. math::

   \psi_{k,i} = w_{k,i-1} - \mu \nabla J_k(w_{k,i-1}) \qquad \text{(98, adaptation)}

   \phi_{k,i} = \psi_{k,i} + w_{k,i-1} - \psi_{k,i-1} \qquad \text{(99, correction)}

   w_{k,i} = \sum_{\ell} \bar a_{\ell k} \phi_{\ell,i} \qquad \text{(100, combination)}

- :math:`w_{k,-1}` = :meth:`~distributed_resource_optimization.ExactDiffusionActor.start`,
  :math:`\psi_{k,-1} = w_{k,-1}`; :math:`\nabla J_k` =
  :meth:`~distributed_resource_optimization.ExactDiffusionActor.gradient`
- one packet per edge per round
  (:class:`~distributed_resource_optimization.ExactDiffusionMessage`): :math:`\phi`, degree
- optional ``round_timeout`` (lossy links): round *i* ends at ``t0 + (i+1)·round_timeout``; a
  missing packet's weight stays with :math:`k` that round

Stop and protocol
-----------------

A :class:`~distributed_resource_optimization.ConsensusCoordinator` (``lam0 = ()``) starts the run
and ends it once every participant reports ``max_iter``; there is no convergence stop (the
algorithm repeats for :math:`i = 0, 1, \dots`).

.. code-block:: python

   algo = ExactDiffusionAlgorithm(finish_callback, actor, step=0.1, max_iter=10_000)
