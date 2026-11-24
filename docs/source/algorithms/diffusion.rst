Diffusion
=========

Fully distributed combine-then-adapt (CTA) diffusion with a sum constraint (de Azevedo et al.
2017, §III): every participant holds an estimate :math:`w_i` of all entries (one row per
entry, one column per period), its own cost depends only on its own entry, and the estimates keep
:math:`\sum_j w[j] = P_N`. The target :math:`P_N` is learnt by a second diffusion (information
sharing) that runs alongside in the same packet.

Algorithm
---------

Weights: Metropolis eq. (12) as printed, :math:`a_{ij} = 1/\max(n_i, n_j)` with :math:`n_i = |N_i|`
the neighbourhood size including the node itself (degree + 1, as Xiao–Boyd's Metropolis–Hastings
rule) from the neighbours' degrees carried in the packets, :math:`a_{ii} = 1 - \sum_j a_{ij}`
(:func:`~distributed_resource_optimization.algorithm.firstorder.diffusion.diffusion.metropolis`).
Per round :math:`k`:

.. math::

   \text{information sharing (13):}\quad
   \varphi_i = \sum_j a_{ij} x_j, \qquad x_i = \varphi_i - \mu_{IS}\,(\varphi_i^{prev} - \varphi_i)

   \text{optimization (16, 20, 21):}\quad
   \phi_i = \sum_j a_{ij} w_j, \qquad
   \Delta = -\mu \nabla J_i(\phi_i[i]), \qquad
   w_i = \phi_i + \Delta\,e_i + \tfrac{P_N - \sum \phi_i - \Delta}{m - 1} \sum_{j \ne i,\ j\ \text{active}} e_j

- :math:`x_i` starts at :meth:`~distributed_resource_optimization.DiffusionActor.measure` and
  tends to the mean; :math:`\nabla J_i` is
  :meth:`~distributed_resource_optimization.DiffusionActor.gradient`
- *active* entries (e.g. dispatchable units) adapt and take the nudge, split equally over the
  other active entries (Fig. 8); an inactive participant only combines and its entry stays 0
- convergence of the information sharing by flags among neighbours: flag
  :math:`\max|\Delta x_i| \le \varepsilon_{IS}`, counter
  :math:`c_i = 1 + \min(c_i, c_j)` while it holds; past the diameter :math:`d` the earliest
  proposed reset round is flooded, and in that round every participant sets
  :math:`P_N = n\,x_i` (eq. 10) and restarts :math:`x_i` from its measurement; the first reset
  starts the optimization at :math:`w = 0`; :math:`\nabla J_i` is only asked of active
  participants
- :math:`n` = the participants of the start message (the paper floods an ID database, §III-A;
  like :math:`d` it is known at setup)
- one packet per edge per round (:class:`~distributed_resource_optimization.DiffusionMessage`):
  :math:`x`, :math:`w`, degree, counter, reset proposal
- eq. (13)'s :math:`\hat s_i = \varphi_i^{prev} - \varphi_i`: :math:`\mu_{IS} \ge 0` speeds the
  averaging up (the paper's Fig. 6 / Table I; the opposite sign only slows it)
- eq. (21)'s deficit is taken on the combined estimate :math:`\phi_i`, so every active estimate
  sums to :math:`P_N` each round (the previous estimate's sum :math:`\sum w_{i,k-1}` lets the
  sum error grow as :math:`e_k = (A - D)e_{k-1}`); with one :math:`\mu` for all the fixed point
  has equal gradients up to an :math:`O(\mu)` offset
- optional ``round_timeout`` (lossy links): round *k* ends at ``t0 + (k+1)·round_timeout``; a
  missing packet's weights stay with :math:`i` that round; no neighbour is dropped (the paper's
  :math:`d`-round drop is for nodes that leave, not for lost packets)

Stop and protocol
-----------------

A :class:`~distributed_resource_optimization.ConsensusCoordinator` (``lam0 = ()``; the number
of periods is the length of the measurement) starts the run and ends it once every participant
reports ``max_iter``; there is no convergence stop (the paper tracks continuously).

.. code-block:: python

   algo = DiffusionAlgorithm(finish_callback, actor, own=i, active=active, entries=n,
                             step=0.01, step_is=0.0, eps_is=1e-4, diameter=d,
                             max_iter=10_000)
