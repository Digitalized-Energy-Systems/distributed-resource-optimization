DEED-ADMM
=========

DEED-ADMM (Zhu et al. 2025, Algorithm 1) solves :math:`\min \sum_i f_i(x_i, y_i)` subject to
:math:`\sum_i (x_i - y_i) = 0` with neighbour messages only: each participant keeps a local copy
:math:`\lambda_i` of the multiplier and a tracker :math:`\xi_i` of the mean coupling
:math:`x - y` (dynamic consensus), both vectors in :math:`\mathbb{R}^T`.

Algorithm
---------

With :math:`W = I - \beta L` of the topology (:math:`w_{ij} = \beta` for a neighbour), per
iteration :math:`k`:

.. math::

   \tilde\lambda_i = \lambda_i + \beta \sum_{j} (\lambda_j - \lambda_i), \qquad
   \tilde\xi_i = \xi_i + \beta \sum_{j} (\xi_j - \xi_i)

   \Delta_i = \mathrm{update}_i(\tilde\lambda_i, \tilde\xi_i)
     = (x_{i,k+1} - x_{i,k}) - (y_{i,k+1} - y_{i,k})

   \xi_i \leftarrow \tilde\xi_i + \Delta_i, \qquad
   \lambda_i \leftarrow \tilde\lambda_i + \tfrac{\gamma}{2} \xi_i

- the local update (Algorithm 1 lines 5–11 and 14: the primal steps, the projection onto the
  local box :math:`\hat p_i`, the multiplier :math:`\nu_i`) is the participant's
  (:class:`~distributed_resource_optimization.DEEDParticipant`); :math:`\xi_{i,0}` is its
  initial coupling :math:`x_{i,0} - y_{i,0}`
- Assumption 1 (:math:`W` doubly stochastic, :math:`\rho(W - \mathbf{1}\mathbf{1}^\top/n) < 1`)
  needs :math:`\beta \, \lambda_{\max}(L) < 2`; then :math:`\sum_i \xi_i = \sum_i (x_i - y_i)`
  every iteration (Lemma 2)
- one message per edge per iteration carries :math:`(\lambda_j, \xi_j)`
  (:class:`~distributed_resource_optimization.DEEDMessage`); rounds are synchronous
- optional ``round_timeout`` (lossy links): round *k* ends at ``t0 + (k+1)·round_timeout``;
  a missing :math:`(\lambda_j, \xi_j)` drops edge :math:`(i, j)` for that round and its weight
  stays with :math:`i`. A lost :math:`\xi_j` breaks Lemma 2 for good, so under loss the fixed
  point moves (power-balance error), unlike consensus + innovations where loss slows it down

Stop and protocol
-----------------

Started and stopped by a :class:`~distributed_resource_optimization.ConsensusCoordinator` (its
``lam0`` is :math:`\lambda_{i,0}`), with the consensus start / report messages. A participant is
*converged* when :math:`\|\xi_i\|_2 \le \sqrt{T}\,\varepsilon_{abs} + \varepsilon_{rel}\|x_i\|_2`
(primal, :math:`\xi \to 0`) and its participant's
:meth:`~distributed_resource_optimization.DEEDParticipant.converged` (the dual check on
:math:`\hat p_i`, :math:`\nu_i`) is true; flips go to the coordinator, which stops all once every
flag is set. Each participant also stops after ``max_iter`` iterations.

.. code-block:: python

   algo = DEEDADMMAlgorithm(finish_callback, participant, beta=0.2, gamma=1.0,
                            eps_abs=1e-4, eps_rel=1e-3, max_iter=10_000)
