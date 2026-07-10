DEED-ADMM
=========

DEED-ADMM combines ADMM primal/dual updates with a dynamic-consensus tracker so
that economic dispatch runs fully peer-to-peer — no coordinator (Zhu et al. 2025,
*"DEED-ADMM: A Scalable Distributed Algorithm for Economic Dispatch in Multi-Energy
Systems With Energy Storage"*). This implementation covers the electricity-only
single-carrier case of the paper's multi-energy hub model
(:math:`A_i = I`, :math:`B = 0`, :math:`z_i = 0`).

Algorithm
---------

Each agent :math:`i` holds a supply variable :math:`x_i`, a demand-side variable
:math:`y_i` (pinned to its demand allocation :math:`d_i`), box-projected copies
:math:`\hat p`, a dual :math:`\lambda_i`, and a consensus tracker :math:`\xi_i`.
Per iteration (Algorithm 1):

.. math::

   \tilde\lambda_i = \sum_j w_{ij} \lambda_j, \quad
   \tilde\xi_i = \sum_j w_{ij} \xi_j
   \qquad\text{(uniform } w_{ij} = 1/n\text{)}

.. math::

   b_x = \gamma x - \tilde\lambda + (\gamma \hat p_x - v_x)
         - \tfrac{\gamma}{2}\tilde\xi - \nabla c_i^{\text{lin}}, \qquad
   x \leftarrow H_x b_x

followed by the :math:`y`-projection onto :math:`A_i y = d_i`, the box projection
:math:`\hat p \leftarrow \operatorname{clip}(x + v/\gamma,\, p_{\min},\, p_{\max})`,
and the tracker/dual updates

.. math::

   \xi \leftarrow \tilde\xi + \Delta x - \Delta y, \qquad
   \lambda \leftarrow \tilde\lambda + \tfrac{\gamma}{2}\xi, \qquad
   v \leftarrow v + \gamma\,(x - \hat p).

The tracker :math:`\xi` measures the running supply–demand mismatch; at convergence
:math:`\sum_i x_i = \sum_i d_i` and :math:`\lambda` agrees across agents (the
market-clearing price).

Deviations from the paper (by design)
-------------------------------------

- Ramp-rate constraints (eqs. 16–17) are not modelled — the benchmark networks
  specify none.
- Fixed ``max_iter`` instead of a residual-based stop; the reported schedule is the
  box-projected :math:`\hat p_x` (generators) or the SOC-feasible :math:`x`
  (storage), so it stays feasible even when cut off early.
- Storage SOC dynamics use a greedy forward projection with terminal-SOC bisection
  (mirroring the diffusion storage actor) instead of the paper's linear-inequality
  :math:`M_i` mechanism — convergence guarantees do not formally carry over, but it
  keeps the per-iteration cost trivial and works well empirically.

Usage
-----

.. doctest::

   >>> from distributed_resource_optimization import (
   ...     DEEDADMMMessage,
   ...     create_deed_admm_thermal_participant,
   ...     start_distributed_optimization,
   ... )
   >>> results = {}
   >>> def make_finish(aid):
   ...     def cb(algorithm, carrier):
   ...         results[aid] = algorithm.P.copy()
   ...     return cb
   >>> demand = np.array([100.0])
   >>> actors = [
   ...     create_deed_admm_thermal_participant(
   ...         make_finish(i),
   ...         p_min=0.0, p_max=100.0, marginal_cost=1.0,
   ...         d_i=demand / 2, n_agents=2,
   ...     )
   ...     for i in range(2)
   ... ]
   >>> start = DEEDADMMMessage(lam=np.zeros(1), xi=np.zeros(1), k=0, data=None, initial=True)
   >>> asyncio.run(start_distributed_optimization(actors, start))
   >>> bool(np.allclose(results[0] + results[1], demand, atol=1.0))
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
     - Called with ``(algorithm, carrier)`` when ``max_iter`` is reached; the final
       schedule is in the algorithm's ``P`` attribute
   * - ``d_i``
     - —
     - Per-step demand allocation; the sum over all agents must equal total demand
       (storage units get zeros)
   * - ``gamma``
     - ``0.05``
     - ADMM penalty parameter :math:`\gamma`
   * - ``max_iter``
     - ``500``
     - Number of iterations (no residual-based stop)
   * - ``n_agents``
     - ``1``
     - Total number of participants; must match the actual (complete-graph)
       neighbourhood size + 1, since the uniform weights are :math:`1/n`

Participant factories
---------------------

- :func:`~distributed_resource_optimization.create_deed_admm_thermal_participant` —
  generator with cost :math:`a_i x^2 + b_i x` and box limits.
- :func:`~distributed_resource_optimization.create_deed_admm_renewable_participant` —
  zero-marginal-cost unit whose upper bound is a per-step availability time series.
- :func:`~distributed_resource_optimization.create_deed_admm_storage_participant` —
  battery/reservoir with power bounds, efficiencies, SOC bounds, and a terminal-SOC
  target; its ``E`` attribute carries the SOC trajectory at termination.

.. note::

   The uniform weight matrix assumes a **complete communication graph**: every agent
   averages over all :math:`n` participants each round. Running on a sparser
   topology with ``n_agents`` unchanged silently mis-weights the consensus.

See Also
--------

- :class:`~distributed_resource_optimization.DEEDADMMAlgorithm`,
  :class:`~distributed_resource_optimization.DEEDADMMStorageAlgorithm`
- :class:`~distributed_resource_optimization.DEEDADMMMessage`
