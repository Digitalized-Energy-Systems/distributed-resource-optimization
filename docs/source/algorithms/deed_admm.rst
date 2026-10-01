DEED-ADMM
=========

DEED-ADMM combines ADMM primal/dual updates with a dynamic-consensus tracker so
that economic dispatch runs fully peer-to-peer — no coordinator (Zhu et al. 2025,
*"DEED-ADMM: A Scalable Distributed Algorithm for Economic Dispatch in Multi-Energy
Systems With Energy Storage"*). This implementation covers the electricity-only
single-carrier case of the paper's energy-hub model: :math:`A_i = I`, no hub power
limit (empty :math:`Y` block), and storage as the paper's :math:`z_i` block.

Algorithm
---------

Each agent :math:`i` is an energy hub with an optional supplier :math:`x_i`, the
hub input :math:`y_i`, optional storage :math:`z_i = [P^{Sd}_i; P^{Sc}_i]`, a local
demand :math:`d_i` (hub balance :math:`y_i + B z_i = d_i`, :math:`B = [I, -I]`),
box-projected copies :math:`\hat p`, a dual :math:`\lambda_i`, and a consensus
tracker :math:`\xi_i`. Per iteration (Algorithm 1):

.. math::

   \tilde\lambda_i = \sum_j w_{ij} \lambda_j, \quad
   \tilde\xi_i = \sum_j w_{ij} \xi_j
   \qquad\text{with } W = I - \beta L

.. math::

   b_x = \gamma x - \tilde\lambda + (\gamma \hat x - \nu_x)
         - \tfrac{\gamma}{2}\tilde\xi - \nabla c_i^{\text{lin}}, \qquad
   x \leftarrow H_x b_x

.. math::

   b_y = \gamma y + \tilde\lambda + \tfrac{\gamma}{2}\tilde\xi, \qquad
   b_z = Z^\top(\gamma \hat z - \nu_z)

.. math::

   y \leftarrow H_y\bigl(b_y + H_i(d - H_y b_y - B H_z b_z)\bigr), \qquad
   z \leftarrow H_z\bigl(b_z + B^\top H_i(d - H_y b_y - B H_z b_z)\bigr)

followed by the box projection
:math:`\hat p \leftarrow \operatorname{clip}(M p + \nu/\gamma,\, \hat p_{\min},\, \hat p_{\max})`
and the tracker/dual updates

.. math::

   \xi \leftarrow \tilde\xi + \Delta x - \Delta y, \qquad
   \lambda \leftarrow \tilde\lambda + \tfrac{\gamma}{2}\xi, \qquad
   \nu \leftarrow \nu + \gamma\,(M p - \hat p).

Without storage the :math:`y`-update collapses to :math:`y = d_i`. With storage,
:math:`Z = [I_{2\tau}; Z_1^\top]` where :math:`Z_1^\top z = E - E_0` is the energy
level change (eq. 9), so the power limits (5), the energy limits (8) and the
terminal reserve :math:`E_\tau \ge \omega E_{\max}` (11) are box constraints on
:math:`\hat z`. The tracker :math:`\xi` measures the running supply–demand
mismatch; at convergence :math:`\sum_i x_i = \sum_i y_i` and :math:`\lambda`
agrees across agents (the market-clearing price).

Protocol
--------

Rounds are synchronous: an agent broadcasts its round-:math:`k` state and steps
once it holds round-:math:`k` state from every neighbour (state from a neighbour
that is one round ahead is buffered). Each agent stops by itself after
``max_iter`` rounds, so no protocol message outlives a run. Messages carry an
``epoch``; a newer epoch starts a new run, while traffic from older epochs or from
a finished run is ignored. An agent without neighbours runs all rounds locally.

Deviations from the paper (by design)
-------------------------------------

- Ramp-rate constraints (eqs. 16–17) are not modelled — the benchmark networks
  specify none.
- Fixed ``max_iter`` instead of a residual-based stop; the reported schedule is the
  box-projected :math:`\hat x` (suppliers) or :math:`\hat P^{Sd} - \hat P^{Sc}`
  (storage), so power limits hold even when cut off early.

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
     - Total number of participants; sets the default ``beta = 1/n``
   * - ``beta``
     - ``1/n_agents``
     - Laplacian step of :math:`W = I - \beta L` (:math:`w_{ij} = \beta` per
       neighbour, :math:`w_{ii} = 1 - \beta |N_i|`). On a complete graph the
       default gives the uniform average; any connected graph works as long as
       :math:`\beta |N_i| < 1` (checked at start-up). The paper uses
       :math:`\beta = 0.2` on a sparse graph.

Participant factories
---------------------

- :func:`~distributed_resource_optimization.create_deed_admm_thermal_participant` —
  generator with cost :math:`a_i x^2 + b_i x` and (optionally per-step) box limits.
- :func:`~distributed_resource_optimization.create_deed_admm_renewable_participant` —
  unit whose upper bound is a per-step availability time series, with an optional
  linear ``marginal_cost`` (zero for wind/solar, non-zero for e.g. biomass).
- :func:`~distributed_resource_optimization.create_deed_admm_storage_participant` —
  battery/pumped hydro with power bounds, efficiencies, SOC bounds and a terminal
  reserve :math:`E_\tau \ge e_{\text{final}} E_{\max}`; its ``E`` attribute carries
  the SOC trajectory at termination. By the paper's Theorem 1 the relaxation is
  exact when :math:`\eta_c \eta_d < 1`: optimal schedules never charge and
  discharge at once.

See Also
--------

- :class:`~distributed_resource_optimization.DEEDADMMAlgorithm`,
  :class:`~distributed_resource_optimization.DEEDADMMStorageAlgorithm`
- :class:`~distributed_resource_optimization.DEEDADMMMessage`
