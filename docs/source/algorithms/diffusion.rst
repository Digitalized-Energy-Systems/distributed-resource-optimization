Diffusion & Exact Diffusion
===========================

The Diffusion algorithm coordinates :math:`N` agents on a shared price signal
:math:`\lambda(t)` over a scheduling horizon via peer-to-peer *adapt-then-combine*
iterations (Ces et al. 2025, *"Fully distributed economic dispatch with storage via
diffusion strategies"*). Exact Diffusion (same paper, Sec. 3.4; underlying theory:
Yuan, Ling & Sayed 2018, *"Exact Diffusion for Distributed Optimization and
Learning"*) adds a correction stage that removes classical Diffusion's constant-step-size
steady-state bias.

Classical Diffusion (adapt-then-combine)
----------------------------------------

Each participant maintains a price estimate :math:`\lambda_i` and a power iterate
:math:`\varphi_i`. Per iteration it

.. math::

   \varphi_i^{k} = \lambda_i^{k} - \varepsilon \, \nabla J_i(\lambda_i^{k}),
   \qquad
   \lambda_i^{k+1} = w_{ii} \varphi_i^{k} + \sum_{j \in \mathcal{N}_i} w_{ij} \, \varphi_j^{k}

broadcasting :math:`\varphi_i` between the two steps. The combination weights
:math:`w_{ij}` come from the Mean-Metropolis rule by default
(:math:`w_{ij} = 2/(n_i + n_j + 1)`, Ces et al. eq. 19); ``weight_rule`` selects one of
four rules (``mean_metropolis``, ``averaging``, ``relative_degree``, ``hastings``).
Classical Diffusion requires a doubly-stochastic weight matrix to converge, and the
implementation assumes (and verifies at runtime, via the degree attached to every
message) a **degree-regular communication graph** — true for the complete-graph
topologies this package wires up.

Exact Diffusion (adapt-correct-combine)
---------------------------------------

Classical Diffusion with a constant step size converges to a *biased* point whenever
local costs differ. Exact Diffusion inserts a correction between adapt and combine:

.. math::

   \varphi_i^{k} = \lambda_i^{k-1} - \varepsilon_i \nabla J_i(\lambda_i^{k-1}),
   \qquad
   \bar\varphi_i^{k} = \varphi_i^{k} + \lambda_i^{k-1} - \varphi_i^{k-1},
   \qquad
   \lambda_i^{k} = \sum_{j} \bar w_{ij} \bar\varphi_j^{k}

using :math:`\bar W = (I + W)/2` — the averaging with :math:`I` keeps the combine
matrix positive semi-definite, which the convergence proof requires. The per-agent
step :math:`\varepsilon_i = \varepsilon / p_i` uses the Perron-vector entry
:math:`p_i` (``perron_scale``, computed centrally via
:func:`~distributed_resource_optimization.algorithm._weight_rules.perron_vector`);
on a degree-regular graph :math:`p_i \equiv 1`.

Termination
-----------

Both variants share the same stopping machinery:

- **Tolerance criterion**: a participant flags itself converged once
  :math:`\max_t |\lambda^{k} - \lambda^{k-1}| \le` ``tol`` has held for ``patience``
  consecutive rounds; the run ends in the first round where *every* participant is
  flagged. On the complete graphs this package uses, all participants terminate
  simultaneously.
- **max_iter failsafe**: if the tolerance never fires, the run stops after
  ``max_iter`` rounds and the algorithm's ``converged`` attribute stays ``False``.

Usage
-----

.. doctest::

   >>> from distributed_resource_optimization import (
   ...     LinearCostEconomicDispatchDiffusionActor,
   ...     create_diffusion_participant,
   ...     create_diffusion_start,
   ...     start_distributed_optimization,
   ... )
   >>> def on_finish(algo, carrier):
   ...     pass
   >>> actors = [
   ...     create_diffusion_participant(
   ...         on_finish,
   ...         LinearCostEconomicDispatchDiffusionActor(cost=1.0, p_max=100.0, n_guess=3),
   ...         horizon=1,
   ...     )
   ...     for _ in range(3)
   ... ]
   >>> start = create_diffusion_start(10.0, data=90.0, horizon=1)
   >>> asyncio.run(start_distributed_optimization(actors, start))
   >>> total = sum(a.actor.P for a in actors)
   >>> bool(np.allclose(total, 90.0, atol=1.0))
   True

Exact Diffusion is a drop-in replacement — swap the factory:

.. doctest::

   >>> from distributed_resource_optimization import create_exact_diffusion_participant
   >>> actors = [
   ...     create_exact_diffusion_participant(
   ...         on_finish,
   ...         LinearCostEconomicDispatchDiffusionActor(cost=1.0, p_max=100.0, n_guess=3),
   ...         horizon=1,
   ...     )
   ...     for _ in range(3)
   ... ]
   >>> asyncio.run(start_distributed_optimization(actors, create_diffusion_start(10.0, data=90.0, horizon=1)))
   >>> bool(np.allclose(sum(a.actor.P for a in actors), 90.0, atol=1.0))
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
     - Called with ``(algorithm, carrier)`` when the run terminates
   * - ``diffusion_actor``
     - ``None``
     - :class:`~distributed_resource_optimization.DiffusionActor` supplying the gradient
       :math:`\nabla J`; ``None`` means zero gradient
   * - ``initial_lam``
     - ``10.0``
     - Starting scalar broadcast to all :math:`\lambda` dimensions
   * - ``epsilon``
     - ``0.1``
     - Gradient step size (Exact Diffusion: base step before Perron scaling)
   * - ``perron_scale``
     - ``1.0``
     - *(Exact Diffusion only)* this agent's Perron-vector entry :math:`p_i`
   * - ``max_iter``
     - ``300``
     - Failsafe iteration cap
   * - ``tol``
     - ``1e-4``
     - Per-round :math:`\lambda`-change convergence tolerance
   * - ``patience``
     - ``3``
     - Consecutive sub-``tol`` rounds required before flagging convergence
   * - ``horizon``
     - ``24``
     - Number of schedule time steps
   * - ``weight_rule``
     - ``"mean_metropolis"``
     - Combination-weight rule for the combine step

Economic Dispatch & Storage Actors
----------------------------------

Two built-in :class:`~distributed_resource_optimization.DiffusionActor`
implementations drive the price signal toward an economic dispatch:

- :class:`~distributed_resource_optimization.LinearCostEconomicDispatchDiffusionActor` —
  responds with :math:`P(\lambda) = \operatorname{clip}((\lambda - c)/\varepsilon,
  P_{\min}, P_{\max})` and returns the gradient :math:`P(\lambda) - P_{\text{target}}/n`.
- :class:`~distributed_resource_optimization.ReservoirStorageDiffusionActor` — a
  battery/reservoir whose price-driven schedule is projected onto SOC-feasible
  trajectories (power limits, efficiencies, terminal-SOC target) via the shared
  helper in :mod:`~distributed_resource_optimization.misc.util`.

Both clip their power response *inside* the gradient, so respecting limits is part of
the converged fixed point — the paper's outer saturation-flag loop is deliberately
not ported.

.. note::

   For a fixed :math:`\varepsilon`, GW-scale networks can oscillate instead of
   balancing. The benchmark scenarios scale the gradient step from the network's
   size (``0.5 · n · band / Σ p_nom``); pick ``epsilon`` accordingly when wiring
   the algorithm yourself.

See Also
--------

- :func:`~distributed_resource_optimization.create_diffusion_participant`,
  :class:`~distributed_resource_optimization.DiffusionAlgorithm`
- :func:`~distributed_resource_optimization.create_exact_diffusion_participant`,
  :class:`~distributed_resource_optimization.ExactDiffusionAlgorithm`
- :class:`~distributed_resource_optimization.DiffusionActor`,
  :class:`~distributed_resource_optimization.NoDiffusionActor`
