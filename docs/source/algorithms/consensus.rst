Consensus + Innovations
=======================

Consensus + innovations (Kar & Hug 2012; Hug, Kar & Wu 2015) lets :math:`N` participants
agree on a shared vector :math:`\lambda` (e.g. a price per time step) while each one's local
*innovation* — its mismatch for the current :math:`\lambda` — drives the agreed value to the
point where all mismatches sum to zero.

Algorithm
---------

Participant :math:`j` holds :math:`\lambda_j`. In iteration :math:`i = 1, 2, \dots`:

.. math::

   \lambda_j \leftarrow \lambda_j
     - \beta_i \sum_{l \in \omega_j} \bigl(\lambda_j - \lambda_l\bigr)
     - \alpha_i \, \mathrm{innovation}_j(\lambda_j)

- :math:`\omega_j` — the participant's peers: topology neighbours that take part
- :math:`\beta_i` — consensus step, :math:`\alpha_i` — innovation step, both functions of
  :math:`i` (:func:`~distributed_resource_optimization.power_step` gives :math:`c / i^p`)
- decaying steps with :math:`\beta_i / \alpha_i \to \infty` reach the exact solution;
  constant steps converge faster but only to a neighbourhood of it (Kar & Hug 2012, §V)
- optional ``lam_min``: after each update :math:`\lambda_j \leftarrow \max(\lambda_j,`
  ``lam_min``:math:`)` (default :math:`-\infty`, the papers' unbounded update) — keeps
  :math:`\lambda` finite when a surplus cannot be absorbed by any participant
- optional ``settle_s`` on the coordinator: when all flags say converged it waits ``settle_s``
  and stops only if they still do — set it to the longest report delay when reports travel
  different distances (multi-hop routes), so a late "not converged" flip is not missed
- optional ``round_timeout`` (lossy links, Kar & Hug 2012 §IV): round *k* ends for every
  participant at ``t0 + (k+1)·round_timeout`` (``t0`` from the start message, set by the
  coordinator's ``start_latency``); peers whose λ has not arrived by then are failed links of that
  round (ω_q(t)); a late λ of a finished round is dropped. ``rounds_by_deadline`` /
  ``missing_edges`` count them. Default ``None`` = wait for every peer

Rounds are synchronous: iteration :math:`k` waits for every peer's :math:`\lambda` of
iteration :math:`k`.

Protocol
--------

1. A :class:`~distributed_resource_optimization.ConsensusCoordinator` sends a
   :class:`~distributed_resource_optimization.ConsensusStart` (initial :math:`\lambda`,
   participant list, its own address) to every participant.
2. Participants exchange :class:`~distributed_resource_optimization.ConsensusMessage`\ s
   with their peers and iterate.
3. A participant is *converged* when :math:`\max|\Delta\lambda| <` ``eps`` and its actor's
   :meth:`~distributed_resource_optimization.ConsensusActor.converged` is true; it reports
   every flip of that flag to the coordinator
   (:class:`~distributed_resource_optimization.ConsensusReport`).
4. When all participants are converged the coordinator returns a
   :class:`~distributed_resource_optimization.ConsensusResult` (stop reason, iterations and the
   carrier ``time`` of the stop — simulated time under a mango simulation); with the mango carrier its
   ``OptimizationFinishedMessage`` stops the participants. Each participant also stops on
   its own after ``max_iter`` iterations and then calls ``finish_callback``.

The local problem
-----------------

Subclass :class:`~distributed_resource_optimization.ConsensusActor`:

.. code-block:: python

   class Quadratic(ConsensusActor):
       """x(λ) = clip((λ − b)/a, 0, cap) against a local demand d."""

       def innovation(self, lam):
           self.prev, self.x = self.x, np.clip((lam - self.b) / self.a, 0.0, self.cap)
           return self.x - self.d

       def converged(self):
           return self.prev is not None and np.max(np.abs(self.x - self.prev)) < 1e-6

A complete run on an irregular graph in a mango world is in
``tests/consensus/test_innovations.py``.

See Also
--------

- :class:`~distributed_resource_optimization.ConsensusInnovationsAlgorithm`,
  :class:`~distributed_resource_optimization.ConsensusCoordinator`
- :class:`~distributed_resource_optimization.ConsensusActor`,
  :func:`~distributed_resource_optimization.power_step`
