"""First-order distributed methods.

Gradient-, averaging-, and dual-subgradient-based distributed
algorithms, as distinct from the operator-splitting ADMM family and the
combinatorial heuristics:

* :mod:`.consensus` — distributed averaging with optional gradient correction.
* :mod:`.diffusion` — adapt-then-combine diffusion.
* :mod:`.gossip_qp` — primal-dual (dual-subgradient) token-passing QP
  with a single coupling multiplier.
"""
