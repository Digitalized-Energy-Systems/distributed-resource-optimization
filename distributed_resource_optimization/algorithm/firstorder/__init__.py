"""First-order distributed methods.

Gradient-, averaging-, and dual-subgradient-based distributed
algorithms, as distinct from the operator-splitting ADMM family and the
combinatorial heuristics:

* :mod:`.consensus` — distributed averaging (incl. economic dispatch).
* :mod:`.diffusion` — adapt-then-combine diffusion and exact diffusion
  (incl. economic dispatch and reservoir storage).
* :mod:`.fdgdm` — fast distributed gradient descent (Bai et al. 2022)
  economic dispatch.
* :mod:`.gossip_qp` — primal-dual (dual-subgradient) token-passing QP
  with a single coupling multiplier.
"""
