"""Combination-weight rules for diffusion-type distributed algorithms.

Ces et al. (2025) use the "Mean Metropolis" rule (their eq. 19,
``w_ij = 2/(n_i + n_j + 1)``, borrowed from the paper's consensus derivation)
for classical Diffusion (Sec. 3.3), and evaluate four rules -- averaging,
relative degree, Mean Metropolis, and Hastings -- for Exact Diffusion
(Sec. 3.5, Table I), attributing the weight-matrix constructions to
Y. He, W. Wang, X. Wu, "Multi-agent based fully distributed economic
dispatch in microgrid using exact diffusion strategy," IEEE Access 8 (2020)
7020-7031. That paper sits behind the IEEE Xplore paywall and could not be
retrieved; the averaging, relative-degree, and Metropolis/Hastings formulas
below instead follow the standard graph-theoretic definitions collected in
A. H. Sayed, "Diffusion Adaptation over Networks," arXiv:1205.4220, Table 7
-- the canonical source these rule names trace back to (Sayed's own
"Metropolis rule" entry is jointly cited to Metropolis and Hastings, i.e.
the classical Metropolis-Hastings weight). Only Mean Metropolis (eq. 19's
distinctive arithmetic-mean form) is taken verbatim from Ces et al.; the
other three should be revisited against He/Wang/Wu directly if exact
fidelity to that paper's numbers is required.

All rules need each node's own degree; averaging/relative-degree/Hastings
additionally need each *neighbor's* degree. On a degree-regular graph (this
codebase always wires diffusion over a complete graph; the paper's own
network model is a 2-regular ring) every neighbor's degree equals the
node's own, so :func:`regular_graph_weights` covers that case directly from
local degree alone, without a full per-neighbor degree exchange. Callers
(:mod:`.diffusion.diffusion`, :mod:`.diffusion.exact_diffusion`) still verify
this assumption at runtime -- each message carries the sender's own degree,
and a mismatch raises rather than silently producing a non-stochastic matrix.
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np

WeightRule = Literal["mean_metropolis", "averaging", "relative_degree", "hastings"]


def combination_weights(
    own_degree: int,
    neighbor_degrees: Mapping[object, int],
    rule: WeightRule = "mean_metropolis",
) -> tuple[float, dict[object, float]]:
    """Compute combination weights for one node from its and its neighbors' degrees.

    :param own_degree: Number of communication neighbors of this node (``n_k``),
                       not counting itself.
    :param neighbor_degrees: Map from neighbor address to that neighbor's own
                             degree (``n_l``).
    :param rule: Which combination rule to apply.
    :returns: ``(self_weight, {addr: neighbor_weight})``; all weights sum to 1.
    :raises ValueError: If *rule* is not recognised.
    """
    n_k = own_degree

    if rule == "mean_metropolis":
        # Ces et al. eq. (19): w_ij = 2 / (n_i + n_j + 1)
        weights = {addr: 2.0 / (n_k + n_l + 1) for addr, n_l in neighbor_degrees.items()}
    elif rule == "averaging":
        # Sayed Table 7, rule 1: a_lk = 1/n_k' with n_k' = n_k + 1 (closed neighborhood)
        w = 1.0 / (n_k + 1)
        weights = {addr: w for addr in neighbor_degrees}
    elif rule == "relative_degree":
        # Sayed Table 7, rule 6: a_lk = n_l' / sum_{m in closed N_k} n_m'
        closed_self = n_k + 1
        closed_neighbors = {addr: n_l + 1 for addr, n_l in neighbor_degrees.items()}
        denom = closed_self + sum(closed_neighbors.values())
        self_weight = closed_self / denom
        weights = {addr: n_l1 / denom for addr, n_l1 in closed_neighbors.items()}
        return self_weight, weights
    elif rule == "hastings":
        # Sayed Table 7, rule 5 (jointly cited to Metropolis & Hastings):
        # a_lk = 1 / max(n_k', n_l')
        closed_self = n_k + 1
        weights = {addr: 1.0 / max(closed_self, n_l + 1) for addr, n_l in neighbor_degrees.items()}
    else:
        raise ValueError(f"Unknown weight rule: {rule!r}")

    self_weight = 1.0 - sum(weights.values())
    return self_weight, weights


def regular_graph_weights(
    own_degree: int, rule: WeightRule = "mean_metropolis"
) -> tuple[float, float]:
    """Self- and (uniform) neighbor-weight on a degree-regular graph.

    Valid whenever every node has the same degree as *own_degree* -- true for
    this codebase's complete-graph topologies and for the paper's own ring
    network -- and avoids needing a degree handshake between neighbors.

    :param own_degree: Number of communication neighbors of this node.
    :param rule: Which combination rule to apply.
    :returns: ``(self_weight, neighbor_weight)``; ``self_weight + own_degree *
             neighbor_weight == 1``.
    """
    if own_degree == 0:
        return 1.0, 0.0
    dummy_neighbor_degrees = {i: own_degree for i in range(own_degree)}
    self_weight, weights = combination_weights(own_degree, dummy_neighbor_degrees, rule)
    neighbor_weight = next(iter(weights.values()))
    return self_weight, neighbor_weight


def perron_vector(weight_matrix: np.ndarray) -> np.ndarray:
    """Return the Perron eigenvector ``p`` (eigenvalue 1) of a left-stochastic matrix.

    Used for Exact Diffusion's per-agent feedback gain (Ces et al. eq. 31:
    ``epsilon_i = epsilon / p_i``). Normalised so ``sum(p) == len(p)`` -- on a
    regular graph (where the combination matrix reduces to a uniform doubly
    stochastic matrix) this yields ``p_i == 1`` for all agents, i.e. no
    per-agent scaling.

    :param weight_matrix: Square left-stochastic combination matrix (columns sum to 1).
    :returns: Real, positive Perron eigenvector, normalised to sum to ``N``.
    """
    eigvals, eigvecs = np.linalg.eig(weight_matrix)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    p = np.real(eigvecs[:, idx])
    p = np.abs(p)
    return p * (len(p) / p.sum())
