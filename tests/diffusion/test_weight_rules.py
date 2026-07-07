"""Unit tests for the combination-weight rules shared by Diffusion and Exact Diffusion."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization.algorithm._weight_rules import (
    combination_weights,
    perron_vector,
    regular_graph_weights,
)

RULES = ["mean_metropolis", "averaging", "relative_degree", "hastings"]


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_regular_graph_weights_sum_to_one(rule, degree):
    self_w, neighbor_w = regular_graph_weights(degree, rule)
    assert self_w + degree * neighbor_w == pytest.approx(1.0)


def test_mean_metropolis_matches_ces_et_al_eq19():
    """w_ij = 2/(n_i+n_j+1) — Ces et al. eq. (19), checked directly (not via the
    regular-graph shortcut) for an asymmetric pair of degrees."""
    self_w, weights = combination_weights(3, {"a": 5}, "mean_metropolis")
    assert weights["a"] == pytest.approx(2.0 / (3 + 5 + 1))


def test_averaging_matches_current_uniform_behaviour():
    """Averaging rule reduces to the plain 1/(N+1) split classical Diffusion
    used before the Mean-Metropolis fix."""
    self_w, neighbor_w = regular_graph_weights(4, "averaging")
    assert self_w == pytest.approx(1 / 5)
    assert neighbor_w == pytest.approx(1 / 5)


def test_relative_degree_and_hastings_reduce_to_averaging_on_regular_graph():
    """On a degree-regular graph, relative-degree and Hastings collapse to the
    same uniform weight as averaging — only Mean Metropolis (eq. 19's
    distinctive arithmetic-mean form) differs. This is a well-known
    consequence of the formulas, not a bug."""
    avg = regular_graph_weights(3, "averaging")
    rel = regular_graph_weights(3, "relative_degree")
    hastings = regular_graph_weights(3, "hastings")
    assert rel == pytest.approx(avg)
    assert hastings == pytest.approx(avg)


def test_relative_degree_weights_favor_higher_degree_neighbors():
    """On an irregular graph, relative-degree gives more weight to neighbors
    with more connections."""
    self_w, weights = combination_weights(3, {"low": 1, "high": 5}, "relative_degree")
    assert weights["high"] > weights["low"]


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="Unknown weight rule"):
        combination_weights(2, {"a": 2}, "nonexistent")  # type: ignore[arg-type]


def test_perron_vector_uniform_on_regular_doubly_stochastic_matrix():
    n = 4
    W = np.full((n, n), 1.0 / n)
    p = perron_vector(W)
    assert np.allclose(p, np.ones(n))


def test_perron_vector_normalised_to_sum_n():
    # A left-stochastic (column-sum-1) but non-uniform matrix.
    W = np.array(
        [
            [0.5, 0.2, 0.3],
            [0.3, 0.6, 0.3],
            [0.2, 0.2, 0.4],
        ]
    )
    p = perron_vector(W)
    assert p.sum() == pytest.approx(3.0)
    assert np.all(p > 0)
