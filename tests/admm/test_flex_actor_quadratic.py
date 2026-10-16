"""Quadratic cost term of the box-bounded ADMM flex actor."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from distributed_resource_optimization import create_admm_flex_actor_box_bounded
from distributed_resource_optimization.algorithm.admm.flex_actor import _local_update


def test_box_closed_form_matches_qp_with_quadratic_cost():
    lb, u = np.zeros(3), np.full(3, 10.0)
    S = np.array([2.0, 5.0, 8.0])
    Q = np.array([0.1, 0.5, 0.0])
    v = np.array([-6.0, -6.0, -6.0])
    rho = 0.5
    actor = create_admm_flex_actor_box_bounded(lb, u, S=S, Q=Q)
    x = cp.Variable(3)
    cp.Problem(
        cp.Minimize(S @ x + cp.sum(cp.multiply(Q, cp.square(x))) + rho / 2 * cp.sum_squares(x + v)),
        [x >= lb, x <= u],
    ).solve()
    np.testing.assert_allclose(_local_update(actor, v, rho), x.value, atol=1e-4)


def test_zero_q_is_the_linear_actor():
    lb, u, S, v = np.zeros(2), np.full(2, 5.0), np.array([1.0, 2.0]), np.array([-3.0, -1.0])
    linear = create_admm_flex_actor_box_bounded(lb, u, S=S)
    zero_q = create_admm_flex_actor_box_bounded(lb, u, S=S, Q=0.0)
    assert zero_q.Q is None
    assert np.array_equal(_local_update(linear, v, 0.1), _local_update(zero_q, v, 0.1))


def test_negative_q_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        create_admm_flex_actor_box_bounded(np.zeros(1), np.ones(1), Q=-1.0)
