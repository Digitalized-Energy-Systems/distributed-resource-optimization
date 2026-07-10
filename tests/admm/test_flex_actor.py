"""Unit tests for ADMMFlexActor's local QP and coupling-constraint builder.

``_local_update`` and ``_create_C_and_d`` were previously exercised only
inside full coordinated ADMM runs; these tests pin their math directly
against hand-computed optima.
"""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization.algorithm.admm.flex_actor import (
    _create_C_and_d,
    _local_update,
    create_admm_flex_actor_box_bounded,
    create_admm_flex_actor_one_to_many,
)


class TestLocalUpdateBoxOnly:
    """Box-only actors use the closed form x* = clip(-v - S/rho, lb, u)."""

    def test_unconstrained_optimum(self):
        actor = create_admm_flex_actor_box_bounded([-10.0], [10.0], S=[0.5])
        x = _local_update(actor, v=np.array([0.2]), rho=2.0)
        # -v - S/rho = -0.2 - 0.25 = -0.45, inside the box.
        assert x == pytest.approx([-0.45])

    def test_clipped_at_bounds(self):
        actor = create_admm_flex_actor_box_bounded([0.0, 0.0], [1.0, 1.0])
        x = _local_update(actor, v=np.array([-5.0, 5.0]), rho=1.0)
        assert x == pytest.approx([1.0, 0.0])

    def test_rho_scales_cost_influence(self):
        actor = create_admm_flex_actor_box_bounded([-10.0], [10.0], S=[1.0])
        x_small_rho = _local_update(actor, v=np.array([0.0]), rho=0.5)
        x_large_rho = _local_update(actor, v=np.array([0.0]), rho=10.0)
        assert x_small_rho == pytest.approx([-2.0])  # S/rho dominates
        assert x_large_rho == pytest.approx([-0.1])


class TestLocalUpdateCoupled:
    """With coupling constraints the QP is solved via cvxpy/OSQP."""

    def test_one_to_many_ratio_is_enforced(self):
        """tech_capacity [1, 2] ties x0/1 == x1/2; minimising
        rho/2*||x||^2 + (rho*v)^T x over that line with v = [-1, -1] gives
        t* = (1 + 2)/(5) = 0.6, i.e. x = [0.6, 1.2]."""
        actor = create_admm_flex_actor_one_to_many(1.0, [1.0, 2.0])
        x = _local_update(actor, v=np.array([-1.0, -1.0]), rho=1.0)
        assert x == pytest.approx([0.6, 1.2], abs=1e-3)
        assert x[1] == pytest.approx(2.0 * x[0], abs=1e-3)

    def test_capacity_row_caps_total_allocation(self):
        actor = create_admm_flex_actor_one_to_many(1.0, [1.0, 2.0])
        # Strong pull toward the upper bounds.
        x = _local_update(actor, v=np.array([-100.0, -100.0]), rho=1.0)
        assert np.sum(np.abs(x)) <= 3.0 + 1e-6
        assert x[1] == pytest.approx(2.0 * x[0], abs=1e-3)


class TestCreateCAndD:
    def test_capacity_row_signs_and_rhs(self):
        C, d = _create_C_and_d(np.array([1.0, -2.0, 3.0]))
        np.testing.assert_allclose(C[0], [1.0, -1.0, 1.0])
        assert d[0] == 6.0  # sum of absolute capacities

    def test_ratio_rows_tie_outputs_to_last(self):
        C, d = _create_C_and_d(np.array([2.0, 4.0]))
        assert C.shape == (3, 2)
        # Rows 1/2 encode x0/2 == x1/4 as two opposing inequalities.
        np.testing.assert_allclose(C[1], [0.5, -0.25])
        np.testing.assert_allclose(C[2], [-0.5, 0.25])
        np.testing.assert_allclose(d[1:], 0.0)

    def test_zero_capacity_rows_are_skipped(self):
        C, d = _create_C_and_d(np.array([0.0, 4.0]))
        # The ratio pair for the zero-capacity output must be all-zero rows.
        np.testing.assert_allclose(C[1:], 0.0)

    def test_one_to_many_box_bounds_respect_sign(self):
        actor = create_admm_flex_actor_one_to_many(10.0, [0.5, -1.0])
        np.testing.assert_allclose(actor.lb, [0.0, -10.0])
        np.testing.assert_allclose(actor.u, [5.0, 0.0])
