"""Shared storage model (misc/storage.py, decision Q15)."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from distributed_resource_optimization.misc.storage import StorageModel


def _battery(**kw) -> StorageModel:
    base = dict(e_max=10.0, p_charge_max=4.0, p_discharge_max=5.0, e_initial=0.5)
    base.update(kw)
    return StorageModel(**base)


def test_price_response_matches_an_independent_solve():
    model = _battery(discharge_cost=2.0)
    lam = np.array([1.0, 1.0, 20.0, 25.0, 3.0, 30.0])
    eps = 0.5
    p, e = model.price_response(lam, eps)

    T = lam.size
    d, c = cp.Variable(T), cp.Variable(T)
    lower = np.tril(np.ones((T, T)))
    path = model.e0 + lower @ (0.95 * c - d / 0.95)
    cp.Problem(
        cp.Minimize(
            2.0 * cp.sum(d) - lam @ (d - c) + eps / 2 * (cp.sum_squares(d) + cp.sum_squares(c))
        ),
        [d >= 0, c >= 0, d <= 5.0, c <= 4.0, path >= 0, path <= 10.0, path[-1] >= model.e0],
    ).solve(solver=cp.CLARABEL)
    np.testing.assert_allclose(p, d.value - c.value, atol=1e-5)
    assert e[-1] >= model.e0 - 1e-6  # E_T >= E_0


def test_proximal_response_fixed_point_is_unbiased():
    """Iterated (with a vanishing price drift, as in a converging price
    iteration), the centred response reaches the exact best response: fill the
    battery in the cheap hours, empty it back to E_0 in the expensive ones.
    The zero-centred price_response stays pulled toward idle storage."""
    model = _battery()
    lam = np.array([9.0, 9.0, 11.0, 11.0])
    p = None
    for k in range(300):
        p, e = model.proximal_response(lam + 1e-9 * k, 1.0, p)
    assert e.max() == pytest.approx(10.0, abs=1e-3)  # full
    assert e[-1] == pytest.approx(model.e0, abs=1e-3)  # back to E_0
    biased, _ = model.price_response(lam, 1.0)
    assert np.abs(biased).sum() < 0.5 * np.abs(p).sum()


def test_price_response_charges_cheap_and_discharges_expensive_hours():
    model = _battery()
    p, _ = model.price_response(np.array([1.0, 1.0, 50.0, 50.0]), 0.1)
    assert p[0] < 0 and p[1] < 0
    assert p[2] > 0 and p[3] > 0
    d, c = np.maximum(p, 0), np.maximum(-p, 0)
    assert model.is_feasible(d, c)


def test_projection_keeps_feasible_targets_and_repairs_infeasible_ones():
    model = _battery()
    feasible = np.array([-2.0, -2.0, 1.0, 2.0])
    p, _ = model.project(feasible)
    np.testing.assert_allclose(p, feasible, atol=1e-5)

    p, e = model.project(np.array([5.0, 5.0, 5.0, 5.0]))  # would empty the battery
    d, c = np.maximum(p, 0), np.maximum(-p, 0)
    assert model.is_feasible(d, c, tol=1e-5)
    assert e[-1] >= model.e0 - 1e-5


def test_is_feasible_detects_a_drained_battery():
    model = _battery()
    assert not model.is_feasible(np.array([5.0]), np.array([0.0]))  # ends below E_0


def test_terminal_target_is_respected():
    model = _battery(e_final=0.8)
    p, e = model.price_response(np.full(6, 100.0), 0.1)  # wants to discharge
    assert e[-1] >= 8.0 - 1e-5


@pytest.mark.parametrize(
    "bad", [dict(e_max=0.0), dict(eta_charge=0.0), dict(soc_min=0.8, soc_max=0.5)]
)
def test_invalid_parameters(bad):
    with pytest.raises(ValueError):
        _battery(**bad)
