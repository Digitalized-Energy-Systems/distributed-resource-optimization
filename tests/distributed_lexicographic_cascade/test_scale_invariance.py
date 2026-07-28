"""Scale-freeness of the lexicographic cascade.

``rho``, ``inner_abs_tol``, ``r_regularization`` and both residuals are
absolute constants carrying MW (or MW^2), while ``r`` is dimensionless and the
coupling constraint ``sigma + sum_i r_i c_i <= B - theta`` is homogeneous.  The
optimal ``r`` is therefore invariant under a uniform rescaling of the data, but
the solver is not: shrink a working case to LV magnitudes and every residual
drops below ``inner_abs_tol`` on the first iteration of each tier -- before the
``r``-update has seen the newly-binding constraint through ``z``/``u`` -- so the
cascade returns its ``r = 0`` initialisation and reports ``converged=True``.

These tests pin the failure, the fix, and that the fix is off by default.
"""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    SectorDemand,
    solve_cp_distributed_lexicographic_cascade,
)
from distributed_resource_optimization.algorithm.admm.lexicographic.kernel import (
    _characteristic_scale,
    _local_regularization,
)
from distributed_resource_optimization.algorithm.admm.types import CPSpec

SCALE_FREE = {
    "normalize": True,
    "r_regularization_relative": True,
    "minimize_usage": True,
}


#: Heat left uncovered by base supply, i.e. what the converters must deliver.
LV_DEFICIT_MW = 0.008


def _lv_case(scale: float = 1.0, n_cps: int = 8):
    """An LV feeder: 3.6 kW P2H units against an 8 kW heat deficit.

    Magnitudes and shape are those of ``simbench_lv_*`` -- in particular the
    shortfall sits in tiers 3-4 while base supply already covers tiers 1-2, so
    the constraint first binds several tiers into the cascade.  ``scale``
    multiplies every MW quantity, which must leave the optimal ``r`` untouched.

    ``n_cps`` matters: the per-CP share of the deficit shrinks as the fleet
    grows, and below ~8 CPs the first ``r``-step is large enough to clear
    ``inner_abs_tol`` on its own and the defect does not reproduce.
    """
    cps = [
        CPSpec(
            cp_id=f"p2h-{i}",
            capacity_by_sector={
                "electricity": 0.0050 * scale,
                "heat": -0.0036 * scale,
            },
        )
        for i in range(n_cps)
    ]
    demands = [
        SectorDemand(
            sector="electricity",
            demand_by_tier={1: np.zeros(1)},
            base_supply=np.array([0.5 * scale]),
        ),
        SectorDemand(
            sector="heat",
            demand_by_tier={
                1: np.array([0.002 * scale]),
                2: np.array([0.002 * scale]),
                3: np.array([0.004 * scale]),
                4: np.array([0.004 * scale]),
            },
            base_supply=np.array([0.004 * scale]),
        ),
    ]
    return cps, demands


def _heat_served(result) -> float:
    return float(sum(v[0] for v in result.served_by_sector_tier["heat"].values()))


def _heat_delivered(result, cps) -> float:
    return -sum(
        float(result.factor_by_cp[c.cp_id][0]) * c.capacity_by_sector["heat"] for c in cps
    )


def test_lv_scale_defaults_stop_after_one_iteration_per_tier():
    """Regression: the shipped defaults dispatch ~nothing on an LV feeder.

    Not an iteration-budget problem -- the cascade *stops early*, exactly one
    iteration per tier, and reports success.  Inside an iteration the ``r``
    update runs before the ``(z, sigma)`` update that first sees the tier's
    binding constraint, so ``r`` cannot respond in a tier's first pass; on LV
    data the residuals of that non-response are all below ``inner_abs_tol``.
    """
    cps, demands = _lv_case()
    result = solve_cp_distributed_lexicographic_cascade(
        cps, demands, inner_abs_tol=1.0e-3, inner_iters_max=200
    )
    assert result.iterations == 4  # four tiers, one iteration each
    assert result.converged is True  # ... and it claims success
    assert _heat_delivered(result, cps) < 0.01 * LV_DEFICIT_MW


def test_lv_scale_dispatches_under_scale_free_settings():
    """Same case, same tolerance, same iteration cap: the deficit is covered."""
    cps, demands = _lv_case()
    result = solve_cp_distributed_lexicographic_cascade(
        cps, demands, inner_abs_tol=1.0e-3, inner_iters_max=200, **SCALE_FREE
    )
    assert result.iterations > 4
    assert _heat_delivered(result, cps) == pytest.approx(LV_DEFICIT_MW, rel=0.05)


def test_scale_free_accuracy_improves_with_a_tighter_tolerance():
    """inner_abs_tol now acts on normalised data, so it means what it says.

    Pins that the residual few-percent gap at 1e-3 is tolerance tightness and
    not a systematic bias: tightening the tolerance walks it to the deficit.
    """
    cps, demands = _lv_case()
    loose, tight = (
        solve_cp_distributed_lexicographic_cascade(
            cps, demands, inner_abs_tol=tol, inner_iters_max=2000, **SCALE_FREE
        )
        for tol in (1.0e-3, 1.0e-5)
    )
    assert tight.iterations > loose.iterations
    assert abs(_heat_delivered(tight, cps) - LV_DEFICIT_MW) < abs(
        _heat_delivered(loose, cps) - LV_DEFICIT_MW
    )
    assert _heat_delivered(tight, cps) == pytest.approx(LV_DEFICIT_MW, rel=0.01)


@pytest.mark.parametrize("scale", [1.0, 1.0e1, 1.0e3, 1.0e5])
def test_r_is_invariant_under_uniform_rescaling(scale):
    """The whole point: identical dimensionless r at any grid magnitude."""
    base, _ = _lv_case()
    ref = solve_cp_distributed_lexicographic_cascade(
        *_lv_case(1.0), inner_abs_tol=1.0e-3, **SCALE_FREE
    )
    got = solve_cp_distributed_lexicographic_cascade(
        *_lv_case(scale), inner_abs_tol=1.0e-3, **SCALE_FREE
    )
    for cp in base:
        assert float(got.factor_by_cp[cp.cp_id][0]) == pytest.approx(
            float(ref.factor_by_cp[cp.cp_id][0]), rel=1e-9
        )


@pytest.mark.parametrize("scale", [1.0e1, 1.0e3])
def test_defaults_are_not_invariant_under_rescaling(scale):
    """Companion to the above: without the fix, scale changes the answer.

    This is the hypothesis-free proof that the formulation, not any single
    constant, is scale-blind -- the problem is homogeneous in the data.
    """
    small = solve_cp_distributed_lexicographic_cascade(
        *_lv_case(1.0), inner_abs_tol=1.0e-3
    )
    big = solve_cp_distributed_lexicographic_cascade(
        *_lv_case(scale), inner_abs_tol=1.0e-3
    )
    assert float(big.factor_by_cp["p2h-0"][0]) > 100.0 * float(small.factor_by_cp["p2h-0"][0])


@pytest.mark.parametrize("scale", [1.0e1, 1.0e3])
def test_served_is_returned_in_input_units(scale):
    """sigma is solved for in the normalised frame and must come back as MW."""
    ref = solve_cp_distributed_lexicographic_cascade(
        *_lv_case(1.0), inner_abs_tol=1.0e-3, **SCALE_FREE
    )
    got = solve_cp_distributed_lexicographic_cascade(
        *_lv_case(scale), inner_abs_tol=1.0e-3, **SCALE_FREE
    )
    assert _heat_served(got) == pytest.approx(_heat_served(ref) * scale, rel=1e-9)


def test_normalisation_is_off_by_default():
    """Byte-parity guard: the defaults must reproduce prior campaigns."""
    cps, demands = _lv_case()
    a = solve_cp_distributed_lexicographic_cascade(cps, demands, inner_abs_tol=1.0e-3)
    b = solve_cp_distributed_lexicographic_cascade(
        cps, demands, inner_abs_tol=1.0e-3, normalize=False, r_regularization_relative=False
    )
    assert a.iterations == b.iterations
    for k, v in a.factor_by_cp.items():
        assert float(v[0]) == float(b.factor_by_cp[k][0])


def test_normalisation_preserves_a_well_scaled_solve():
    """On O(1) data the fix must not move an already-correct answer."""
    cps = [CPSpec(cp_id="p2h-1", capacity_by_sector={"electricity": 5.0, "heat": -4.5})]
    demands = [
        SectorDemand(
            sector="electricity", demand_by_tier={1: np.zeros(1)}, base_supply=np.array([5.0])
        ),
        SectorDemand(
            sector="heat", demand_by_tier={1: np.array([4.0])}, base_supply=np.array([0.0])
        ),
    ]
    result = solve_cp_distributed_lexicographic_cascade(cps, demands, **SCALE_FREE)
    assert result.served_by_sector_tier["heat"][1][0] == pytest.approx(4.0, abs=1e-3)
    assert 4.0 / 4.5 - 1e-2 <= float(result.factor_by_cp["p2h-1"][0]) <= 1.0 + 1e-9


def test_characteristic_scale_uses_the_round_peak():
    demands = [
        SectorDemand(
            sector="heat",
            demand_by_tier={1: np.array([0.3]), 2: np.array([0.4])},
            base_supply=np.array([0.2]),
        ),
        SectorDemand(
            sector="electricity", demand_by_tier={1: np.array([0.1])}, base_supply=np.array([2.5])
        ),
    ]
    assert _characteristic_scale(demands) == pytest.approx(2.5)


def test_characteristic_scale_falls_back_to_one_on_empty_data():
    demands = [
        SectorDemand(
            sector="heat", demand_by_tier={1: np.zeros(1)}, base_supply=np.zeros(1)
        )
    ]
    assert _characteristic_scale(demands) == 1.0


def test_local_regularization_absolute_vs_relative():
    assert _local_regularization(0.1, False, 1.37e-5) == 0.1
    assert _local_regularization(0.1, True, 1.37e-5) == pytest.approx(1.37e-6)
