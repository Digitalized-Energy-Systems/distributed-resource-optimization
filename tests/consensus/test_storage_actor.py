"""ReservoirStorageConsensusActor / project_storage_schedule_from_price tests.

The storage actor price-drives a desired schedule (discharge above
``discharge_cost + charge_cost``, charge below ``charge_cost``) and then
projects it onto an SOC-feasible one, bisecting over an additive bias so the
terminal SOC lands on ``e_final``. The same helper backs the ADMM and
diffusion storage actors.
"""

from __future__ import annotations

import numpy as np

from distributed_resource_optimization import ReservoirStorageConsensusActor
from distributed_resource_optimization.misc.util import (
    project_storage_schedule_from_price,
)


def _actor(**overrides) -> ReservoirStorageConsensusActor:
    defaults = dict(
        e_max=10.0,
        p_charge_max=5.0,
        p_discharge_max=5.0,
        eta_charge=1.0,
        eta_discharge=1.0,
        e_initial=0.5,
        e_final=0.5,
        soc_min=0.0,
        soc_max=1.0,
        charge_cost=0.0,
        discharge_cost=0.0,
        epsilon=1.0,
    )
    defaults.update(overrides)
    return ReservoirStorageConsensusActor(**defaults)


class TestPriceResponse:
    """Sign and magnitude of the price-driven schedule."""

    def test_discharges_when_price_above_threshold(self):
        # desired = (price - 0) / epsilon = 2; e_final matches the outcome so
        # the bias search stays inactive.
        actor = _actor(e_final=0.3)
        p = actor.project_power(np.array([2.0]), data=None)
        assert np.allclose(p, [2.0])

    def test_charges_when_price_below_charge_cost(self):
        actor = _actor(e_final=0.7)
        p = actor.project_power(np.array([-2.0]), data=None)
        assert np.allclose(p, [-2.0])

    def test_idles_in_dead_band(self):
        # charge_cost < price < charge_cost + discharge_cost → no action.
        actor = _actor(charge_cost=1.0, discharge_cost=2.0)
        p = actor.project_power(np.array([2.0, 2.5]), data=None)
        assert np.allclose(p, [0.0, 0.0])
        assert np.allclose(actor.E, [5.0, 5.0])


class TestFeasibility:
    """Power limits, SOC bounds, and energy availability are never violated."""

    def test_discharge_capped_at_p_discharge_max(self):
        actor = _actor(e_max=100.0, e_initial=1.0, p_discharge_max=4.0, e_final=0.96)
        p = actor.project_power(np.array([1000.0]), data=None)
        assert np.allclose(p, [4.0])

    def test_charge_capped_at_p_charge_max(self):
        actor = _actor(e_max=100.0, e_initial=0.0, p_charge_max=3.0, e_final=0.03)
        p = actor.project_power(np.array([-1000.0]), data=None)
        assert np.allclose(p, [-3.0])

    def test_discharge_limited_by_stored_energy(self):
        # Only 2 MWh in the reservoir: total discharge across the horizon
        # cannot exceed it, regardless of the price signal.
        actor = _actor(e_initial=0.2, e_final=0.0, p_discharge_max=50.0)
        p = actor.project_power(np.array([100.0, 100.0, 100.0]), data=None)
        assert np.all(p >= 0.0)
        assert np.sum(p) <= 2.0 + 1e-9

    def test_soc_and_power_bounds_hold_under_alternating_prices(self):
        actor = _actor(p_charge_max=20.0, p_discharge_max=20.0, epsilon=0.1)
        lam = np.array([50.0, -50.0, 50.0, -50.0])
        p, e_path, _ = project_storage_schedule_from_price(actor, lam)
        assert np.all(np.abs(p) <= 20.0 + 1e-9)
        assert np.all(e_path >= -1e-9)
        assert np.all(e_path <= 10.0 + 1e-9)


class TestEfficiency:
    """Charge/discharge efficiencies enter the SOC bookkeeping."""

    def test_charging_stores_p_times_eta(self):
        # 4 MW charged for one step at η=0.5 stores 2 MWh.
        actor = _actor(eta_charge=0.5, e_initial=0.0, e_final=0.2, p_charge_max=4.0)
        p, _, e_end = project_storage_schedule_from_price(actor, np.array([-1000.0]))
        assert np.allclose(p, [-4.0])
        assert e_end == 2.0

    def test_discharging_drains_p_over_eta(self):
        # Delivering 5 MW at η=0.5 drains 10 MWh — the whole reservoir — so
        # the SOC headroom caps the deliverable power at (e·η) = 5 MW.
        actor = _actor(eta_discharge=0.5, e_initial=1.0, e_final=0.0, p_discharge_max=20.0)
        p, _, e_end = project_storage_schedule_from_price(actor, np.array([1000.0]))
        assert np.allclose(p, [5.0])
        assert e_end == 0.0


class TestTerminalSocBiasSearch:
    """The bias bisection steers the terminal SOC toward e_final."""

    def test_terminal_soc_reaches_target(self):
        # Unbiased, the schedule discharges everything (e_end=0); the search
        # must find the bias whose schedule ends at e_final = 3 MWh.
        actor = _actor(e_final=0.3)
        p, _, e_end = project_storage_schedule_from_price(actor, np.array([10.0, 10.0]))
        assert abs(e_end - 3.0) < 1e-2
        # Net discharged energy equals the SOC drop (η = 1).
        assert np.isclose(np.sum(p), 5.0 - e_end)


class TestActorState:
    """project_power mirrors the projected schedule onto actor.P / actor.E."""

    def test_updates_P_and_E(self):
        actor = _actor(e_final=0.3)
        lam = np.array([2.0, 0.0, 0.0])
        p = actor.project_power(lam, data=None)
        assert actor.P.shape == lam.shape
        assert actor.E.shape == lam.shape
        assert np.allclose(actor.P, p)
