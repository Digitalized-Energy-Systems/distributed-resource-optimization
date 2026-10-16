"""ReservoirStorageConsensusActor on the shared storage model (decision Q15).

The actor answers a price vector with the exact SOC-feasible best response
``argmin Σ c_d·p_d − λ·p + (ε/2)(p_d² + p_c²)`` under the storage constraints
and ``E_T ≥ e_final·e_max`` (see ``misc/storage.py``).
"""

from __future__ import annotations

import numpy as np

from distributed_resource_optimization import ReservoirStorageConsensusActor


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
        discharge_cost=0.0,
        epsilon=1.0,
    )
    defaults.update(overrides)
    return ReservoirStorageConsensusActor(**defaults)


class TestPriceResponse:
    def test_interior_response_is_lambda_over_epsilon(self):
        # Charge at λ = −2, discharge at λ = +2: (λ − c)/ε = ∓2, net energy 0.
        p = _actor().project_power(np.array([-2.0, 2.0]), data=None)
        np.testing.assert_allclose(p, [-2.0, 2.0], atol=1e-5)

    def test_discharge_cost_shifts_the_discharge_threshold(self):
        actor = _actor(e_final=0.0, discharge_cost=1.0)
        p = actor.project_power(np.array([0.5, 3.0]), data=None)
        np.testing.assert_allclose(p, [0.0, 2.0], atol=1e-5)  # (3 − 1)/1

    def test_flat_low_price_does_not_drain_below_terminal_level(self):
        # E_T >= E_0: with nothing to gain the battery may not end emptier.
        actor = _actor()
        actor.project_power(np.array([5.0, 5.0, 5.0]), data=None)
        assert actor.E[-1] >= 5.0 - 1e-5


class TestFeasibility:
    def test_power_limits(self):
        actor = _actor(e_max=100.0, p_discharge_max=4.0, p_charge_max=3.0, e_final=0.0)
        p = actor.project_power(np.array([1000.0, -1000.0]), data=None)
        assert p[0] <= 4.0 + 1e-6 and p[1] >= -3.0 - 1e-6

    def test_energy_limits_under_alternating_prices(self):
        actor = _actor(p_charge_max=20.0, p_discharge_max=20.0, epsilon=0.1)
        actor.project_power(np.array([50.0, -50.0, 50.0, -50.0]), data=None)
        assert np.all(actor.E >= -1e-6) and np.all(actor.E <= 10.0 + 1e-6)

    def test_discharge_limited_by_stored_energy(self):
        actor = _actor(e_initial=0.2, e_final=0.0, p_discharge_max=50.0)
        p = actor.project_power(np.array([100.0, 100.0, 100.0]), data=None)
        assert np.sum(p) <= 2.0 + 1e-5


class TestEfficiency:
    def test_discharging_drains_p_over_eta(self):
        actor = _actor(eta_discharge=0.5, e_initial=1.0, e_final=0.0, p_discharge_max=20.0)
        p = actor.project_power(np.array([1000.0]), data=None)
        np.testing.assert_allclose(p, [5.0], atol=1e-5)  # 10 MWh · η
        assert abs(actor.E[-1]) < 1e-5


class TestActorState:
    def test_updates_P_and_E(self):
        actor = _actor()
        p = actor.project_power(np.array([2.0, -2.0, 0.0]), data=None)
        assert np.array_equal(actor.P, p)
        assert actor.E.shape == (4,)
        np.testing.assert_allclose(actor.E[0], 5.0)
