"""LinearCostEconomicDispatchConsensusActor unit tests."""

from __future__ import annotations

import numpy as np

from distributed_resource_optimization import LinearCostEconomicDispatchConsensusActor


class TestProjectPower:
    """Unit tests for the project_power method (eq. 23)."""

    def _actor(self, cost=10.0, p_max=30.0, epsilon=0.1, p_min=0.0):
        return LinearCostEconomicDispatchConsensusActor(
            cost=cost, p_max=p_max, epsilon=epsilon, p_min=p_min
        )

    def test_below_cost_clips_to_p_min(self):
        """When λ < cost, P clips to p_min (default 0)."""
        actor = self._actor(cost=10.0, p_max=30.0, epsilon=0.1)
        lam = np.array([5.0])  # below cost
        p = actor.project_power(lam, data=None)
        assert np.allclose(p, [0.0])

    def test_above_max_clips_to_p_max(self):
        """When λ >> cost, P clips to p_max."""
        actor = self._actor(cost=10.0, p_max=5.0, epsilon=0.1)
        lam = np.array([100.0])  # far above cost
        p = actor.project_power(lam, data=None)
        assert np.allclose(p, [5.0])

    def test_unclipped_matches_eq_23(self):
        """Within bounds, P = (λ - cost) / epsilon, matching eq. 23."""
        actor = self._actor(cost=10.0, p_max=30.0, epsilon=0.1)
        lam = np.array([11.0])
        p = actor.project_power(lam, data=None)
        # P = (11-10)/0.1 = 10
        assert np.allclose(p, [10.0])

    def test_updates_internal_P(self):
        """project_power must update actor.P for tracking."""
        actor = self._actor(cost=10.0, p_max=20.0, epsilon=0.1)
        lam = np.array([11.0])
        actor.project_power(lam, data=None)
        assert np.allclose(actor.P, [10.0])

    def test_vectorised_lam(self):
        """Works correctly for multi-dimensional λ vectors."""
        actor = self._actor(cost=10.0, p_max=30.0, epsilon=0.1)
        lam = np.array([11.0, 12.0])
        p = actor.project_power(lam, data=None)
        assert np.allclose(p, [10.0, 20.0])

    def test_data_argument_is_unused(self):
        """The auxiliary `data` payload doesn't affect the projection."""
        actor = self._actor(cost=10.0, p_max=30.0, epsilon=0.1)
        lam = np.array([11.0])
        p_with_data = actor.project_power(lam, data=np.array([999.0]))
        p_without_data = actor.project_power(lam, data=None)
        assert np.allclose(p_with_data, p_without_data)
