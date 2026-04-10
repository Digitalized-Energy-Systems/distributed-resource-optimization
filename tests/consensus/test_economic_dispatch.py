"""LinearCostEconomicDispatchConsensusActor unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import LinearCostEconomicDispatchConsensusActor


class TestGradientTerm:
    """Unit tests for the gradient_term method."""

    def _actor(self, cost=10.0, p_max=30.0, rho=0.05, epsilon=0.1, p_min=0.0, n_guess=3):
        return LinearCostEconomicDispatchConsensusActor(
            cost=cost, p_max=p_max, rho=rho, epsilon=epsilon, p_min=p_min, n_guess=n_guess
        )

    def test_below_cost_clips_to_p_min(self):
        """When λ < cost, P = p_min (default 0) so gradient = -ρ*(0 - target/N)."""
        actor = self._actor(cost=10.0, p_max=30.0, rho=0.05, epsilon=0.1, n_guess=3)
        lam = np.array([5.0])     # below cost
        p_target = np.array([30.0])
        grad = actor.gradient_term(lam, p_target)
        # P = clip((5-10)/0.1, 0, 30) = 0
        # grad = -0.05 * (0 - 30/3) = -0.05 * (-10) = +0.5
        assert np.allclose(grad, [0.5])

    def test_above_max_clips_to_p_max(self):
        """When λ >> cost, P = p_max so gradient = -ρ*(p_max - target/N)."""
        actor = self._actor(cost=10.0, p_max=5.0, rho=0.05, epsilon=0.1, n_guess=3)
        lam = np.array([100.0])   # far above cost
        p_target = np.array([30.0])
        grad = actor.gradient_term(lam, p_target)
        # P = clip((100-10)/0.1, 0, 5) = 5
        # grad = -0.05 * (5 - 30/3) = -0.05 * (5 - 10) = +0.25
        assert np.allclose(grad, [0.25])

    def test_at_clearing_price_zero_gradient(self):
        """At the clearing price λ* = cost + ε*(p_target/N), gradient = 0."""
        cost, epsilon, p_max, rho, n_guess = 10.0, 0.1, 30.0, 0.05, 3
        p_target = np.array([30.0])
        # clearing price: λ* = cost + ε * (p_target/N) = 10 + 0.1*(30/3) = 11
        lam = np.array([11.0])
        actor = self._actor(cost=cost, p_max=p_max, rho=rho, epsilon=epsilon, n_guess=n_guess)
        grad = actor.gradient_term(lam, p_target)
        assert np.allclose(grad, [0.0], atol=1e-10)

    def test_updates_internal_P(self):
        """gradient_term must update actor.P for tracking."""
        actor = self._actor(cost=10.0, p_max=20.0, epsilon=0.1, n_guess=1)
        lam = np.array([11.0])
        p_target = np.array([5.0])
        actor.gradient_term(lam, p_target)
        # P = clip((11-10)/0.1, 0, 20) = clip(10, 0, 20) = 10
        assert np.allclose(actor.P, [10.0])

    def test_vectorised_lam(self):
        """Works correctly for multi-dimensional λ / p_target vectors."""
        actor = self._actor(cost=10.0, p_max=30.0, rho=0.05, epsilon=0.1, n_guess=2)
        lam = np.array([11.0, 12.0])
        p_target = np.array([20.0, 40.0])
        grad = actor.gradient_term(lam, p_target)
        # P[0] = clip((11-10)/0.1, 0, 30) = 10; target/N=10  → grad[0] = -0.05*(10-10)=0
        # P[1] = clip((12-10)/0.1, 0, 30) = 20; target/N=20  → grad[1] = -0.05*(20-20)=0
        assert np.allclose(grad, [0.0, 0.0], atol=1e-10)

    def test_none_p_target_treated_as_zero(self):
        """p_target=None → treated as 0, so gradient = -ρ*P."""
        actor = self._actor(cost=10.0, p_max=30.0, rho=0.05, epsilon=0.1, n_guess=1)
        lam = np.array([11.0])
        grad = actor.gradient_term(lam, None)
        # P = clip(10, 0, 30) = 10; grad = -0.05*(10 - 0) = -0.5
        assert np.allclose(grad, [-0.5])
