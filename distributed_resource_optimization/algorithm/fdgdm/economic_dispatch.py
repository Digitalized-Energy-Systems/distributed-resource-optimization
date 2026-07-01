"""Economic-dispatch actors for the FDGDM algorithm.

Two :class:`~.fdgdm.FDGDMActor` implementations:

* :class:`LinearCostEconomicDispatchFDGDMActor` — quadratic cost function
  ``F(P) = (ε/2)P² + c·P`` whose gradient and curvature are available in
  closed form.  Covers thermal generators and (approximately) renewables.
* :class:`ReservoirStorageFDGDMActor` — battery/reservoir storage with a
  piecewise-linear cost and quadratic regularisation.  Power limits are
  enforced exactly; SOC coupling across time steps is ignored (each time
  step is treated independently).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .fdgdm import FDGDMActor

# ---------------------------------------------------------------------------
# Linear cost economic dispatch
# ---------------------------------------------------------------------------


@dataclass
class LinearCostEconomicDispatchFDGDMActor(FDGDMActor):
    """FDGDM actor for economic dispatch with quadratic cost.

    The assumed cost function is ``F(P) = (ε/2)·P² + c·P``, giving gradient
    ``∇F(P) = ε·P + c`` and constant second derivative ``ε``.

    :param cost: Linear cost coefficient *c* (e.g. ``marginal_cost`` from PyPSA).
    :param p_max: Maximum power output (scalar or per-timestep vector).
    :param epsilon: Quadratic cost coefficient / curvature bound (default 0.1).
    :param p_min: Minimum power output (default 0).
    """

    cost: float
    p_max: float | np.ndarray
    epsilon: float = 0.1
    p_min: float = 0.0
    initial_schedule: np.ndarray | None = None

    P: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def __post_init__(self) -> None:
        self._p_max = np.asarray(self.p_max, dtype=float)

    def gradient(self, P: np.ndarray, data: Any) -> np.ndarray:
        """Return ``ε·P + c`` (gradient of the quadratic cost function)."""
        return self.epsilon * P + self.cost

    def curvature_bound(self) -> float:
        """Return ε — the exact (constant) second derivative of F."""
        return self.epsilon

    def project(self, P: np.ndarray) -> np.ndarray:
        """Clip *P* to ``[p_min, p_max]`` and cache the result in ``self.P``.

        If ``initial_schedule`` is set, it is used instead of *P* on the very
        first call and then cleared.  This lets the scenario supply a per-agent
        demand-feasible starting point without changing the FDGDM kickoff
        protocol (which sends a single shared value to all participants).
        """
        if self.initial_schedule is not None:
            P = np.asarray(self.initial_schedule, dtype=float)
            self.initial_schedule = None
        self.P = np.clip(P, self.p_min, self._p_max)
        return self.P.copy()


# ---------------------------------------------------------------------------
# Reservoir / battery storage (simplified)
# ---------------------------------------------------------------------------


@dataclass
class ReservoirStorageFDGDMActor(FDGDMActor):
    """FDGDM actor for battery/reservoir storage.

    The cost model is piecewise-linear with quadratic regularisation:

    * Discharging (P > 0): ``F(P) = discharge_cost·P + (ε/2)·P²``
    * Charging   (P < 0): ``F(P) = charge_cost·(-P) + (ε/2)·P²``

    resulting in gradient ``∇F(P) = sign_cost(P) + ε·P`` and constant
    curvature bound ``ε``.

    Power limits are enforced exactly via :meth:`project`.  SOC coupling
    across time steps is **not** modelled — each time step is treated
    independently.  For a full SOC-aware actor, see the Diffusion or ADMM
    equivalents.

    :param p_charge_max: Maximum charging power (MW, positive value).
    :param p_discharge_max: Maximum discharging power (MW, positive value).
    :param charge_cost: Marginal cost for charging (default 0).
    :param discharge_cost: Marginal benefit for discharging (default 0).
    :param epsilon: Quadratic regularisation / curvature bound (default 0.1).
    """

    p_charge_max: float
    p_discharge_max: float
    charge_cost: float = 0.0
    discharge_cost: float = 0.0
    epsilon: float = 0.1

    P: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def gradient(self, P: np.ndarray, data: Any) -> np.ndarray:
        """Return the piecewise gradient of the storage cost function."""
        linear = np.where(P >= 0.0, self.discharge_cost, -self.charge_cost)
        return linear + self.epsilon * P

    def curvature_bound(self) -> float:
        """Return ε — the constant second derivative of the regularised cost."""
        return self.epsilon

    def project(self, P: np.ndarray) -> np.ndarray:
        """Clip *P* to ``[-p_charge_max, p_discharge_max]`` and cache in ``self.P``."""
        self.P = np.clip(P, -self.p_charge_max, self.p_discharge_max)
        return self.P.copy()
