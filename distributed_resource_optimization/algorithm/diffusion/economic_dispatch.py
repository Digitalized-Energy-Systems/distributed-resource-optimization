"""Economic-dispatch actors for the diffusion algorithm.

Two :class:`~distributed_resource_optimization.algorithm.diffusion.diffusion.DiffusionActor`
implementations:

* :class:`LinearCostEconomicDispatchDiffusionActor` — linearised inverted
  quadratic cost response.
* :class:`ReservoirStorageDiffusionActor` — battery/reservoir storage whose
  charge/discharge schedule responds to a time-varying price signal λ(t).

Both actors clip their power response to the feasible box *inside*
``gradient_term`` itself, so respecting the limits is part of the converged
fixed point.  This is why the paper's outer saturation-flag loop (Ces et al.
2025, Sec. 4 — re-running the optimization with limit-violating agents pinned
to their ratings) is deliberately not ported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...misc.storage import StorageModel
from .diffusion import DiffusionActor

# ---------------------------------------------------------------------------
# Linear cost economic dispatch
# ---------------------------------------------------------------------------


@dataclass
class LinearCostEconomicDispatchDiffusionActor(DiffusionActor):
    """Economic dispatch via a linearised inverted quadratic cost function.

    :param cost: Marginal cost coefficient *c* in the cost function
                 ``cP + (ε/2)P²`` (marginal cost ``c + εP``, hence the
                 response ``P(λ) = (λ - c)/ε`` below).
    :param p_max: Maximum power output.
    :param epsilon: Sensitivity of power response to price (default 0.1).
    :param p_min: Minimum power output (default 0).
    :param n_guess: Estimated number of participants for target normalisation.
    """

    cost: float
    # Can be a scalar (same limit for all timesteps) or a vector
    # (one max value per timestep).
    p_max: float | np.ndarray
    epsilon: float = 0.1
    p_min: float = 0.0
    n_guess: int = 10

    P: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def gradient_term(self, lam: np.ndarray, p_target: Any) -> np.ndarray:
        """Compute the gradient for the adapt step.

        :param lam: Current price vector λ.
        :param p_target: Total target power (scalar or array); normalised by
                         :attr:`n_guess` to get the per-participant share.
        :returns: ``P(λ) - p_target / n_guess`` where ``P(λ)`` is the optimal
                  local power response.
        """
        p_max_arr = np.asarray(self.p_max, dtype=float)
        # If p_max is a vector, it should match the number of λ dimensions.
        # Broadcasting is fine for scalar p_max.
        if p_max_arr.ndim > 0 and p_max_arr.shape != lam.shape:
            if p_max_arr.size == lam.size:
                p_max_arr = p_max_arr.reshape(lam.shape)
            else:
                raise ValueError(
                    f"p_max shape {p_max_arr.shape} does not match lam shape {lam.shape}"
                )

        self.P = np.clip((lam - self.cost) / self.epsilon, self.p_min, p_max_arr)
        p_target_arr = np.asarray(p_target if p_target is not None else 0.0)
        return self.P - p_target_arr / self.n_guess


# ---------------------------------------------------------------------------
# Reservoir / battery storage
# ---------------------------------------------------------------------------


@dataclass
class ReservoirStorageDiffusionActor(DiffusionActor):
    """Reservoir/battery storage actor for diffusion-based dispatch.

    The storage actor answers the price λ(t) with the exact SOC-feasible best
    response of the shared storage model
    (:class:`~distributed_resource_optimization.misc.storage.StorageModel`,
    decision Q15): it minimises ``Σ c_d·p_d + c_c·p_c − λ·p + (ε/2)(p_d² + p_c²)``
    under the SOC dynamics, power and energy bounds and ``E_T ≥ e_final·e_max``.

    :param e_max: Maximum energy capacity (MWh).
    :param p_charge_max: Maximum charging power (MW).
    :param p_discharge_max: Maximum discharging power (MW).
    :param eta_charge: Charging efficiency (default 0.95).
    :param eta_discharge: Discharging efficiency (default 0.95).
    :param e_initial: Initial energy level as a fraction of ``e_max``.
    :param e_final: Minimum final energy level as a fraction of ``e_max``.
    :param soc_min: Minimum state of charge (fraction).
    :param soc_max: Maximum state of charge (fraction).
    :param charge_cost: Cost per MWh charged (0 by convention, Q15).
    :param discharge_cost: Cost per MWh discharged.
    :param epsilon: Proximal weight ε of the response, centred on the previous
        schedule (default 0.1; see :func:`_storage_response`).
    :param n_guess: Estimated number of participants for target normalisation.
    """

    e_max: float
    p_charge_max: float
    p_discharge_max: float
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    e_initial: float = 0.5
    e_final: float = 0.5
    soc_min: float = 0.0
    soc_max: float = 1.0
    charge_cost: float = 0.0
    discharge_cost: float = 0.0
    epsilon: float = 0.1
    n_guess: int = 10

    P: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    E: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def __post_init__(self) -> None:
        self._model = StorageModel(
            e_max=self.e_max,
            p_charge_max=self.p_charge_max,
            p_discharge_max=self.p_discharge_max,
            eta_charge=self.eta_charge,
            eta_discharge=self.eta_discharge,
            e_initial=self.e_initial,
            e_final=self.e_final,
            soc_min=self.soc_min,
            soc_max=self.soc_max,
            discharge_cost=self.discharge_cost,
            charge_cost=self.charge_cost,
        )

    def gradient_term(self, lam: np.ndarray, p_target: Any) -> np.ndarray:
        """Compute the gradient for the adapt step.

        :param lam: Current price vector λ.
        :param p_target: Total target power (scalar or array); normalised by
                         :attr:`n_guess` to get the per-participant share.
        :returns: ``P(λ) - p_target / n_guess`` where ``P(λ)`` is the
                  SOC-feasible price-driven schedule.
        """
        # Proximal-point response of the shared storage model (Q15).
        self.P, self.E = _storage_response(self._model, lam, self.epsilon, self.P)

        p_target_arr = np.asarray(p_target if p_target is not None else 0.0)
        return self.P - p_target_arr / self.n_guess


def _storage_response(model: Any, lam: Any, epsilon: float, previous: Any) -> tuple:
    """Proximal-point response of the shared storage model to the price *lam*:
    ``argmin cost(p) − λᵀp + (ε/2)‖p − p_prev‖²`` = ``prox(p_prev + λ/ε, ε)``.

    Centring the proximal term on the previous schedule (instead of on zero)
    keeps the price iteration stable without biasing its fixed point: at a
    fixed point p is an exact best response to λ. Centred on zero, storage was
    pulled toward idling (DE-AT, quadratic costs: 15 % of the optimal battery
    throughput; centred: the optimal throughput, schedule within 4 % for exact
    diffusion).
    """
    lam = np.asarray(lam, dtype=float)
    prev = np.asarray(previous, dtype=float)
    if prev.shape != lam.shape:
        prev = np.zeros_like(lam)
    return model.prox(prev + lam / epsilon, epsilon)
