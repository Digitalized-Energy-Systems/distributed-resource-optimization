"""Factory functions for DEED-ADMM participants (single energy carrier)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .deed_admm import DEEDADMMAlgorithm, DEEDADMMStorageAlgorithm


def create_deed_admm_thermal_participant(
    finish_callback: Callable,
    *,
    p_min: float,
    p_max: float,
    marginal_cost: float,
    cost_quad: float = 0.0,
    d_i: np.ndarray,
    gamma: float = 0.05,
    max_iter: int = 500,
    n_agents: int = 1,
) -> DEEDADMMAlgorithm:
    """DEED-ADMM participant for a thermal generator.

    :param finish_callback: ``(algorithm, carrier) -> None``.
    :param p_min: Minimum generation limit (MW, scalar).
    :param p_max: Maximum generation limit (MW, scalar).
    :param marginal_cost: Linear cost coefficient bᵢ (currency/MWh).
    :param cost_quad: Quadratic cost coefficient aᵢ (currency/MW²h).
        Use 0 for purely linear cost (common in PyPSA models).
    :param d_i: Demand allocation per time step (MW, shape τ).
    :param gamma: ADMM penalty γ.
    :param max_iter: Maximum iterations.
    :param n_agents: Total number of agents n.
    """
    tau = len(d_i)
    return DEEDADMMAlgorithm(
        finish_callback=finish_callback,
        cost_quad=np.full(tau, float(cost_quad)),
        cost_lin=np.full(tau, float(marginal_cost)),
        x_min=np.full(tau, float(p_min)),
        x_max=np.full(tau, float(p_max)),
        d_i=d_i,
        gamma=gamma,
        max_iter=max_iter,
        n_agents=n_agents,
    )


def create_deed_admm_renewable_participant(
    finish_callback: Callable,
    *,
    p_max_timeseries: np.ndarray,
    d_i: np.ndarray,
    gamma: float = 0.05,
    max_iter: int = 500,
    n_agents: int = 1,
) -> DEEDADMMAlgorithm:
    """DEED-ADMM participant for a renewable generator (zero marginal cost).

    :param finish_callback: ``(algorithm, carrier) -> None``.
    :param p_max_timeseries: Available capacity per time step (MW, shape τ).
    :param d_i: Demand allocation per time step (MW, shape τ).
    :param gamma: ADMM penalty γ.
    :param max_iter: Maximum iterations.
    :param n_agents: Total number of agents n.
    """
    tau = len(p_max_timeseries)
    assert len(d_i) == tau, "p_max_timeseries and d_i must have the same length"
    return DEEDADMMAlgorithm(
        finish_callback=finish_callback,
        cost_quad=np.zeros(tau),
        cost_lin=np.zeros(tau),
        x_min=np.zeros(tau),
        x_max=np.asarray(p_max_timeseries, dtype=float),
        d_i=d_i,
        gamma=gamma,
        max_iter=max_iter,
        n_agents=n_agents,
    )


def create_deed_admm_storage_participant(
    finish_callback: Callable,
    *,
    e_max: float,
    p_charge_max: float,
    p_discharge_max: float,
    eta_charge: float = 0.95,
    eta_discharge: float = 0.95,
    e_initial: float = 0.5,
    e_final: float | None = None,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    tau: int,
    gamma: float = 0.05,
    max_iter: int = 500,
    n_agents: int = 1,
) -> DEEDADMMStorageAlgorithm:
    """DEED-ADMM participant for a battery / reservoir storage unit.

    The storage unit receives ``d_i = 0`` (zero demand allocation) and
    optimises its charge/discharge schedule in response to the shared price
    signal λ̃, with SOC constraints enforced by a greedy forward-pass
    projection and bisection for the terminal energy target.

    :param finish_callback: ``(algorithm, carrier) -> None``.
    :param e_max: Energy capacity (MWh).
    :param p_charge_max: Maximum charging rate (MW, positive value).
    :param p_discharge_max: Maximum discharging rate (MW, positive value).
    :param eta_charge: Charging efficiency (0 < η ≤ 1).
    :param eta_discharge: Discharging efficiency (0 < η ≤ 1).
    :param e_initial: Initial state of charge as fraction of e_max.
    :param e_final: Target terminal SOC fraction; defaults to e_initial.
    :param soc_min: Minimum SOC fraction.
    :param soc_max: Maximum SOC fraction.
    :param tau: Number of scheduling time steps.
    :param gamma: ADMM penalty γ.
    :param max_iter: Maximum iterations.
    :param n_agents: Total number of agents n (generators + storage).
    """
    if e_final is None:
        e_final = e_initial
    return DEEDADMMStorageAlgorithm(
        finish_callback=finish_callback,
        cost_quad=np.zeros(tau),
        cost_lin=np.zeros(tau),
        x_min=np.full(tau, -float(p_charge_max)),
        x_max=np.full(tau, float(p_discharge_max)),
        d_i=np.zeros(tau),
        gamma=gamma,
        max_iter=max_iter,
        n_agents=n_agents,
        e_max=float(e_max),
        p_charge_max=float(p_charge_max),
        p_discharge_max=float(p_discharge_max),
        eta_charge=float(eta_charge),
        eta_discharge=float(eta_discharge),
        e_initial=float(e_initial),
        e_final=float(e_final),
        soc_min=float(soc_min),
        soc_max=float(soc_max),
    )
