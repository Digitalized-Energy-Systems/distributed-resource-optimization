"""Factory functions for DEED-ADMM participants (single energy carrier)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .deed_admm import DEEDADMMAlgorithm, DEEDADMMStorageAlgorithm


def create_deed_admm_thermal_participant(
    finish_callback: Callable,
    *,
    p_min: np.ndarray | float,
    p_max: np.ndarray | float,
    marginal_cost: float,
    cost_quad: float = 0.0,
    d_i: np.ndarray,
    gamma: float = 0.05,
    max_iter: int = 500,
    n_agents: int = 1,
    beta: float | None = None,
    round_timeout: float | None = None,
    tol: float | None = None,
    diameter: int = 1,
) -> DEEDADMMAlgorithm:
    """DEED-ADMM participant for a thermal generator.

    :param finish_callback: ``(algorithm, carrier) -> None``.
    :param p_min: Minimum generation limit (MW, scalar or shape τ).
    :param p_max: Maximum generation limit (MW, scalar or shape τ).
    :param marginal_cost: Linear cost coefficient bᵢ (currency/MWh).
    :param cost_quad: Quadratic cost coefficient aᵢ (currency/MW²h).
        Use 0 for purely linear cost (common in PyPSA models).
    :param d_i: Demand allocation per time step (MW, shape τ).
    :param gamma: ADMM penalty γ.
    :param max_iter: Number of iterations.
    :param n_agents: Total number of agents n (default ``beta = 1/n``).
    :param beta: Laplacian step of the weight matrix W = I − βL.
    :param round_timeout: Round deadline in seconds; see :class:`~.deed_admm.DEEDADMMAlgorithm`.
    :param tol: Convergence tolerance (MW); see :class:`~.deed_admm.DEEDADMMAlgorithm`.
    :param diameter: Communication-graph diameter; see :class:`~.deed_admm.DEEDADMMAlgorithm`.
    """
    tau = len(d_i)
    return DEEDADMMAlgorithm(
        finish_callback=finish_callback,
        cost_quad=np.full(tau, float(cost_quad)),
        cost_lin=np.full(tau, float(marginal_cost)),
        x_min=np.broadcast_to(np.asarray(p_min, dtype=float), (tau,)),
        x_max=np.broadcast_to(np.asarray(p_max, dtype=float), (tau,)),
        d_i=d_i,
        gamma=gamma,
        max_iter=max_iter,
        n_agents=n_agents,
        beta=beta,
        round_timeout=round_timeout,
        tol=tol,
        diameter=diameter,
    )


def create_deed_admm_renewable_participant(
    finish_callback: Callable,
    *,
    p_max_timeseries: np.ndarray,
    d_i: np.ndarray,
    marginal_cost: float = 0.0,
    gamma: float = 0.05,
    max_iter: int = 500,
    n_agents: int = 1,
    beta: float | None = None,
    round_timeout: float | None = None,
    tol: float | None = None,
    diameter: int = 1,
) -> DEEDADMMAlgorithm:
    """DEED-ADMM participant for a renewable generator.

    :param finish_callback: ``(algorithm, carrier) -> None``.
    :param p_max_timeseries: Available capacity per time step (MW, shape τ).
    :param d_i: Demand allocation per time step (MW, shape τ).
    :param marginal_cost: Linear cost coefficient (currency/MWh); 0 for
        wind/solar, but e.g. biomass is classified renewable and is not free.
    :param gamma: ADMM penalty γ.
    :param max_iter: Number of iterations.
    :param n_agents: Total number of agents n (default ``beta = 1/n``).
    :param beta: Laplacian step of the weight matrix W = I − βL.
    :param round_timeout: Round deadline in seconds; see :class:`~.deed_admm.DEEDADMMAlgorithm`.
    :param tol: Convergence tolerance (MW); see :class:`~.deed_admm.DEEDADMMAlgorithm`.
    :param diameter: Communication-graph diameter; see :class:`~.deed_admm.DEEDADMMAlgorithm`.
    """
    tau = len(p_max_timeseries)
    assert len(d_i) == tau, "p_max_timeseries and d_i must have the same length"
    return DEEDADMMAlgorithm(
        finish_callback=finish_callback,
        cost_quad=np.zeros(tau),
        cost_lin=np.full(tau, float(marginal_cost)),
        x_min=np.zeros(tau),
        x_max=np.asarray(p_max_timeseries, dtype=float),
        d_i=d_i,
        gamma=gamma,
        max_iter=max_iter,
        n_agents=n_agents,
        beta=beta,
        round_timeout=round_timeout,
        tol=tol,
        diameter=diameter,
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
    marginal_cost: float = 0.0,
    dt: float = 1.0,
    tau: int,
    gamma: float = 0.05,
    max_iter: int = 500,
    n_agents: int = 1,
    beta: float | None = None,
    round_timeout: float | None = None,
    tol: float | None = None,
    diameter: int = 1,
) -> DEEDADMMStorageAlgorithm:
    """DEED-ADMM participant for a battery / pumped-hydro storage unit.

    Modelled as a hub without supplier or own demand (x = 0, dᵢ = 0) whose
    storage block z = [P^Sd; P^Sc] shifts the hub input; SOC dynamics and the
    terminal reserve are linear constraints inside the ADMM, as in the paper.

    :param finish_callback: ``(algorithm, carrier) -> None``.
    :param e_max: Energy capacity (MWh).
    :param p_charge_max: Maximum charging rate (MW, positive value).
    :param p_discharge_max: Maximum discharging rate (MW, positive value).
    :param eta_charge: Charging efficiency (0 < η ≤ 1).
    :param eta_discharge: Discharging efficiency (0 < η ≤ 1).
    :param e_initial: Initial state of charge as fraction of e_max.
    :param e_final: Terminal reserve: E_τ ≥ e_final·e_max (defaults to
        e_initial).
    :param soc_min: Minimum SOC fraction.
    :param soc_max: Maximum SOC fraction.
    :param marginal_cost: Cost per MWh discharged.
    :param dt: Time-step length in hours.
    :param tau: Number of scheduling time steps.
    :param gamma: ADMM penalty γ.
    :param max_iter: Number of iterations.
    :param n_agents: Total number of agents n (generators + storage).
    :param beta: Laplacian step of the weight matrix W = I − βL.
    :param round_timeout: Round deadline in seconds; see :class:`~.deed_admm.DEEDADMMAlgorithm`.
    :param tol: Convergence tolerance (MW); see :class:`~.deed_admm.DEEDADMMAlgorithm`.
    :param diameter: Communication-graph diameter; see :class:`~.deed_admm.DEEDADMMAlgorithm`.
    """
    return DEEDADMMStorageAlgorithm(
        finish_callback=finish_callback,
        cost_quad=np.zeros(tau),
        cost_lin=np.zeros(tau),
        x_min=np.zeros(tau),
        x_max=np.zeros(tau),
        d_i=np.zeros(tau),
        gamma=gamma,
        max_iter=max_iter,
        n_agents=n_agents,
        beta=beta,
        round_timeout=round_timeout,
        tol=tol,
        diameter=diameter,
        e_max=float(e_max),
        p_charge_max=float(p_charge_max),
        p_discharge_max=float(p_discharge_max),
        eta_charge=float(eta_charge),
        eta_discharge=float(eta_discharge),
        e_initial=float(e_initial),
        e_final=e_final,
        soc_min=float(soc_min),
        soc_max=float(soc_max),
        discharge_cost=float(marginal_cost),
        dt=float(dt),
    )
