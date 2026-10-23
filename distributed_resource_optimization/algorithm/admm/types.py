"""Shared data contracts for the priority-cascade ADMM variants.

These dataclasses are consumed by both the waterfall variant
(:mod:`..waterfall.core`) and the lexicographic variants
(:mod:`..lexicographic`).  They live here, one level above either
variant, so neither has to import the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CPSpec:
    """Per-CP specification consumed by :func:`..waterfall.core.solve_cp_priority_admm`.

    :param cp_id: Stable identifier for the coupling point.
    :param capacity_by_sector: Per-sector signed effective capacity (load
        convention).  A P2H with 10 MW input and η = 0.95 is described
        as ``{"electricity": 10.0, "heat": -9.5}``.  A CHP with 10 MW
        gas input, 3.5 MW electricity output, 4.5 MW heat output is
        ``{"gas": 10.0, "electricity": -3.5, "heat": -4.5}``.
    """

    cp_id: str
    capacity_by_sector: dict[str, float]


@dataclass(frozen=True)
class SectorDemand:
    """Per-sector demand profile over the horizon.

    :param sector: Sector identifier (e.g. ``"electricity"``).
    :param demand_by_tier: ``tier -> length-H array`` of MW per priority
        tier (lower tier index = higher priority).
    :param base_supply: Length-H array of MW available to this sector
        before any CP contribution (load convention: positive = supply
        from base generators).
    """

    sector: str
    demand_by_tier: dict[int, np.ndarray]
    base_supply: np.ndarray


@dataclass(frozen=True)
class CPAdmmResult:
    """Output of :func:`..waterfall.core.solve_cp_priority_admm`.

    :param factor_by_cp: ``cp_id -> length-H regulation factor in [0, 1]``.
    :param served_by_sector_tier: ``sector -> tier -> length-H served MW``.
    :param iterations: Number of ADMM iterations executed.
    :param primal_residual: Final primal residual ``||x - z||``.
    :param dual_residual: Final dual residual ``rho * ||z - z_prev||``.
    :param converged: ``True`` if the per-step damped factor move fell
        below ``abs_tol`` before ``max_iters``.
    :param history: Optional per-iteration diagnostics when the kernel
        was invoked with ``record_history=True``.
    """

    factor_by_cp: dict[str, np.ndarray]
    served_by_sector_tier: dict[str, dict[int, np.ndarray]]
    iterations: int
    primal_residual: float
    dual_residual: float
    converged: bool
    history: dict[str, Any] = field(default_factory=dict)
