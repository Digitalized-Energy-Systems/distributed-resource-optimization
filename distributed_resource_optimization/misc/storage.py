"""Shared battery/reservoir model and its exact SOC-feasible responses (decision Q15).

Every algorithm and central dispatch use the same storage model:

* energy dynamics ``E[t+1] = E[t] − Δt·p_d[t]/η_d + Δt·p_c[t]·η_c``,
* ``soc_min·E_max ≤ E[t] ≤ soc_max·E_max``, ``0 ≤ p_d ≤ p_discharge_max``,
  ``0 ≤ p_c ≤ p_charge_max``,
* terminal condition ``E[T] ≥ e_final·E_max`` (``e_final`` defaults to the
  initial SOC, i.e. ``E_T ≥ E_0``),
* discharge-only cost ``discharge_cost · p_d`` (charging energy is paid for
  through the generators that produce it); ``charge_cost`` exists for
  completeness and defaults to 0.

The net power ``p = p_d − p_c`` (positive = discharge) is what the algorithms
exchange.  Two exact responses are offered, each a convex QP over the whole
horizon that is compiled once (cvxpy parameters) and re-solved with OSQP warm
starts — the per-round cost of a storage participant:

* :meth:`StorageModel.price_response` — the storage counterpart of a
  generator's price response ``P(λ) = clip((λ − c)/ε)``: it minimises
  ``Σ c_d·p_d + c_c·p_c − λ·p + (ε/2)(p_d² + p_c²)``.  The proximal term with
  the same ε a generator of that size gets makes the response unique (a pure
  price response of a linear-cost battery is bang-bang) and discourages
  simultaneous charging and discharging.  Used by consensus and diffusion.
* :meth:`StorageModel.prox` — the proximal step
  ``argmin cost(p) + (ρ/2)‖p − center‖²``.  Used by ADMM (center ``−v``).
* :meth:`StorageModel.project` — the Euclidean projection of a desired net
  schedule onto the feasible set (``prox`` without the cost).  Used by FDGDM,
  whose update is a gradient step followed by a projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

_OSQP = {
    "eps_abs": 1e-9,
    "eps_rel": 1e-9,
    "polishing": True,
    "max_iter": 200_000,
    "warm_start": True,
    "verbose": False,
}
#: Tie-break against simultaneous charge/discharge in :meth:`StorageModel.project`.
_PROJECTION_TIEBREAK = 1e-7


@dataclass
class StorageModel:
    """One storage unit (see the module docstring for the model).

    :param e_max: Energy capacity (MWh).
    :param p_charge_max: Maximum charging power (MW, ≥ 0).
    :param p_discharge_max: Maximum discharging power (MW, ≥ 0).
    :param eta_charge: Charging efficiency η_c ∈ (0, 1].
    :param eta_discharge: Discharging efficiency η_d ∈ (0, 1].
    :param e_initial: Initial SOC as a fraction of ``e_max``.
    :param e_final: Minimum terminal SOC as a fraction of ``e_max``
        (``None`` → ``e_initial``, i.e. ``E_T ≥ E_0``).
    :param soc_min: Minimum SOC fraction.
    :param soc_max: Maximum SOC fraction.
    :param discharge_cost: Cost per MWh discharged.
    :param charge_cost: Cost per MWh charged (0 by convention).
    :param dt: Timestep length in hours.
    """

    e_max: float
    p_charge_max: float
    p_discharge_max: float
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    e_initial: float = 0.5
    e_final: float | None = None
    soc_min: float = 0.0
    soc_max: float = 1.0
    discharge_cost: float = 0.0
    charge_cost: float = 0.0
    dt: float = 1.0

    _price_problems: dict = field(default_factory=dict, init=False, repr=False)
    _projection_problems: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.e_max <= 0:
            raise ValueError(f"e_max must be positive, got {self.e_max!r}")
        if not (0 < self.eta_charge <= 1 and 0 < self.eta_discharge <= 1):
            raise ValueError("efficiencies must lie in (0, 1]")
        if not 0.0 <= self.soc_min <= self.soc_max <= 1.0:
            raise ValueError("need 0 <= soc_min <= soc_max <= 1")

    # -- bounds ---------------------------------------------------------------------

    @property
    def e_min_abs(self) -> float:
        return self.soc_min * self.e_max

    @property
    def e_max_abs(self) -> float:
        return self.soc_max * self.e_max

    @property
    def e0(self) -> float:
        """Initial energy (MWh), clipped to the SOC bounds."""
        return float(np.clip(self.e_initial * self.e_max, self.e_min_abs, self.e_max_abs))

    @property
    def e_terminal_min(self) -> float:
        """Minimum terminal energy (MWh), clipped to the SOC bounds."""
        frac = self.e_initial if self.e_final is None else self.e_final
        return float(np.clip(frac * self.e_max, self.e_min_abs, self.e_max_abs))

    # -- evaluation -----------------------------------------------------------------

    def energy(self, p_d: np.ndarray, p_c: np.ndarray) -> np.ndarray:
        """Energy path ``E[0..T]`` (MWh) for a charge/discharge schedule."""
        delta = self.dt * (np.asarray(p_c) * self.eta_charge - np.asarray(p_d) / self.eta_discharge)
        return self.e0 + np.concatenate([[0.0], np.cumsum(delta)])

    def is_feasible(self, p_d: np.ndarray, p_c: np.ndarray, tol: float = 1e-6) -> bool:
        """Whether a schedule satisfies every constraint of the model (within *tol*)."""
        p_d, p_c = np.asarray(p_d, dtype=float), np.asarray(p_c, dtype=float)
        e = self.energy(p_d, p_c)
        return bool(
            np.all(p_d >= -tol)
            and np.all(p_c >= -tol)
            and np.all(p_d <= self.p_discharge_max + tol)
            and np.all(p_c <= self.p_charge_max + tol)
            and np.all(e >= self.e_min_abs - tol)
            and np.all(e <= self.e_max_abs + tol)
            and e[-1] >= self.terminal_requirement(p_d.size) - tol
        )

    # -- exact responses --------------------------------------------------------------

    def terminal_requirement(self, T: int) -> float:
        """Terminal energy floor for a *T*-step horizon (MWh).

        ``e_terminal_min``, capped at what full-power charging can reach within
        the horizon, so a short horizon never makes the model infeasible (an
        unreachable target means "charge as much as possible").
        """
        reachable = self.e0 + T * self.dt * self.p_charge_max * self.eta_charge
        return min(self.e_terminal_min, reachable, self.e_max_abs)

    def _constraints(self, p_d: cp.Variable, p_c: cp.Variable, T: int) -> list:
        lower = np.tril(np.ones((T, T))) * self.dt
        e_path = self.e0 + lower @ (self.eta_charge * p_c - p_d / self.eta_discharge)
        return [
            p_d >= 0,
            p_c >= 0,
            p_d <= self.p_discharge_max,
            p_c <= self.p_charge_max,
            e_path >= self.e_min_abs,
            e_path <= self.e_max_abs,
            e_path[T - 1] >= self.terminal_requirement(T),
        ]

    def price_response(self, lam: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        """Exact SOC-feasible response to the price vector *lam*.

        :returns: ``(p, E)`` — net power (positive = discharge) and the energy
            path ``E[0..T]``.
        """
        lam = np.asarray(lam, dtype=float).ravel()
        T = lam.size
        key = (T, float(epsilon))
        if key not in self._price_problems:
            if epsilon <= 0:
                raise ValueError(f"epsilon must be positive, got {epsilon!r}")
            p_d, p_c = cp.Variable(T), cp.Variable(T)
            price = cp.Parameter(T)
            objective = cp.Minimize(
                self.discharge_cost * cp.sum(p_d)
                + self.charge_cost * cp.sum(p_c)
                - price @ (p_d - p_c)
                + (epsilon / 2.0) * (cp.sum_squares(p_d) + cp.sum_squares(p_c))
            )
            self._price_problems[key] = (
                cp.Problem(objective, self._constraints(p_d, p_c, T)),
                price,
                p_d,
                p_c,
            )
        problem, price, p_d, p_c = self._price_problems[key]
        price.value = lam
        return self._solve(problem, p_d, p_c)

    def prox(
        self, center: np.ndarray, rho: float, with_cost: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Proximal step ``argmin cost(p) + (ρ/2)‖p − center‖²`` over the feasible set.

        :param with_cost: Include the discharge/charge cost (``False`` gives the
            Euclidean projection, see :meth:`project`).
        :returns: ``(p, E)`` as for :meth:`price_response`.
        """
        center = np.asarray(center, dtype=float).ravel()
        T = center.size
        key = (T, bool(with_cost))
        if key not in self._projection_problems:
            p_d, p_c = cp.Variable(T), cp.Variable(T)
            # (ρ/2)‖p − y‖² = ½‖s·p − z‖² with s = √ρ, z = √ρ·y keeps the
            # problem DPP, so it is compiled once and only re-solved.
            z = cp.Parameter(T)
            scale = cp.Parameter(nonneg=True)
            cost = (
                self.discharge_cost * cp.sum(p_d) + self.charge_cost * cp.sum(p_c)
                if with_cost
                else 0
            )
            objective = cp.Minimize(
                cost
                + 0.5 * cp.sum_squares(scale * (p_d - p_c) - z)
                + _PROJECTION_TIEBREAK * cp.sum(p_d + p_c)
            )
            self._projection_problems[key] = (
                cp.Problem(objective, self._constraints(p_d, p_c, T)),
                (z, scale),
                p_d,
                p_c,
            )
        problem, (z, scale), p_d, p_c = self._projection_problems[key]
        root = float(np.sqrt(rho))
        scale.value = root
        z.value = root * center
        return self._solve(problem, p_d, p_c)

    def project(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Euclidean projection of a desired net schedule onto the feasible set.

        :returns: ``(p, E)`` as for :meth:`price_response`.
        """
        return self.prox(target, 1.0, with_cost=False)

    def _solve(
        self, problem: cp.Problem, p_d: cp.Variable, p_c: cp.Variable
    ) -> tuple[np.ndarray, np.ndarray]:
        problem.solve(solver=cp.OSQP, **_OSQP)
        if p_d.value is None:
            problem.solve(solver=cp.CLARABEL)
        if p_d.value is None:
            raise RuntimeError(f"storage QP failed (status={problem.status})")
        d = np.clip(np.asarray(p_d.value, dtype=float), 0.0, self.p_discharge_max)
        c = np.clip(np.asarray(p_c.value, dtype=float), 0.0, self.p_charge_max)
        return d - c, self.energy(d, c)
