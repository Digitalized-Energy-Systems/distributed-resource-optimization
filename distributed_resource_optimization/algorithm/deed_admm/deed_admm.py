"""DEED-ADMM: Distributed Economic Dispatch via ADMM with dynamic consensus.

Zhu et al. 2025, "DEED-ADMM: A Scalable Distributed Algorithm for Economic
Dispatch in Multi-Energy Systems With Energy Storage".

Each agent i runs the following update per iteration k (Algorithm 1):

  Collect from neighbours:  λ̃ᵢ = Σⱼ wᵢⱼ λⱼ,ₖ   ξ̃ᵢ = Σⱼ wᵢⱼ ξⱼ,ₖ

  Compute b-vectors:
    bₓ = γx − λ̃ + Xᵀ(γx̂ − vₓ) − (γ/2)ξ̃
    bᵧ = γy + λ̃ + Yᵀ(γŷ − vᵧ) + (γ/2)ξ̃

  Primal update (closed form):
    x ← Hₓ bₓ
    y ← Hᵧ(bᵧ + AᵀHᵢ(d − AHᵧbᵧ))        [generator case, B=0]
    p̂ ← clip(Mp + v/γ, p̂_min, p̂_max)     [box projection]

  Auxiliary / dual update:
    ξ ← ξ̃ + (x_new − x_old) − (y_new − y_old)
    λ ← λ̃ + (γ/2)ξ
    v ← v + γ(Mp − p̂)

For the electricity-only PyPSA benchmark the multi-energy hub model
simplifies to Aᵢ = Iᵀ (no energy conversion), B = 0, zᵢ = 0.  The
weight matrix is uniform: wᵢⱼ = 1/n for all j (including self).

Deviations from the paper (by design):
* The paper's ramp-rate constraints (16)-(17) are not modelled — none of
  the benchmark networks specify ramp limits.
* Iterations run for a fixed ``max_iter`` instead of the paper's
  "while not converged"; the final schedule is read from the box-projected
  p̂ₓ so it stays feasible even when cut off early.
* Storage SOC dynamics are enforced by a greedy projection inside the
  x-update (see :class:`DEEDADMMStorageAlgorithm`) rather than by the
  paper's linear-inequality Mᵢ mechanism.

Communication pattern: peer-to-peer (no central coordinator), identical
to the averaging-consensus pattern.  Each agent broadcasts its (λ, ξ)
after every primal update and waits for ALL neighbours before advancing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..core import DistributedAlgorithm, OptimizationMessage

if TYPE_CHECKING:
    from ...carrier.core import Carrier


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


@dataclass
class DEEDADMMMessage(OptimizationMessage):
    """Message exchanged between DEED-ADMM participants.

    :param lam: Sender's current local dual estimate λᵢ,ₖ (shape τ).
    :param xi: Sender's current consensus tracker ξᵢ,ₖ (shape τ).
    :param k: Current iteration counter.
    :param data: Auxiliary payload forwarded on the initial kick-off.
    :param initial: If ``True`` this is the kick-off message; the recipient
                    (re-)initialises its state.
    """

    lam: np.ndarray
    xi: np.ndarray
    k: int
    data: Any
    initial: bool = False


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


class DEEDADMMAlgorithm(DistributedAlgorithm):
    """Peer-to-peer DEED-ADMM participant for a single electricity generator.

    This implementation covers the electricity-only (single-carrier) case:
    * Aᵢ = Iᵀ  (no energy-carrier transformation)
    * B  = 0    (no storage coupling; use a dedicated storage subclass for that)
    * zᵢ = 0

    The variable xᵢ (supply) converges to the generator's optimal schedule.
    The hub-input variable yᵢ is pinned to dᵢ from the first iteration by
    the per-agent constraint Aᵢyᵢ = dᵢ (Aᵢ = I here); only the global sums
    Σxᵢ = Σyᵢ = Σdᵢ are matched by the ξ tracker, not xᵢ = yᵢ per agent.

    :param finish_callback: ``(algorithm, carrier) -> None`` called when done.
    :param cost_quad: Per-step quadratic cost coefficient aᵢ (shape τ or
        scalar); cost is c(xₜ) = aᵢ xₜ² + bᵢ xₜ.  Set to 0 for linear cost.
    :param cost_lin: Per-step linear cost coefficient bᵢ (shape τ or scalar).
    :param x_min: Lower bound on generator output per step (shape τ or scalar).
    :param x_max: Upper bound on generator output per step (shape τ or scalar).
    :param d_i: Demand allocated to this agent per step (shape τ).  The sum
        over all agents must equal total system demand at each step.
    :param gamma: ADMM penalty parameter γ > 0 (paper default: 0.05).
    :param max_iter: Maximum number of DEED-ADMM iterations.
    :param n_agents: Total number of participating agents n (used to compute
        the self-weight wᵢᵢ = 1/n in the uniform doubly-stochastic matrix).
    """

    def __init__(
        self,
        finish_callback: Callable,
        cost_quad: np.ndarray | float,
        cost_lin: np.ndarray | float,
        x_min: np.ndarray | float,
        x_max: np.ndarray | float,
        d_i: np.ndarray,
        gamma: float = 0.05,
        max_iter: int = 500,
        n_agents: int = 1,
    ) -> None:
        self.finish_callback = finish_callback
        self.gamma = float(gamma)
        self.max_iter = max_iter
        self.n_agents = n_agents

        tau = len(np.atleast_1d(d_i))

        self._cost_quad = np.broadcast_to(np.atleast_1d(cost_quad).astype(float), (tau,)).copy()
        self._cost_lin = np.broadcast_to(np.atleast_1d(cost_lin).astype(float), (tau,)).copy()
        self._x_min = np.broadcast_to(np.atleast_1d(x_min).astype(float), (tau,)).copy()
        self._x_max = np.broadcast_to(np.atleast_1d(x_max).astype(float), (tau,)).copy()
        self._d_i = np.asarray(d_i, dtype=float).copy()
        self._tau = tau

        # Precomputed inverse matrices (diagonal → stored as 1-D arrays).
        #
        # With X = Y = I (box constraints only):
        #   Hₓᵢ = (∇²cᵢ + 2γ I)⁻¹  →  diag: 1 / (2aᵢ + 2γ)
        #   Hᵧ  = (2γ I)⁻¹           →  scalar: 1/(2γ)
        #   Hᵢ  = (Aᵢ Hᵧ Aᵢᵀ)⁻¹     →  Aᵢ=I  ⇒  Hᵢ = 1/Hᵧ = 2γ I
        g = self.gamma
        self._H_x = 1.0 / (2.0 * self._cost_quad + 2.0 * g)  # shape (τ,)
        self._H_y_scalar = 1.0 / (2.0 * g)                    # scalar
        self._H_i_scalar = 2.0 * g                            # scalar (= 1/H_y)

        # Runtime state (initialised in _reset_state)
        self._x: np.ndarray = np.zeros(tau)
        self._y: np.ndarray = np.zeros(tau)
        self._p_hat_x: np.ndarray = np.zeros(tau)
        self._p_hat_y: np.ndarray = np.zeros(tau)
        self._xi: np.ndarray = np.zeros(tau)
        self._lam: np.ndarray = np.zeros(tau)
        self._v_x: np.ndarray = np.zeros(tau)
        self._v_y: np.ndarray = np.zeros(tau)
        self._k: int = 0

        self._msg_queue: dict[int, list[DEEDADMMMessage]] = {}
        self._first_message: bool = True
        self._started: bool = False

        # Public result — written by finish_callback convention
        self.P: np.ndarray = np.zeros(tau)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_x(self, x_raw: np.ndarray) -> np.ndarray:
        """Hook applied to the raw x-update before it is stored.

        The base class applies no projection (the box bounds are handled by
        the p̂ variable); subclasses insert schedule-feasibility projections
        here (e.g. SOC dynamics for storage).
        """
        return x_raw

    def _extract_schedule(self) -> np.ndarray:
        """Final schedule reported at termination.

        p̂ₓ is the box-projected supply variable; at convergence x = p̂ₓ.
        Using p̂ₓ (not raw x) keeps the schedule feasible even when the
        algorithm hasn't fully converged.
        """
        return self._p_hat_x.copy()

    def _reset_state(self) -> None:
        """Initialise all iteration variables to zero."""
        tau = self._tau
        self._x = np.zeros(tau)
        self._y = np.zeros(tau)
        self._p_hat_x = np.zeros(tau)
        self._p_hat_y = np.zeros(tau)
        self._xi = np.zeros(tau)
        self._lam = np.zeros(tau)
        self._v_x = np.zeros(tau)
        self._v_y = np.zeros(tau)
        self._k = 0
        self._msg_queue.clear()

    def _step(self, neighbour_messages: list[DEEDADMMMessage]) -> None:
        """Run one DEED-ADMM primal/dual update using collected neighbour messages.

        Implements Algorithm 1 of Zhu et al. 2025 for the single-carrier,
        no-storage (B=0, zᵢ=0, Aᵢ=I) case.
        """
        g = self.gamma
        n = self.n_agents

        # --- 1. Weighted neighbour averages (uniform W = (1/n)J) ---
        # Include self contribution with weight 1/n; neighbour messages each
        # get weight 1/n.  The weighted sum = (self + Σ neighbours) / n.
        lam_sum = self._lam.copy()
        xi_sum = self._xi.copy()
        for msg in neighbour_messages:
            lam_sum += msg.lam
            xi_sum += msg.xi
        lam_tilde = lam_sum / n
        xi_tilde = xi_sum / n

        # --- 2. b-vectors (X = Y = I, so Xᵀ = Yᵀ = I) ---
        # The x-stationarity condition is:
        #   (∇²cᵢ + γ(I + XᵀX)) x = b_xi − ∇cᵢ_linear
        # so we subtract the constant part of the cost gradient (= cost_lin)
        # before multiplying by Hₓᵢ.
        b_x = (
            g * self._x
            - lam_tilde
            + (g * self._p_hat_x - self._v_x)
            - (g / 2.0) * xi_tilde
            - self._cost_lin   # linear cost gradient (constant part of ∇cᵢ)
        )
        b_y = (
            g * self._y
            + lam_tilde
            + (g * self._p_hat_y - self._v_y)
            + (g / 2.0) * xi_tilde
        )

        # --- 3. Primal updates ---
        # x update: x ← Hₓ bₓ, then the subclass feasibility hook.
        x_new = self._project_x(self._H_x * b_x)

        # y update (Aᵢ=I, Hᵧ=scalar, Hᵢ=scalar):
        #   inner = d − Hᵧ bᵧ
        #   y ← Hᵧ(bᵧ + Hᵢ inner)  =  Hᵧ bᵧ + Hᵧ Hᵢ inner
        # With Hᵧ = 1/(2γ) and Hᵢ = 2γ:  Hᵧ Hᵢ = 1, so y = dᵢ exactly at
        # every iteration (projection onto Aᵢy = dᵢ).  The first-iteration
        # jump y: 0 → dᵢ is how demand enters the ξ tracker.  Consequently
        # p̂_y and v_y below cannot influence the result; they are kept only
        # to mirror the paper's Algorithm 1 structure.
        Hy_b_y = self._H_y_scalar * b_y
        inner = self._d_i - Hy_b_y
        y_new = Hy_b_y + self._H_i_scalar * (self._H_y_scalar * inner)

        # p̂ projection: clip(Mp + v/γ, p̂_min, p̂_max)
        # M = blkdiag(I, I) → Mx = [x; y], p̂ = [p̂_x; p̂_y]
        p_hat_x_new = np.clip(x_new + self._v_x / g, self._x_min, self._x_max)
        p_hat_y_new = np.clip(y_new + self._v_y / g, self._x_min, self._x_max)

        # --- 4. Auxiliary update ---
        xi_new = xi_tilde + (x_new - self._x) - (y_new - self._y)

        # --- 5. Dual updates ---
        lam_new = lam_tilde + (g / 2.0) * xi_new
        v_x_new = self._v_x + g * (x_new - p_hat_x_new)
        v_y_new = self._v_y + g * (y_new - p_hat_y_new)

        # Store
        self._x = x_new
        self._y = y_new
        self._p_hat_x = p_hat_x_new
        self._p_hat_y = p_hat_y_new
        self._xi = xi_new
        self._lam = lam_new
        self._v_x = v_x_new
        self._v_y = v_y_new

    # ------------------------------------------------------------------
    # DistributedAlgorithm interface
    # ------------------------------------------------------------------

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: DEEDADMMMessage,
        meta: Any,
    ) -> None:
        """Handle one incoming DEED-ADMM message from a neighbour."""
        neighbours = carrier.others("")

        # --- Termination path ---
        if message_data.k >= self.max_iter:
            if not self._started:
                return
            self.P = self._extract_schedule()
            self.finish_callback(self, carrier)
            self._first_message = True
            self._started = False
            self._msg_queue.clear()
            return

        # After termination ignore stale pre-convergence messages.
        if self._first_message and self._started and not message_data.initial:
            return

        # --- Initialisation path ---
        if self._first_message or message_data.initial:
            self._first_message = False
            self._started = True
            self._reset_state()
            # Broadcast our initial (λ, ξ) = (0, 0) to all neighbours
            for addr in neighbours:
                carrier.send_to_other(
                    DEEDADMMMessage(
                        lam=self._lam.copy(),
                        xi=self._xi.copy(),
                        k=0,
                        data=message_data.data,
                    ),
                    addr,
                )
            # External kick-off (initial=True from a non-neighbour leader): done.
            # A first real-neighbour message (initial=False) carries k=0 state
            # — fall through to queue/advance so this step is counted.
            if message_data.initial:
                return

        # --- Normal path: queue and advance when all neighbours ready ---
        queue = self._msg_queue.setdefault(message_data.k, [])
        queue.append(message_data)

        if len(queue) < len(neighbours):
            return

        # All neighbours' messages for iteration k received — run one step.
        self._step(queue)
        del self._msg_queue[message_data.k]
        self._k += 1

        next_k = self._k
        for addr in neighbours:
            carrier.send_to_other(
                DEEDADMMMessage(
                    lam=self._lam.copy(),
                    xi=self._xi.copy(),
                    k=next_k,
                    data=message_data.data,
                ),
                addr,
            )


# ---------------------------------------------------------------------------
# Storage variant
# ---------------------------------------------------------------------------


class DEEDADMMStorageAlgorithm(DEEDADMMAlgorithm):
    """DEED-ADMM participant for an electricity storage unit.

    Extends :class:`DEEDADMMAlgorithm` with SOC-aware projection so that the
    multi-step charge/discharge schedule always satisfies:

    * Power bounds: −p_charge_max ≤ xₜ ≤ p_discharge_max
    * Energy capacity: soc_min·e_max ≤ SOCₜ ≤ soc_max·e_max
    * SOC dynamics: SOCₜ₊₁ = SOCₜ − xₜ/η_d  (discharge)
                                              − xₜ·η_c  (charge, xₜ < 0)
    * Terminal energy target: SOC_τ ≈ e_final·e_max  (bisection enforced)

    The λ/ξ consensus protocol and the ``_step`` update are inherited
    unchanged from the parent class; only the :meth:`_project_x` hook is
    overridden to insert the SOC projection after the raw x-update.

    Note this deviates from the paper, which folds the SOC dynamics
    (eqs. 9-11) into the exact ADMM as linear inequalities via the Mᵢ
    matrix.  The greedy projection used here is a heuristic — the paper's
    convergence guarantees do not formally carry over — but it keeps the
    per-iteration cost trivial and works well empirically.

    The storage unit gets ``d_i = 0`` (zero demand allocation): it
    contributes net injection, not demand consumption.  The global energy
    balance Σ x_gen + Σ x_stor = demand is still enforced by the ξ tracker.

    :param e_max: Energy capacity in MWh.
    :param p_charge_max: Maximum charging rate (MW, positive value).
    :param p_discharge_max: Maximum discharging rate (MW, positive value).
    :param eta_charge: Charging efficiency η_c ∈ (0, 1].
    :param eta_discharge: Discharging efficiency η_d ∈ (0, 1].
    :param e_initial: Initial state of charge as a fraction of e_max.
    :param e_final: Target terminal state of charge fraction (defaults to
        e_initial, i.e. return-to-origin).
    :param soc_min: Minimum SOC fraction (lower bound).
    :param soc_max: Maximum SOC fraction (upper bound).
    """

    def __init__(
        self,
        finish_callback: Callable,
        cost_quad: np.ndarray | float,
        cost_lin: np.ndarray | float,
        x_min: np.ndarray | float,
        x_max: np.ndarray | float,
        d_i: np.ndarray,
        gamma: float = 0.05,
        max_iter: int = 500,
        n_agents: int = 1,
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
    ) -> None:
        super().__init__(
            finish_callback=finish_callback,
            cost_quad=cost_quad,
            cost_lin=cost_lin,
            x_min=x_min,
            x_max=x_max,
            d_i=d_i,
            gamma=gamma,
            max_iter=max_iter,
            n_agents=n_agents,
        )
        self._e_max = float(e_max)
        self._p_charge_max = float(p_charge_max)
        self._p_discharge_max = float(p_discharge_max)
        self._eta_c = float(eta_charge)
        self._eta_d = float(eta_discharge)
        self._e_initial_frac = float(e_initial)
        self._e_final_frac = float(e_initial if e_final is None else e_final)
        self._soc_min = float(soc_min)
        self._soc_max = float(soc_max)

        # Public SOC trajectory (τ+1 points), written at termination.
        self.E: np.ndarray = np.zeros(self._tau + 1)

    # ------------------------------------------------------------------
    # SOC projection helpers
    # ------------------------------------------------------------------

    def _project_soc_forward(
        self, x_desired: np.ndarray, bias: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Greedy forward-pass SOC projection with additive bias.

        Returns ``(x_projected, e_path, e_terminal)``.
        """
        tau = self._tau
        e_max = self._e_max
        e_min_abs = self._soc_min * e_max
        e_max_abs = self._soc_max * e_max

        x = np.empty(tau)
        e = np.empty(tau + 1)
        e[0] = self._e_initial_frac * e_max

        for t in range(tau):
            e_t = e[t]
            max_discharge = (e_t - e_min_abs) * self._eta_d
            max_charge = (e_max_abs - e_t) / max(self._eta_c, 1e-9)

            p_lo = max(-self._p_charge_max, -max_charge)
            p_hi = min(self._p_discharge_max, max_discharge)
            p_lo = min(p_lo, p_hi)  # guard against empty interval

            x[t] = float(np.clip(x_desired[t] + bias, p_lo, p_hi))

            if x[t] >= 0.0:
                e[t + 1] = e_t - x[t] / max(self._eta_d, 1e-9)
            else:
                e[t + 1] = e_t - x[t] * self._eta_c

            e[t + 1] = float(np.clip(e[t + 1], e_min_abs, e_max_abs))

        return x, e, float(e[-1])

    def _project_soc(self, x_desired: np.ndarray) -> np.ndarray:
        """Project desired schedule onto SOC-feasible region.

        Uses bisection over an additive bias to also hit the target terminal
        energy (mirrors ``ReservoirStorageDiffusionActor``).
        """
        e_target = self._e_final_frac * self._e_max

        def f(bias: float) -> float:
            _, _, e_fin = self._project_soc_forward(x_desired, bias)
            return e_fin - e_target

        # Fast path — no terminal correction needed.
        x0, e0, e_fin0 = self._project_soc_forward(x_desired, 0.0)
        if abs(e_fin0 - e_target) < 1e-3:
            return x0

        # Exponential expansion to bracket the root. f(bias) is decreasing in
        # bias (larger bias -> more discharge -> lower terminal energy), so
        # when f0 < 0 (ended below target) the bias must go negative (more
        # charging) to raise it back up, and vice versa.
        f0 = e_fin0 - e_target
        best_x, best_err = x0, abs(f0)
        lo, hi = 0.0, (-1.0 if f0 < 0.0 else 1.0)
        f_lo, f_hi = f0, f(hi)
        if abs(f_hi) < best_err:
            best_x, best_err = self._project_soc_forward(x_desired, hi)[0], abs(f_hi)
        for _ in range(20):
            if f_lo * f_hi <= 0.0:
                break
            hi *= 2.0
            f_hi = f(hi)
            if abs(f_hi) < best_err:
                best_x, best_err = self._project_soc_forward(x_desired, hi)[0], abs(f_hi)
        else:
            # Couldn't bracket — return the best bias tried so far.
            return best_x

        # 35-iteration bisection.
        for _ in range(35):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            x_mid, _, _ = self._project_soc_forward(x_desired, mid)
            if abs(f_mid) < best_err:
                best_err = abs(f_mid)
                best_x = x_mid
            if f_lo * f_mid <= 0.0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid

        return best_x

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def _project_x(self, x_raw: np.ndarray) -> np.ndarray:
        """Insert the SOC projection after the raw x-update."""
        return self._project_soc(x_raw)

    def _extract_schedule(self) -> np.ndarray:
        """Report the SOC-feasible x (not p̂ₓ) and record the SOC trajectory.

        p̂ₓ is only power-box projected; before full convergence it may
        violate the SOC dynamics.  ``self._x`` is SOC-feasible by
        construction (every ``_step`` passes through ``_project_soc``).
        Re-running the forward pass on it is idempotent and yields the
        energy path, stored in :attr:`E`.
        """
        x, e, _ = self._project_soc_forward(self._x, 0.0)
        self.E = e
        return x
