"""Exchange ADMM — Boyd et al. 2011 §7.3.2 (and the sharing form §7.3 with g = indicator{≥ 0}).

Every participant i holds a vector x_i (e.g. a net injection per hour); the exchange
constraint couples them: Σ x_i = 0, or with ``surplus=True`` Σ x_i ≥ 0 (a surplus is allowed
and free). The coordinator keeps the averages, z̄ and the scaled dual u (one price vector ρu):

* z̄ = 0 (exchange) or max(x̄ + u, 0) element-wise (indicator{N z̄ ≥ 0}),
* u ← u + x̄ - z̄,
* participant i solves min f_i(x_i) + ρ/2 ||x_i - x_i^k + x̄^k - z̄^k + u^k||².

The correction sent to participant i is ``v = -(x_i - x̄ + z̄ - u)``, the convention of
:class:`.flex_actor.ADMMFlexActor` (x minimises ρ/2||x + v||² + its own cost). Start x = 0, u = 0.
"""

from __future__ import annotations

import numpy as np

from .core import ADMMGenericCoordinator
from .sharing_admm import ADMMSharingGlobalActor, ADMMTargetDistanceObjective


class ADMMExchangeGlobalActor(ADMMSharingGlobalActor):
    """Global actor of exchange ADMM: the sharing actor with g = indicator{0} or {≥ 0}.

    :param surplus: allow Σ x_i ≥ 0 (z̄ = max(x̄ + u, 0)) instead of Σ x_i = 0.
    """

    def __init__(self, surplus: bool = True) -> None:
        super().__init__(ADMMTargetDistanceObjective())
        self.surplus = surplus

    def z_update(self, input_data, x, u, z, rho, n) -> np.ndarray:
        # 2026-11-13 Q-E1 (D-117): the surplus slack as g = indicator{Σx ≥ 0}
        x_avg = sum(x) / len(x)
        return np.maximum(x_avg + u, 0.0) if self.surplus else np.zeros_like(x_avg)

    def init_z(self, n: int, m: int) -> np.ndarray:
        return np.zeros(m)  # 2026-11-13 Q-E4 (D-120): flat start


def create_exchange_admm_coordinator(
    surplus: bool = True, **coordinator_kwargs
) -> ADMMGenericCoordinator:
    """:class:`~.core.ADMMGenericCoordinator` for exchange ADMM; *coordinator_kwargs* go to it
    (rho, max_iters, abs_tol, rel_tol, mu, tau, round_timeout). Start it with
    ``create_admm_start(None, length)`` from :mod:`.core`."""
    return ADMMGenericCoordinator(ADMMExchangeGlobalActor(surplus), **coordinator_kwargs)
