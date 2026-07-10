"""Unit tests for the generic ADMM coordinator and its norm helpers.

The coordinator's iteration/termination logic was previously exercised only
through full algorithm runs; these tests isolate it with a scripted carrier
whose participants respond synchronously.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from distributed_resource_optimization.algorithm.admm.consensus_admm import (
    ADMMConsensusGlobalActor,
    create_admm_start_consensus,
)
from distributed_resource_optimization.algorithm.admm.core import (
    ADMMAnswer,
    ADMMGenericCoordinator,
    ADMMMessage,
    _deepcopy_z,
    _max_diff_norm,
    _max_norm,
)
from distributed_resource_optimization.carrier.core import Carrier

# ---------------------------------------------------------------------------
# Scripted carrier
# ---------------------------------------------------------------------------


class ScriptedCarrier(Carrier):
    """Carrier whose participants are plain functions ``ADMMMessage -> np.ndarray``.

    ``send_awaitable`` resolves immediately with the responder's answer and
    counts how many rounds each participant saw.
    """

    def __init__(self, responders: list) -> None:
        self.responders = responders
        self.calls = [0] * len(responders)

    def send_to_other(self, content, receiver, meta=None):
        raise NotImplementedError

    def reply_to_other(self, content, meta):
        raise NotImplementedError

    def send_awaitable(self, content: ADMMMessage, receiver: int, meta=None) -> asyncio.Future:
        self.calls[receiver] += 1
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        fut.set_result(ADMMAnswer(x=self.responders[receiver](content)))
        return fut

    def others(self, participant_id: str) -> list[int]:
        return list(range(len(self.responders)))

    def get_address(self) -> int:
        return -1


def _box_responder(lb: np.ndarray, ub: np.ndarray, cost: np.ndarray):
    """Closed-form ADMMFlexActor box response: x = clip(-v - S/rho, lb, ub)."""

    def respond(message: ADMMMessage) -> np.ndarray:
        return np.clip(-np.asarray(message.v) - cost / message.rho, lb, ub)

    return respond


# ---------------------------------------------------------------------------
# Coordinator iteration / termination
# ---------------------------------------------------------------------------


class TestCoordinatorTermination:
    async def test_converges_and_stops_before_max_iters(self):
        """Exchange ADMM with two cheap/expensive box actors must hit the
        residual-based stop well before max_iters, and the solution must sum
        to the target with the cheap unit dispatched first (merit order)."""
        target = np.array([8.0])
        responders = [
            _box_responder(np.zeros(1), np.full(1, 10.0), np.array([1.0])),  # cheap
            _box_responder(np.zeros(1), np.full(1, 10.0), np.array([5.0])),  # expensive
        ]
        carrier = ScriptedCarrier(responders)
        coordinator = ADMMGenericCoordinator(
            global_actor=ADMMConsensusGlobalActor(alpha=0.0),
            rho=1.0,
            max_iters=500,
        )

        result = await coordinator.start_optimization(
            carrier, create_admm_start_consensus(target), {}
        )

        total = sum(result)
        assert total == pytest.approx(target, abs=1e-2)
        # Merit order: the cheap unit covers the demand.
        assert result[0] == pytest.approx([8.0], abs=0.1)
        assert result[1] == pytest.approx([0.0], abs=0.1)
        # Early stop actually happened.
        assert 0 < carrier.calls[0] < 500
        assert carrier.calls[0] == carrier.calls[1]

    async def test_max_iters_failsafe_bounds_rounds(self, caplog):
        """Responders that never satisfy the consensus constraint must stop at
        max_iters (with a warning) instead of iterating forever."""

        def stubborn(message: ADMMMessage) -> np.ndarray:
            return np.array([100.0])  # ignores every correction

        carrier = ScriptedCarrier([stubborn, stubborn])
        coordinator = ADMMGenericCoordinator(
            global_actor=ADMMConsensusGlobalActor(alpha=0.0),
            rho=1.0,
            max_iters=7,
        )

        with caplog.at_level("WARNING"):
            await coordinator.start_optimization(
                carrier, create_admm_start_consensus(np.array([1.0])), {}
            )

        assert carrier.calls == [7, 7]
        assert any("max iterations" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Norm helpers
# ---------------------------------------------------------------------------


class TestNormHelpers:
    def test_max_norm_array(self):
        assert _max_norm(np.array([1.0, -3.0, 2.0])) == 3.0

    def test_max_norm_list_of_arrays(self):
        v = [np.array([3.0, 4.0]), np.array([1.0, 0.0])]
        assert _max_norm(v) == pytest.approx(5.0)  # max of L2 norms

    def test_max_diff_norm_array(self):
        a = np.array([1.0, 5.0])
        b = np.array([2.0, 3.0])
        assert _max_diff_norm(a, b) == 2.0

    def test_max_diff_norm_list_of_arrays(self):
        a = [np.array([1.0, 1.0]), np.array([0.0, 0.0])]
        b = [np.array([1.0, 1.0]), np.array([3.0, 4.0])]
        assert _max_diff_norm(a, b) == pytest.approx(5.0)

    def test_deepcopy_z_is_independent(self):
        z_list = [np.array([1.0]), np.array([2.0])]
        copy_list = _deepcopy_z(z_list)
        copy_list[0][0] = 99.0
        assert z_list[0][0] == 1.0

        z_arr = np.array([1.0, 2.0])
        copy_arr = _deepcopy_z(z_arr)
        copy_arr[0] = 99.0
        assert z_arr[0] == 1.0
