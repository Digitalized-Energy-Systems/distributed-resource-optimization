"""LocalSearchDecider unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from distributed_resource_optimization import (
    LocalSearchDecider,
    create_cohda_participant_with_decider,
    create_cohda_start_message,
    start_distributed_optimization,
)
from distributed_resource_optimization.algorithm.heuristic.cohda.decider import (
    _find_new_value,
    _find_in_local_search_room,
)


def _make_decider(
    initial: list[float],
    corridors: list[tuple[float, float]],
    local_perf=None,
) -> LocalSearchDecider:
    if local_perf is None:
        local_perf = lambda _: 0.0
    return LocalSearchDecider(
        initial_schedule=np.array(initial, dtype=float),
        corridors=corridors,
        local_performance=local_perf,
        max_iterations=20,
        sample_size_per_value=20,
    )


class TestLocalSearchDeciderInit:
    def test_initial_schedule_returned(self):
        d = _make_decider([1.0, 2.0], [(0.0, 5.0), (0.0, 5.0)])
        from distributed_resource_optimization.algorithm.heuristic.cohda.core import WorkingMemory, SystemConfig
        mem = WorkingMemory(target_params=None, system_config=SystemConfig(), solution_candidate=None)
        result = d.initial_schedule(mem)
        assert np.allclose(result, [1.0, 2.0])

    def test_corridors_stored(self):
        d = _make_decider([0.5], [(0.0, 1.0)])
        assert d.corridors == [(0.0, 1.0)]

    def test_default_distribution_is_uniform(self):
        d = _make_decider([0.5], [(0.0, 1.0)])
        sampler = d.distribution(0.0, 1.0)
        samples = [sampler() for _ in range(100)]
        assert all(0.0 <= s <= 1.0 for s in samples)


class TestFindNewValue:
    def test_result_within_corridor(self):
        d = _make_decider([0.5], [(0.0, 1.0)])
        schedule = np.array([0.5])
        val = _find_new_value(d, 0, schedule, 0.0)
        lo, hi = d.corridors[0]
        assert lo <= val <= hi

    def test_pushes_toward_positive_delta(self):
        """With delta_to_target > 0, convergence force should prefer larger values."""
        d = _make_decider(
            [0.0],
            [(0.0, 10.0)],
            local_perf=lambda _: 0.0,
        )
        d.convergence_force_factor = 1.0
        schedule = np.array([0.0])
        # Large positive delta_to_target → prefer larger values
        vals = [_find_new_value(d, 0, schedule, 5.0) for _ in range(10)]
        assert np.mean(vals) > 3.0  # should be clearly above midpoint


class TestFindInLocalSearchRoom:
    def test_output_shape(self):
        d = _make_decider([1.0, 2.0], [(0.0, 5.0), (0.0, 5.0)])
        current = np.array([1.0, 2.0])
        open_sched = np.array([1.0, 1.0])
        result = _find_in_local_search_room(d, current, open_sched)
        assert result.shape == (2,)

    def test_values_within_corridors(self):
        d = _make_decider([2.0, 3.0], [(1.0, 4.0), (2.0, 5.0)])
        current = np.array([2.0, 3.0])
        open_sched = np.array([0.5, 0.5])
        result = _find_in_local_search_room(d, current, open_sched)
        assert 1.0 <= result[0] <= 4.0
        assert 2.0 <= result[1] <= 5.0


@pytest.mark.asyncio
async def test_cohda_with_local_search_decider_converges():
    """End-to-end: two participants with LocalSearchDecider reach a valid solution."""
    corridors = [(0.0, 5.0), (0.0, 5.0)]
    initial = [1.0, 1.0]

    d1 = LocalSearchDecider(
        initial_schedule=np.array(initial),
        corridors=corridors,
        local_performance=lambda _: 0.0,
        max_iterations=10,
        sample_size_per_value=10,
    )
    d2 = LocalSearchDecider(
        initial_schedule=np.array(initial),
        corridors=corridors,
        local_performance=lambda _: 0.0,
        max_iterations=10,
        sample_size_per_value=10,
    )

    p1 = create_cohda_participant_with_decider(1, d1)
    p2 = create_cohda_participant_with_decider(2, d2)

    start = create_cohda_start_message([4.0, 4.0])
    await start_distributed_optimization([p1, p2], start)

    # LocalSearchDecider is gradient-based — perf is not explicitly evaluated
    assert p1.memory.solution_candidate is not None
    assert 1 in p1.memory.solution_candidate.present
    assert 2 in p1.memory.solution_candidate.present
