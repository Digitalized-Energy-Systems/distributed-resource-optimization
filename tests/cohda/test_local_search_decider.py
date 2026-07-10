"""LocalSearchDecider unit tests."""

from __future__ import annotations

import random

import numpy as np
import pytest

from distributed_resource_optimization import (
    LocalSearchDecider,
    create_cohda_participant_with_decider,
    create_cohda_start_message,
    start_distributed_optimization,
)
from distributed_resource_optimization.algorithm.heuristic.cohda.decider import (
    _find_in_local_search_room,
    _find_new_value,
)


def _make_decider(
    initial: list[float],
    corridors: list[tuple[float, float]],
    local_perf=None,
) -> LocalSearchDecider:
    if local_perf is None:

        def local_perf(_: object) -> float:
            return 0.0

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
        from distributed_resource_optimization.algorithm.heuristic.cohda.core import (
            SystemConfig,
            WorkingMemory,
        )

        mem = WorkingMemory(
            target_params=None, system_config=SystemConfig(), solution_candidate=None
        )
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

    def test_moves_toward_residual_not_past_it(self):
        """The force term rewards *closing* the residual, not maximising the value.

        With current value 0 and residual 5 inside corridor (0, 10), the best
        sample is the one closest to 5 — values near 10 overshoot and must not
        win (the pre-fix force term was monotone in the value and always
        pushed to the corridor's upper end).
        """
        random.seed(7)
        d = _make_decider([0.0], [(0.0, 10.0)])
        d.convergence_force_factor = 1.0
        schedule = np.array([0.0])
        vals = [_find_new_value(d, 0, schedule, 5.0) for _ in range(20)]
        assert all(abs(v - 5.0) < 2.5 for v in vals)
        assert abs(np.mean(vals) - 5.0) < 1.0

    def test_returns_best_sample_not_last(self):
        """With a deterministic 'distribution' cycling through known values,
        the returned value must be the best-performing one regardless of
        sampling order."""
        values = iter([9.0, 2.0, 7.0, 1.0, 8.0])

        def distribution(lo: float, hi: float):
            return lambda: next(values)

        d = LocalSearchDecider(
            initial_schedule=np.array([0.0]),
            corridors=[(0.0, 10.0)],
            local_performance=lambda _: 0.0,
            convergence_force_factor=1.0,
            max_iterations=10,
            sample_size_per_value=5,
            distribution=distribution,
        )
        # Residual 2.0 from current 0.0 → sample 2.0 closes it exactly.
        val = _find_new_value(d, 0, np.array([0.0]), 2.0)
        assert val == 2.0

    def test_keeps_current_value_when_no_sample_improves(self):
        """If every sample is worse than staying put, the current value wins."""

        def distribution(lo: float, hi: float):
            return lambda: 10.0  # all samples far from the residual

        d = LocalSearchDecider(
            initial_schedule=np.array([1.0]),
            corridors=[(0.0, 10.0)],
            local_performance=lambda _: 0.0,
            convergence_force_factor=1.0,
            max_iterations=10,
            sample_size_per_value=5,
            distribution=distribution,
        )
        # Current value 1.0 with residual 0 → any move away is penalised.
        val = _find_new_value(d, 0, np.array([1.0]), 0.0)
        assert val == 1.0


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
    """End-to-end: two participants with LocalSearchDecider close in on the target.

    Both agents start at [1, 1] (sum [2, 2]) against target [4, 4], which is
    reachable inside the corridors (e.g. 2 + 2 per dimension).  The converged
    candidate must be complete, its performance must clearly improve on the
    starting point's −4.0, and the aggregate schedule must be near the target.
    """
    random.seed(42)
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

    candidate = p1.memory.solution_candidate
    assert candidate is not None
    assert candidate.present == frozenset({1, 2})
    # Starting point scores −4.0; random search must get substantially closer.
    assert candidate.perf is not None and candidate.perf > -1.0
    np.testing.assert_allclose(candidate.schedules.sum(axis=0), [4.0, 4.0], atol=1.0)


# ---------------------------------------------------------------------------
# decide(): publication and monotone improvement
# ---------------------------------------------------------------------------


class TestDecideStep:
    def _setup(self, distribution_value: float, current_row: list[float]):
        from distributed_resource_optimization import (
            ScheduleSelection,
            SolutionCandidate,
            SystemConfig,
            TargetParams,
        )
        from distributed_resource_optimization.algorithm.heuristic.cohda.decider import decide

        decider = LocalSearchDecider(
            initial_schedule=np.array(current_row),
            corridors=[(0.0, 5.0), (0.0, 5.0)],
            local_performance=lambda _: 0.0,
            convergence_force_factor=0.1,
            max_iterations=5,
            sample_size_per_value=3,
            distribution=lambda lo, hi: lambda: distribution_value,
        )
        participant = create_cohda_participant_with_decider(1, decider)
        participant.memory.target_params = TargetParams(
            schedule=np.array([2.0, 2.0]), weights=np.ones(2)
        )
        participant.counter = 1
        sysconfig = SystemConfig(
            {1: ScheduleSelection(schedule=np.array(current_row), counter=1)}
        )
        candidate = SolutionCandidate(
            participant_id=1,
            schedules=np.array([current_row]),
            perf=None,
            present=frozenset({1}),
        )
        return decide, participant, decider, sysconfig, candidate

    def test_improved_schedule_is_published_to_sysconfig(self):
        """When the search finds a better schedule, that schedule (not the old
        one) must be written to the system config, and the candidate must
        carry its evaluated performance.

        Regression test: decide() used to publish the *old* schedule and
        adopt the new candidate unconditionally.
        """
        decide, participant, decider, sysconfig, candidate = self._setup(
            distribution_value=2.0, current_row=[0.0, 0.0]
        )
        new_sysconfig, new_candidate = decide(participant, decider, sysconfig, candidate)

        # [2, 2] hits the target exactly: perf 0 beats the incumbent's -4.
        assert new_candidate.perf == pytest.approx(0.0)
        published = new_sysconfig.schedule_choices[1]
        np.testing.assert_allclose(published.schedule, [2.0, 2.0])
        assert published.counter == 2  # counter advanced with the change

    def test_worse_search_keeps_incumbent_and_sysconfig(self):
        """When no sample improves on the incumbent, the incumbent candidate
        and the published schedule stay untouched (no counter churn)."""
        decide, participant, decider, sysconfig, candidate = self._setup(
            distribution_value=3.0, current_row=[2.0, 2.0]
        )
        new_sysconfig, new_candidate = decide(participant, decider, sysconfig, candidate)

        assert new_candidate.perf == pytest.approx(0.0)  # incumbent already optimal
        published = new_sysconfig.schedule_choices[1]
        np.testing.assert_allclose(published.schedule, [2.0, 2.0])
        assert published.counter == 1  # unchanged — no spurious version bump
