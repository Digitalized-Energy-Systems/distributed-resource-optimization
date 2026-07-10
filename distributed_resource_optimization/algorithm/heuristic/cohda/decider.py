"""LocalSearchDecider — continuous-valued COHDA local search.

This module provides a gradient-free local search strategy for participants
whose feasible set is a continuous corridor rather than a finite enumeration.
The decider samples random values in each corridor dimension, evaluates a
combined local+global performance, and narrows the corridor by pruning
undesirable regions.


"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable

import numpy as np

from .core import (
    COHDAAlgorithmData,
    LocalDecider,
    ScheduleSelection,
    SolutionCandidate,
    SystemConfig,
    WorkingMemory,
    _evaluated,
    create_from_updated_sysconf,
)

if TYPE_CHECKING:
    pass


class LocalSearchDecider(LocalDecider):
    """Random local search within per-dimension corridors.

    For each schedule dimension *i*, ``find_new_value`` samples candidate
    values from ``corridors[i]``, evaluates a combined local+global
    performance, and iteratively prunes the search space toward the best
    region found.

    :param initial_schedule: Starting schedule vector.
    :param corridors: List of ``(low, high)`` bounds for each dimension.
    :param local_performance: ``(schedule) -> float`` local objective.
    :param convergence_force_factor: Weight of the global deviation term
                                     (pushes the agent toward the target).
    :param max_iterations: Pruning iterations per dimension.
    :param sample_size_per_value: Number of random samples drawn initially.
    :param distribution: Factory ``(low, high) -> Callable[[], float]``
                         that produces random values (default: uniform).
    """

    def __init__(
        self,
        initial_schedule: np.ndarray,
        corridors: list[tuple[float, float]],
        local_performance: Callable[[np.ndarray], float],
        convergence_force_factor: float = 0.1,
        max_iterations: int = 10,
        sample_size_per_value: int = 10,
        distribution: Callable[[float, float], Callable[[], float]] | None = None,
    ) -> None:
        self._initial_schedule = np.asarray(initial_schedule, dtype=float)
        self.corridors = corridors
        self.local_performance = local_performance
        self.convergence_force_factor = convergence_force_factor
        self.max_iterations = max_iterations
        self.sample_size_per_value = sample_size_per_value
        self.distribution = (
            distribution
            if distribution is not None
            else (lambda lo, hi: lambda: random.uniform(lo, hi))
        )

    def initial_schedule(self, memory: WorkingMemory) -> np.ndarray:
        return self._initial_schedule


# ---------------------------------------------------------------------------
# Algorithmic functions
# ---------------------------------------------------------------------------


def _local_performance_with_global_share(
    decider: LocalSearchDecider,
    schedule: np.ndarray,
    new_value: float,
    current_value: float,
    delta_to_target: float,
) -> float:
    """Combined local + convergence-force performance.

    The convergence force penalises the residual that would *remain* in this
    dimension after moving from *current_value* to *new_value*: changing the
    own contribution by ``new_value - current_value`` shrinks the current
    global residual ``delta_to_target`` by exactly that amount, so the term
    is maximal (zero) when the move closes the residual completely.
    """
    remaining_residual = delta_to_target - (new_value - current_value)
    return decider.local_performance(schedule) - decider.convergence_force_factor * abs(
        remaining_residual
    )


def _find_new_value(
    decider: LocalSearchDecider,
    current_index: int,
    current_best_schedule: np.ndarray,
    delta_to_target: float,
) -> float:
    """Find an improved value for dimension *current_index* via random search.

    Samples are drawn from the corridor once, evaluated (and consumed) one at
    a time, and the *best-performing* value is returned — keeping the current
    value when no sample beats it.  After every three evaluations the
    remaining samples are pruned toward the best-looking region.
    """
    lo, hi = decider.corridors[current_index]
    sampler = decider.distribution(lo, hi)
    possible = [sampler() for _ in range(decider.sample_size_per_value)]
    current_value = float(current_best_schedule[current_index])

    def _performance(value: float) -> float:
        candidate_schedule = current_best_schedule.copy()
        candidate_schedule[current_index] = value
        return _local_performance_with_global_share(
            decider, candidate_schedule, value, current_value, delta_to_target
        )

    best_value = current_value
    best_perf = _performance(current_value)
    window: list[tuple[float, float]] = []  # (value, performance)
    iteration = 1

    while possible and iteration <= decider.max_iterations:
        idx = random.randrange(len(possible))
        value = possible.pop(idx)
        perf = _performance(value)
        if perf > best_perf:
            best_value, best_perf = value, perf
        window.append((value, perf))

        if len(window) == 3:
            # Sort by value ascending; prune based on monotonicity
            window.sort(key=lambda t: t[0])
            v1, p1 = window[0]
            v2, p2 = window[1]
            v3, p3 = window[2]

            if p1 > p2 > p3:
                possible = [v for v in possible if v < v2]
            elif p3 > p2 > p1:
                possible = [v for v in possible if v > v2]
            elif (p2 > p1 > p3) or (p3 > p1 > p2):
                lo2, hi2 = min(v2, v3), max(v2, v3)
                possible = [v for v in possible if lo2 < v < hi2]
            window.clear()

        iteration += 1

    return best_value


def _find_in_local_search_room(
    decider: LocalSearchDecider,
    current_best_schedule: np.ndarray,
    open_schedule: np.ndarray,
) -> np.ndarray:
    """Search each dimension independently for an improved value."""
    new_solution = current_best_schedule.copy()
    for i in range(len(current_best_schedule)):
        new_solution[i] = _find_new_value(
            decider, i, current_best_schedule, float(open_schedule[i])
        )
    return new_solution


def decide(
    cohda_data: COHDAAlgorithmData,
    decider: LocalSearchDecider,
    sysconfig: SystemConfig,
    candidate: SolutionCandidate,
) -> tuple[SystemConfig, SolutionCandidate]:
    """LocalSearchDecider decide step.

    Searches for a better schedule in the local search corridor.  The
    *open_schedule* (residual distance from the current candidate sum to the
    target) guides the convergence-force term.  The searched schedule is
    adopted — and published to the system config — only when the resulting
    candidate outperforms the incumbent, mirroring the monotone-improvement
    rule of the enumerating default decider.

    :returns: Updated ``(system_config, candidate)`` pair.
    """
    participant_id = cohda_data.participant_id
    perf_func = cohda_data.performance_function
    target = cohda_data.memory.target_params

    current_best_schedule = candidate.schedules[participant_id - 1].copy()
    open_schedule = target.schedule - candidate.schedules.sum(axis=0)

    incumbent = candidate
    if incumbent.perf is None:
        incumbent = _evaluated(incumbent, perf_func, target)

    new_best_schedule = _find_in_local_search_room(decider, current_best_schedule, open_schedule)
    new_candidate = create_from_updated_sysconf(participant_id, sysconfig, new_best_schedule)
    new_candidate = _evaluated(new_candidate, perf_func, target)

    if new_candidate.perf > incumbent.perf:  # type: ignore[operator]
        chosen_candidate = new_candidate
        chosen_schedule = new_best_schedule
    else:
        chosen_candidate = incumbent
        chosen_schedule = current_best_schedule

    existing = sysconfig.schedule_choices.get(participant_id)
    if existing is None or not np.array_equal(chosen_schedule, existing.schedule):
        sysconfig.schedule_choices[participant_id] = ScheduleSelection(
            schedule=chosen_schedule,
            counter=cohda_data.counter + 1,
        )
        cohda_data.counter += 1

    return sysconfig, chosen_candidate
