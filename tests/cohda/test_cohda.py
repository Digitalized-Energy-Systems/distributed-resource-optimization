"""COHDA unit tests."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from distributed_resource_optimization import (
    ScheduleSelection,
    SolutionCandidate,
    SystemConfig,
    TargetParams,
    WorkingMemory,
    cohda_default_performance,
    create_cohda_participant,
    create_cohda_start_message,
    decide,
    merge_candidates,
    merge_sysconfigs,
    perceive,
)
from tests.carrier import TestCarrier

# ---------------------------------------------------------------------------
# Performance function
# ---------------------------------------------------------------------------


class TestCohdaDefaultPerformance:
    def test_uniform_weights(self):
        mat = np.array([[1, 2, 3], [1, 2, 3], [1, 1, 2]], dtype=float)
        target = TargetParams(
            schedule=np.array([3.0, 3.0, 5.0]),
            weights=np.array([1.0, 1.0, 1.0]),
        )
        assert cohda_default_performance(mat, target) == -5.0

    def test_nonuniform_weights(self):
        mat = np.array([[1, 2, 3], [1, 2, 3], [1, 1, 2]], dtype=float)
        target = TargetParams(
            schedule=np.array([3.0, 3.0, 5.0]),
            weights=np.array([1.0, 2.0, 3.0]),
        )
        assert cohda_default_performance(mat, target) == -13.0


# ---------------------------------------------------------------------------
# merge_sysconfigs
# ---------------------------------------------------------------------------

MERGE_SYSCONFIG_CASES = [
    # (i_dict, j_dict, expected_dict) — dicts map id -> (schedule, counter)
    (
        {1: ([1, 2], 42), 2: ([4, 2], 4)},
        {1: ([10, 20], 45), 2: ([40, 20], 5)},
        {1: ([10, 20], 45), 2: ([40, 20], 5)},
    ),
    (
        {1: ([1, 2], 42)},
        {1: ([10, 20], 40), 2: ([40, 20], 5)},
        {1: ([1, 2], 42), 2: ([40, 20], 5)},
    ),
    (
        {1: ([1, 2], 42)},
        {2: ([40, 20], 5)},
        {1: ([1, 2], 42), 2: ([40, 20], 5)},
    ),
    (
        {1: ([1, 2], 42)},
        {1: ([40, 20], 5)},
        {1: ([1, 2], 42)},
    ),
    (
        {1: ([1, 2], 42), 2: ([40, 20], 5)},
        {1: ([1, 2], 42), 2: ([40, 20], 5)},
        {1: ([1, 2], 42), 2: ([40, 20], 5)},
    ),
]


def _make_sysconfig(d: dict) -> SystemConfig:
    return SystemConfig(
        {
            pid: ScheduleSelection(schedule=np.array(s, dtype=float), counter=c)
            for pid, (s, c) in d.items()
        }
    )


@pytest.mark.parametrize("i_dict,j_dict,expected_dict", MERGE_SYSCONFIG_CASES)
class TestMergeSysconfigs:
    def test_merge(self, i_dict, j_dict, expected_dict):
        sc_i = _make_sysconfig(i_dict)
        sc_j = _make_sysconfig(j_dict)
        expected = _make_sysconfig(expected_dict)
        merged = merge_sysconfigs(sc_i, sc_j)
        assert merged == expected

    def test_identity_preserved(self, i_dict, j_dict, expected_dict):
        """If merged == i, the returned object must be the same object."""
        sc_i = _make_sysconfig(i_dict)
        sc_j = _make_sysconfig(j_dict)
        merged = merge_sysconfigs(sc_i, sc_j)
        expected = _make_sysconfig(expected_dict)
        if sc_i == expected:
            assert merged is sc_i


# ---------------------------------------------------------------------------
# merge_candidates
# ---------------------------------------------------------------------------


def _sum_schedule(cluster_schedule, _):
    return float(np.sum(cluster_schedule))


MERGE_CANDIDATES_CASES = [
    # (schedules_i, present_i, pid_i, perf_i,
    #  schedules_j, present_j, pid_j, perf_j,
    #  own_id,
    #  expected_schedules, expected_pid, expected_perf)
    (
        [[1, 2], [4, 2]],
        [1, 2],
        1,
        0.5,
        [[10, 20], [40, 20]],
        [1, 2],
        2,
        0.5,
        3,
        [[1, 2], [4, 2]],
        1,
        0.5,
    ),
    (
        [[1, 2], [4, 2]],
        [1, 2],
        1,
        0.4,
        [[10, 20], [40, 20]],
        [1, 2],
        2,
        0.5,
        3,
        [[10, 20], [40, 20]],
        2,
        0.5,
    ),
    (
        [[1, 2], [4, 2]],
        [1, 2],
        1,
        0.4,
        [[0, 0], [40, 20]],
        [1],
        2,
        0.5,
        3,
        [[1, 2], [4, 2]],
        1,
        0.4,
    ),
    (
        [[1, 2], [0, 0]],
        [1],
        1,
        0.4,
        [[10, 20], [40, 20]],
        [1, 2],
        2,
        0.5,
        3,
        [[10, 20], [40, 20]],
        2,
        0.5,
    ),
    (
        [[1, 2], [0, 0]],
        [1],
        1,
        0.4,
        [[0, 0], [40, 20]],
        [2],
        2,
        0.5,
        3,
        [[1, 2], [40, 20]],
        3,
        None,
    ),
]


@pytest.mark.parametrize(
    "si,pi,pid_i,perf_i,sj,pj,pid_j,perf_j,own_id,exp_s,exp_pid,exp_perf",
    MERGE_CANDIDATES_CASES,
)
class TestMergeCandidates:
    def test_merge(
        self, si, pi, pid_i, perf_i, sj, pj, pid_j, perf_j, own_id, exp_s, exp_pid, exp_perf
    ):
        ci = SolutionCandidate(
            participant_id=pid_i,
            schedules=np.array(si, dtype=float),
            perf=perf_i,
            present=frozenset(pi),
        )
        cj = SolutionCandidate(
            participant_id=pid_j,
            schedules=np.array(sj, dtype=float),
            perf=perf_j,
            present=frozenset(pj),
        )
        merged = merge_candidates(ci, cj, own_id, _sum_schedule, None)
        assert merged.participant_id == exp_pid
        assert np.allclose(merged.schedules, np.array(exp_s, dtype=float))
        assert merged.perf == exp_perf


# ---------------------------------------------------------------------------
# perceive
# ---------------------------------------------------------------------------


class TestSimplePerceiveStart:
    def test_target_params_set(self):
        cohda = create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]])
        input_wm = WorkingMemory(
            target_params=TargetParams(
                schedule=np.array([1.0, 2.0, 3.0]),
                weights=np.array([1.0, 1.0, 1.0]),
            ),
            system_config=SystemConfig(),
            solution_candidate=None,
        )
        perceive(cohda, [input_wm])
        assert cohda.memory.target_params == TargetParams(
            schedule=np.array([1.0, 2.0, 3.0]),
            weights=np.array([1.0, 1.0, 1.0]),
        )


# ---------------------------------------------------------------------------
# perceive + decide
# ---------------------------------------------------------------------------


class TestSelectionMultiplePerceiveDecide:
    def test_best_schedule_chosen(self):
        cohda = create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3], [1, 1, 1], [4, 2, 3]])
        init_wm = WorkingMemory(
            target_params=TargetParams(
                schedule=np.array([1.0, 2.0, 1.0]),
                weights=np.array([1.0, 1.0, 1.0]),
            ),
            system_config=SystemConfig(),
            solution_candidate=SolutionCandidate(
                participant_id=1,
                schedules=np.zeros((1, 3)),
                perf=0.0,
                present=frozenset(),
            ),
        )
        sysconf, candidate = perceive(cohda, [init_wm])
        sysconf, candidate = decide(cohda, cohda.decider, sysconf, candidate)

        assert np.allclose(candidate.schedules[0], [1, 1, 1])
        assert sysconf.schedule_choices[1].counter == 2


# ---------------------------------------------------------------------------
# on_exchange_message (direct call)
# ---------------------------------------------------------------------------


class TestOnExchangeCohda:
    def test_two_participant_exchange(self):
        test_carrier = TestCarrier(test_neighbors={1})
        part1 = create_cohda_participant(1, [[1, 1, 0.0], [1, 1, 1], [4, 2, 1], [0, 1, 0]])
        part2 = create_cohda_participant(2, [[0.0, 1, 2], [1, 2.0, 3], [1, 1, 1], [4, 2, 3]])
        init_wm = create_cohda_start_message([1, 2.0, 1])

        asyncio.get_event_loop().run_until_complete(
            part1.on_exchange_message(test_carrier, init_wm, None)
        )
        wm = test_carrier.test_neighbor_messages[1][-1]
        assert np.allclose(wm.solution_candidate.schedules, [[1, 1, 1]])

        asyncio.get_event_loop().run_until_complete(
            part2.on_exchange_message(test_carrier, wm, None)
        )
        wm = test_carrier.test_neighbor_messages[1][-1]
        asyncio.get_event_loop().run_until_complete(
            part1.on_exchange_message(test_carrier, wm, None)
        )
        wm = test_carrier.test_neighbor_messages[1][-1]

        assert part1.participant_id == 1
        assert part1.memory.target_params == init_wm.target_params

        expected_sc1 = {
            2: ScheduleSelection(schedule=np.array([0.0, 1.0, 2.0]), counter=1),
            1: ScheduleSelection(schedule=np.array([1.0, 1.0, 0.0]), counter=3),
        }
        for pid, expected_sel in expected_sc1.items():
            actual = part1.memory.system_config.schedule_choices[pid]
            assert actual == expected_sel, f"Mismatch for participant {pid}"

        assert np.allclose(wm.solution_candidate.schedules, [[1, 1, 0], [0, 1, 2]])

        asyncio.get_event_loop().run_until_complete(
            part2.on_exchange_message(test_carrier, wm, None)
        )
        wm = test_carrier.test_neighbor_messages[1][-1]

        assert wm is not None
        assert np.allclose(part1.memory.solution_candidate.schedules, [[1, 1, 0], [0, 1, 2]])
        assert np.allclose(part2.memory.solution_candidate.schedules, [[1, 1, 0], [0, 1, 2]])

        len_before = len(test_carrier.test_neighbor_messages[1])
        asyncio.get_event_loop().run_until_complete(
            part1.on_exchange_message(test_carrier, wm, None)
        )
        # No new message should have been sent (nothing changed)
        assert len_before == len(test_carrier.test_neighbor_messages[1])


# ---------------------------------------------------------------------------
# Hinrichs convergence test (10 participants)
# ---------------------------------------------------------------------------

S_HINRICHS = [
    [[1.0, 1, 1, 1, 1], [4, 3, 3, 3, 3], [6, 6, 6, 6, 6], [9, 8, 8, 8, 8], [11, 11, 11, 11, 11]],
    [
        [13, 12, 12, 12, 12],
        [15, 15, 15, 14, 14],
        [18, 17, 17, 17, 17],
        [20, 20, 20, 19, 19],
        [23, 22, 22, 22, 22],
    ],
    [
        [25, 24, 23, 23, 23],
        [27, 26, 26, 25, 25],
        [30, 29, 28, 28, 28],
        [32, 31, 31, 30, 30],
        [35, 34, 33, 33, 33],
    ],
    [
        [36, 35, 35, 34, 34],
        [39, 38, 37, 36, 36],
        [41, 40, 40, 39, 39],
        [44, 43, 42, 41, 41],
        [46, 45, 45, 44, 44],
    ],
    [
        [48, 47, 46, 45, 45],
        [50, 49, 48, 48, 47],
        [53, 52, 51, 50, 50],
        [55, 54, 53, 53, 52],
        [58, 57, 56, 55, 55],
    ],
    [
        [60, 58, 57, 56, 56],
        [62, 61, 60, 59, 58],
        [65, 63, 62, 61, 61],
        [67, 66, 65, 64, 63],
        [70, 68, 67, 66, 66],
    ],
    [
        [71, 70, 68, 67, 67],
        [74, 72, 71, 70, 69],
        [76, 75, 73, 72, 72],
        [79, 77, 76, 75, 74],
        [81, 80, 78, 77, 77],
    ],
    [
        [83, 81, 80, 78, 78],
        [85, 83, 82, 81, 80],
        [88, 86, 85, 83, 83],
        [90, 88, 87, 86, 85],
        [93, 91, 90, 88, 88],
    ],
    [
        [95, 92, 91, 90, 89],
        [97, 95, 93, 92, 91],
        [100, 97, 96, 95, 94],
        [102, 100, 98, 97, 96],
        [105, 102, 101, 100, 99],
    ],
    [
        [106, 104, 102, 101, 100],
        [109, 106, 105, 103, 102],
        [111, 109, 107, 106, 105],
        [114, 111, 110, 108, 107],
        [116, 114, 112, 111, 110],
    ],
]


class TestOnExchangeCohdaHinrichs:
    def test_convergence_10_participants(self):
        """Participants are run in round-robin order; each always processes the
        latest message.  The loop stops when a participant produces no new
        output (i.e. the algorithm has converged for that round).
        """
        test_carrier = TestCarrier(test_neighbors={1})
        parts = [
            create_cohda_participant(i + 1, schedule_set)
            for i, schedule_set in enumerate(S_HINRICHS)
        ]

        wm = create_cohda_start_message([542, 528, 519, 511, 509.0])

        last_length = -1
        while last_length == -1 or last_length < len(
            test_carrier.test_neighbor_messages.get(1, [])
        ):
            for part in parts:
                if last_length != -1:
                    last_length = len(test_carrier.test_neighbor_messages.get(1, []))
                else:
                    last_length = 0

                asyncio.get_event_loop().run_until_complete(
                    part.on_exchange_message(test_carrier, wm, None)
                )
                # always grab the latest message
                if test_carrier.test_neighbor_messages.get(1):
                    wm = test_carrier.test_neighbor_messages[1][-1]

                # if no new message was added, stop this inner pass
                if last_length == len(test_carrier.test_neighbor_messages.get(1, [])):
                    break

        assert parts[0].memory.solution_candidate.perf == pytest.approx(-5, abs=0.1)
        total = parts[0].memory.solution_candidate.schedules.sum(axis=0)
        assert np.allclose(total, [543, 529, 520, 512, 510])
