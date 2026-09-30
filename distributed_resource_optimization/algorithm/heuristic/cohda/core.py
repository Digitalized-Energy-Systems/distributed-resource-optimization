"""COHDA — Combinatorial Optimization Heuristic for Distributed Agents.

Each agent maintains a :class:`WorkingMemory` consisting of:

* A :class:`TargetParams` describing the global target schedule and weights.
* A :class:`SystemConfig` — the agent's view of every participant's current
  schedule choice (with a monotonic counter for version control).
* A :class:`SolutionCandidate` — the best complete solution known so far.

Agents exchange :class:`WorkingMemory` objects.  Upon receipt each agent runs
*perceive → decide → act* and forwards its updated memory to all neighbours.

References
----------
Hinrichs et al. (2014) "COHDA: A Combinatorial Optimization Heuristic for
Distributed Agents".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ...core import DistributedAlgorithm, OptimizationMessage

if TYPE_CHECKING:
    from ....carrier.core import Carrier


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScheduleSelection:
    """A participant's chosen schedule together with its version counter."""

    schedule: np.ndarray
    counter: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScheduleSelection):
            return NotImplemented
        return np.array_equal(self.schedule, other.schedule) and self.counter == other.counter

    def __hash__(self) -> int:
        return hash((tuple(float(v) for v in self.schedule), self.counter))

    def __repr__(self) -> str:
        return f"ScheduleSelection(schedule={self.schedule.tolist()}, counter={self.counter})"


@dataclass
class SystemConfig:
    """Each participant's schedule choice, keyed by 1-indexed participant ID."""

    schedule_choices: dict[int, ScheduleSelection] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SystemConfig):
            return NotImplemented
        if set(self.schedule_choices) != set(other.schedule_choices):
            return False
        return all(
            self.schedule_choices[k] == other.schedule_choices[k] for k in self.schedule_choices
        )

    def __hash__(self) -> int:
        return hash(frozenset((k, v) for k, v in self.schedule_choices.items()))

    def __repr__(self) -> str:
        return f"SystemConfig(schedule_choices={dict(self.schedule_choices)})"


@dataclass
class SolutionCandidate:
    """The best complete solution known to a participant.

    :param participant_id: ID of the agent that last updated this candidate.
    :param schedules: 2-D array of shape ``(max_id, n_intervals)`` where row
                      ``participant_id - 1`` holds that participant's schedule
                      (1-indexed IDs → 0-indexed rows).
    :param perf: Cached performance value (``None`` if not yet evaluated).
    :param present: Frozen set of participant IDs whose schedule is included.
    """

    participant_id: int
    schedules: np.ndarray
    perf: float | None
    present: frozenset[int]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SolutionCandidate):
            return NotImplemented
        return (
            self.participant_id == other.participant_id
            and np.array_equal(self.schedules, other.schedules)
            and self.perf == other.perf
            and self.present == other.present
        )

    def __hash__(self) -> int:
        return hash((self.participant_id, self.schedules.tobytes(), self.perf, self.present))

    def __repr__(self) -> str:
        return (
            f"SolutionCandidate(participant_id={self.participant_id}, "
            f"perf={self.perf}, present={set(self.present)}, "
            f"schedules={self.schedules.tolist()})"
        )


@dataclass
class TargetParams:
    """Global target schedule and per-interval weights."""

    schedule: np.ndarray
    weights: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TargetParams):
            return NotImplemented
        return np.array_equal(self.schedule, other.schedule) and np.array_equal(
            self.weights, other.weights
        )

    def __hash__(self) -> int:
        return hash(
            (
                tuple(float(v) for v in self.schedule),
                tuple(float(v) for v in self.weights),
            )
        )

    def __repr__(self) -> str:
        return f"TargetParams(schedule={self.schedule.tolist()}, weights={self.weights.tolist()})"


@dataclass
class WorkingMemory(OptimizationMessage):
    """State shared between COHDA participants.

    :param target_params: Global optimisation target (set once, on first recv).
    :param system_config: Current view of all participants' schedule choices.
    :param solution_candidate: Best complete solution known to this agent.
    """

    target_params: TargetParams | None
    system_config: SystemConfig
    solution_candidate: SolutionCandidate | None


# ---------------------------------------------------------------------------
# Default performance function
# ---------------------------------------------------------------------------


def cohda_default_performance(
    cluster_schedule: np.ndarray,
    target_params: TargetParams,
) -> float:
    """Score a candidate by weighted distance from the target (higher = better).

    :param cluster_schedule: Array of shape ``(n_participants, n_intervals)``.
    :param target_params: Target schedule and weights.
    :returns: ``-sum(weights * abs(target - column_sums))``.
    """
    sum_cs = cluster_schedule.sum(axis=0)  # (n_intervals,)
    diff = np.abs(target_params.schedule - sum_cs)
    return -float(np.sum(diff * target_params.weights))


# ---------------------------------------------------------------------------
# Local decider hierarchy
# ---------------------------------------------------------------------------


class LocalDecider:
    """Abstract strategy for selecting a local schedule in the decide step."""

    def initial_schedule(self, memory: WorkingMemory) -> np.ndarray:
        raise NotImplementedError

    def decide(
        self,
        cohda_data: "COHDAAlgorithmData",
        sysconfig: SystemConfig,
        candidate: SolutionCandidate,
    ) -> tuple[SystemConfig, SolutionCandidate]:
        """Pick this participant's schedule; return ``(sysconfig, candidate)``."""
        raise NotImplementedError


class DefaultLocalDecider(LocalDecider):
    """Enumerate all feasible schedules and pick the globally best one.

    :param schedule_provider: ``(WorkingMemory) -> list[array-like]`` — returns
                              all feasible schedules for this participant.
    :param is_local_acceptable: Predicate filtering individual schedules.
    """

    def __init__(
        self,
        schedule_provider: Callable[[WorkingMemory], list],
        is_local_acceptable: Callable[[np.ndarray], bool] = lambda _: True,
    ) -> None:
        self.schedule_provider = schedule_provider
        self.is_local_acceptable = is_local_acceptable

    def initial_schedule(self, memory: WorkingMemory) -> np.ndarray:
        return np.array(self.schedule_provider(memory)[0], dtype=float)

    def decide(
        self,
        cohda_data: "COHDAAlgorithmData",
        sysconfig: SystemConfig,
        candidate: SolutionCandidate,
    ) -> tuple[SystemConfig, SolutionCandidate]:
        """Evaluate all feasible schedules; keep the best-performing candidate."""
        possible = [np.array(s, dtype=float) for s in self.schedule_provider(cohda_data.memory)]
        current_best = candidate
        if current_best.perf is None:
            current_best = _evaluated(
                current_best, cohda_data.performance_function, cohda_data.memory.target_params
            )

        current_best_schedule = current_best.schedules[cohda_data.participant_id - 1].copy()

        for schedule in possible:
            if self.is_local_acceptable(schedule):
                new_cand = create_from_updated_sysconf(
                    cohda_data.participant_id, sysconfig, schedule
                )
                new_cand = _evaluated(
                    new_cand, cohda_data.performance_function, cohda_data.memory.target_params
                )
                if new_cand.perf > current_best.perf:  # type: ignore[operator]
                    current_best = new_cand
                    current_best_schedule = schedule

        publish_own_schedule(cohda_data, sysconfig, current_best_schedule)
        return sysconfig, current_best


# ---------------------------------------------------------------------------
# COHDAAlgorithmData
# ---------------------------------------------------------------------------


class COHDAAlgorithmData(DistributedAlgorithm):
    """Per-participant COHDA state machine.

    :param participant_id: 1-indexed unique participant ID.
    :param decider: Local schedule selection strategy.
    :param performance_function: Scores a full solution matrix.
    """

    def __init__(
        self,
        participant_id: int,
        decider: LocalDecider,
        performance_function: Callable = cohda_default_performance,
    ) -> None:
        self.participant_id = participant_id
        self.counter: int = 0
        self.memory: WorkingMemory = WorkingMemory(
            target_params=None,
            system_config=SystemConfig(),
            solution_candidate=None,
        )
        self.performance_function = performance_function
        self.decider = decider

    async def on_exchange_message(
        self,
        carrier: "Carrier",
        message_data: WorkingMemory,
        meta: Any,
    ) -> None:
        """Run one perceive → decide → act cycle and forward updates to neighbours.

        Skipped entirely if nothing changed during perception.

        .. note::
           The :class:`WorkingMemory` is forwarded **by reference** — an
           in-process carrier like ``SimpleCarrier`` delivers the live object, so
           receivers may observe later in-place mutations (harmless under
           single-threaded asyncio, since counters only grow, but different from
           a serialising network transport, which delivers immutable snapshots).
        """
        old_sysconf = self.memory.system_config
        old_candidate = self.memory.solution_candidate

        sysconf, candidate = perceive(self, [message_data])

        if sysconf != old_sysconf or candidate != old_candidate:
            sysconf, candidate = decide(self, sysconf, candidate)
            wm = act(self, sysconf, candidate)
            for other in carrier.others():
                carrier.send_to_other(wm, other)


# ---------------------------------------------------------------------------
# Core algorithmic functions
# ---------------------------------------------------------------------------


def merge_sysconfigs(
    sysconfig_i: SystemConfig,
    sysconfig_j: SystemConfig,
) -> SystemConfig:
    """Merge two system configs, keeping the higher-counter entry per agent.

    Returns *sysconfig_i* unchanged if it already dominates *sysconfig_j*.
    """
    choices_i = sysconfig_i.schedule_choices
    choices_j = sysconfig_j.schedule_choices
    all_ids = sorted(set(choices_i) | set(choices_j))

    new_choices: dict[int, ScheduleSelection] = {}
    modified = False
    for aid in all_ids:
        if aid in choices_i and (
            aid not in choices_j or choices_i[aid].counter >= choices_j[aid].counter
        ):
            new_choices[aid] = choices_i[aid]
        else:
            new_choices[aid] = choices_j[aid]
            modified = True

    return SystemConfig(new_choices) if modified else sysconfig_i


def merge_candidates(
    candidate_i: SolutionCandidate,
    candidate_j: SolutionCandidate | None,
    participant_id: int,
    perf_func: Callable,
    target_params: TargetParams | None,
) -> SolutionCandidate:
    """Merge two solution candidates using the COHDA dominance rules.

    1. If ``K_i ⊂ K_j`` (proper subset) → use *j* (more complete).
    2. If ``K_i == K_j`` → compare performance; break ties by lower agent ID.
    3. If ``K_j`` has IDs not in ``K_i`` → build a merged candidate.
    4. Otherwise keep *i*.
    """
    if candidate_j is None:
        return candidate_i

    keyset_i = candidate_i.present
    keyset_j = candidate_j.present
    candidate = candidate_i  # default

    if keyset_i < keyset_j:
        candidate = candidate_j
    elif keyset_i == keyset_j:
        # Lazy-evaluate performance
        if candidate_i.perf is None:
            candidate_i = _evaluated(candidate_i, perf_func, target_params)
        if candidate_j.perf is None:
            candidate_j = _evaluated(candidate_j, perf_func, target_params)

        if candidate_j.perf > candidate_i.perf:  # type: ignore[operator]
            candidate = candidate_j
        elif candidate_j.perf == candidate_i.perf:
            if candidate_j.participant_id < candidate_i.participant_id:
                candidate = candidate_j
    elif keyset_j - keyset_i:
        # j contributes new participants — build a merged candidate
        both = keyset_i | keyset_j
        n_rows = max(both)
        n_cols = candidate_i.schedules.shape[1]
        base = np.zeros((n_rows, n_cols))
        for k in both:
            src = candidate_i if k in keyset_i else candidate_j
            base[k - 1] = src.schedules[k - 1]
        candidate = SolutionCandidate(
            participant_id=participant_id,
            schedules=base,
            perf=None,
            present=frozenset(both),
        )

    return candidate


def _evaluated(
    candidate: SolutionCandidate,
    perf_func: Callable,
    target_params: TargetParams | None,
) -> SolutionCandidate:
    """Return a copy of *candidate* with :attr:`~SolutionCandidate.perf` set."""
    return SolutionCandidate(
        participant_id=candidate.participant_id,
        schedules=candidate.schedules,
        perf=perf_func(candidate.schedules, target_params),
        present=candidate.present,
    )


def perceive(
    cohda_data: COHDAAlgorithmData,
    working_memories: list[WorkingMemory],
) -> tuple[SystemConfig, SolutionCandidate]:
    """Incorporate received working memories into local state.

    Initialises the local schedule and candidate the first time they are
    needed, then merges each incoming memory.

    :returns: Updated ``(system_config, solution_candidate)`` pair.
    """
    current_sysconfig: SystemConfig | None = None
    current_candidate: SolutionCandidate | None = None
    own_id = cohda_data.participant_id
    own_memory = cohda_data.memory
    initial: np.ndarray | None = None  # computed at most once per call

    for new_wm in working_memories:
        if own_memory.target_params is None:
            own_memory.target_params = new_wm.target_params

        if current_sysconfig is None:
            if own_id not in own_memory.system_config.schedule_choices:
                initial = cohda_data.decider.initial_schedule(own_memory)
                own_memory.system_config.schedule_choices[own_id] = ScheduleSelection(
                    schedule=initial,
                    counter=cohda_data.counter + 1,
                )
                cohda_data.counter += 1
                current_sysconfig = SystemConfig(dict(own_memory.system_config.schedule_choices))
            else:
                current_sysconfig = own_memory.system_config

        if current_candidate is None:
            if (
                own_memory.solution_candidate is None
                or own_id not in own_memory.solution_candidate.present
            ):
                if initial is None:
                    initial = cohda_data.decider.initial_schedule(own_memory)
                base = np.zeros((own_id, len(initial)))
                base[own_id - 1] = initial
                own_memory.solution_candidate = SolutionCandidate(
                    participant_id=own_id,
                    schedules=base,
                    perf=None,
                    present=frozenset([own_id]),
                )
            current_candidate = own_memory.solution_candidate

        current_sysconfig = merge_sysconfigs(current_sysconfig, new_wm.system_config)
        current_candidate = merge_candidates(
            current_candidate,
            new_wm.solution_candidate,
            own_id,
            cohda_data.performance_function,
            own_memory.target_params,
        )

    return current_sysconfig, current_candidate


def create_from_updated_sysconf(
    participant_id: int,
    sysconfig: SystemConfig,
    new_schedule: np.ndarray,
) -> SolutionCandidate:
    """Build a fresh :class:`SolutionCandidate` from *sysconfig* + *new_schedule*."""
    max_id = max(sysconfig.schedule_choices)
    n_intervals = len(new_schedule)
    base = np.zeros((max_id, n_intervals))
    for pid, sel in sysconfig.schedule_choices.items():
        base[pid - 1] = np.array(sel.schedule, dtype=float)
    base[participant_id - 1] = new_schedule
    return SolutionCandidate(
        participant_id=participant_id,
        schedules=base,
        perf=None,
        present=frozenset(sysconfig.schedule_choices),
    )


def decide(
    cohda_data: COHDAAlgorithmData,
    sysconfig: SystemConfig,
    candidate: SolutionCandidate,
) -> tuple[SystemConfig, SolutionCandidate]:
    """Run the participant's own :meth:`LocalDecider.decide`."""
    return cohda_data.decider.decide(cohda_data, sysconfig, candidate)


def publish_own_schedule(
    cohda_data: COHDAAlgorithmData,
    sysconfig: SystemConfig,
    schedule: np.ndarray,
) -> None:
    """Write *schedule* to *sysconfig*, bumping the counter only if it changed."""
    existing = sysconfig.schedule_choices.get(cohda_data.participant_id)
    if existing is None or not np.array_equal(schedule, existing.schedule):
        cohda_data.counter += 1
        sysconfig.schedule_choices[cohda_data.participant_id] = ScheduleSelection(
            schedule=schedule,
            counter=cohda_data.counter,
        )


def act(
    cohda_data: COHDAAlgorithmData,
    new_sysconfig: SystemConfig,
    new_candidate: SolutionCandidate,
) -> WorkingMemory:
    """Commit *new_sysconfig* and *new_candidate* to memory; return memory."""
    cohda_data.memory.system_config = new_sysconfig
    cohda_data.memory.solution_candidate = new_candidate
    return cohda_data.memory


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_cohda_start_message(
    target_schedule: list[float] | np.ndarray,
    weights: list[float] | np.ndarray | None = None,
) -> WorkingMemory:
    """Create the initial :class:`WorkingMemory` that kicks off a COHDA run.

    :param target_schedule: Global target vector.
    :param weights: Per-interval weights (default: ones).
    """
    target = np.array(target_schedule, dtype=float)
    w = np.ones(len(target)) if weights is None else np.array(weights, dtype=float)
    return WorkingMemory(
        target_params=TargetParams(schedule=target, weights=w),
        system_config=SystemConfig(),
        solution_candidate=None,
    )


def create_cohda_participant(
    participant_id: int,
    schedule_set: list | Callable,
    performance_function: Callable = cohda_default_performance,
) -> COHDAAlgorithmData:
    """Create a COHDA participant with a :class:`DefaultLocalDecider`.

    :param participant_id: 1-indexed unique ID.
    :param schedule_set: A list of feasible schedules **or** a callable
                         ``(WorkingMemory) -> list[array-like]``.
    :param performance_function: Scoring function (default performance).
    """
    if isinstance(schedule_set, list):
        _frozen = [[float(x) for x in s] for s in schedule_set]

        def provider(_: Any, _frozen: list = _frozen) -> list:
            return [np.array(s) for s in _frozen]
    else:
        provider = schedule_set

    decider = DefaultLocalDecider(schedule_provider=provider)
    return create_cohda_participant_with_decider(participant_id, decider, performance_function)


#: Create a COHDA participant with an explicit local decider (same signature
#: as :class:`COHDAAlgorithmData`).
create_cohda_participant_with_decider = COHDAAlgorithmData
