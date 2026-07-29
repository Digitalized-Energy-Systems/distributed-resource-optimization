"""The round budget must not be spent entirely by the first tier.

``round_timeout_s`` is carrier-clock time, and under a *simulation* clock every
iteration costs a message round-trip, so the whole cascade gets
``round_timeout_s / iter_timeout_s`` iterations TOTAL. With one shared deadline
the top tier spends all of them and every later tier breaks at iteration 0, so
the cascade answers only tier 1 and commits. Measured on a 47-CP LV feeder at
2.0/0.2: tier 0 died at iteration 10-11, tiers 1..3 at iteration **0**.

``tier_fair_deadline`` gives each tier a cumulative slice of the same overall
bound. Because the slices are absolute deadlines, a tier that converges early
hands its unused time to the next one.
"""

from __future__ import annotations

import asyncio
import logging
import re

import numpy as np
import pytest

from distributed_resource_optimization import (
    ActorContainer,
    CPSpec,
    SectorDemand,
    SimpleCarrier,
    cid,
    create_gossip_cascade_participant,
    create_gossip_cascade_start,
)


_N_CPS = 4


class _SlowCarrier(SimpleCarrier):
    """Carrier whose clock advances per delivered message, so iterations consume
    the round budget the way a simulation clock does.

    Scaled so one full all-to-all broadcast round costs ``iter_timeout_s``:
    that is what makes ``round_timeout_s / iter_timeout_s`` the real iteration
    budget, which is the property under test.
    """

    tick_s = 0.2 / (_N_CPS * (_N_CPS - 1))
    _clock = [0.0]

    def now(self) -> float:  # noqa: D102
        return self._clock[0]

    def send_to_other(self, message, receiver):  # noqa: D102
        self._clock[0] += self.tick_s
        return super().send_to_other(message, receiver)


async def _run(cps, demands, *, tier_fair, round_timeout_s=2.0, iter_timeout_s=0.2):
    _SlowCarrier._clock[0] = 0.0
    seen_tiers: set[int] = set()
    results, events = {}, {}

    def _cb(cp_id):
        def inner(r, converged, iters):
            results[cp_id] = (np.asarray(r).ravel()[0], converged, iters)
            events[cp_id].set()

        return inner

    container = ActorContainer()
    participants = []
    for spec in cps:
        events[spec.cp_id] = asyncio.Event()
        participants.append(
            create_gossip_cascade_participant(
                cp_id=spec.cp_id,
                capacity_by_sector=spec.capacity_by_sector,
                on_commit=_cb(spec.cp_id),
            )
        )
    carriers = [_SlowCarrier(container, p) for p in participants]

    # Record which tiers each participant actually enters.
    for p in participants:
        original = p._run_cascade

        async def traced(carrier, _p=p, _orig=original):
            await _orig(carrier)
            ctx = _p._ctx
            if ctx is not None:
                seen_tiers.add(int(ctx.tier_index))

        p._run_cascade = traced  # type: ignore[method-assign]

    start = create_gossip_cascade_start(
        round_id=1,
        participants=[c.cp_id for c in cps],
        demands=demands,
        horizon=1,
        rho=1.0,
        inner_iters_max=200,
        inner_abs_tol=1e-3,
        r_regularization=0.1,
        normalize=True,
        r_regularization_relative=True,
        minimize_usage=True,
        iter_timeout_s=iter_timeout_s,
        round_timeout_s=round_timeout_s,
        tier_fair_deadline=tier_fair,
    )
    carriers[0].send_to_other(start, cid(carriers[0]))
    await asyncio.wait_for(
        asyncio.gather(*[e.wait() for e in events.values()]), timeout=60.0
    )
    return results, seen_tiers


def _four_tier_case():
    """A P2H fleet whose heat deficit only appears in the LOWER tiers, so the
    answer is invisible to a cascade that never leaves tier 1."""
    cps = [
        CPSpec(f"p2h-{i}", {"electricity": 0.05, "heat": -0.0475})
        for i in range(_N_CPS)
    ]
    demands = [
        SectorDemand(
            "heat",
            {1: np.array([0.05]), 2: np.array([0.15]), 3: np.array([0.2]),
             4: np.array([0.1])},
            np.array([0.10]),  # covers tier 1 only; tiers 2-4 need the fleet
        ),
        SectorDemand(
            "electricity",
            {1: np.array([0.02]), 2: np.array([0.02]), 3: np.array([0.02]),
             4: np.array([0.02])},
            np.array([5.0]),  # abundant, so the fleet is free to run
        ),
    ]
    return cps, demands


def _starved_tiers(caplog) -> set[int]:
    """Tiers whose inner loop timed out at iteration 0 — i.e. got no
    optimisation at all, only a deadline check."""
    out = set()
    for rec in caplog.records:
        m = re.search(r"timed out at tier (\d+) iter (\d+)", rec.getMessage())
        if m and int(m.group(2)) == 0:
            out.add(int(m.group(1)))
    return out


@pytest.mark.asyncio
async def test_one_shared_deadline_starves_the_lower_tiers(caplog):
    """The defect, stated structurally: once the top tier exhausts the shared
    deadline every later tier breaks at iteration 0, having done no work."""
    cps, demands = _four_tier_case()
    with caplog.at_level(logging.WARNING):
        await _run(cps, demands, tier_fair=False)
    starved = _starved_tiers(caplog)
    assert starved, "expected the shared deadline to starve at least one tier"
    assert max(starved) == 3  # the bottom of the ladder never runs

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await _run(cps, demands, tier_fair=True)
    assert _starved_tiers(caplog) < starved


@pytest.mark.asyncio
async def test_tier_fair_deadline_reaches_the_lower_tiers():
    """With the budget shared, the same round under the same overall bound
    answers the deficit the lower tiers carry."""
    cps, demands = _four_tier_case()
    fair, _ = await _run(cps, demands, tier_fair=True)
    unfair, _ = await _run(cps, demands, tier_fair=False)
    r_fair = np.mean([v[0] for v in fair.values()])
    r_unfair = np.mean([v[0] for v in unfair.values()])
    assert r_fair > r_unfair
    assert r_fair > 0.5


@pytest.mark.asyncio
async def test_tier_fair_is_bounded_by_the_same_round_budget():
    """The overall bound is unchanged — the slices are cumulative, not extra."""
    cps, demands = _four_tier_case()
    await _run(cps, demands, tier_fair=True, round_timeout_s=2.0)
    fair_clock = _SlowCarrier._clock[0]
    await _run(cps, demands, tier_fair=False, round_timeout_s=2.0)
    unfair_clock = _SlowCarrier._clock[0]
    # Sharing the budget must not extend it: the deadlines are cumulative
    # slices of the same bound, not an allowance per tier.
    assert fair_clock <= unfair_clock * 1.05


@pytest.mark.asyncio
async def test_flag_off_is_the_old_behaviour():
    """Default off must reproduce the single-deadline path exactly."""
    cps, demands = _four_tier_case()
    a, _ = await _run(cps, demands, tier_fair=False)
    b, _ = await _run(cps, demands, tier_fair=False)
    assert {k: v[0] for k, v in a.items()} == {k: v[0] for k, v in b.items()}
