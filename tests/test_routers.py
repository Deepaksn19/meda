"""Tests for the comparison routers of Sec. V-B / VI-C (baseline and formal)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from meda_routing.core.actions import Action, max_reliable_step
from meda_routing.core.biochip import DegradationConfig, MEDABiochip
from meda_routing.core.dynamics import plan_move
from meda_routing.core.geometry import Droplet, Rect
from meda_routing.core.jobs import JobSampler, JobSamplerConfig, RoutingJob, hazard_bounds
from meda_routing.core.movement import action_flags, footprint_mask, move_distribution
from meda_routing.routers.base import Router, RoutingState, run_job
from meda_routing.routers.baseline import (
    ShortestPathRouter,
    parse_step_mode,
    shortest_path_action,
)
from meda_routing.routers.formal import (
    FormalRouter,
    _build_transitions,
    default_horizon,
    estimate_degradation,
    synthesize,
)


# --------------------------------------------------------------------- helpers
def ideal_chip(width: int, height: int, seed: int = 0) -> MEDABiochip:
    """Chip whose MCs never wear out (``tau = 1`` => ``D = 1``): deterministic moves."""
    chip = MEDABiochip(width, height, rng=np.random.default_rng(seed))
    chip.tau[:] = 1.0
    return chip


def set_level(chip: MEDABiochip, cells, level: int) -> None:
    """Make the sensed health of ``cells`` equal ``level`` (0 = visible fault)."""
    for x, y in cells:
        if level == 0:
            chip.faults[x, y] = True
        else:
            # D = tau ** (n / c) = tau with n = c; floor(4 * tau) = level.
            chip.tau[x, y] = (level + 0.5) / chip.health_levels
            chip.actuations[x, y] = int(chip.c[x, y])
            chip.c[x, y] = chip.actuations[x, y]
    assert all(chip.health()[x, y] == level for x, y in cells)


def chebyshev(a: Rect, b: Rect) -> int:
    return max(abs(a.xa - b.xa), abs(a.ya - b.ya))


def wall_scenario():
    """20 x 12 chip, 2 x 2 droplet routed east; column x=8 is faulty in rows 4..9.

    The hazard zone spans rows 2..9, so the only gap is rows 2-3; the optimal
    detour still takes max(|dx|, |dy|) = 13 single-step cycles.
    """
    chip = ideal_chip(20, 12)
    start, goal = Droplet.at(2, 5, 2, 2), Droplet.at(15, 5, 2, 2)
    job = RoutingJob(start, goal, hazard_bounds(start, goal, 20, 12, margin=3))
    wall = [(8, y) for y in range(4, 10)]
    return chip, job, wall


def corridor_scenario(level: int, cells=((8, 5), (8, 6))):
    """2 x 2 droplet in a 2-MC-high corridor; ``cells`` on its path get ``level``."""
    chip = ideal_chip(20, 12)
    start, goal = Droplet.at(2, 5, 2, 2), Droplet.at(15, 5, 2, 2)
    job = RoutingJob(start, goal, Rect(0, 5, 19, 6))
    set_level(chip, cells, level)
    return chip, job


def sampled_jobs(width, height, n, seed, sizes=((2, 2), (3, 3), (4, 4), (3, 4))):
    sampler = JobSampler(
        width, height, JobSamplerConfig(droplet_sizes=list(sizes)), rng=np.random.default_rng(seed)
    )
    return [sampler.sample() for _ in range(n)]


class SensorOnlyChip:
    """Exposes only what a real controller can sense; any other access fails."""

    def __init__(self, chip: MEDABiochip) -> None:
        self._chip = chip

    def health(self) -> np.ndarray:
        return self._chip.health()

    @property
    def health_levels(self) -> int:
        return self._chip.health_levels


# -------------------------------------------------------------------- baseline
def test_parse_step_mode():
    assert parse_step_mode("single") == (False, 1)
    assert parse_step_mode("double") == (False, 2)
    assert parse_step_mode("adaptive") == (True, 1)
    with pytest.raises(ValueError):
        parse_step_mode("triple")
    with pytest.raises(ValueError):
        FormalRouter(step_mode="bogus")


@pytest.mark.parametrize(
    "offset, expected",
    [
        ((3, 0), Action.E),
        ((-2, 0), Action.W),
        ((0, 4), Action.N),
        ((0, -1), Action.S),
        ((5, 2), Action.NE),
        ((-1, 7), Action.NW),
        ((2, -2), Action.SE),
        ((-3, -9), Action.SW),
    ],
)
def test_shortest_path_action(offset, expected):
    droplet = Droplet.at(10, 10, 3, 3)
    goal = droplet.shift(*offset)
    assert shortest_path_action(droplet, goal) == expected
    assert ShortestPathRouter().name == "baseline"


@pytest.mark.parametrize("step_mode", ["single", "double", "adaptive"])
def test_baseline_minimum_cycles_on_ideal_chip(step_mode):
    router = ShortestPathRouter(step_mode)
    for i, job in enumerate(sampled_jobs(30, 20, 25, seed=1)):
        chip = ideal_chip(30, 20, seed=i)
        res = run_job(router, job, chip, np.random.default_rng(i), record_path=True)
        dx, dy = abs(job.goal.xa - job.start.xa), abs(job.goal.ya - job.start.ya)
        if step_mode == "single":
            expected = max(dx, dy)
        elif step_mode == "double":
            expected = max(math.ceil(dx / 2), math.ceil(dy / 2))
        else:
            lx, ly = max_reliable_step(job.start)
            expected = max(math.ceil(dx / lx), math.ceil(dy / ly))
        assert res.success and res.invalid_actions == 0
        assert res.cycles == expected
        # Never overshoots: the offsets to the goal shrink monotonically.
        offs = [(abs(job.goal.xa - d.xa), abs(job.goal.ya - d.ya)) for d in res.path]
        for (ax, ay), (bx, by) in zip(offs, offs[1:]):
            assert bx <= ax and by <= ay


def test_baseline_never_invalid_on_degraded_chips():
    cfg = DegradationConfig(initial_actuations="uniform", max_initial_actuations=600)

    class Checked(ShortestPathRouter):
        def act(self, state: RoutingState) -> Action:
            action = super().act(state)
            plan = plan_move(state.droplet, state.goal, state.hazard, action, False, 1)
            assert plan.valid and plan.target != state.droplet
            return action

    router = Checked()
    for i, job in enumerate(sampled_jobs(24, 16, 20, seed=2)):
        rng = np.random.default_rng(100 + i)
        chip = MEDABiochip(24, 16, cfg, rng=rng)
        chip.reset()
        chip.inject_faults(0.2, protect=(job.start, job.goal))
        res = run_job(router, job, chip, rng)
        assert res.invalid_actions == 0


def test_baseline_gets_stuck_at_wall():
    chip, job, wall = wall_scenario()
    chip.set_faults(wall)
    res = run_job(ShortestPathRouter(), job, chip, np.random.default_rng(0), record_path=True)
    assert not res.success
    # Fig. 14(b)-(c): the droplet stops in front of the degraded MCs.
    assert res.path[-1] == Droplet.at(6, 5, 2, 2)
    assert res.path[4:] == [Droplet.at(6, 5, 2, 2)] * (len(res.path) - 4)


# ---------------------------------------------------------------------- formal
def test_default_horizon_and_estimate():
    a, b = Droplet.at(0, 0, 2, 2), Droplet.at(5, 2, 2, 2)
    assert default_horizon(a, b) == math.ceil(1.5 * 7) + 1 == 12
    assert default_horizon(a, a) == 1
    est = estimate_degradation(np.array([[0, 1], [2, 3]]), 4)
    np.testing.assert_allclose(est, [[0.0, 1 / 3], [2 / 3, 1.0]])


def test_formal_equals_shortest_path_on_ideal_chip():
    for i, job in enumerate(sampled_jobs(30, 20, 12, seed=3)):
        formal = FormalRouter()
        rng_f, rng_b = np.random.default_rng(i), np.random.default_rng(i)
        res_f = run_job(formal, job, ideal_chip(30, 20), rng_f, record_path=True)
        res_b = run_job(ShortestPathRouter(), job, ideal_chip(30, 20), rng_b, record_path=True)
        assert formal.success_probability == pytest.approx(1.0)
        assert res_f.success and res_f.invalid_actions == 0
        assert res_f.cycles == chebyshev(job.start, job.goal)
        assert res_f.path == res_b.path


@pytest.mark.parametrize("step_mode", ["double", "adaptive"])
def test_formal_multi_step_modes_reach_goal_in_minimum_cycles(step_mode):
    for i, job in enumerate(sampled_jobs(30, 20, 6, seed=4)):
        rng_f, rng_b = np.random.default_rng(i), np.random.default_rng(i)
        res_f = run_job(FormalRouter(step_mode), job, ideal_chip(30, 20), rng_f)
        res_b = run_job(ShortestPathRouter(step_mode), job, ideal_chip(30, 20), rng_b)
        assert res_f.success and res_b.success
        assert res_f.cycles == res_b.cycles
        assert res_f.invalid_actions == 0


def test_formal_reaches_goal_on_realistic_healthy_chip():
    rng = np.random.default_rng(5)
    chip = MEDABiochip(30, 20, rng=rng)
    chip.reset()
    for job in sampled_jobs(30, 20, 8, seed=5):
        formal = FormalRouter()
        res = run_job(formal, job, chip, rng)
        assert res.success
        assert formal.success_probability == pytest.approx(1.0)


def test_formal_routes_around_wall():
    chip, job, wall = wall_scenario()
    chip.set_faults(wall)
    formal = FormalRouter()
    res = run_job(formal, job, chip, np.random.default_rng(0), record_path=True)
    assert formal.strategy.horizon == 21
    assert formal.success_probability == pytest.approx(1.0)
    # Optimal detour without stalling in front of the wall.
    assert res.success and res.cycles == 13 and res.invalid_actions == 0
    faulty = np.zeros(chip.shape, dtype=bool)
    faulty[tuple(np.array(wall).T)] = True
    for d in res.path:
        sx, sy = d.slices()
        assert not faulty[sx, sy].any()


@pytest.mark.parametrize("level", [3, 2, 1, 0])
def test_formal_success_probability_exact(level):
    # 13 moves east in K = 21 cycles; only the move whose frontier is column 8
    # is uncertain (p = level / 3), so it can be retried 21 - 12 = 9 times.
    chip, job = corridor_scenario(level)
    formal = FormalRouter()
    formal.reset(job, chip)
    p = level / 3
    assert formal.strategy.horizon == 21
    assert formal.success_probability == pytest.approx(1 - (1 - p) ** 9, abs=1e-12)


def test_formal_success_probability_decreases_with_degradation():
    probs = []
    for level in (3, 2, 1, 0):
        # One of the two frontier MCs of the corridor crossing at x = 8.
        chip, job = corridor_scenario(level, cells=((8, 5),))
        formal = FormalRouter(horizon=14)  # one spare cycle
        formal.reset(job, chip)
        probs.append(formal.success_probability)
    assert probs[0] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(probs, probs[1:]))
    assert probs[-1] == pytest.approx(0.75)  # p = (0 + 1) / 2, two attempts


def test_formal_prefers_reliable_detour_over_risky_shortcut():
    # A partially degraded wall (p = 1/2 through the shortcut) with plenty of
    # time: the max-probability strategy detours through the healthy gap.
    chip, job, wall = wall_scenario()
    set_level(chip, [(8, y) for y in range(4, 10) if y != 5], 0)
    formal = FormalRouter()
    formal.reset(job, chip)
    assert formal.success_probability == pytest.approx(1.0)
    res = run_job(formal, job, chip, np.random.default_rng(0), record_path=True)
    assert res.success
    assert min(d.ya for d in res.path) == 2  # went through the gap at rows 2-3


def test_formal_uses_sensor_data_only():
    chip, job, wall = wall_scenario()
    clean = FormalRouter()
    clean.reset(job, chip)
    hidden_chip = chip.copy()
    hidden_chip.set_faults(wall, hidden=True)
    hidden = FormalRouter()
    hidden.reset(job, SensorOnlyChip(hidden_chip))  # would fail on any other access
    assert hidden.success_probability == clean.success_probability == pytest.approx(1.0)
    np.testing.assert_array_equal(hidden.strategy.policy, clean.strategy.policy)
    np.testing.assert_array_equal(hidden.strategy.values, clean.strategy.values)
    # The hidden wall blocks the (health-agnostic-looking) straight route.
    res = run_job(hidden, job, hidden_chip, np.random.default_rng(0))
    assert not res.success
    # Visible faults, by contrast, change the strategy.
    visible_chip = chip.copy()
    visible_chip.set_faults(wall)
    visible = FormalRouter()
    visible.reset(job, SensorOnlyChip(visible_chip))
    assert not np.array_equal(visible.strategy.policy, clean.strategy.policy)


def test_formal_value_matches_monte_carlo():
    rng = np.random.default_rng(7)
    chip = ideal_chip(12, 8)
    chip.c[:] = 1e6  # negligible wear during the runs
    start, goal = Droplet.at(0, 0, 2, 2), Droplet.at(9, 5, 2, 2)
    job = RoutingJob(start, goal, Rect(0, 0, 11, 7))
    protected = np.zeros(chip.shape, dtype=bool)
    for d in (start, goal):
        protected[d.slices()] = True
    draws = rng.random(chip.shape)
    for x, y in zip(*np.nonzero((draws < 0.6) & ~protected)):
        chip.tau[x, y] = 1 / 3 if draws[x, y] < 0.3 else 2 / 3
        chip.actuations[x, y] = int(chip.c[x, y])
    chip.set_faults([(x, y) for x, y in zip(*np.nonzero((draws > 0.95) & ~protected))])
    # With tau in {1/3, 2/3, 1} the estimate H / 3 equals the true D.
    est = estimate_degradation(chip.health(), chip.health_levels)
    np.testing.assert_allclose(est, chip.effective_degradation(), atol=1e-4)

    horizon = chebyshev(start, goal) + 2
    formal = FormalRouter(horizon=horizon, cache_size=1)
    formal.reset(job, chip)
    p = formal.success_probability
    assert 0.3 < p < 0.95
    n = 1500
    wins = sum(run_job(formal, job, chip.copy(), rng, k_max=horizon).success for _ in range(n))
    assert formal.last_cache_hit
    sigma = math.sqrt(p * (1 - p) / n)
    assert abs(wins / n - p) < 4 * sigma + 0.005


def test_formal_values_are_monotone_in_horizon():
    chip, job, wall = wall_scenario()
    set_level(chip, wall[:3], 1)
    set_level(chip, wall[3:], 2)
    strategy = synthesize(
        job.goal, job.hazard, estimate_degradation(chip.health(), chip.health_levels), 30
    )
    v = strategy.values
    assert np.all(v >= 0) and np.all(v <= 1 + 1e-12)
    assert np.all(np.diff(v, axis=0) >= -1e-9)
    assert v[:, strategy.state_id(job.goal)].min() == 1.0
    # V_1 is non-zero only one move away from the goal.
    near = [s for s in range(strategy.num_states) if v[1, s] > 0]
    assert len(near) > 1
    assert all(chebyshev(strategy.droplet(s), job.goal) <= 1 for s in near)


@pytest.mark.parametrize("step_mode", ["single", "double"])
def test_formal_values_are_exact_and_optimal(step_mode):
    """Independent check of the vectorized synthesis on a small model.

    ``values`` must be the exact value of the stored policy table (policy
    evaluation with :func:`move_distribution`) and satisfy the Bellman
    optimality equation up to the tie tolerance.
    """
    chip = ideal_chip(10, 7)
    set_level(chip, [(4, 1), (4, 2), (4, 3), (5, 4)], 0)
    set_level(chip, [(4, 4), (6, 2), (2, 5)], 1)
    set_level(chip, [(4, 5), (7, 3), (3, 1)], 2)
    est = estimate_degradation(chip.health(), chip.health_levels)
    hazard, goal = Rect(0, 0, 9, 6), Droplet.at(7, 2, 2, 2)
    adaptive, fixed = parse_step_mode(step_mode)
    strategy = synthesize(goal, hazard, est, 9, adaptive, fixed)
    states = [strategy.droplet(s) for s in range(strategy.num_states)]
    dist = {}
    for d in states:
        for a in Action:
            plan = plan_move(d, goal, hazard, a, adaptive, fixed)
            if plan.valid and plan.target != d:
                pattern = footprint_mask(est.shape, plan.target)
                dist[d, a] = move_distribution(est, pattern, d, hazard, action_flags(a))
            else:
                dist[d, a] = {d: 1.0}
    v_pi = {d: float(d == goal) for d in states}
    fractional = 0
    for t in range(1, strategy.horizon + 1):
        prev = strategy.values[t - 1]
        new = {}
        for d in states:
            if d == goal:
                new[d] = 1.0
                continue
            q = [sum(p * prev[strategy.state_id(n)] for n, p in dist[d, a].items()) for a in Action]
            assert strategy.value(d, t) >= max(q) - 1e-8  # Bellman optimality
            a = strategy.action(d, t)
            new[d] = sum(p * v_pi[n] for n, p in dist[d, a].items())
        v_pi = new
        for d in states:
            assert strategy.value(d, t) == pytest.approx(v_pi[d], abs=1e-9)
            fractional += 0.0 < v_pi[d] < 1.0
    assert fractional > 0  # the model is genuinely stochastic


def test_formal_lost_bound_uses_full_horizon_action():
    # Where the goal is out of reach within t cycles, every action is worth 0
    # and pi_t must fall back to pi_K instead of the health-agnostic nominal
    # order, which would push into the visible wall until the horizon ran out.
    chip, job, wall = wall_scenario()
    chip.set_faults(wall)
    formal = FormalRouter()
    formal.reset(job, chip)
    strategy = formal.strategy
    k_full = strategy.horizon
    lost = strategy.values[1:] == 0.0
    assert lost.any() and not lost.all()
    expected = np.broadcast_to(strategy.policy[k_full], lost.shape)
    np.testing.assert_array_equal(strategy.policy[1:][lost], expected[lost])
    front = Droplet.at(6, 5, 2, 2)  # facing the wall, 10 cycles from the goal
    assert strategy.value(front, 9) == 0.0 and strategy.value(front, 10) > 0.0
    for r in range(1, 10):
        assert strategy.action(front, r) == strategy.action(front, k_full) != Action.E

    class LateFormal(FormalRouter):
        """Acts as if the job had started 12 cycles late (9 of 21 cycles left)."""

        def act(self, state: RoutingState) -> Action:
            return super().act(dataclasses.replace(state, k=state.k + 12))

    late = LateFormal()
    res = run_job(late, job, chip, np.random.default_rng(0), record_path=True)
    assert late.strategy.value(job.start, k_full - 12) == 0.0
    assert res.success and res.cycles == 13  # no stalling in front of the wall
    assert Droplet.at(6, 5, 2, 2) not in res.path


def test_formal_parameter_validation():
    for bad in (0.0, -1e-9, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            FormalRouter(tie_tolerance=bad)
    with pytest.raises(TypeError):
        FormalRouter(horizon=10.5)  # not silently truncated
    with pytest.raises(ValueError):
        FormalRouter(horizon=0)
    assert FormalRouter(horizon=np.int64(12)).horizon == 12
    goal, hazard = Droplet.at(1, 1, 2, 2), Rect(0, 0, 9, 5)
    with pytest.raises(ValueError):
        synthesize(goal, hazard, np.ones((6, 6)), 5)  # map narrower than the zone
    with pytest.raises(ValueError):
        synthesize(goal, hazard, np.ones((10, 6)), 5, tie_tolerance=-1e-9)
    with pytest.raises(ValueError):  # goal sticks out of the zone
        synthesize(Droplet.at(9, 4, 2, 2), hazard, np.ones((10, 6)), 5)
    assert synthesize(goal, hazard, np.ones((10, 6)), 5).value(goal) == 1.0


@pytest.mark.parametrize("step_mode", ["single", "double", "adaptive"])
def test_transition_rows_are_distributions(step_mode):
    chip, job, wall = wall_scenario()
    set_level(chip, wall, 1)
    est = estimate_degradation(chip.health(), chip.health_levels)
    adaptive, fixed = parse_step_mode(step_mode)
    pair, nxt, prob, nominal, goal_id = _build_transitions(
        job.goal, job.hazard, job.goal.size, est, adaptive, fixed
    )
    num_states = nominal.shape[0]
    totals = np.bincount(pair, weights=prob, minlength=num_states * 8).reshape(num_states, 8)
    expected = np.ones((num_states, 8))
    expected[goal_id] = 0.0  # absorbing goal has no rows
    np.testing.assert_allclose(totals, expected, atol=1e-12)
    assert nxt.min() >= 0 and nxt.max() < num_states
    if step_mode != "single":
        return
    strategy = synthesize(job.goal, job.hazard, est, 5)
    # Spot-check one state-action pair against move_distribution directly.
    d = Droplet.at(6, 5, 2, 2)
    plan = plan_move(d, job.goal, job.hazard, Action.E, False, 1)
    dist = move_distribution(
        est, footprint_mask(est.shape, plan.target), d, job.hazard, action_flags(Action.E)
    )
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist[d] == pytest.approx(2 / 3)  # frontier (8,5),(8,6) at level 1 -> p = 1/3
    assert strategy.state_id(d) == (6 - 0) * (job.hazard.height - 1) + (5 - 2)
    assert strategy.droplet(strategy.state_id(d)) == d


def test_formal_act_policy_indexing_and_errors():
    chip, job, wall = wall_scenario()
    chip.set_faults(wall)
    formal = FormalRouter(horizon=25)
    with pytest.raises(RuntimeError):
        formal.act(RoutingState(chip, job.start, job.goal, job.hazard, 0, 10))
    formal.reset(job, chip)
    strategy = formal.strategy
    d = Droplet.at(6, 5, 2, 2)
    sid = strategy.state_id(d)
    for k in (0, 5, 24):
        state = RoutingState(chip, d, job.goal, job.hazard, k, 100)
        assert formal.act(state) == Action(int(strategy.policy[25 - k, sid]))
    for k in (25, 40):  # horizon exhausted: fall back to pi_K
        state = RoutingState(chip, d, job.goal, job.hazard, k, 100)
        assert formal.act(state) == Action(int(strategy.policy[25, sid]))
    other_goal = Droplet.at(15, 6, 2, 2)
    with pytest.raises(RuntimeError):
        formal.act(RoutingState(chip, d, other_goal, job.hazard, 0, 100))
    with pytest.raises(ValueError):
        strategy.state_id(Droplet.at(6, 5, 3, 3))
    with pytest.raises(ValueError):
        FormalRouter(horizon=0)


def test_formal_cache():
    chip, job, wall = wall_scenario()
    formal = FormalRouter(cache_size=2)
    formal.reset(job, chip)
    first = formal.strategy
    assert not formal.last_cache_hit
    formal.reset(job, chip)
    assert formal.last_cache_hit and formal.strategy is first
    chip.set_faults(wall)  # health changed inside the hazard zone -> resynthesize
    formal.reset(job, chip)
    assert not formal.last_cache_hit and formal.strategy is not first
    uncached = FormalRouter()
    uncached.reset(job, chip)
    uncached.reset(job, chip)
    assert not uncached.last_cache_hit


def test_formal_synthesis_time_large_zone():
    chip = MEDABiochip(60, 30, rng=np.random.default_rng(8))
    chip.reset()
    start, goal = Droplet.at(0, 0, 4, 4), Droplet.at(56, 26, 4, 4)
    job = RoutingJob(start, goal, Rect(0, 0, 59, 29))
    chip.inject_faults(0.1, protect=(start, goal))
    formal = FormalRouter()
    formal.reset(job, chip)
    assert formal.strategy.num_states == 57 * 27
    assert 0 < formal.last_synthesis_seconds < 5.0
    assert 0.0 <= formal.success_probability <= 1.0


def test_run_job_integration_on_faulty_chips():
    cfg = DegradationConfig(initial_actuations="uniform", max_initial_actuations=1500)
    routers = [ShortestPathRouter(), FormalRouter(), ShortestPathRouter("adaptive")]
    wins = {r.name + r.step_mode: 0 for r in routers}
    jobs = sampled_jobs(20, 20, 12, seed=9)
    for i, job in enumerate(jobs):
        base_chip = MEDABiochip(20, 20, cfg, rng=np.random.default_rng(200 + i))
        base_chip.reset()
        base_chip.inject_faults(0.15, protect=(job.start, job.goal))
        for router in routers:
            assert isinstance(router, Router)
            chip = base_chip.copy()
            rng = np.random.default_rng(i)
            res = run_job(router, job, chip, rng, kmax_alpha=1.5, record_path=True)
            assert res.cycles <= math.ceil(1.5 * (job.hazard.width + job.hazard.height))
            assert res.invalid_actions == 0
            assert all(job.hazard.contains(d) for d in res.path)
            assert res.success == (res.path[-1] == job.goal)
            wins[router.name + router.step_mode] += int(res.success)
    assert wins["formalsingle"] >= wins["baselinesingle"]
    assert wins["formalsingle"] > 0
