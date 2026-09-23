"""Probabilistic droplet movement: frontier sets, patterns and validity (Sec. III-A).

The paper adopts "the probabilistic transitions modeling from [18]" without
spelling it out; :mod:`meda_routing.core.movement` follows the authors'
reference implementation (``melfar87/MEDA``, ``envs/meda.py``,
``MEDAEnv._updatePattern``).  :func:`reference_update_pattern` below is a
line-by-line re-implementation of that method on the reference's half-open
coordinates ``(x0, y0, x1, y1)``, in which every ``random.random() <= p``
draw is replaced by an exact branch, so it yields the complete outcome
distribution.  The stable API (:func:`plan_move` + :func:`move_distribution`)
must reproduce validity, collision flags, actuation pattern and outcome
distribution exactly.  :func:`sample_move` must in turn match
:func:`move_distribution` statistically.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

from meda_routing.core.actions import (
    DIRECTIONS,
    Action,
    adaptive_step,
    max_reliable_step,
    unit_step,
)
from meda_routing.core.biochip import MEDABiochip
from meda_routing.core.dynamics import MovePlan, execute_move, plan_move, step_size
from meda_routing.core.geometry import Droplet, Rect
from meda_routing.core.jobs import hazard_bounds
from meda_routing.core.movement import (
    action_flags,
    clamp_into,
    footprint_mask,
    is_valid_action,
    move_distribution,
    sample_move,
    unit_step_probs,
)

_NORTH = {Action.N, Action.NE, Action.NW}
_SOUTH = {Action.S, Action.SE, Action.SW}
_EAST = {Action.E, Action.NE, Action.SE}
_WEST = {Action.W, Action.NW, Action.SW}
_SHIFT = {  # reference tmpShift, (x0, y0, x1, y1) increments
    Action.N: (0, 1, 0, 1),
    Action.S: (0, -1, 0, -1),
    Action.E: (1, 0, 1, 0),
    Action.W: (-1, 0, -1, 0),
    Action.NE: (1, 1, 1, 1),
    Action.NW: (-1, 1, -1, 1),
    Action.SE: (1, -1, 1, -1),
    Action.SW: (-1, -1, -1, -1),
}


# =========================================================== reference model
def half_open(rect: Rect) -> np.ndarray:
    return np.array([rect.xa, rect.ya, rect.xb + 1, rect.yb + 1], dtype=np.int64)


def inclusive(arr: Sequence[int]) -> Droplet:
    return Droplet(int(arr[0]), int(arr[1]), int(arr[2]) - 1, int(arr[3]) - 1)


def reference_update_pattern(
    droplet: Droplet,
    goal: Droplet,
    hazard: Rect,
    degradation: np.ndarray,
    action: Action,
    parm_step: bool = True,
):
    """``MEDAEnv._updatePattern`` with its random draws enumerated.

    Returns ``(valid, collision, pattern, target, distribution)`` where
    ``collision`` is ``None`` for a valid action (the reference leaves its
    flags untouched then) and ``distribution`` maps inclusive droplets to
    probabilities.
    """
    x_min, y_min, x_max, y_max = half_open(hazard)
    dr0 = half_open(droplet)
    x0, y0, x1, y1 = dr0
    move_n = move_s = move_e = move_w = False
    valid, collision = True, None
    if action == Action.N and y1 < y_max:
        move_n = True
    elif action == Action.S and y0 > y_min:
        move_s = True
    elif action == Action.E and x1 < x_max:
        move_e = True
    elif action == Action.W and x0 > x_min:
        move_w = True
    elif action == Action.NE and y1 < y_max and x1 < x_max:
        move_n = move_e = True
    elif action == Action.NW and y1 < y_max and x0 > x_min:
        move_n = move_w = True
    elif action == Action.SE and y0 > y_min and x1 < x_max:
        move_s = move_e = True
    elif action == Action.SW and y0 > y_min and x0 > x_min:
        move_s = move_w = True
    else:
        valid = False
        collision = (
            bool(x0 == x_min and action in _WEST),
            bool(y0 == y_min and action in _SOUTH),
            bool(x1 == x_max and action in _EAST),
            bool(y1 == y_max and action in _NORTH),
        )
    shift = np.array(_SHIFT[action] if valid else (0, 0, 0, 0), dtype=np.int64)
    gl = half_open(goal)
    if parm_step:
        radius = np.floor_divide(dr0[[2, 3]] - dr0[[0, 1]], 2)
        tmp = dr0 + shift * radius[[0, 1, 0, 1]]
        if (y0 < gl[1] < tmp[1]) or (y0 > gl[1] > tmp[1]):
            tmp[[1, 3]] = gl[[1, 3]]
        if (x0 < gl[0] < tmp[0]) or (x0 > gl[0] > tmp[0]):
            tmp[[0, 2]] = gl[[0, 2]]
    else:
        tmp = dr0 + shift
    if tmp[0] < x_min:
        tmp[[0, 2]] += x_min - tmp[0]
    elif tmp[2] > x_max:
        tmp[[0, 2]] -= tmp[2] - x_max
    if tmp[1] < y_min:
        tmp[[1, 3]] += y_min - tmp[1]
    elif tmp[3] > y_max:
        tmp[[1, 3]] -= tmp[3] - y_max
    pattern = np.zeros(degradation.shape, dtype=np.uint8)
    pattern[tmp[0] : tmp[2], tmp[1] : tmp[3]] = 1
    flags = (move_n, move_s, move_e, move_w)
    dist = reference_movement(droplet, hazard, pattern, degradation, flags)
    return valid, collision, pattern.astype(bool), inclusive(tmp), dist


def reference_movement(
    droplet: Droplet,
    hazard: Rect,
    pattern: np.ndarray,
    degradation: np.ndarray,
    flags: Tuple[bool, bool, bool, bool],
) -> Dict[Droplet, float]:
    """The ``while (moveN or moveS or moveE or moveW)`` loop of ``_updatePattern``, enumerated."""
    x_min, y_min, x_max, y_max = half_open(hazard)
    pattern = pattern.astype(np.uint8)

    def front_prob(front: np.ndarray, deg: np.ndarray) -> float:
        total = front.sum()
        return float(np.dot(front, deg) / total) if total > 0 else 0.0

    dist: Dict[Droplet, float] = {}
    stack = [(tuple(int(v) for v in half_open(droplet)), *flags, 1.0)]
    while stack:
        dr, mn, ms, me, mw, mass = stack.pop()
        if not (mn or ms or me or mw):
            key = inclusive(dr)
            dist[key] = dist.get(key, 0.0) + mass
            continue
        p_n = p_s = p_e = p_w = 0.0
        if mn:
            if dr[3] < y_max:
                cols = slice(max(dr[0] - 1, 0), dr[2] + 1)
                p_n = front_prob(pattern[cols, dr[3]], degradation[cols, dr[3]])
        elif ms:
            if dr[1] > y_min:
                cols = slice(max(dr[0] - 1, 0), dr[2] + 1)
                p_s = front_prob(pattern[cols, dr[1] - 1], degradation[cols, dr[1] - 1])
        if me:
            if dr[2] < x_max:
                rows = slice(max(dr[1] - 1, 0), dr[3] + 1)
                p_e = front_prob(pattern[dr[2], rows], degradation[dr[2], rows])
        elif mw:
            if dr[0] > x_min:
                rows = slice(max(dr[1] - 1, 0), dr[3] + 1)
                p_w = front_prob(pattern[dr[0] - 1, rows], degradation[dr[0] - 1, rows])
        # moveX = (random.random() <= probX), enumerated
        y_branches = [(mn, ms, 1.0)]
        if mn:
            y_branches = [(True, False, p_n), (False, False, 1.0 - p_n)]
        elif ms:
            y_branches = [(False, True, p_s), (False, False, 1.0 - p_s)]
        x_branches = [(me, mw, 1.0)]
        if me:
            x_branches = [(True, False, p_e), (False, False, 1.0 - p_e)]
        elif mw:
            x_branches = [(False, True, p_w), (False, False, 1.0 - p_w)]
        for nn, ns, qy in y_branches:
            for ne, nw, qx in x_branches:
                q = mass * qy * qx
                if q <= 0.0:
                    continue
                dx = (1 if ne else 0) - (1 if nw else 0)
                dy = (1 if nn else 0) - (1 if ns else 0)
                nxt = (dr[0] + dx, dr[1] + dy, dr[2] + dx, dr[3] + dy)
                stack.append((nxt, nn, ns, ne, nw, q))
    return dist


# ================================================================== helpers
def chi2_upper(dof: int, z: float) -> float:
    """Wilson-Hilferty approximation of the chi-square quantile at normal score ``z``."""
    k = float(dof)
    return k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3


def assert_matches_distribution(
    samples: List[Droplet], exact: Dict[Droplet, float], z: float = 4.5
) -> None:
    """Monte-Carlo frequencies agree with the exact distribution (per outcome and chi-square)."""
    n = len(samples)
    counts = Counter(samples)
    assert set(counts) <= set(exact), f"impossible outcomes sampled: {set(counts) - set(exact)}"
    for outcome, p in exact.items():
        freq = counts.get(outcome, 0) / n
        assert abs(freq - p) <= z * math.sqrt(p * (1.0 - p) / n) + 1.0 / n, (outcome, freq, p)
    bins = [(counts.get(o, 0), n * p) for o, p in exact.items() if n * p >= 5.0]
    rare_obs = n - sum(o for o, _ in bins)
    rare_exp = n - sum(e for _, e in bins)
    if rare_exp > 1e-9:
        if rare_exp >= 5.0 or not bins:
            bins.append((rare_obs, rare_exp))
        else:
            o, e = bins[-1]
            bins[-1] = (o + rare_obs, e + rare_exp)
    if len(bins) >= 2:
        stat = sum((o - e) ** 2 / e for o, e in bins)
        assert stat <= chi2_upper(len(bins) - 1, z), (stat, bins)


def random_case(rng: np.random.Generator, max_size: int = 6, min_size: int = 1):
    """Random chip size, same-size start/goal, routing zone and degradation matrix."""
    width, height = int(rng.integers(6, 19)), int(rng.integers(6, 19))
    w = int(rng.integers(min_size, min(max_size, width - 1) + 1))
    h = int(rng.integers(min_size, min(max_size, height - 1) + 1))
    start = Droplet.at(
        int(rng.integers(0, width - w + 1)), int(rng.integers(0, height - h + 1)), w, h
    )
    goal = Droplet.at(
        int(rng.integers(0, width - w + 1)), int(rng.integers(0, height - h + 1)), w, h
    )
    if rng.random() < 0.7:
        hazard = hazard_bounds(start, goal, width, height, margin=int(rng.integers(0, 4)))
    else:
        hazard = Rect(0, 0, width - 1, height - 1)
    deg = rng.uniform(0.0, 1.0, size=(width, height))
    deg[rng.random((width, height)) < 0.2] = 0.0
    deg[rng.random((width, height)) < 0.2] = 1.0
    return start, goal, hazard, deg


def stable_distribution(
    plan: MovePlan, droplet: Droplet, hazard: Rect, deg: np.ndarray
) -> Dict[Droplet, float]:
    """Outcome distribution of :func:`execute_move` for ``plan``."""
    if not plan.valid or plan.target == droplet:
        return {droplet: 1.0}
    pattern = footprint_mask(deg.shape, plan.target)
    return move_distribution(deg, pattern, droplet, hazard, action_flags(plan.action))


# ============================================================= basic helpers
def test_footprint_mask_and_action_flags():
    mask = footprint_mask((6, 5), Rect(1, 2, 3, 4))
    assert mask.dtype == bool and mask.shape == (6, 5)
    assert {(int(x), int(y)) for x, y in zip(*np.nonzero(mask))} == set(Rect(1, 2, 3, 4).cells())
    for action, (ux, uy) in DIRECTIONS.items():
        assert action_flags(action) == (uy, ux)  # (y-direction, x-direction)


def test_step_size_selects_algorithm_1_or_fixed_step():
    d, g = Droplet.at(3, 3, 4, 4), Droplet.at(20, 20, 4, 4)
    assert step_size(d, g, Action.NE) == adaptive_step(d, g, Action.NE) == (2, 2)
    assert step_size(d, g, Action.NE, adaptive=False) == unit_step(d, g, Action.NE, 1) == (1, 1)
    assert step_size(d, g, Action.NE, adaptive=False, fixed_step=2) == (2, 2)


# ================================================== validity and clamping
def test_is_valid_action_matches_reference_exhaustively():
    hazard = Rect(2, 1, 9, 7)
    x_min, y_min, x_max, y_max = half_open(hazard)
    checked = 0
    for w in range(1, 5):
        for h in range(1, 5):
            for xa in range(hazard.xa, hazard.xb - w + 2):
                for ya in range(hazard.ya, hazard.yb - h + 2):
                    d = Droplet.at(xa, ya, w, h)
                    x0, y0, x1, y1 = half_open(d)
                    ok = {"N": y1 < y_max, "S": y0 > y_min, "E": x1 < x_max, "W": x0 > x_min}
                    for action in Action:
                        expected = all(ok[c] for c in action.name)  # "NE" needs N and E
                        assert is_valid_action(d, hazard, action) == expected
                        checked += 1
    assert checked > 3000


def test_clamp_into(rng):
    zone = Rect(3, 2, 14, 11)
    for _ in range(3000):
        w, h = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        r = Droplet.at(int(rng.integers(-8, 20)), int(rng.integers(-8, 20)), w, h)
        c = clamp_into(r, zone)
        assert type(c) is Droplet and c.size == r.size
        assert zone.contains(c)
        if zone.contains(r):
            assert c == r
        # minimal shift, equal to the reference's clamping of tmpDr
        dx = max(zone.xa - r.xa, 0) - max(r.xb - zone.xb, 0)
        dy = max(zone.ya - r.ya, 0) - max(r.yb - zone.yb, 0)
        assert c == r.shift(dx, dy)


def test_invalid_action_holds_position_and_flags_collision():
    hazard = Rect(0, 0, 9, 9)
    d = Droplet.at(0, 7, 3, 3)  # touches the west and north boundaries
    plan = plan_move(d, Droplet.at(6, 0, 3, 3), hazard, Action.NW)
    assert not plan.valid and plan.target == d
    assert plan.collision == (True, False, False, True)  # (west, south, east, north)
    plan = plan_move(d, Droplet.at(6, 0, 3, 3), hazard, Action.NE)  # north blocked only
    assert not plan.valid and plan.collision == (False, False, False, True)
    plan = plan_move(d, Droplet.at(6, 0, 3, 3), hazard, Action.SE)
    assert plan.valid and plan.collision == (False, False, False, False)
    new, pattern = execute_move(
        plan_move(d, d, hazard, Action.W), d, hazard, np.ones((10, 10)), np.random.default_rng(0)
    )
    assert new == d  # holding position ...
    assert np.array_equal(pattern, footprint_mask((10, 10), d))  # ... with the holding pattern


# =========================================================== frontier sets
def test_frontier_probabilities_by_hand():
    width, height = 10, 8
    # every MC has a distinct degradation level, so each frontier set is identifiable
    deg = (1.0 + np.arange(width * height, dtype=float).reshape(width, height)) / (
        width * height + 1
    )
    everything = np.ones((width, height), dtype=bool)
    chip_zone = Rect(0, 0, width - 1, height - 1)
    d = Droplet(2, 2, 4, 3)

    def probs(pattern: np.ndarray, drop: Droplet, flags: Tuple[int, int], zone: Rect = chip_zone):
        return unit_step_probs(deg, pattern, drop, zone, flags)

    # row beyond the leading edge, columns xa-1 .. xb+1
    assert probs(everything, d, (1, 0))[0] == pytest.approx(deg[1:6, 4].mean())
    assert probs(everything, d, (-1, 0))[0] == pytest.approx(deg[1:6, 1].mean())
    # column beyond the leading edge, rows ya-1 .. yb+1
    assert probs(everything, d, (0, 1))[1] == pytest.approx(deg[5, 1:5].mean())
    assert probs(everything, d, (0, -1))[1] == pytest.approx(deg[1, 1:5].mean())
    # both axes of an ordinal move use the same droplet position
    py, px = probs(everything, d, (1, -1))
    assert (py, px) == (pytest.approx(deg[1:6, 4].mean()), pytest.approx(deg[1, 1:5].mean()))
    # the frontier is clipped at the chip boundary
    corner = Droplet(0, 0, 2, 1)
    assert probs(everything, corner, (1, 0))[0] == pytest.approx(deg[0:4, 2].mean())
    assert probs(everything, corner, (0, 1))[1] == pytest.approx(deg[3, 0:3].mean())
    far = Droplet(7, 5, 9, 7)
    assert probs(everything, far, (-1, 0))[0] == pytest.approx(deg[6:10, 4].mean())
    assert probs(everything, far, (0, -1))[1] == pytest.approx(deg[6, 4:8].mean())
    # only actuated MCs of the frontier count
    partial = np.zeros((width, height), dtype=bool)
    partial[3:6, 4] = True
    assert probs(partial, d, (1, 0))[0] == pytest.approx(deg[3:6, 4].mean())
    # no actuated frontier MC: the axis cannot move
    assert probs(np.zeros_like(partial), d, (1, 0)) == (0.0, 0.0)
    # the droplet cannot leave the routing zone, even under an actuated pattern
    assert probs(everything, far, (1, 1)) == (0.0, 0.0)
    zone = Rect(2, 2, 8, 3)
    assert probs(everything, d, (1, 0), zone)[0] == 0.0
    assert probs(everything, d, (0, 1), zone)[1] == pytest.approx(deg[5, 1:5].mean())


def test_movement_uses_true_degradation_not_health_reading():
    chip = MEDABiochip(8, 6)
    chip.tau[:] = 1.0
    frontier = [(x, 3) for x in range(0, 8)]
    for x, y in frontier:  # D = 0.74 exactly: 2-bit reading floor(2.96) = 2 (0.5)
        chip.tau[x, y], chip.c[x, y], chip.actuations[x, y] = 0.74, 1.0, 1
    assert all(chip.health()[x, y] == 2 for x, y in frontier)
    d = Droplet.at(2, 1, 2, 2)
    plan = plan_move(d, d.shift(0, 3), Rect(0, 0, 7, 5), Action.N, adaptive=False)
    dist = stable_distribution(plan, d, Rect(0, 0, 7, 5), chip.effective_degradation())
    assert dist == {d.shift(0, 1): pytest.approx(0.74), d: pytest.approx(0.26)}


# =================================================== exact distribution
def test_move_distribution_is_a_distribution_inside_the_box(rng):
    for _ in range(300):
        start, goal, hazard, deg = random_case(rng)
        action = Action(int(rng.integers(8)))
        plan = plan_move(
            start, goal, hazard, action, adaptive=bool(rng.random() < 0.7), fixed_step=2
        )
        dist = stable_distribution(plan, start, hazard, deg)
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-12)
        assert all(p > 0.0 for p in dist.values())
        lo_x, hi_x = sorted((start.xa, plan.target.xa))
        lo_y, hi_y = sorted((start.ya, plan.target.ya))
        for outcome in dist:
            assert type(outcome) is Droplet and outcome.size == start.size
            assert hazard.contains(outcome)
            # never beyond the target, never backwards
            assert lo_x <= outcome.xa <= hi_x and lo_y <= outcome.ya <= hi_y


def test_matches_reference_update_pattern(rng):
    """Validity, collisions, pattern, target and outcome distribution == reference code."""
    cases = 0
    for _ in range(150):
        start, goal, hazard, deg = random_case(rng)
        for action in Action:
            for parm in (True, False):
                valid, collision, pattern, target, ref = reference_update_pattern(
                    start, goal, hazard, deg, action, parm
                )
                plan = plan_move(start, goal, hazard, action, adaptive=parm, fixed_step=1)
                assert plan.valid == valid == is_valid_action(start, hazard, action)
                if not valid:
                    assert plan.collision == collision
                assert plan.target == target
                assert np.array_equal(footprint_mask(deg.shape, plan.target), pattern)
                dist = stable_distribution(plan, start, hazard, deg)
                assert set(dist) <= set(ref)
                for outcome, p in ref.items():
                    assert dist.get(outcome, 0.0) == pytest.approx(p, abs=1e-12), (
                        start,
                        goal,
                        action,
                        parm,
                    )
                cases += 1
    assert cases == 150 * 16


def test_healthy_chip_moves_deterministically_to_target(rng):
    """``D = 1``: every valid move reaches its target (steps of at most the droplet size)."""
    for _ in range(300):
        start, goal, hazard, _ = random_case(rng, min_size=2)
        deg = np.ones((hazard.xb + 1, hazard.yb + 1))
        action = Action(int(rng.integers(8)))
        adaptive = bool(rng.random() < 0.6)
        plan = plan_move(
            start, goal, hazard, action, adaptive=adaptive, fixed_step=int(rng.integers(1, 3))
        )
        assert stable_distribution(plan, start, hazard, deg) == {plan.target: 1.0}
        new, pattern = execute_move(plan, start, hazard, deg, rng)
        assert new == plan.target
        assert np.array_equal(pattern, footprint_mask(deg.shape, plan.target))
        if plan.valid:
            assert plan.target != start  # a valid action always actuates a new footprint
        else:
            assert new == start


def test_step_longer_than_droplet_cannot_move():
    """A footprint that is not adjacent to the droplet exerts no force (reference model too).

    Algorithm 1 never produces such steps (``lambda <= floor(size / 2)``), but a
    fixed double step of a droplet that is one MC tall does.
    """
    hazard = Rect(0, 0, 9, 9)
    deg = np.ones((10, 10))
    d = Droplet.at(0, 2, 5, 1)
    plan = plan_move(d, Droplet.at(0, 8, 5, 1), hazard, Action.N, adaptive=False, fixed_step=2)
    assert plan.valid and plan.target == d.shift(0, 2)
    assert stable_distribution(plan, d, hazard, deg) == {d: 1.0}
    pattern = footprint_mask(deg.shape, plan.target)
    assert reference_movement(d, hazard, pattern, deg, (True, False, False, False)) == {d: 1.0}
    # the (reference) single step moves it
    assert stable_distribution(
        plan_move(d, d, hazard, Action.N, adaptive=False), d, hazard, deg
    ) == {d.shift(0, 1): 1.0}


def test_fully_degraded_frontier_blocks_movement(rng):
    width, height = 16, 12
    hazard = Rect(0, 0, width - 1, height - 1)
    d = Droplet.at(5, 4, 4, 4)  # Lambda = (2, 2)
    far = Droplet.at(13, 8, 4, 4)
    deg = np.ones((width, height))
    deg[:, 8] = 0.0  # row beyond the north edge: dead
    # cardinal north: stays
    plan = plan_move(d, far, hazard, Action.N)
    assert stable_distribution(plan, d, hazard, deg) == {d: 1.0}
    assert all(execute_move(plan, d, hazard, deg, rng)[0] == d for _ in range(20))
    # ordinal north-east: the north axis stops at once.  The east frontier
    # (column xb+1, actuated rows 6..8) contains the dead corner MC of row 8,
    # so every unit step east succeeds with probability 2/3.
    plan = plan_move(d, far, hazard, Action.NE)
    assert plan.target == d.shift(2, 2)
    dist = stable_distribution(plan, d, hazard, deg)
    assert dist == {
        d: pytest.approx(1 / 3),
        d.shift(1, 0): pytest.approx(2 / 3 * 1 / 3),
        d.shift(2, 0): pytest.approx(4 / 9),
    }
    # both frontiers dead: stays
    deg[9, :] = 0.0
    assert stable_distribution(plan, d, hazard, deg) == {d: 1.0}
    # one healthy MC in the actuated frontier: success probability 1 / 4
    deg = np.ones((width, height))
    deg[9, 3:9] = 0.0
    deg[9, 5] = 1.0
    plan = plan_move(d, far, hazard, Action.E, adaptive=False)
    # frontier = column 9, rows 3..8, of which only the target rows 4..7 are actuated
    assert stable_distribution(plan, d, hazard, deg) == {
        d.shift(1, 0): pytest.approx(0.25),
        d: pytest.approx(0.75),
    }


def test_hidden_defects_block_movement_but_not_sensing():
    chip = MEDABiochip(12, 10)
    chip.tau[:] = 1.0
    chip.set_faults([(6, y) for y in range(10)], hidden=True)
    d = Droplet.at(2, 3, 4, 4)
    plan = plan_move(d, Droplet.at(8, 3, 4, 4), Rect(0, 0, 11, 9), Action.E)
    assert np.all(chip.health() == 3)
    sensed = stable_distribution(plan, d, Rect(0, 0, 11, 9), chip.degradation())
    actual = stable_distribution(plan, d, Rect(0, 0, 11, 9), chip.effective_degradation())
    assert sensed == {plan.target: 1.0} and actual == {d: 1.0}


# ======================================================== Monte-Carlo checks
MC_SCENARIOS = [
    # (name, droplet, goal, action, adaptive)
    ("cardinal E, unit step", Droplet.at(6, 6, 2, 2), Droplet.at(15, 6, 2, 2), Action.E, False),
    ("cardinal N, adaptive 2", Droplet.at(6, 3, 4, 4), Droplet.at(6, 11, 4, 4), Action.N, True),
    ("ordinal SW, unit step", Droplet.at(9, 9, 2, 2), Droplet.at(1, 1, 2, 2), Action.SW, False),
    (
        "ordinal NE, adaptive (2, 2)",
        Droplet.at(4, 3, 4, 4),
        Droplet.at(12, 10, 4, 4),
        Action.NE,
        True,
    ),
    ("multi-step W, adaptive 3", Droplet.at(12, 5, 6, 6), Droplet.at(0, 5, 6, 6), Action.W, True),
    (
        "ordinal NW, adaptive (3, 2)",
        Droplet.at(11, 2, 6, 5),
        Droplet.at(1, 9, 6, 5),
        Action.NW,
        True,
    ),
    ("capped SE, adaptive (1, 2)", Droplet.at(5, 9, 5, 5), Droplet.at(6, 3, 5, 5), Action.SE, True),
]


@pytest.mark.parametrize(
    "name, droplet, goal, action, adaptive", MC_SCENARIOS, ids=[s[0] for s in MC_SCENARIOS]
)
def test_sample_move_matches_exact_distribution(name, droplet, goal, action, adaptive):
    rng = np.random.default_rng(zlib.crc32(name.encode()))  # hash() is salted per process
    width, height = 20, 16
    hazard = Rect(0, 0, width - 1, height - 1)
    deg = rng.uniform(0.35, 0.95, size=(width, height))
    plan = plan_move(droplet, goal, hazard, action, adaptive=adaptive)
    assert plan.valid
    step = (plan.target.xa - droplet.xa, plan.target.ya - droplet.ya)
    assert tuple((v > 0) - (v < 0) for v in step) == DIRECTIONS[action]
    if adaptive:
        lx, ly = max_reliable_step(droplet)
        assert abs(step[0]) <= lx and abs(step[1]) <= ly
    exact = stable_distribution(plan, droplet, hazard, deg)
    assert sum(exact.values()) == pytest.approx(1.0, abs=1e-12)
    assert len(exact) >= 2
    pattern = footprint_mask(deg.shape, plan.target)
    flags = action_flags(action)
    n = 20000
    samples = [sample_move(deg, pattern, droplet, hazard, flags, rng) for _ in range(n)]
    assert_matches_distribution(samples, exact)
    # execute_move draws from the same distribution
    via_execute = [execute_move(plan, droplet, hazard, deg, rng)[0] for _ in range(4000)]
    assert_matches_distribution(via_execute, exact)


def test_monte_carlo_check_detects_a_wrong_distribution():
    """Sanity check of the statistical helper itself."""
    rng = np.random.default_rng(1)
    a, b = Droplet(0, 0, 0, 0), Droplet(1, 0, 1, 0)
    samples = [a if rng.random() < 0.5 else b for _ in range(20000)]
    assert_matches_distribution(samples, {a: 0.5, b: 0.5})
    with pytest.raises(AssertionError):
        assert_matches_distribution(samples, {a: 0.47, b: 0.53})
