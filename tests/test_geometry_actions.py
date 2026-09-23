"""Droplet geometry (Sec. III-A, Fig. 1) and the parameterized action space (Sec. III-B).

Algorithm 1 (:func:`adaptive_step`) is checked three ways:

* against the worked numbers of the paper (Example 1 and Fig. 1);
* against its defining properties (maximal reliable step ``floor(size / 2)``,
  never overshooting the goal) on many random droplet/goal pairs;
* against a re-implementation of the parameterized step of the authors'
  reference code (``melfar87/MEDA``, ``envs/meda.py``,
  ``MEDAEnv._updatePattern`` with ``b_parm_step=True``), which works on
  half-open coordinates ``(x0, y0, x1, y1)`` and snaps an overshooting target
  onto the goal instead of using the indicator form of the paper.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import Tuple

import numpy as np
import pytest

from meda_routing.core.actions import (
    DIRECTIONS,
    NUM_ACTIONS,
    Action,
    adaptive_step,
    apply_step,
    max_reliable_step,
    unit_step,
)
from meda_routing.core.geometry import Droplet, Rect, chip_rect
from meda_routing.core.jobs import PAPER_DROPLET_SIZES, JobSamplerConfig, RoutingJob

# Reference ``tmpShift`` of every direction, as ``(x0, y0, x1, y1)`` increments.
REF_SHIFT = {
    Action.N: (0, 1, 0, 1),
    Action.S: (0, -1, 0, -1),
    Action.E: (1, 0, 1, 0),
    Action.W: (-1, 0, -1, 0),
    Action.NE: (1, 1, 1, 1),
    Action.NW: (-1, 1, -1, 1),
    Action.SE: (1, -1, 1, -1),
    Action.SW: (-1, -1, -1, -1),
}


def half_open(rect: Rect) -> np.ndarray:
    """Inclusive ``(xa, ya, xb, yb)`` -> reference ``(x0, y0, x1, y1)``."""
    return np.array([rect.xa, rect.ya, rect.xb + 1, rect.yb + 1], dtype=np.int64)


def reference_step(
    droplet: Droplet, goal: Droplet, action: Action, parm_step: bool = True
) -> Tuple[int, int]:
    """Signed step of the reference ``_updatePattern`` (routing-zone clamping omitted)."""
    dr, gl = half_open(droplet), half_open(goal)
    x0, y0 = dr[0], dr[1]
    shift = np.array(REF_SHIFT[action], dtype=np.int64)
    if parm_step:
        radius = np.floor_divide(dr[[2, 3]] - dr[[0, 1]], 2)
        tmp = dr + shift * radius[[0, 1, 0, 1]]
        # "Prevent target overshooting" in y, then in x
        if (y0 < gl[1] < tmp[1]) or (y0 > gl[1] > tmp[1]):
            tmp[[1, 3]] = gl[[1, 3]]
        if (x0 < gl[0] < tmp[0]) or (x0 > gl[0] > tmp[0]):
            tmp[[0, 2]] = gl[[0, 2]]
    else:
        tmp = dr + shift
    return int(tmp[0] - x0), int(tmp[1] - y0)


def random_pair(
    rng: np.random.Generator, max_size: int = 8, spread: int = 25
) -> Tuple[Droplet, Droplet]:
    """Random droplet and a random goal of the same size (both anywhere)."""
    w, h = int(rng.integers(1, max_size + 1)), int(rng.integers(1, max_size + 1))
    x, y = (int(v) for v in rng.integers(-spread, spread + 1, size=2))
    # small offsets are over-represented: they exercise the capping branch
    scale = int(rng.choice([2, 4, 8, spread]))
    dx, dy = (int(v) for v in rng.integers(-scale, scale + 1, size=2))
    return Droplet.at(x, y, w, h), Droplet.at(x + dx, y + dy, w, h)


def sign(v: int) -> int:
    return (v > 0) - (v < 0)


# ====================================================================== geometry
def test_rect_bounds_are_inclusive():
    r = Rect(2, 3, 5, 4)
    assert (r.width, r.height, r.area, r.size) == (4, 2, 8, (4, 2))
    single = Rect(7, 7, 7, 7)
    assert single.size == (1, 1) and single.area == 1
    assert list(single.cells()) == [(7, 7)]
    assert r.as_tuple() == (2, 3, 5, 4)
    xa, ya, xb, yb = r  # __iter__ unpacking
    assert (xa, ya, xb, yb) == (2, 3, 5, 4)


@pytest.mark.parametrize("coords", [(3, 0, 2, 0), (0, 3, 0, 2), (1, 1, 0, 0)])
def test_degenerate_rect_rejected(coords):
    with pytest.raises(ValueError):
        Rect(*coords)
    with pytest.raises(ValueError):
        Droplet(*coords)


def test_droplet_at_and_chip_rect():
    d = Droplet.at(3, 2, 4, 3)
    assert d == Droplet(3, 2, 6, 4) and d.size == (4, 3)  # the droplet of Fig. 1
    assert type(d) is Droplet
    assert chip_rect(30, 20) == Rect(0, 0, 29, 19)
    assert chip_rect(30, 20).area == 600


def test_shift_keeps_type_and_size():
    d = Droplet(1, 2, 4, 3)
    s = d.shift(3, -2)
    assert type(s) is Droplet and s == Droplet(4, 0, 7, 1) and s.size == d.size
    r = Rect(0, 0, 1, 1).shift(-1, 5)
    assert type(r) is Rect and r == Rect(-1, 5, 0, 6)


def test_rects_are_frozen_and_hashable():
    d = Droplet(0, 0, 1, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.xa = 3  # type: ignore[misc]
    table = {d: 1, Droplet(0, 0, 1, 1): 2}
    assert table == {Droplet(0, 0, 1, 1): 2}


def test_contains_point_and_rect_inclusive_edges():
    zone = Rect(2, 2, 6, 5)
    for x, y in [(2, 2), (6, 5), (2, 5), (6, 2), (4, 3)]:
        assert zone.contains_point(x, y)
    for x, y in [(1, 2), (7, 5), (2, 1), (6, 6)]:
        assert not zone.contains_point(x, y)
    assert zone.contains(zone)
    assert zone.contains(Rect(2, 2, 3, 3)) and zone.contains(Rect(5, 4, 6, 5))
    assert not zone.contains(Rect(5, 4, 7, 5))  # one column outside
    assert not Rect(3, 3, 4, 4).contains(zone)


def test_intersection_inclusive_edges():
    a = Rect(0, 0, 3, 3)
    assert a.intersection(Rect(3, 3, 5, 5)) == Rect(3, 3, 3, 3)  # sharing a corner MC
    assert not a.intersects(Rect(4, 0, 5, 3))  # adjacent, no shared MC
    assert a.intersection(Rect(4, 0, 5, 3)) is None
    assert a.intersection(Rect(1, -2, 2, 8)) == Rect(1, 0, 2, 3)


def test_set_operations_match_cell_sets(rng):
    """contains / intersects / intersection / cells / slices agree with brute-force cell sets."""
    for _ in range(300):
        xa, ya = (int(v) for v in rng.integers(0, 10, size=2))
        a = Rect(xa, ya, xa + int(rng.integers(0, 5)), ya + int(rng.integers(0, 5)))
        xb, yb = (int(v) for v in rng.integers(0, 10, size=2))
        b = Rect(xb, yb, xb + int(rng.integers(0, 5)), yb + int(rng.integers(0, 5)))
        ca, cb = set(a.cells()), set(b.cells())
        assert len(ca) == a.area
        assert all(a.contains_point(x, y) for x, y in ca)
        assert a.contains(b) == (cb <= ca)
        assert a.intersects(b) == bool(ca & cb) == b.intersects(a)
        inter = a.intersection(b)
        assert (set(inter.cells()) if inter is not None else set()) == ca & cb
        grid = np.zeros((16, 16), dtype=bool)
        grid[a.slices()] = True  # slices index an [x, y] array
        assert {(int(x), int(y)) for x, y in zip(*np.nonzero(grid))} == ca
        assert grid[a.slices()].shape == (a.width, a.height)


def test_manhattan_distance(rng):
    a, b = Droplet(1, 1, 3, 2), Droplet(5, -2, 7, -1)
    assert a.manhattan(b) == 4 + 3 == b.manhattan(a)
    assert a.manhattan(a) == 0
    for _ in range(200):
        d, g = random_pair(rng)
        dist = d.manhattan(g)
        # the same between SW corners and NE corners (sizes are preserved)
        assert dist == abs(d.xa - g.xa) + abs(d.ya - g.ya) == abs(d.xb - g.xb) + abs(d.yb - g.yb)
        assert (dist == 0) == (d == g)
        assert d.shift(3, -4).manhattan(g.shift(3, -4)) == dist
        assert d.manhattan(Rect(*g.as_tuple())) == dist  # a Rect argument is fine


# ========================================================= equality semantics
def test_droplet_never_equals_rect_with_same_coordinates():
    d, r = Droplet(1, 2, 3, 4), Rect(1, 2, 3, 4)
    assert d != r and r != d
    assert d.as_tuple() == r.as_tuple()
    assert d == Droplet(1, 2, 3, 4)
    assert len({d, r}) == 2


def test_operations_return_droplets_so_goal_checks_work():
    """The env tests ``droplet == goal``: every droplet-producing operation keeps the type."""
    d = Droplet.at(2, 2, 3, 3)
    assert type(apply_step(d, adaptive_step(d, d.shift(9, 9), Action.NE))) is Droplet
    job = RoutingJob(Rect(0, 0, 1, 1), Rect(5, 5, 6, 6), Rect(0, 0, 9, 9))
    # RoutingJob coerces Rect endpoints so that "droplet == goal" can succeed
    assert type(job.start) is Droplet and type(job.goal) is Droplet
    assert job.start.shift(5, 5) == job.goal


# ================================================================ action space
def test_action_space_matches_paper():
    # A = {aN, aS, aE, aW, aNE, aNW, aSE, aSW} in the order of Sec. III-B
    assert [a.name for a in Action] == ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
    assert NUM_ACTIONS == 8 == len(DIRECTIONS)
    # north is +y (rows), east is +x (columns), Fig. 1
    assert DIRECTIONS[Action.N] == (0, 1) and DIRECTIONS[Action.E] == (1, 0)
    for action, (ux, uy) in DIRECTIONS.items():
        assert ux in (-1, 0, 1) and uy in (-1, 0, 1) and (ux, uy) != (0, 0)
        assert ("E" in action.name) == (ux == 1) and ("W" in action.name) == (ux == -1)
        assert ("N" in action.name) == (uy == 1) and ("S" in action.name) == (uy == -1)
    assert len(set(DIRECTIONS.values())) == 8


@pytest.mark.parametrize("w, h", [(1, 1), (2, 2), (3, 3), (4, 3), (5, 4), (6, 6), (7, 2)])
def test_max_reliable_step_is_half_the_size(w, h):
    assert max_reliable_step(Droplet.at(5, 5, w, h)) == (w // 2, h // 2)


# =================================================================== Algorithm 1
def test_paper_example_1_and_fig_1():
    # Fig. 1 (1-based coordinates; Algorithm 1 only uses differences)
    droplet = Droplet(3, 2, 6, 4)
    assert droplet.size == (4, 3)
    assert max_reliable_step(droplet) == (2, 1)  # (floor(w/2), floor(h/2))
    far = Droplet(20, 20, 23, 22)
    # NE moves two MCs east and one north (the prose of Example 1 swaps the
    # words "east"/"north"; the formula and Fig. 1 agree on (2, 1)).
    assert adaptive_step(droplet, far, Action.NE) == (2, 1)
    assert apply_step(droplet, (2, 1)) == Droplet(5, 3, 8, 5)  # delta^(k+1) of Fig. 1
    # Example 1: goal (4, 3, 7, 5) caps the step at (1, 1)
    goal = Droplet(4, 3, 7, 5)
    assert adaptive_step(droplet, goal, Action.NE) == (1, 1)
    assert apply_step(droplet, (1, 1)) == goal
    # identical in the 0-based coordinates used by the code
    assert adaptive_step(droplet.shift(-1, -1), goal.shift(-1, -1), Action.NE) == (1, 1)


@pytest.mark.parametrize("action", list(Action))
@pytest.mark.parametrize("w, h", [(4, 4), (5, 3), (6, 5)])
def test_all_directions(action, w, h):
    ux, uy = DIRECTIONS[action]
    d = Droplet.at(20, 20, w, h)
    lx, ly = max_reliable_step(d)
    full = (ux * lx, uy * ly)
    # goal far ahead, far behind, or at the droplet: full reliable step
    assert adaptive_step(d, d.shift(15 * ux, 15 * uy), action) == full
    assert adaptive_step(d, d.shift(-15 * ux, -15 * uy), action) == full
    assert adaptive_step(d, d, action) == full
    # goal exactly one reliable step ahead: land on it
    assert adaptive_step(d, d.shift(*full), action) == full
    # goal 1 MC ahead on every moving axis (1 < Lambda for these sizes): capped
    assert adaptive_step(d, d.shift(ux, uy), action) == (ux, uy)
    # only the axes of the action move, whatever the goal offset
    for gx, gy in [(1, 1), (-1, -1), (7, -3), (-5, 9)]:
        sx, sy = adaptive_step(d, d.shift(gx, gy), action)
        assert (sx == 0) == (ux == 0) and (sy == 0) == (uy == 0)
        assert sign(sx) == ux and sign(sy) == uy


def test_adaptive_step_properties_random(rng):
    for _ in range(20000):
        d, g = random_pair(rng)
        action = Action(int(rng.integers(NUM_ACTIONS)))
        lam = adaptive_step(d, g, action)
        cap = max_reliable_step(d)
        for u, step, dist, big in zip(DIRECTIONS[action], lam, (g.xa - d.xa, g.ya - d.ya), cap):
            if u == 0:
                assert step == 0
                continue
            assert abs(step) <= big  # never beyond the reliable distance
            if 0 < u * dist < big:
                assert step == dist  # capped: lands exactly on the goal coordinate
            else:
                assert step == u * big  # otherwise the full reliable step
            if u * dist > 0:
                assert 0 <= (dist - step) * u  # never overshoots the goal


def test_adaptive_step_equals_reference_parameterized_step(rng):
    """Paper's indicator form == reference code's snap-to-goal form."""
    for _ in range(20000):
        d, g = random_pair(rng)
        action = Action(int(rng.integers(NUM_ACTIONS)))
        assert adaptive_step(d, g, action) == reference_step(d, g, action, parm_step=True), (
            d,
            g,
            action,
        )


# ===================================================================== unit_step
def test_unit_step_single_step_ignores_goal(rng):
    for _ in range(2000):
        d, g = random_pair(rng)
        action = Action(int(rng.integers(NUM_ACTIONS)))
        assert unit_step(d, g, action) == DIRECTIONS[action]
        # reference code without the parameterized action space (b_parm_step=False)
        assert unit_step(d, g, action, 1) == reference_step(d, g, action, parm_step=False)


def test_unit_step_double_step_caps_one_mc_short_goal():
    d = Droplet.at(10, 10, 2, 2)
    assert unit_step(d, d.shift(9, 9), Action.NE, 2) == (2, 2)
    assert unit_step(d, d.shift(1, 1), Action.NE, 2) == (1, 1)
    assert unit_step(d, d.shift(1, 5), Action.NE, 2) == (1, 2)
    assert unit_step(d, d.shift(-1, 0), Action.W, 2) == (-1, 0)
    assert unit_step(d, d.shift(0, -2), Action.S, 2) == (0, -2)
    assert unit_step(d, d.shift(0, 1), Action.S, 2) == (0, -2)  # goal behind: full step


def test_unit_step_properties_random(rng):
    for _ in range(5000):
        d, g = random_pair(rng)
        action = Action(int(rng.integers(NUM_ACTIONS)))
        m = int(rng.integers(1, 5))
        step = unit_step(d, g, action, m)
        for u, s, dist in zip(DIRECTIONS[action], step, (g.xa - d.xa, g.ya - d.ya)):
            if u == 0:
                assert s == 0
            elif 0 < u * dist < m:
                assert s == dist
            else:
                assert s == u * m
        # for square droplets Algorithm 1 is a fixed step of Lambda MCs
        if d.width == d.height and d.width >= 2:
            lam = max_reliable_step(d)[0]
            assert unit_step(d, g, action, lam) == adaptive_step(d, g, action)


# ============================================================== droplet sizes
def test_paper_droplet_sizes():
    """Sec. IV-A: w, h in {2..6} with w / h in [0.8, 1.25] (exact rational test)."""
    expected = {
        (w, h)
        for w in range(2, 7)
        for h in range(2, 7)
        if Fraction(4, 5) <= Fraction(w, h) <= Fraction(5, 4)
    }
    assert expected == {(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (4, 5), (5, 4), (5, 6), (6, 5)}
    assert len(PAPER_DROPLET_SIZES) == 9 == len(set(PAPER_DROPLET_SIZES))
    assert set(PAPER_DROPLET_SIZES) == expected
    # the default training distribution uses all of them
    assert set(map(tuple, JobSamplerConfig().droplet_sizes)) == expected
