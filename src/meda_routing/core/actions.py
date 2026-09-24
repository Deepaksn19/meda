"""Parameterized action space (Sec. III-B, Algorithm 1).

The agent chooses only a *direction* ``a`` from
``A = {aN, aS, aE, aW, aNE, aNW, aSE, aSW}``; the number of MCs the droplet is
moved in each axis (the signed distance ``(lambda_x, lambda_y)``) is derived
from the droplet size and its position relative to the goal by
:func:`adaptive_step` (Algorithm 1).  This keeps ``|A| = 8`` for every droplet
size while still allowing moves of more than two MCs per control cycle.

Where each default comes from is tagged next to it:

* ``[PAPER ...]`` -- the value is stated in the paper (section, figure, table).
* ``[REF-CODE]`` -- not in the paper; taken from the first author's public
  code ``melfar87/MEDA`` (incl. the Stable-Baselines PPO2 defaults, saved
  model and training log of that code).
* ``[ASSUMED]`` -- not fixed by the paper or the reference code; our choice.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, Tuple

from .geometry import Droplet, Rect


class Action(IntEnum):
    """Movement directions, in the order listed in the paper."""

    N = 0
    S = 1
    E = 2
    W = 3
    NE = 4
    NW = 5
    SE = 6
    SW = 7


NUM_ACTIONS = len(Action)  # [PAPER Sec. III-B] 8 directions: N, S, E, W, NE, NW, SE, SW

#: Unit direction ``(ux, uy)`` of every action; north is ``+y``, east is ``+x``.
DIRECTIONS: Dict[Action, Tuple[int, int]] = {
    Action.N: (0, 1),
    Action.S: (0, -1),
    Action.E: (1, 0),
    Action.W: (-1, 0),
    Action.NE: (1, 1),
    Action.NW: (-1, 1),
    Action.SE: (1, -1),
    Action.SW: (-1, -1),
}

_NORTH = {Action.N, Action.NE, Action.NW}
_SOUTH = {Action.S, Action.SE, Action.SW}
_EAST = {Action.E, Action.NE, Action.SE}
_WEST = {Action.W, Action.NW, Action.SW}


def _ind(condition: bool) -> int:
    """Indicator function ``1{.}``."""
    return 1 if condition else 0


def max_reliable_step(droplet: Rect) -> Tuple[int, int]:
    """``(Lambda_x, Lambda_y) = (floor((xb-xa+1)/2), floor((yb-ya+1)/2))``.

    The largest per-axis displacement that still leaves an overlap of at least
    half the droplet with its current footprint (Example 1 of the paper: a
    ``4 x 3`` droplet can move ``(2, 1)`` MCs per cycle).
    """
    return (droplet.xb - droplet.xa + 1) // 2, (droplet.yb - droplet.ya + 1) // 2


def adaptive_step(droplet: Rect, goal: Rect, action: Action) -> Tuple[int, int]:
    """Algorithm 1 — signed distance ``(lambda_x, lambda_y)`` for ``action``.

    The step along an axis is the maximum reliable step ``Lambda`` in the
    requested direction, capped to the remaining distance to the goal when the
    goal lies strictly between 0 and ``Lambda`` MCs away in that direction, so
    that the droplet never overshoots the goal.
    """
    # The whole step rule is [PAPER Algorithm 1]: Lambda = floor(size / 2), capped at the goal.
    action = Action(action)
    lam_x, lam_y = 0, 0
    dx, dy = goal.xa - droplet.xa, goal.ya - droplet.ya  # line 2
    cap_x, cap_y = max_reliable_step(droplet)  # line 3: (Lambda_x, Lambda_y)
    if action in _NORTH:  # line 4
        lam_y = cap_y - (cap_y - dy) * _ind(0 < dy < cap_y)
    if action in _SOUTH:  # line 5
        lam_y = -cap_y + (cap_y + dy) * _ind(-cap_y < dy < 0)
    if action in _EAST:  # line 6
        lam_x = cap_x - (cap_x - dx) * _ind(0 < dx < cap_x)
    if action in _WEST:  # line 7
        lam_x = -cap_x + (cap_x + dx) * _ind(-cap_x < dx < 0)
    return lam_x, lam_y  # line 8


def unit_step(droplet: Rect, goal: Rect, action: Action, max_step: int = 1) -> Tuple[int, int]:
    """Non-adaptive step: ``max_step`` MCs per axis, never overshooting the goal.

    Used for ablations of the parameterized action space and by baselines that
    only support single- (``max_step=1``) or double-step (``max_step=2``)
    movements, as in the formal-synthesis model of [18].
    """
    ux, uy = DIRECTIONS[Action(action)]
    dx, dy = goal.xa - droplet.xa, goal.ya - droplet.ya

    def _axis(u: int, d: int) -> int:
        if u == 0:
            return 0
        step = max_step
        if 0 < u * d < step:  # goal strictly inside the step range: don't overshoot
            step = u * d
        return u * step

    return _axis(ux, dx), _axis(uy, dy)


def apply_step(droplet: Droplet, step: Tuple[int, int]) -> Droplet:
    return droplet.shift(step[0], step[1])
