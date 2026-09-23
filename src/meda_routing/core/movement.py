"""Probabilistic droplet movement on a MEDA biochip.

The paper (Sec. III-A) adopts the probabilistic transition model of Elfar et
al. [18]: the probability that a movement succeeds depends on the health of
the *frontier set*, the actuated MCs adjacent to the droplet boundary in the
direction of motion.  The paper does not spell the model out; we follow the
authors' reference implementation (``melfar87/MEDA``, ``envs/meda.py``,
``MEDAEnv._updatePattern``):

1. The actuation pattern ``U`` is the footprint of the *target* droplet
   location (the current location shifted by the step computed from
   Algorithm 1, clamped to the routing zone).
2. The droplet then advances one MC at a time towards the target.  In every
   iteration each still-active axis (N/S and E/W, handled independently for
   ordinal moves) advances with probability

       p = mean(D_ij for (i, j) in frontier ∩ U)

   where the frontier is the row (column) directly beyond the droplet's
   leading edge, extended by one MC on either side.  An axis whose frontier
   contains no actuated MC (target reached) or that fails its draw stops for
   the rest of the control cycle.

``D`` is the *true* degradation (the physics), not the quantized health
reading seen by the agent.

:func:`sample_move` draws one outcome; :func:`move_distribution` enumerates
all outcomes with their exact probabilities (used by the model-based
baseline and by the tests).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from .actions import DIRECTIONS, Action
from .geometry import Droplet, Rect

# (y-direction, x-direction) flags, each in {-1, 0, +1}
Flags = Tuple[int, int]


def footprint_mask(shape: Tuple[int, int], rect: Rect) -> np.ndarray:
    """Actuation pattern ``U`` (bool ``W x H``) covering ``rect``."""
    mask = np.zeros(shape, dtype=bool)
    sx, sy = rect.slices()
    mask[sx, sy] = True
    return mask


def _frontier_prob(
    degradation: np.ndarray,
    pattern: np.ndarray,
    droplet: Droplet,
    hazard: Rect,
    axis: str,
    sign: int,
) -> float:
    """Success probability of one unit step of ``droplet`` along ``axis``."""
    width, height = pattern.shape
    if axis == "y":
        if sign > 0:
            if droplet.yb >= hazard.yb:
                return 0.0
            line = droplet.yb + 1
        else:
            if droplet.ya <= hazard.ya:
                return 0.0
            line = droplet.ya - 1
        lo, hi = max(droplet.xa - 1, 0), min(droplet.xb + 1, width - 1)
        front = pattern[lo : hi + 1, line]
        health = degradation[lo : hi + 1, line]
    else:
        if sign > 0:
            if droplet.xb >= hazard.xb:
                return 0.0
            line = droplet.xb + 1
        else:
            if droplet.xa <= hazard.xa:
                return 0.0
            line = droplet.xa - 1
        lo, hi = max(droplet.ya - 1, 0), min(droplet.yb + 1, height - 1)
        front = pattern[line, lo : hi + 1]
        health = degradation[line, lo : hi + 1]
    count = int(np.count_nonzero(front))
    if count == 0:
        return 0.0
    return float(health[front].sum() / count)


def unit_step_probs(
    degradation: np.ndarray,
    pattern: np.ndarray,
    droplet: Droplet,
    hazard: Rect,
    flags: Flags,
) -> Tuple[float, float]:
    """Per-axis success probabilities ``(p_y, p_x)`` for the active axes."""
    fy, fx = flags
    py = _frontier_prob(degradation, pattern, droplet, hazard, "y", fy) if fy else 0.0
    px = _frontier_prob(degradation, pattern, droplet, hazard, "x", fx) if fx else 0.0
    return py, px


def action_flags(action: Action) -> Flags:
    ux, uy = DIRECTIONS[Action(action)]
    return uy, ux


def sample_move(
    degradation: np.ndarray,
    pattern: np.ndarray,
    droplet: Droplet,
    hazard: Rect,
    flags: Flags,
    rng: np.random.Generator,
) -> Droplet:
    """Simulate one control cycle and return the resulting droplet location."""
    fy, fx = flags
    while fy or fx:
        py, px = unit_step_probs(degradation, pattern, droplet, hazard, (fy, fx))
        if fy:
            fy = fy if rng.random() < py else 0
        if fx:
            fx = fx if rng.random() < px else 0
        droplet = droplet.shift(fx, fy)
    return droplet


def move_distribution(
    degradation: np.ndarray,
    pattern: np.ndarray,
    droplet: Droplet,
    hazard: Rect,
    flags: Flags,
    prob_fn: Callable[..., Tuple[float, float]] = unit_step_probs,
) -> Dict[Droplet, float]:
    """Exact distribution over resulting droplet locations for one cycle."""
    result: Dict[Droplet, float] = {}
    frontier = {(droplet, flags): 1.0}
    while frontier:
        nxt: Dict[Tuple[Droplet, Flags], float] = {}
        for (d, (fy, fx)), mass in frontier.items():
            if not fy and not fx:
                result[d] = result.get(d, 0.0) + mass
                continue
            py, px = prob_fn(degradation, pattern, d, hazard, (fy, fx))
            y_outcomes = [(fy, py), (0, 1.0 - py)] if fy else [(0, 1.0)]
            x_outcomes = [(fx, px), (0, 1.0 - px)] if fx else [(0, 1.0)]
            for ny, qy in y_outcomes:
                for nx, qx in x_outcomes:
                    q = mass * qy * qx
                    if q <= 0.0:
                        continue
                    key = (d.shift(nx, ny), (ny, nx))
                    nxt[key] = nxt.get(key, 0.0) + q
        frontier = nxt
    return result


def clamp_into(rect: Droplet, zone: Rect) -> Droplet:
    """Shift ``rect`` (without resizing) so that it lies inside ``zone``.

    Mirrors the reference implementation, which clamps an over-long step to
    the routing zone instead of rejecting it.
    """
    dx = dy = 0
    if rect.xa < zone.xa:
        dx = zone.xa - rect.xa
    elif rect.xb > zone.xb:
        dx = zone.xb - rect.xb
    if rect.ya < zone.ya:
        dy = zone.ya - rect.ya
    elif rect.yb > zone.yb:
        dy = zone.yb - rect.yb
    return rect.shift(dx, dy)


def is_valid_action(droplet: Droplet, hazard: Rect, action: Action) -> bool:
    """An action is invalid if it pushes the droplet out of the routing zone.

    Following the reference implementation, an action is invalid when the
    droplet already touches the routing-zone boundary in (any of) the
    requested direction(s); the droplet then holds its position and the agent
    receives the action penalty ``r_act``.
    """
    ux, uy = DIRECTIONS[Action(action)]
    if uy > 0 and droplet.yb >= hazard.yb:
        return False
    if uy < 0 and droplet.ya <= hazard.ya:
        return False
    if ux > 0 and droplet.xb >= hazard.xb:
        return False
    if ux < 0 and droplet.xa <= hazard.xa:
        return False
    return True
