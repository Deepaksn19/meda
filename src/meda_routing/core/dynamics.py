"""One control cycle of a routing job, independent of any RL wrapper.

:func:`plan_move` turns a direction into the actuation pattern (target
footprint) and :func:`execute_move` samples the droplet's resulting location.
The Gymnasium environment and the multi-droplet bioassay scheduler both use
these functions, so every policy is evaluated with exactly the same physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .actions import DIRECTIONS, Action, adaptive_step, unit_step
from .geometry import Droplet, Rect
from .movement import action_flags, clamp_into, footprint_mask, is_valid_action, sample_move


@dataclass(frozen=True)
class MovePlan:
    action: Action
    valid: bool
    #: Target droplet location whose footprint is actuated (``U``).
    target: Droplet
    #: ``(west, south, east, north)`` routing-zone collisions of an invalid action.
    collision: Tuple[bool, bool, bool, bool]


def step_size(
    droplet: Droplet, goal: Rect, action: Action, adaptive: bool = True, fixed_step: int = 1
) -> Tuple[int, int]:
    """Signed step ``(lambda_x, lambda_y)``: Algorithm 1 or a fixed step."""
    if adaptive:
        return adaptive_step(droplet, goal, action)
    return unit_step(droplet, goal, action, fixed_step)


def plan_move(
    droplet: Droplet,
    goal: Rect,
    hazard: Rect,
    action: Action,
    adaptive: bool = True,
    fixed_step: int = 1,
) -> MovePlan:
    action = Action(int(action))
    if not is_valid_action(droplet, hazard, action):
        ux, uy = DIRECTIONS[action]
        collision = (
            ux < 0 and droplet.xa <= hazard.xa,
            uy < 0 and droplet.ya <= hazard.ya,
            ux > 0 and droplet.xb >= hazard.xb,
            uy > 0 and droplet.yb >= hazard.yb,
        )
        return MovePlan(action, False, droplet, collision)
    step = step_size(droplet, goal, action, adaptive, fixed_step)
    target = clamp_into(droplet.shift(*step), hazard)
    return MovePlan(action, True, target, (False, False, False, False))


def execute_move(
    plan: MovePlan,
    droplet: Droplet,
    hazard: Rect,
    degradation: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[Droplet, np.ndarray]:
    """Sample the new droplet location; returns it with the actuation pattern."""
    pattern = footprint_mask(degradation.shape, plan.target)
    if plan.valid and plan.target != droplet:
        droplet = sample_move(degradation, pattern, droplet, hazard, action_flags(plan.action), rng)
    return droplet, pattern
