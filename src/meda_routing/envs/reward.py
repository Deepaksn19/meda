"""Reward function (Sec. III-D).

The paper defines ``r = a_dis * r_dis + a_ter * r_ter + a_act * r_act`` with

* ``r_dis = D(delta_k, delta_g) - D(delta_{k+1}, delta_g)`` (Manhattan progress),
* ``r_ter = 1{delta = delta_g}`` (terminal bonus),
* ``r_act`` penalizing invalid actions (leaving the routing zone),

but does not publish the coefficients.  The defaults below are those of the
authors' reference implementation (``_getRewardH`` in ``melfar87/MEDA``),
which weights progress (0.5) and regress (0.8) asymmetrically and charges a
penalty of 1 for every cycle without progress.  Setting
``alpha_dis_away = alpha_dis`` and ``stall_penalty = 0`` gives the plain
linear form of the paper.  Timeouts are *not* penalized (Sec. III-D: the
remaining number of cycles is not observable, so penalizing it would cause
state aliasing).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    alpha_dis: float = 0.5
    alpha_dis_away: float = 0.8
    stall_penalty: float = 1.0
    alpha_ter: float = 100.0
    alpha_act: float = 1.0


def compute_reward(
    config: RewardConfig, prev_dist: int, curr_dist: int, at_goal: bool, invalid: bool
) -> float:
    progress = prev_dist - curr_dist  # r_dis
    if progress > 0:
        reward = config.alpha_dis * progress
    else:
        reward = config.alpha_dis_away * progress - config.stall_penalty
    if at_goal:
        reward += config.alpha_ter  # a_ter * r_ter
    if invalid:
        reward -= config.alpha_act  # a_act * r_act with r_act = -1
    return float(reward)
