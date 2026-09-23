"""Gymnasium environment for single-droplet routing (Sec. III, IV; Algorithm 2).

Cross-checked against the paper and the authors' reference ``MEDAEnv``
(``melfar87/MEDA``, ``envs/meda.py``):

* API and spaces: gymnasium's and Stable-Baselines3's environment checkers;
* observation (Sec. III-C, Fig. 2): channel 0 holds ``H / 2**b`` inside the
  hazard bounds and 0 outside, channel 1 the droplet, channel 2 the goal;
  unified size through ``cv2.INTER_AREA`` resampling (Sec. IV-C);
* reward (Sec. III-D) with the coefficients of the reference ``_getRewardH``;
* episode end (Algorithm 2, line 9): termination at the goal, truncation once
  ``k >= k_max = ceil(alpha * (W_h + H_h))`` (Sec. IV-B);
* wear bookkeeping ``n_ij += U_ij`` every cycle, including the holding
  pattern of an invalid action;
* per-episode fault injection (Sec. V-B) and hidden defects (Sec. VI-C);
* bioassay mode: an externally owned chip whose wear persists across jobs.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from typing import Any, List

import cv2
import gymnasium as gym
import numpy as np
import pytest
import yaml
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from meda_routing.core.actions import Action
from meda_routing.core.biochip import DegradationConfig, MEDABiochip
from meda_routing.core.dynamics import plan_move
from meda_routing.core.geometry import Droplet, Rect, chip_rect
from meda_routing.core.jobs import (
    PAPER_DROPLET_SIZES,
    JobSampler,
    JobSamplerConfig,
    RoutingJob,
    hazard_bounds,
)
from meda_routing.core.movement import footprint_mask
from meda_routing.envs import ENV_ID, EnvConfig, MEDARoutingEnv, RewardConfig
from meda_routing.routers.base import Router, RoutingState, run_job

WORN = {"initial_actuations": "uniform", "max_initial_actuations": 2500}


# ==================================================================== helpers
def greedy_action(droplet: Droplet, goal: Droplet) -> Action:
    """Direction of the goal (north if already there)."""
    sx = (goal.xa > droplet.xa) - (goal.xa < droplet.xa)
    sy = (goal.ya > droplet.ya) - (goal.ya < droplet.ya)
    table = {
        (0, 1): Action.N, (0, -1): Action.S, (1, 0): Action.E, (-1, 0): Action.W,
        (1, 1): Action.NE, (-1, 1): Action.NW, (1, -1): Action.SE, (-1, -1): Action.SW,
    }  # fmt: skip
    return table.get((sx, sy), Action.N)


def expected_native_obs(chip: MEDABiochip, droplet: Rect, goal: Rect, hazard: Rect) -> np.ndarray:
    """Fig. 2 built MC by MC: ``(3, rows = y, cols = x)``."""
    health, levels = chip.health(), chip.health_levels
    width, height = health.shape
    obs = np.zeros((3, height, width), dtype=np.float32)
    for x in range(width):
        for y in range(height):
            if hazard.contains_point(x, y):
                obs[0, y, x] = health[x, y] / levels
            obs[1, y, x] = float(droplet.contains_point(x, y))
            obs[2, y, x] = float(goal.contains_point(x, y))
    return obs


def reference_reward_h(prev_dist: int, curr_dist: int, at_goal: bool, valid: bool) -> float:
    """``MEDAEnv._getRewardH`` of the reference code (timeouts add 0)."""
    reward = 0.0
    if at_goal:
        reward += 100.0
    if curr_dist < prev_dist:
        reward += 0.5 * (prev_dist - curr_dist)
    else:
        reward += 0.8 * (prev_dist - curr_dist) - 1.0
    if not valid:
        reward += -1.0
    return reward


def same(a: Any, b: Any) -> bool:
    """Exact recursive equality of nested tuples / lists / dicts / arrays."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return bool(a == b)


# Reward scenarios on a 24 x 16 chip: 4 x 4 droplet (Lambda = (2, 2)), goal 10 MCs east
W24, H16 = 24, 16
START, GOAL = Droplet.at(4, 4, 4, 4), Droplet.at(14, 4, 4, 4)
JOB = RoutingJob(START, GOAL, hazard_bounds(START, GOAL, W24, H16))  # zone (1, 1, 20, 10)


def reset_on(env: MEDARoutingEnv, job: RoutingJob, chip: MEDABiochip, **options: Any):
    return env.reset(seed=0, options={"job": job, "chip": chip, **options})


# =============================================================== API checkers
@pytest.mark.parametrize(
    "config",
    [
        dict(width=12, height=10, obs_size=None),
        dict(
            width=14, height=9, obs_size=[30, 30], fault_fraction=0.1, hidden_defect_fraction=0.05
        ),
        dict(
            width=10,
            height=10,
            obs_size=[16, 12],
            mark_collisions=True,
            kmax_alpha=2.0,
            degradation=WORN,
        ),
    ],
    ids=["native", "resized-with-faults", "custom-size-worn"],
)
def test_gymnasium_env_checker(config):
    env = gym.make(ENV_ID, config=config, render_mode="rgb_array")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gym_check_env(env.unwrapped)  # the spec is set: seeding determinism is checked too
        assert [str(w.message) for w in caught] == []
    finally:
        env.close()


@pytest.mark.parametrize(
    "config",
    [
        dict(width=12, height=10, obs_size=[30, 30]),
        dict(width=14, height=9, obs_size=None, fault_fraction=0.1),
    ],
    ids=["resized", "native"],
)
def test_sb3_env_checker(make_env, config):
    env = make_env(config)
    with warnings.catch_warnings():
        # float32 [0, 1] images for a custom CNN extractor are intended
        warnings.filterwarnings("ignore", message=".*image.*", category=UserWarning)
        sb3_check_env(env, warn=True)


def test_spaces_and_info(make_env):
    env = make_env()  # paper defaults: 30 x 30 chip, 30 x 30 observation
    assert env.action_space == gym.spaces.Discrete(8)
    space = env.observation_space
    assert space.shape == (3, 30, 30) and space.dtype == np.float32
    assert float(space.low.min()) == 0.0 and float(space.high.max()) == 1.0
    assert make_env(width=20, height=12, obs_size=None).observation_space.shape == (3, 12, 20)
    assert make_env(width=60, height=30, obs_size=(30, 15)).observation_space.shape == (3, 15, 30)
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, 30, 30) and obs.dtype == np.float32 and obs.flags["C_CONTIGUOUS"]
    assert info == {
        "is_success": False,
        "num_cycles": 0,
        "invalid_action": False,
        "distance": env.droplet.manhattan(env.goal),
        "droplet": env.job.start.as_tuple(),
        "goal": env.goal.as_tuple(),
        "k_max": env.k_max,
    }
    assert np.array_equal(env.get_observation(), obs)


def test_render_rgb_array(make_env, ideal_chip_factory):
    env = make_env(width=12, height=8, obs_size=None, render_mode="rgb_array")
    job = RoutingJob(Droplet.at(0, 0, 2, 2), Droplet.at(9, 5, 2, 2), chip_rect(12, 8))
    reset_on(env, job, ideal_chip_factory(12, 8))
    frame = env.render()
    assert frame.shape == (8 * 8, 12 * 8, 3) and frame.dtype == np.uint8
    # north up: the droplet (blue) is bottom-left, the goal (green) top-right
    r, g, b = (int(v) for v in frame[-4, 4])
    assert b > r and b > g
    r, g, b = (int(v) for v in frame[12, 76])
    assert g > r and g > b
    assert make_env(width=12, height=8).render() is None  # no render mode


# ================================================================ observation
@pytest.mark.parametrize("bits", [2, 3])
def test_observation_channels_native(make_env, bits):
    env = make_env(
        width=16, height=12, obs_size=None, fault_fraction=0.15,
        degradation={"health_bits": bits, **WORN},
    )  # fmt: skip
    start, goal = Droplet.at(2, 2, 2, 2), Droplet.at(6, 5, 2, 2)
    job = RoutingJob(start, goal, hazard_bounds(start, goal, 16, 12, margin=2))  # zone (0, 0, 9, 8)
    obs, _ = env.reset(seed=bits, options={"job": job})  # env-owned chip: worn, faults injected
    levels = 2**bits
    outside = np.ones((12, 16), dtype=bool)
    outside[0:9, 0:10] = False
    for t in range(8):
        np.testing.assert_array_equal(
            obs, expected_native_obs(env.chip, env.droplet, env.goal, env.hazard)
        )
        scaled = obs[0] * levels  # H in {0, ..., 2**b - 1}, scaled by 2**b into [0, 1)
        assert np.array_equal(scaled, np.round(scaled)) and scaled.max() <= levels - 1
        assert len(np.unique(scaled[~outside])) >= 3  # several health levels are visible
        assert not obs[0][outside].any()  # routing zone encoded by masking
        assert np.all(obs[0][env.chip.faults.T & ~outside] == 0.0)
        assert obs[1].sum() == env.droplet.area and obs[2].sum() == env.goal.area
        assert set(np.unique(obs[1:]).tolist()) <= {0.0, 1.0}
        obs, *_ = env.step(t)


def test_default_config_observation_is_not_resampled(make_env):
    env = make_env(degradation=WORN)  # 30 x 30 chip, 30 x 30 observation
    obs, _ = env.reset(seed=4)
    np.testing.assert_array_equal(
        obs, expected_native_obs(env.chip, env.droplet, env.goal, env.hazard)
    )


@pytest.mark.parametrize(
    "width, height, obs_size",
    [(60, 60, (30, 30)), (60, 30, (30, 30)), (60, 30, (30, 15)), (90, 60, (30, 20))],
)
def test_observation_resized_by_pixel_area_relation(make_env, width, height, obs_size):
    """Integer downscaling with INTER_AREA is an exact block average."""
    env = make_env(
        width=width, height=height, obs_size=obs_size, fault_fraction=0.1, degradation=WORN
    )
    obs, _ = env.reset(seed=1)
    for t in range(3):
        native = expected_native_obs(env.chip, env.droplet, env.goal, env.hazard)
        fx, fy = width // obs_size[0], height // obs_size[1]
        expected = native.reshape(3, obs_size[1], fy, obs_size[0], fx).mean(axis=(2, 4))
        assert obs.shape == (3, obs_size[1], obs_size[0]) == env.observation_space.shape
        np.testing.assert_allclose(obs, expected, rtol=0, atol=1e-6)
        assert obs.min() >= 0.0 and obs.max() <= 1.0
        obs, *_ = env.step(t + 2)


@pytest.mark.parametrize("obs_size", [(30, 30), (30, 20), (40, 25)])
def test_observation_resized_non_integer_factor(make_env, obs_size):
    env = make_env(width=24, height=18, obs_size=obs_size, degradation=WORN)
    obs, _ = env.reset(seed=2)
    native = expected_native_obs(env.chip, env.droplet, env.goal, env.hazard)
    expected = cv2.resize(native.transpose(1, 2, 0), obs_size, interpolation=cv2.INTER_AREA)
    np.testing.assert_allclose(obs, expected.transpose(2, 0, 1), rtol=0, atol=1e-6)


# ===================================================================== reward
@pytest.mark.parametrize(
    "action, moved, reward",
    [
        (Action.E, (2, 0), 0.5 * 2),  # progress: alpha_dis * r_dis
        (Action.W, (-2, 0), 0.8 * -2 - 1.0),  # regress: alpha_dis_away * r_dis - stall penalty
        (Action.N, (0, 2), 0.8 * -2 - 1.0),
        (Action.NE, (2, 2), 0.8 * 0 - 1.0),  # no net progress
        (Action.SE, (2, -2), 0.8 * 0 - 1.0),
        (Action.NW, (-2, 2), 0.8 * -4 - 1.0),
    ],
)
def test_reward_progress_regress_stall(make_env, ideal_chip_factory, action, moved, reward):
    env = make_env(width=W24, height=H16, obs_size=None)
    reset_on(env, JOB, ideal_chip_factory(W24, H16))
    _, r, terminated, truncated, info = env.step(action)
    assert env.droplet == START.shift(*moved)
    assert r == pytest.approx(reward)
    assert not terminated and not truncated and not info["invalid_action"]
    assert info["distance"] == env.droplet.manhattan(GOAL) and info["num_cycles"] == 1


def test_reward_blocked_move_and_invalid_actions(make_env, ideal_chip_factory):
    env = make_env(width=W24, height=H16, obs_size=None)
    chip = ideal_chip_factory(W24, H16)
    chip.set_faults([(8, y) for y in range(3, 9)])  # frontier of the eastward move
    reset_on(env, JOB, chip)
    _, r, _, _, info = env.step(Action.E)
    assert env.droplet == START and not info["invalid_action"]
    assert r == pytest.approx(-1.0)  # stall penalty only
    # droplet in the south-west corner of its routing zone
    corner = RoutingJob(Droplet.at(1, 1, 4, 4), GOAL, Rect(1, 1, 20, 10))
    reset_on(env, corner, ideal_chip_factory(W24, H16))
    for action in (Action.W, Action.S, Action.SW, Action.NW, Action.SE):
        _, r, terminated, truncated, info = env.step(action)
        assert info["invalid_action"] and env.droplet == corner.start
        assert r == pytest.approx(0.8 * 0 - 1.0 - 1.0)  # stall + action penalty
        assert not terminated and not truncated
    _, r, _, _, info = env.step(Action.NE)  # away from both boundaries: valid again
    assert not info["invalid_action"] and env.droplet == Droplet.at(3, 3, 4, 4)
    assert r == pytest.approx(0.5 * (16 - 12))


@pytest.mark.parametrize("start_x, reward", [(12, 100.0 + 0.5 * 2), (13, 100.0 + 0.5 * 1)])
def test_reward_and_termination_at_goal(make_env, ideal_chip_factory, start_x, reward):
    start = Droplet.at(start_x, 4, 4, 4)
    job = RoutingJob(start, GOAL, hazard_bounds(start, GOAL, W24, H16))
    env = make_env(width=W24, height=H16, obs_size=None)
    reset_on(env, job, ideal_chip_factory(W24, H16))
    _, r, terminated, truncated, info = env.step(Action.E)  # capped by Algorithm 1 for x = 13
    assert env.droplet == GOAL and r == pytest.approx(reward)
    assert terminated and not truncated and info["is_success"] and info["distance"] == 0


def test_reward_linear_form_of_paper(make_env, ideal_chip_factory):
    """Paper form ``r = a_dis r_dis + a_ter r_ter + a_act r_act``.

    Obtained with ``alpha_dis_away = alpha_dis`` and no stall penalty.
    """
    reward = {
        "alpha_dis": 0.5,
        "alpha_dis_away": 0.5,
        "stall_penalty": 0.0,
        "alpha_ter": 50.0,
        "alpha_act": 2.0,
    }
    env = make_env(width=W24, height=H16, obs_size=None, reward=reward)
    assert env.config.reward == RewardConfig(**reward)
    expected = {Action.E: 1.0, Action.W: -1.0, Action.NE: 0.0, Action.NW: -2.0}
    for action, value in expected.items():
        reset_on(env, JOB, ideal_chip_factory(W24, H16))
        assert env.step(action)[1] == pytest.approx(value)
    reset_on(
        env,
        RoutingJob(Droplet.at(1, 1, 4, 4), GOAL, Rect(1, 1, 20, 10)),
        ideal_chip_factory(W24, H16),
    )
    assert env.step(Action.W)[1] == pytest.approx(-2.0)
    start = Droplet.at(12, 4, 4, 4)
    reset_on(
        env,
        RoutingJob(start, GOAL, hazard_bounds(start, GOAL, W24, H16)),
        ideal_chip_factory(W24, H16),
    )
    assert env.step(Action.E)[1] == pytest.approx(50.0 + 1.0)


def test_rewards_match_reference_on_random_rollouts(make_env):
    env = make_env(width=16, height=14, obs_size=None, fault_fraction=0.1, degradation=WORN)
    rng = np.random.default_rng(5)
    _, info = env.reset(seed=5)
    kinds: Counter = Counter()
    for _ in range(1500):
        greedy = rng.random() < 0.6
        action = greedy_action(env.droplet, env.goal) if greedy else Action(int(rng.integers(8)))
        prev = info["distance"]
        _, r, terminated, truncated, info = env.step(action)
        curr, at_goal, valid = info["distance"], info["is_success"], not info["invalid_action"]
        assert r == pytest.approx(reference_reward_h(prev, curr, at_goal, valid), abs=1e-12)
        if at_goal:
            kinds["goal"] += 1
        elif curr != prev:
            kinds["progress" if curr < prev else "regress"] += 1
        else:
            kinds["stall"] += 1
        kinds["invalid"] += not valid
        if terminated or truncated:
            kinds["truncated"] += truncated
            _, info = env.reset()
    assert all(kinds[k] > 0 for k in ("goal", "progress", "stall", "regress", "invalid")), kinds


# ============================================== termination and truncation
@pytest.mark.parametrize("alpha, k_max", [(1.0, 20), (1.26, 26), (1.5, 30), (2.0, 40)])
def test_truncation_at_kmax(make_env, ideal_chip_factory, alpha, k_max):
    # routing zone 10 x 10: W_h + H_h = 20
    job = RoutingJob(Droplet.at(1, 1, 2, 2), Droplet.at(8, 8, 2, 2), Rect(1, 1, 10, 10))
    env = make_env(width=12, height=12, obs_size=None, kmax_alpha=alpha)
    _, info = reset_on(env, job, ideal_chip_factory(12, 12))
    assert env.k_max == info["k_max"] == k_max == math.ceil(alpha * 20)
    for k in range(1, k_max + 1):
        _, r, terminated, truncated, info = env.step(Action.SW)  # invalid: never moves
        assert not terminated  # a timeout is not a terminal state (Sec. III-D)
        assert truncated == (k == k_max)
        assert info["num_cycles"] == k and r == pytest.approx(-2.0)  # no timeout penalty


def test_timeout_terminal_option(make_env, ideal_chip_factory):
    """Optional SB2-style episode end: a timeout is reported as terminal."""
    assert EnvConfig().timeout_terminal is False  # default: truncation (Sec. III-D)
    job = RoutingJob(Droplet.at(1, 1, 2, 2), Droplet.at(8, 8, 2, 2), Rect(1, 1, 10, 10))
    env = make_env(width=12, height=12, obs_size=None, timeout_terminal=True)
    reset_on(env, job, ideal_chip_factory(12, 12))
    for k in range(1, 21):
        _, r, terminated, truncated, info = env.step(Action.SW)
        assert not truncated and terminated == (k == 20)
        assert r == pytest.approx(-2.0)  # still no timeout penalty
    assert not info["is_success"]


def test_kmax_uses_hazard_zone_of_sampled_jobs(make_env):
    for alpha in (1.0, 1.5, 2.0):
        env = make_env(width=30, height=30, kmax_alpha=alpha)
        env.reset(seed=3)
        for _ in range(20):
            _, info = env.reset()
            zone = env.hazard
            assert zone == hazard_bounds(env.job.start, env.goal, 30, 30, margin=3)
            assert env.k_max == info["k_max"] == math.ceil(alpha * (zone.width + zone.height))


def test_goal_on_last_cycle_terminates_not_truncates(make_env, ideal_chip_factory):
    start = Droplet.at(12, 4, 4, 4)
    job = RoutingJob(start, GOAL, hazard_bounds(start, GOAL, W24, H16))
    env = make_env(width=W24, height=H16, obs_size=None)
    _, info = reset_on(env, job, ideal_chip_factory(W24, H16), k_max=1)
    assert env.k_max == info["k_max"] == 1
    _, _, terminated, truncated, info = env.step(Action.E)
    assert terminated and not truncated and info["is_success"]


def test_rect_endpoints_still_terminate(make_env, ideal_chip_factory):
    """``droplet == goal`` needs Droplets; RoutingJob converts Rect endpoints."""
    start = Rect(12, 4, 15, 7)
    job = RoutingJob(start, Rect(*GOAL.as_tuple()), hazard_bounds(start, GOAL, W24, H16))
    env = make_env(width=W24, height=H16, obs_size=None)
    reset_on(env, job, ideal_chip_factory(W24, H16))
    assert type(env.droplet) is Droplet and type(env.goal) is Droplet
    assert env.step(Action.E)[2]


# ========================================================== wear bookkeeping
def test_actuation_counts_follow_the_actuated_pattern(make_env):
    env = make_env(width=16, height=12, obs_size=None, fault_fraction=0.1, degradation=WORN)
    rng = np.random.default_rng(11)
    env.reset(seed=11)
    invalid = 0
    for _ in range(300):
        droplet, goal, zone = env.droplet, env.goal, env.hazard
        before = env.chip.actuations.copy()
        action = Action(int(rng.integers(8)))
        plan = plan_move(droplet, goal, zone, action)
        _, _, terminated, truncated, info = env.step(action)
        # U = target footprint; an invalid action actuates the holding pattern
        expected = footprint_mask(env.chip.shape, plan.target if plan.valid else droplet)
        assert np.array_equal(env.chip.actuations - before, expected.astype(np.int64))
        assert np.array_equal(env.last_pattern, expected)
        assert info["invalid_action"] == (not plan.valid)
        invalid += not plan.valid
        if terminated or truncated:
            env.reset()
    assert invalid > 0


def test_move_uses_pre_cycle_degradation_and_obs_shows_new_wear(make_env):
    """Reference order: move with the current ``D``, then ``n += U``, then sense ``H``."""
    chip = MEDABiochip(16, 12, rng=np.random.default_rng(0))
    chip.tau[:] = 1e-6
    chip.c[:] = 1.0  # D = 1 before the first actuation, 1e-6 after it
    start, goal = Droplet.at(2, 4, 4, 4), Droplet.at(10, 4, 4, 4)
    env = make_env(width=16, height=12, obs_size=None)
    reset_on(env, RoutingJob(start, goal, hazard_bounds(start, goal, 16, 12)), chip)
    obs, *_ = env.step(Action.E)
    target = start.shift(2, 0)
    assert env.droplet == target
    assert np.all(chip.actuations[target.slices()] == 1) and chip.actuations.sum() == target.area
    assert np.all(obs[0, target.ya : target.yb + 1, target.xa : target.xb + 1] == 0.0)
    assert obs[0, 4, 0] == 0.75  # never actuated: healthy (saturated reading)


# ============================================================== reproducibility
def test_seeded_episodes_are_reproducible(make_env):
    config = dict(
        width=14, height=12, obs_size=[30, 30], fault_fraction=0.1, hidden_defect_fraction=0.05,
        degradation={"initial_actuations": "uniform", "max_initial_actuations": 1200},
    )  # fmt: skip
    actions = [int(a) for a in np.random.default_rng(0).integers(0, 8, size=80)]

    def run(env: MEDARoutingEnv, seed: int) -> List[Any]:
        record: List[Any] = [env.reset(seed=seed)]
        chip = env.chip
        record.append([chip.tau, chip.c, chip.actuations, chip.faults, chip.hidden_defects])
        record[-1] = [arr.copy() for arr in record[-1]]
        for a in actions:
            out = env.step(a)
            record.append(out)
            if out[2] or out[3]:
                record.append(env.reset())
        return record

    first, second = make_env(config), make_env(config)
    a = run(first, 7)
    assert same(a, run(second, 7))  # two instances, same seed
    assert same(a, run(first, 7))  # re-seeding a used instance
    assert not same(a, run(second, 8))


# ================================================ bioassay mode: external chip
def test_external_chip_keeps_wear_and_gets_no_faults(make_env, ideal_chip_factory):
    env = make_env(
        width=16, height=12, obs_size=None, fault_fraction=0.2, hidden_defect_fraction=0.1
    )
    chip = MEDABiochip(16, 12, DegradationConfig(), rng=np.random.default_rng(3))
    chip.reset()
    tau, c = chip.tau.copy(), chip.c.copy()
    job1 = RoutingJob(Droplet.at(1, 1, 3, 3), Droplet.at(10, 7, 3, 3), Rect(0, 0, 15, 11))
    _, info = env.reset(seed=0, options={"job": job1, "chip": chip})
    assert env.chip is chip and env.job == job1 and info["droplet"] == job1.start.as_tuple()
    assert not chip.faults.any() and not chip.hidden_defects.any()  # no injection on a given chip
    for _ in range(6):
        before = chip.actuations.sum()
        env.step(greedy_action(env.droplet, env.goal))
        assert chip.actuations.sum() == before + 9  # a 3 x 3 footprint per cycle
    wear = chip.actuations.copy()
    job2 = RoutingJob(Droplet.at(11, 8, 3, 3), Droplet.at(2, 2, 3, 3), Rect(0, 0, 15, 11))
    env.reset(options={"job": job2, "chip": chip})
    assert env.chip is chip and np.array_equal(chip.actuations, wear)  # wear persists
    assert np.array_equal(chip.tau, tau) and np.array_equal(chip.c, c)
    env.step(Action.SW)
    assert chip.actuations.sum() == wear.sum() + 9
    # back to the env's own (fresh, faulty) chip; the external one is left alone
    frozen = chip.actuations.copy()
    env.reset(seed=1)
    assert env.chip is not chip and env.chip.faults.mean() >= 0.2
    env.step(Action.N)
    assert np.array_equal(chip.actuations, frozen)


def test_persistent_chip_accumulates_wear_across_episodes(make_env):
    """Online adaptation to one chip: parameters and faults drawn once, wear never reset."""
    env = make_env(
        width=14, height=12, obs_size=None, persistent_chip=True,
        fault_fraction=0.1, hidden_defect_fraction=0.05, degradation=WORN,
    )  # fmt: skip
    env.reset(seed=0)
    chip = env.chip
    names = ("tau", "c", "faults", "hidden_defects")
    drawn = {name: getattr(chip, name).copy() for name in names}
    assert drawn["faults"].mean() >= 0.1 and drawn["hidden_defects"].mean() >= 0.05
    total = chip.actuations.sum()
    for episode in range(8):
        for t in range(6):
            *_, terminated, truncated, _ = env.step((episode + t) % 8)
            if terminated or truncated:
                break
        worn = chip.actuations.sum()
        assert worn > total
        env.reset(seed=episode if episode % 2 else None)  # even a seeded reset keeps the chip
        assert env.chip is chip and chip.actuations.sum() == worn
        for name in names:
            assert np.array_equal(getattr(chip, name), drawn[name])
        dead = chip.faults | chip.hidden_defects
        assert not any(dead[d.slices()].any() for d in (env.job.start, env.job.goal))
        total = worn


def test_chip_seed_models_the_same_chip_in_every_instance(make_env):
    config = dict(
        width=14, height=12, obs_size=None, persistent_chip=True, chip_seed=42,
        fault_fraction=0.1, hidden_defect_fraction=0.05, degradation=WORN,
    )  # fmt: skip
    first, second, other = make_env(config), make_env(config), make_env({**config, "chip_seed": 43})
    first.reset(seed=1)
    second.reset(seed=2)
    other.reset(seed=1)
    for name in ("tau", "c", "actuations", "faults", "hidden_defects"):
        assert np.array_equal(getattr(first.chip, name), getattr(second.chip, name))
    assert not np.array_equal(first.chip.tau, other.chip.tau)
    assert not np.array_equal(first.chip.faults, other.chip.faults)


def test_given_job_on_own_chip_protects_its_endpoints(make_env):
    env = make_env(width=12, height=12, obs_size=None, fault_fraction=0.45)
    job = RoutingJob(Droplet.at(1, 1, 4, 4), Droplet.at(7, 6, 4, 4), Rect(0, 0, 11, 11))
    env.reset(seed=0)
    for _ in range(15):
        env.reset(options={"job": job})
        assert env.job == job and env.chip.faults.mean() >= 0.45
        assert not env.chip.faults[job.start.slices()].any()
        assert not env.chip.faults[job.goal.slices()].any()


# ============================================================ fault injection
def test_faults_redrawn_each_episode_and_endpoints_protected(make_env):
    env = make_env(
        width=20, height=20, obs_size=None, fault_fraction=0.25, hidden_defect_fraction=0.05
    )
    env.reset(seed=0)
    seen = set()
    for _ in range(25):
        obs, _ = env.reset()
        faults, hidden = env.chip.faults, env.chip.hidden_defects
        assert math.ceil(0.25 * 400) <= faults.sum() <= math.ceil(0.25 * 400) + 3  # 2 x 2 clusters
        assert hidden.sum() >= math.ceil(0.05 * 400)
        for drop in (env.job.start, env.job.goal):
            assert not faults[drop.slices()].any() and not hidden[drop.slices()].any()
        # visible faults read 0; hidden defects are invisible in the observation
        zone = np.zeros((20, 20), dtype=bool)
        zone[env.hazard.slices()] = True
        assert np.all(obs[0].T[faults & zone] == 0.0)
        assert np.all(obs[0].T[hidden & ~faults & zone] > 0.0)
        assert np.all(env.chip.effective_degradation()[hidden] == 0.0)
        seen.add(faults.tobytes())
    assert len(seen) == 25  # a new fault map every episode (Sec. V-B)


def test_unprotected_endpoints_can_be_faulty(make_env):
    env = make_env(width=20, height=20, obs_size=None, fault_fraction=0.3, protect_endpoints=False)
    env.reset(seed=0)
    hits = 0
    for _ in range(30):
        env.reset()
        hits += any(env.chip.faults[d.slices()].any() for d in (env.job.start, env.job.goal))
    assert hits > 0


def test_hidden_defects_stop_the_droplet_but_not_the_sensors(make_env, ideal_chip_factory):
    chip = ideal_chip_factory(W24, H16)
    chip.set_faults([(8, y) for y in range(H16)], hidden=True)  # east frontier of START
    env = make_env(width=W24, height=H16, obs_size=None)
    obs, _ = reset_on(env, JOB, chip)
    assert np.all(obs[0, 1:11, 8] == 0.75)  # reads as healthy
    for _ in range(5):
        env.step(Action.E)
        assert env.droplet == START


# ============================================================== configuration
YAML_CONFIG = """
width: 20
height: 14
obs_size: [30, 30]
kmax_alpha: 1.5
fault_fraction: 0.1
hidden_defect_fraction: 0.05
degradation:
  tau_range: [0.55, 0.65]
  c_range: [600, 700]
  health_bits: 3
  initial_actuations: uniform
  max_initial_actuations: 100
jobs:
  droplet_sizes: [[2, 2], [3, 3], [4, 5]]
  sampling: uniform
  hazard_margin: null
reward:
  alpha_dis: 1.0
  alpha_dis_away: 1.0
  stall_penalty: 0.0
"""


def test_env_config_from_yaml_dict(make_env):
    data = yaml.safe_load(YAML_CONFIG)
    cfg = EnvConfig.from_dict(data)
    assert cfg.obs_size == (30, 30) and cfg.kmax_alpha == 1.5
    assert isinstance(cfg.degradation, DegradationConfig)
    assert cfg.degradation.tau_range == (0.55, 0.65) and cfg.degradation.c_range == (600, 700)
    assert isinstance(cfg.degradation.tau_range, tuple)  # YAML lists become tuples
    assert isinstance(cfg.degradation.c_range, tuple)
    assert cfg.degradation.health_bits == 3 and cfg.degradation.initial_actuations == "uniform"
    assert isinstance(cfg.jobs, JobSamplerConfig)
    assert cfg.jobs.droplet_sizes == [(2, 2), (3, 3), (4, 5)] and cfg.jobs.hazard_margin is None
    assert cfg.reward == RewardConfig(alpha_dis=1.0, alpha_dis_away=1.0, stall_penalty=0.0)
    # round trips
    assert EnvConfig.from_dict(cfg.to_dict()) == cfg
    assert EnvConfig.from_dict(yaml.safe_load(yaml.safe_dump(cfg.to_dict()))) == cfg
    assert EnvConfig.from_dict({}) == EnvConfig() == EnvConfig.from_dict(None)
    # the environment honours every section
    env = make_env(data)
    assert env.config == cfg and env.observation_space.shape == (3, 30, 30)
    env.reset(seed=0)
    for _ in range(10):
        env.reset()
        assert env.hazard == chip_rect(20, 14)  # hazard_margin: null -> whole chip
        assert env.job.start.size in {(2, 2), (3, 3), (4, 5)}
        assert env.k_max == math.ceil(1.5 * (20 + 14))
        assert env.chip.health_levels == 8 and env.chip.actuations.max() <= 100
        assert 0.55 <= env.chip.tau.min() and env.chip.tau.max() <= 0.65
        assert env.chip.faults.mean() >= 0.1 and env.chip.hidden_defects.mean() >= 0.05


def test_env_config_errors_and_overrides(make_env):
    with pytest.raises(TypeError):
        EnvConfig.from_dict({"no_such_option": 1})
    with pytest.raises(TypeError):
        EnvConfig.from_dict({"degradation": {"no_such_option": 1}})
    env = make_env({"width": 12, "height": 10}, obs_size=None, kmax_alpha=2.0)
    cfg = env.config
    assert (cfg.width, cfg.height, cfg.obs_size, cfg.kmax_alpha) == (12, 10, None, 2.0)
    assert env.observation_space.shape == (3, 10, 12)
    env = make_env(EnvConfig(width=12, height=10), fault_fraction=0.2)
    assert env.config.fault_fraction == 0.2 and env.config.width == 12
    assert make_env(obs_size=None).observation_space.shape == (3, 30, 30)
    registered = gym.make(ENV_ID, config={"width": 8, "height": 8, "obs_size": None})
    assert registered.observation_space.shape == (3, 8, 8)
    registered.close()


# ================================================== consistency with run_job
class _Greedy(Router):
    name = "greedy"

    def act(self, state: RoutingState) -> Action:
        return greedy_action(state.droplet, state.goal)


def test_env_rollout_matches_run_job(make_env):
    """Environment and router harness share the physics (``core.dynamics``)."""
    cfg = DegradationConfig(initial_actuations="uniform", max_initial_actuations=1500)
    outcomes = Counter()
    for seed in range(8):
        base = MEDABiochip(16, 12, cfg, rng=np.random.default_rng(seed))
        base.reset()
        base.inject_faults(0.15)
        job = JobSampler(16, 12, rng=np.random.default_rng(100 + seed)).sample()
        chip_env, chip_run = base.copy(), base.copy()
        env = make_env(width=16, height=12, obs_size=None)
        env.reset(seed=seed, options={"job": job, "chip": chip_env})  # env RNG == default_rng(seed)
        path, invalid = [env.droplet], 0
        while True:
            _, _, terminated, truncated, info = env.step(greedy_action(env.droplet, env.goal))
            path.append(env.droplet)
            invalid += info["invalid_action"]
            if terminated or truncated:
                break
        result = run_job(_Greedy(), job, chip_run, np.random.default_rng(seed), record_path=True)
        assert result.path == path
        assert result.success == terminated and result.cycles == info["num_cycles"]
        assert result.invalid_actions == invalid
        assert np.array_equal(chip_env.actuations, chip_run.actuations)
        outcomes[result.success] += 1
    assert outcomes[True] > 0


# ================================================= collision cue (reference)
def _cue_job() -> RoutingJob:
    return RoutingJob(Droplet.at(1, 1, 3, 3), Droplet.at(9, 6, 3, 3), Rect(1, 1, 14, 10))


def test_collision_cue_marks_blocked_edges(make_env, ideal_chip_factory):
    env = make_env(width=16, height=12, obs_size=None, mark_collisions=True)
    reset_on(env, _cue_job(), ideal_chip_factory(16, 12))
    obs, *_ = env.step(Action.SW)  # blocked to the west and to the south
    d = env.droplet
    channel = obs[1, d.ya : d.yb + 1, d.xa : d.xb + 1]  # rows = y, cols = x
    assert np.all(channel[:, 0] == 0.5) and np.all(channel[0, :] == 0.5)  # west column, south row
    assert np.all(channel[1:, 1:] == 1.0)
    assert obs[1].sum() == pytest.approx(9 - 5 * 0.5)
    # without the option the droplet channel stays binary
    plain = make_env(width=16, height=12, obs_size=None)
    reset_on(plain, _cue_job(), ideal_chip_factory(16, 12))
    assert set(np.unique(plain.step(Action.SW)[0][1]).tolist()) == {0.0, 1.0}


def test_collision_cue_persists_like_reference(make_env, ideal_chip_factory):
    """Reference ``MEDAEnv.collision``: written by invalid actions only, cleared at reset."""
    env = make_env(width=16, height=12, obs_size=None, mark_collisions=True)
    reset_on(env, _cue_job(), ideal_chip_factory(16, 12))
    env.step(Action.W)  # invalid: collision = (west, -, -, -)
    obs, _, _, _, info = env.step(Action.N)  # valid: the flags persist
    assert not info["invalid_action"] and env.droplet == Droplet.at(1, 2, 3, 3)
    d = env.droplet
    assert np.all(obs[1, d.ya : d.yb + 1, d.xa] == 0.5)  # reference: obs[x0, y0:y1] = 0.5
    assert obs[1].sum() == pytest.approx(9 - 3 * 0.5)
    env.step(Action.S)  # valid, back to the zone's south edge
    obs, _, _, _, info = env.step(Action.S)  # invalid: all four flags are rewritten
    assert info["invalid_action"]
    d = env.droplet
    channel = obs[1, d.ya : d.yb + 1, d.xa : d.xb + 1]
    assert np.all(channel[0, :] == 0.5) and np.all(channel[1:, :] == 1.0)  # south row only
    obs, _ = reset_on(env, _cue_job(), ideal_chip_factory(16, 12))
    assert set(np.unique(obs[1]).tolist()) == {0.0, 1.0}


# ============================================= job distribution vs reference
def reference_hazard(start: Droplet, goal: Droplet, width: int, height: int) -> Rect:
    """``MEDAEnv._resetInitialState``: half-open bounding box grown by 3, clipped to the chip."""
    s = (start.xa, start.ya, start.xb + 1, start.yb + 1)
    g = (goal.xa, goal.ya, goal.xb + 1, goal.yb + 1)
    x0, y0 = max(min(g[0], s[0]) - 3, 0), max(min(g[1], s[1]) - 3, 0)
    x1, y1 = min(max(g[2], s[2]) + 3, width), min(max(g[3], s[3]) + 3, height)
    return Rect(x0, y0, x1 - 1, y1 - 1)


def test_sampled_jobs_follow_paper_and_reference(make_env):
    env = make_env(width=30, height=30)
    env.reset(seed=0)
    sizes = Counter()
    for _ in range(300):
        env.reset()
        job = env.job
        assert job.hazard == reference_hazard(job.start, job.goal, 30, 30)
        assert job.start.size == job.goal.size and job.start != job.goal
        assert chip_rect(30, 30).contains(job.hazard)
        sizes[job.start.size] += 1
    assert set(sizes) == set(PAPER_DROPLET_SIZES)  # one agent for all sizes (Sec. IV-A)


def _reference_axis(n: int) -> List[int]:
    """Reference stratified coordinate set: ``[0]*(n//5) + [1..n-2] + [n-1]*(n//5)``."""
    return [0] * (n // 5) + list(range(1, n - 1)) + [n - 1] * (n // 5)


def test_stratified_axis_matches_reference():
    """Edge-padded coordinate pool of the reference ``_getStratifiedSample`` (``W // 5`` copies)."""
    for n in range(5, 200):
        sampler = JobSampler(n, n, JobSamplerConfig(droplet_sizes=[(2, 2)]))
        assert sorted(sampler._stratified_axis(n)) == sorted(_reference_axis(n)), n
