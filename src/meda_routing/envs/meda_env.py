"""Gymnasium environment for single-droplet routing on MEDA biochips.

This is the virtual training environment of Sec. III: a stochastic model of a
``W x H`` MEDA biochip with per-MC degradation (Eq. 1), the parameterized
8-direction action space with adaptive step size (Algorithm 1), the
3-channel image observation (Fig. 2) and the shaped reward (Sec. III-D).

Episodes (Algorithm 2) start by sampling a routing job ``(delta_s, delta_g,
delta_h)``, degradation parameters ``(tau_ij, c_ij)``, initial actuation
counts ``N`` and, optionally, injected faults.  An episode terminates when
the droplet reaches the goal and is truncated after
``k_max = alpha * (W_h + H_h)`` cycles.

For bioassay execution the environment can also be driven with an explicit
job on an externally owned :class:`MEDABiochip` whose wear persists across
routing jobs: ``env.reset(options={"job": job, "chip": chip})``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.actions import NUM_ACTIONS, Action
from ..core.biochip import DegradationConfig, MEDABiochip
from ..core.dynamics import execute_move, plan_move
from ..core.geometry import Droplet, Rect, chip_rect
from ..core.jobs import JobSampler, JobSamplerConfig, RoutingJob
from .observation import build_observation, observation_shape
from .reward import RewardConfig, compute_reward


@dataclass
class EnvConfig:
    """Configuration of :class:`MEDARoutingEnv`."""

    width: int = 30
    height: int = 30
    #: Unified observation size ``(w, h)`` (Sec. IV-C); ``None`` keeps the
    #: chip resolution ("traditional learning").
    obs_size: Optional[Tuple[int, int]] = (30, 30)
    #: Parameterized action space with adaptive step size (Algorithm 1).  When
    #: disabled every action moves the droplet by ``fixed_step`` MCs per axis.
    adaptive_step: bool = True
    fixed_step: int = 1
    #: Step of the ordinal moves in fixed-step mode (``None``: ``fixed_step``).
    diagonal_step: Optional[int] = None
    #: ``k_max = kmax_alpha * (W_h + H_h)`` (Sec. IV-B, ``alpha in [1, 2]``).
    kmax_alpha: float = 1.0
    #: Fraction of MCs made fully degraded (and visible to the health
    #: sensors) at the start of each episode, placed in ``fault_cluster``-sized
    #: square clusters (Sec. V-B: 10% / 20% in 2x2 clusters).
    fault_fraction: float = 0.0
    fault_cluster: int = 2
    #: Fraction of MCs with defects that are *invisible* to the health
    #: sensors (Sec. VI-C: ~5% inherent defects on the PCB prototypes).
    hidden_defect_fraction: float = 0.0
    #: Never place faults under the start or goal droplet.
    protect_endpoints: bool = True
    #: Mark the droplet edge touching the routing-zone boundary after an
    #: invalid action (reference-code feature, not in the paper; as in the
    #: reference, the marks of the last invalid action persist until the
    #: next invalid action or the end of the episode).
    mark_collisions: bool = False
    #: Report a timeout as ``terminated`` instead of ``truncated``.  Stable
    #: Baselines v2 (used by the authors) cut the return at timeouts; the
    #: default truncation lets PPO bootstrap, avoiding the state aliasing
    #: discussed in Sec. III-D.
    timeout_terminal: bool = False
    #: Online adaptation to one physical chip (Sec. I: offline training in
    #: simulation, then "online DRL training ... to adjust the policy under
    #: different biochip environments"): the chip's degradation parameters,
    #: initial wear and faults are drawn once, and wear keeps accumulating
    #: across episodes instead of being reset.
    persistent_chip: bool = False
    #: With ``persistent_chip``: seed of the physical chip (degradation
    #: parameters, initial wear, faults), so that every environment instance —
    #: training and evaluation alike — models the same chip.  ``None`` draws
    #: the chip from the environment's own random stream.
    chip_seed: Optional[int] = None
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    jobs: JobSamplerConfig = field(default_factory=JobSamplerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "EnvConfig":
        data = dict(data or {})
        nested = {
            "degradation": DegradationConfig,
            "jobs": JobSamplerConfig,
            "reward": RewardConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key in nested and isinstance(value, dict):
                kwargs[key] = nested[key](**_tupleize(value))
            else:
                kwargs[key] = value
        if kwargs.get("obs_size") is not None:
            kwargs["obs_size"] = tuple(kwargs["obs_size"])
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _tupleize(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if k.endswith("_range") and isinstance(v, list):
            v = tuple(v)
        if k == "droplet_sizes" and isinstance(v, list):
            v = [tuple(s) for s in v]
        out[k] = v
    return out


class MEDARoutingEnv(gym.Env):
    """Single-droplet adaptive routing on a MEDA biochip."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[EnvConfig | Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        if config is None or isinstance(config, dict):
            config = EnvConfig.from_dict(_deep_merge(dict(config or {}), overrides))
        elif overrides:
            # nested overrides (e.g. degradation={"health_bits": 3}) update only
            # the given keys of that section
            config = EnvConfig.from_dict(_deep_merge(config.to_dict(), overrides))
        if config.obs_size is not None and not isinstance(config.obs_size, tuple):
            config = dataclasses.replace(config, obs_size=tuple(config.obs_size))
        self.config: EnvConfig = config
        self.render_mode = render_mode
        self.width, self.height = config.width, config.height

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=observation_shape(self.width, self.height, config.obs_size),
            dtype=np.float32,
        )

        self._np_random, _ = gym.utils.seeding.np_random(None)
        self._own_chip = MEDABiochip(self.width, self.height, config.degradation, self._np_random)
        self.chip: MEDABiochip = self._own_chip
        self.sampler = JobSampler(self.width, self.height, config.jobs, self._np_random)
        self.job: Optional[RoutingJob] = None
        self.droplet: Optional[Droplet] = None
        self.k = 0
        self.k_max = 0
        self._collision = (False, False, False, False)
        self.last_pattern: Optional[np.ndarray] = None
        self._chip_ready = False

    # ----------------------------------------------------------- properties
    @property
    def goal(self) -> Droplet:
        assert self.job is not None
        return self.job.goal

    @property
    def hazard(self) -> Rect:
        assert self.job is not None
        return self.job.hazard

    # ------------------------------------------------------------ gym API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            # re-seed the components that share the generator
            self._own_chip.rng = self.np_random
            self.sampler.reseed(self.np_random)
        options = options or {}
        chip = options.get("chip")
        job = options.get("job")

        fresh_chip = chip is None and not (self.config.persistent_chip and self._chip_ready)
        faults_pending = False
        if chip is not None:
            # externally managed chip (bioassay execution): keep its wear
            if chip.shape != (self.width, self.height):
                raise ValueError(
                    f"chip is {chip.width}x{chip.height} but the environment is "
                    f"{self.width}x{self.height}"
                )
            self.chip = chip
        else:
            self.chip = self._own_chip
            if fresh_chip and self.config.persistent_chip and self.config.chip_seed is not None:
                # the same physical chip in every environment instance
                own_rng = self.chip.rng
                self.chip.rng = np.random.default_rng(self.config.chip_seed)
                self.chip.reset(resample_parameters=True)
                self._inject_faults(protect=())
                self.chip.rng = own_rng
            elif fresh_chip:
                self.chip.reset(resample_parameters=True)
                faults_pending = True  # placed around this episode's job below
            self._chip_ready = True
        if job is not None and not chip_rect(self.width, self.height).contains(job.hazard):
            raise ValueError(
                f"job routing zone {job.hazard.as_tuple()} does not fit the "
                f"{self.width}x{self.height} chip"
            )
        self.job = job if job is not None else self._sample_job(avoid_faults=not faults_pending)
        if faults_pending:
            protect = (self.job.start, self.job.goal) if self.config.protect_endpoints else ()
            self._inject_faults(protect)

        self.droplet = self.job.start
        self.k = 0
        self.k_max = options.get("k_max") or self._compute_kmax(self.job.hazard)
        self._collision = (False, False, False, False)
        self.last_pattern = None
        return self._get_obs(), self._get_info(invalid=False)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.job is not None and self.droplet is not None, "call reset() first"
        action = Action(int(action))
        self.k += 1
        prev_dist = self.droplet.manhattan(self.goal)

        plan = plan_move(
            self.droplet,
            self.goal,
            self.hazard,
            action,
            self.config.adaptive_step,
            self.config.fixed_step,
            self.config.diagonal_step,
        )
        valid = plan.valid
        if not valid:
            # reference behaviour: an invalid action sets the collision marks,
            # which then persist through valid actions until the next invalid one
            self._collision = plan.collision
        self.droplet, pattern = execute_move(
            plan, self.droplet, self.hazard, self.chip.effective_degradation(), self.np_random
        )
        self.chip.actuate(pattern)  # n_ij += U_ij
        self.last_pattern = pattern

        curr_dist = self.droplet.manhattan(self.goal)
        at_goal = self.droplet == self.goal
        reward = compute_reward(self.config.reward, prev_dist, curr_dist, at_goal, not valid)
        terminated = bool(at_goal)
        truncated = bool(not terminated and self.k >= self.k_max)
        if truncated and self.config.timeout_terminal:
            terminated, truncated = True, False
        return self._get_obs(), reward, terminated, truncated, self._get_info(invalid=not valid)

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        return self.render_frame()

    # ------------------------------------------------------------ helpers
    def _inject_faults(self, protect) -> None:
        cfg = self.config
        self.chip.inject_faults(cfg.fault_fraction, cfg.fault_cluster, protect=protect)
        self.chip.inject_faults(
            cfg.hidden_defect_fraction, cfg.fault_cluster, protect=protect, hidden=True
        )

    def _sample_job(self, avoid_faults: bool) -> RoutingJob:
        """Sample a job; on a persistent chip, avoid endpoints on faulty MCs.

        Gives up after 100 attempts (only possible on extremely faulty chips)
        and then returns the last sample.
        """
        job = self.sampler.sample()
        if not (avoid_faults and self.config.protect_endpoints):
            return job
        dead = self.chip.faults | self.chip.hidden_defects
        for _ in range(100):
            if not any(dead[r.slices()].any() for r in (job.start, job.goal)):
                break
            job = self.sampler.sample()
        return job

    def _compute_kmax(self, hazard: Rect) -> int:
        return max(1, int(np.ceil(self.config.kmax_alpha * (hazard.width + hazard.height))))

    def _get_obs(self) -> np.ndarray:
        return build_observation(
            self.chip.health(),
            self.chip.health_levels,
            self.droplet,
            self.goal,
            self.hazard,
            self.config.obs_size,
            self._collision if self.config.mark_collisions else None,
        )

    def _get_info(self, invalid: bool) -> Dict[str, Any]:
        return {
            "is_success": bool(self.droplet == self.goal),
            "num_cycles": self.k,
            "invalid_action": invalid,
            "distance": int(self.droplet.manhattan(self.goal)),
            "droplet": self.droplet.as_tuple(),
            "goal": self.goal.as_tuple(),
            "k_max": self.k_max,
        }

    def get_observation(self) -> np.ndarray:
        """Current observation (useful when the state was changed externally)."""
        return self._get_obs()

    def render_frame(self, scale: int = 8) -> np.ndarray:
        """RGB frame: health in gray, routing zone lit, faults red, goal green, droplet blue."""
        deg = self.chip.degradation()  # sensed degradation
        img = np.repeat(deg.T[:, :, None], 3, axis=2) * 0.6 + 0.2
        hx, hy = self.hazard.slices()
        inside = np.zeros_like(deg.T, dtype=bool)
        inside[hy, hx] = True
        img[~inside] *= 0.45
        faults = self.chip.faults.T
        img[faults] = (0.85, 0.15, 0.15)
        hidden = self.chip.hidden_defects.T & ~faults
        img[hidden] = (0.55, 0.25, 0.55)
        gx, gy = self.goal.slices()
        img[gy, gx] = img[gy, gx] * 0.3 + np.array([0.1, 0.8, 0.2]) * 0.7
        dx, dy = self.droplet.slices()
        img[dy, dx] = img[dy, dx] * 0.2 + np.array([0.1, 0.35, 0.95]) * 0.8
        img = np.flipud(img)  # north up
        img = np.kron(img, np.ones((scale, scale, 1)))
        return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
