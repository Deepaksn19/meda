"""Formally synthesized health-aware routing strategies (baseline "formal", Fig. 9).

Sec. V-B compares the DRL policy against "formally synthesized strategies
using the PRISM-games model checker" [18], [19]: before every routing job the
synthesizer reads the MC health matrix, builds the probabilistic model of the
single-droplet routing job restricted to its hazard bounds and computes the
strategy that maximizes the probability of reaching the goal within ``K``
cycles, ``Pmax=? [F<=K goal]`` (``mdSynthesizeJobStr`` in the reference
``matlab/MedaSchedulerClass.m``).  PRISM is not available here, so this module
solves the same finite-horizon MDP exactly by value iteration:

* **Model.**  States are all positions of the job's droplet that lie inside
  the hazard bounds; actions are the 8 directions with a fixed step size
  (single-step by default, the ``MEDAY`` model; ``MEDAX`` is double-step).
  Transition probabilities follow the frontier model of
  :mod:`meda_routing.core.movement` evaluated on the *estimated* degradation
  ``D_hat = H / (2**b - 1)`` of the sensed health matrix ``H`` -- the
  synthesizer has sensor data only, never the true ``tau``, ``c``, wear
  counts or hidden defects.
* **Objective.**  ``V_0 = 1{goal}`` and
  ``V_t(s) = max_a sum_s' P(s'|s,a) V_{t-1}(s')``; ``pi_t`` is the maximizer
  with ``t`` cycles remaining.  The default horizon is the reference
  ``pfcnGetKmax``: ``K = ceil(1.5 * D(delta_s, delta_g)) + 1``.
* **Ties.**  Maximizing a bounded-reachability probability leaves many
  actions tied (e.g. every action is worth 1 when there is slack, including
  pushing against a fully degraded frontier).  Ties are broken
  lexicographically by the values with fewer remaining cycles
  ``Q_{t-1}, ..., Q_1`` ("arrive as early as possible", which prevents
  stalling while slack remains), then by the Manhattan distance of the
  nominal (failure-free) move to the goal, then by action index.  On a
  uniformly healthy chip the strategy therefore coincides with the
  shortest-path baseline.
* **Lost bounds.**  Where the goal cannot be reached within ``t`` cycles
  (``V_t(s) = 0``) every action is worth 0, so the order above degenerates to
  the health-agnostic nominal distance and a droplet that fell behind
  schedule would push into a fully degraded frontier until the horizon ran
  out.  There ``pi_t`` is replaced by the full-horizon action ``pi_K``: the
  bounded objective is restarted, as it is once the horizon is used up.  This
  is still optimal for ``F<=t`` (all actions are worth 0) and leaves every
  value unchanged.

The wall-clock synthesis time is recorded per job for comparison with the
5-48 s reported for PRISM (Sec. V-B) and the < 0.1 s of the CNN.
"""

from __future__ import annotations

import math
import operator
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable, List, Optional, Tuple

import numpy as np

from ..core.actions import NUM_ACTIONS, Action
from ..core.biochip import MEDABiochip
from ..core.dynamics import plan_move
from ..core.geometry import Droplet, Rect, chip_rect
from ..core.jobs import RoutingJob
from ..core.movement import action_flags, footprint_mask, move_distribution
from .base import Router, RoutingState
from .baseline import diagonal_step_for, parse_step_mode

#: Values closer than this are treated as equal when breaking ties.
DEFAULT_TIE_TOLERANCE = 1e-9


def _check_horizon(horizon: int) -> int:
    """``horizon`` as an ``int`` >= 1 (floats are rejected, not truncated)."""
    horizon = operator.index(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    return horizon


def _check_tie_tolerance(tie_tolerance: float) -> float:
    # A non-positive tolerance would divide by zero or, if negative, silently
    # turn the maximization into a minimization.
    tie_tolerance = float(tie_tolerance)
    if not (tie_tolerance > 0.0 and math.isfinite(tie_tolerance)):
        raise ValueError(f"tie_tolerance must be positive and finite, got {tie_tolerance}")
    return tie_tolerance


def default_horizon(start: Rect, goal: Rect) -> int:
    """``K = ceil(1.5 * D(start, goal)) + 1`` (reference ``pfcnGetKmax``)."""
    dist = abs(start.xa - goal.xa) + abs(start.ya - goal.ya)
    return int(np.ceil(1.5 * dist)) + 1


def estimate_degradation(health: np.ndarray, levels: int) -> np.ndarray:
    """Degradation estimate ``D_hat = H / (levels - 1)`` from sensed health.

    A fully healthy reading (``levels - 1``) maps to 1 and a fully degraded
    one (0) to 0.
    """
    if levels < 2:
        raise ValueError("need at least two health levels")
    return np.asarray(health, dtype=np.float64) / float(levels - 1)


@dataclass
class FormalStrategy:
    """A synthesized finite-horizon strategy for one routing job."""

    goal: Droplet
    hazard: Rect
    #: Droplet size ``(w, h)``.
    size: Tuple[int, int]
    horizon: int
    adaptive_step: bool
    fixed_step: int
    #: ``policy[t, s]``: action with ``t`` cycles remaining in state ``s``;
    #: where ``values[t, s] == 0`` it is the full-horizon action
    #: ``policy[K, s]`` (row 0 is never used).
    policy: np.ndarray
    #: ``values[t, s]``: probability of reaching the goal within ``t`` cycles
    #: from ``s`` under ``policy`` (maximal up to the tie tolerance).
    values: np.ndarray
    #: Step of the ordinal moves (``None``: same as ``fixed_step``).
    diagonal_step: Optional[int] = None

    # --------------------------------------------------------------- states
    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Number of south-west corner positions ``(nx, ny)`` in the hazard zone."""
        w, h = self.size
        return self.hazard.width - w + 1, self.hazard.height - h + 1

    @property
    def num_states(self) -> int:
        nx, ny = self.grid_shape
        return nx * ny

    def state_id(self, droplet: Rect) -> int:
        """Index of ``droplet`` in the state space (``ValueError`` if absent)."""
        if droplet.size != self.size or not self.hazard.contains(droplet):
            raise ValueError(
                f"droplet {droplet.as_tuple()} is not a state of this strategy "
                f"(size {self.size}, hazard {self.hazard.as_tuple()})"
            )
        _, ny = self.grid_shape
        return (droplet.xa - self.hazard.xa) * ny + (droplet.ya - self.hazard.ya)

    def droplet(self, sid: int) -> Droplet:
        _, ny = self.grid_shape
        ix, iy = divmod(int(sid), ny)
        return Droplet.at(self.hazard.xa + ix, self.hazard.ya + iy, *self.size)

    # -------------------------------------------------------------- queries
    def _horizon_index(self, remaining: Optional[int]) -> int:
        if remaining is None or remaining < 1 or remaining > self.horizon:
            return self.horizon
        return int(remaining)

    def action(self, droplet: Rect, remaining: Optional[int] = None) -> Action:
        """``pi_r(droplet)``; ``remaining`` outside ``[1, K]`` uses ``pi_K``."""
        t = self._horizon_index(remaining)
        return Action(int(self.policy[t, self.state_id(droplet)]))

    def value(self, droplet: Rect, remaining: Optional[int] = None) -> float:
        """``V_r(droplet)``; ``remaining=None`` gives the full horizon ``K``."""
        t = self._horizon_index(remaining)
        return float(self.values[t, self.state_id(droplet)])


def _build_transitions(
    goal: Droplet,
    hazard: Rect,
    size: Tuple[int, int],
    degradation: np.ndarray,
    adaptive: bool,
    fixed_step: int,
    diagonal_step: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Sparse transition table of the job MDP.

    Returns ``(pair, nxt, prob, nominal, goal_id)``: entry ``i`` says that
    state-action pair ``pair[i] = s * 8 + a`` leads to ``nxt[i]`` with
    probability ``prob[i]``; ``nominal[s, a]`` is the Manhattan distance to the
    goal after the failure-free move.  The goal is absorbing and has no rows.
    """
    w, h = size
    nx, ny = hazard.width - w + 1, hazard.height - h + 1
    x0, y0 = hazard.xa, hazard.ya
    goal_id = (goal.xa - x0) * ny + (goal.ya - y0)
    pairs: List[int] = []
    nxt: List[int] = []
    probs: List[float] = []
    nominal = np.empty((nx * ny, NUM_ACTIONS), dtype=np.int64)
    actions = list(Action)
    flags = [action_flags(a) for a in actions]
    for ix in range(nx):
        for iy in range(ny):
            s = ix * ny + iy
            droplet = Droplet.at(x0 + ix, y0 + iy, w, h)
            for a in actions:
                plan = plan_move(droplet, goal, hazard, a, adaptive, fixed_step, diagonal_step)
                nominal[s, a] = plan.target.manhattan(goal)
                if s == goal_id:
                    continue
                if plan.valid and plan.target != droplet:
                    pattern = footprint_mask(degradation.shape, plan.target)
                    dist = move_distribution(degradation, pattern, droplet, hazard, flags[a])
                else:  # invalid action or zero step: the droplet holds its position
                    dist = {droplet: 1.0}
                base = s * NUM_ACTIONS + int(a)
                for d, p in dist.items():
                    pairs.append(base)
                    nxt.append((d.xa - x0) * ny + (d.ya - y0))
                    probs.append(p)
    return (
        np.asarray(pairs, dtype=np.int64),
        np.asarray(nxt, dtype=np.int64),
        np.asarray(probs, dtype=np.float64),
        nominal,
        goal_id,
    )


def synthesize(
    goal: Droplet,
    hazard: Rect,
    degradation: np.ndarray,
    horizon: int,
    adaptive: bool = False,
    fixed_step: int = 1,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    diagonal_step: Optional[int] = None,
) -> FormalStrategy:
    """Solve ``Pmax=? [F<=horizon goal]`` for a droplet of ``goal``'s size.

    ``degradation`` is the ``W x H`` (``[x, y]``-indexed) success-degradation
    map the model is built on; only its values inside ``hazard`` matter.
    """
    horizon = _check_horizon(horizon)
    tie_tolerance = _check_tie_tolerance(tie_tolerance)
    degradation = np.asarray(degradation, dtype=np.float64)
    if degradation.ndim != 2 or not chip_rect(*degradation.shape).contains(hazard):
        raise ValueError(
            f"degradation map of shape {degradation.shape} does not cover the hazard "
            f"bounds {hazard.as_tuple()}"
        )
    if not hazard.contains(goal):
        raise ValueError("goal must lie inside the hazard bounds")
    goal = Droplet(*goal.as_tuple())
    size = goal.size
    pair, nxt, prob, nominal, goal_id = _build_transitions(
        goal, hazard, size, degradation, adaptive, fixed_step, diagonal_step
    )
    num_states = nominal.shape[0]
    rows = np.arange(num_states)
    slots = np.broadcast_to(np.arange(NUM_ACTIONS), (num_states, NUM_ACTIONS))
    scale = 1.0 / tie_tolerance

    policy = np.zeros((horizon + 1, num_states), dtype=np.int8)
    values = np.zeros((horizon + 1, num_states), dtype=np.float64)
    values[0, goal_id] = 1.0

    # rank[s, a] = position of a in the tie-break order of s (0 = preferred);
    # initially by nominal distance to the goal, then by action index.
    order = np.lexsort((slots, nominal), axis=-1)
    rank = np.empty((num_states, NUM_ACTIONS), dtype=np.int64)
    np.put_along_axis(rank, order, slots, axis=-1)
    policy[0] = order[:, 0]
    for t in range(1, horizon + 1):
        q = np.bincount(
            pair, weights=prob * values[t - 1][nxt], minlength=num_states * NUM_ACTIONS
        ).reshape(num_states, NUM_ACTIONS)
        # Lexicographic order (Q_t, Q_{t-1}, ..., Q_1, nominal, index): sort by
        # the quantized Q_t and resolve ties with the previous rank.
        order = np.lexsort((rank, -np.rint(q * scale)), axis=-1)
        np.put_along_axis(rank, order, slots, axis=-1)
        best = order[:, 0]
        policy[t] = best
        values[t] = q[rows, best]
        values[t, goal_id] = 1.0
    # Lost bounds (module docstring): where V_t = 0 up to the tolerance every
    # action is tied at 0; use the full-horizon action there.  The values of
    # such states are 0 under any action, so no value changes.
    lost = np.rint(values[1:] * scale) == 0
    policy[1:] = np.where(lost, policy[horizon], policy[1:])
    return FormalStrategy(
        goal=goal,
        hazard=hazard,
        size=size,
        horizon=int(horizon),
        adaptive_step=adaptive,
        fixed_step=fixed_step,
        policy=policy,
        values=values,
        diagonal_step=diagonal_step,
    )


class FormalRouter(Router):
    """Stand-in for the PRISM-games synthesizer of [18], [19] ("formal" in Fig. 9).

    Parameters
    ----------
    step_mode:
        ``"single"`` (default, as the ``MEDAY`` model of [18]), ``"double"``
        (``MEDAX``) or ``"adaptive"`` (Algorithm 1).
    horizon:
        Fixed bound ``K`` of ``Pmax=? [F<=K goal]``; ``None`` uses
        :func:`default_horizon` of the job.
    cache_size:
        Keep up to this many strategies keyed by the job and the sensed health
        inside its hazard bounds, like the reference ``fcnLookupStrategy``.
        ``0`` (default) re-synthesizes before every job, which is what the
        synthesis times of Sec. V-B measure.
    tie_tolerance:
        Probabilities closer than this are considered tied.

    After :meth:`reset`, ``success_probability`` is ``V_K`` at the start
    location under the *estimated* model and ``last_synthesis_seconds`` the
    wall time spent reading the health matrix and synthesizing.
    """

    name = "formal"

    def __init__(
        self,
        step_mode: str = "single",
        horizon: Optional[int] = None,
        cache_size: int = 0,
        tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    ) -> None:
        self.step_mode = step_mode
        self.adaptive_step, self.fixed_step = parse_step_mode(step_mode)
        self.diagonal_step = diagonal_step_for(step_mode)
        self.horizon = None if horizon is None else _check_horizon(horizon)
        self.cache_size = int(cache_size)
        self.tie_tolerance = _check_tie_tolerance(tie_tolerance)
        self.strategy: Optional[FormalStrategy] = None
        self.success_probability: float = float("nan")
        self.last_synthesis_seconds: float = float("nan")
        self.last_cache_hit: bool = False
        self._cache: "OrderedDict[Hashable, FormalStrategy]" = OrderedDict()

    def reset(self, job: RoutingJob, chip: MEDABiochip) -> None:
        start_time = time.perf_counter()
        # Sensor data only: the quantized health matrix (Eq. 1).
        health = chip.health()
        levels = chip.health_levels
        k = self.horizon if self.horizon is not None else default_horizon(job.start, job.goal)
        key = None
        strategy = None
        if self.cache_size > 0:
            sx, sy = job.hazard.slices()
            key = (
                job.goal.as_tuple(),
                job.hazard.as_tuple(),
                k,
                self.adaptive_step,
                self.fixed_step,
                self.diagonal_step,
                levels,
                np.ascontiguousarray(health[sx, sy]).tobytes(),
            )
            strategy = self._cache.get(key)
            if strategy is not None:
                self._cache.move_to_end(key)
        self.last_cache_hit = strategy is not None
        if strategy is None:
            strategy = synthesize(
                job.goal,
                job.hazard,
                estimate_degradation(health, levels),
                k,
                self.adaptive_step,
                self.fixed_step,
                self.tie_tolerance,
                self.diagonal_step,
            )
            if key is not None:
                self._cache[key] = strategy
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        self.strategy = strategy
        self.success_probability = strategy.value(job.start)
        self.last_synthesis_seconds = time.perf_counter() - start_time

    def act(self, state: RoutingState) -> Action:
        strategy = self.strategy
        if strategy is None:
            raise RuntimeError("FormalRouter.act() called before reset()")
        if (
            state.goal.as_tuple() != strategy.goal.as_tuple()
            or state.hazard.as_tuple() != strategy.hazard.as_tuple()
        ):
            raise RuntimeError("state belongs to a different routing job; call reset() first")
        # pi_r with r cycles remaining; once the horizon is exhausted, restart
        # with the full-horizon strategy pi_K (the table already does so where
        # the goal is out of reach within r cycles, see "Lost bounds").
        return strategy.action(state.droplet, strategy.horizon - state.k)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(step_mode={self.step_mode!r}, horizon={self.horizon!r}, "
            f"cache_size={self.cache_size})"
        )
