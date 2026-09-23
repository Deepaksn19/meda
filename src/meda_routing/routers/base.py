"""Common interface for droplet-routing policies ("routers").

A router controls one routing job at a time.  :meth:`Router.reset` is called
when a job starts (a formal synthesizer computes its strategy there) and
:meth:`Router.act` once per control cycle.  Routers only see what a real
controller sees: droplet sensing (the droplet location) and the quantized
health matrix — never the hidden degradation parameters or hidden defects.

:func:`run_job` executes a single routing job on a biochip with the shared
physics from :mod:`meda_routing.core.dynamics`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..core.actions import Action
from ..core.biochip import MEDABiochip
from ..core.dynamics import execute_move, plan_move
from ..core.geometry import Droplet, Rect
from ..core.jobs import RoutingJob


@dataclass
class RoutingState:
    """What a router observes at control cycle ``k`` of a routing job."""

    chip: MEDABiochip
    droplet: Droplet
    goal: Droplet
    hazard: Rect
    k: int
    k_max: int
    #: ``(west, south, east, north)`` routing-zone collisions of the last
    #: invalid action in this job (persisting, as in the reference).
    collision: tuple = (False, False, False, False)

    def health(self) -> np.ndarray:
        return self.chip.health()


class Router(ABC):
    """A droplet-routing policy."""

    #: Short name used in logs, tables and plot legends.
    name: str = "router"
    #: Use Algorithm 1's adaptive step (True) or ``fixed_step`` MCs per axis.
    adaptive_step: bool = True
    fixed_step: int = 1

    def reset(self, job: RoutingJob, chip: MEDABiochip) -> None:  # noqa: B027
        """Called once at the start of every routing job."""

    @abstractmethod
    def act(self, state: RoutingState) -> Action:
        """Return the direction to actuate in this control cycle."""


@dataclass
class JobResult:
    success: bool
    cycles: int
    invalid_actions: int
    path: List[Droplet] = field(default_factory=list)


def run_job(
    router: Router,
    job: RoutingJob,
    chip: MEDABiochip,
    rng: np.random.Generator,
    k_max: Optional[int] = None,
    kmax_alpha: float = 1.0,
    record_path: bool = False,
) -> JobResult:
    """Route a single droplet from ``job.start`` to ``job.goal`` on ``chip``.

    ``chip`` is mutated: every actuation increments its wear counters, so
    successive calls model a chip that ages over its lifetime.
    """
    if k_max is None:
        k_max = max(1, int(np.ceil(kmax_alpha * (job.hazard.width + job.hazard.height))))
    droplet = job.start
    path = [droplet] if record_path else []
    if droplet == job.goal:  # nothing to route
        return JobResult(True, 0, 0, path)
    router.reset(job, chip)
    collision = (False, False, False, False)
    invalid = 0
    for k in range(1, k_max + 1):
        state = RoutingState(chip, droplet, job.goal, job.hazard, k - 1, k_max, collision)
        action = router.act(state)
        plan = plan_move(droplet, job.goal, job.hazard, action, router.adaptive_step, router.fixed_step)
        invalid += int(not plan.valid)
        if not plan.valid:  # marks of the last invalid action persist (reference)
            collision = plan.collision
        droplet, pattern = execute_move(plan, droplet, job.hazard, chip.effective_degradation(), rng)
        chip.actuate(pattern)
        if record_path:
            path.append(droplet)
        if droplet == job.goal:
            return JobResult(True, k, invalid, path)
    return JobResult(False, k_max, invalid, path)
