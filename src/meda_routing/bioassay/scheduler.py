"""Bioassay execution: concurrent routing jobs on one aging biochip (Sec. V-B).

Fig. 9 reports the probability that a whole bioassay completes within ``k``
control cycles.  The paper gives no details of the execution; this module
follows the authors' reference scheduler (``melfar87/MEDA``,
``meda_scheduler.py``: ``MedaScheduler.tick``, ``processStates``,
``updateControlActions`` and ``_assignEnv``).  Every control cycle (tick):

1. **MO state machine** (``processStates``): a ``READY`` MO becomes ``BUSY``
   once every MO in its ``pre`` and ``cond`` lists is ``DONE``, which starts
   its routing jobs; a ``BUSY`` MO becomes ``DONE`` when all its jobs are
   done.  The pass is repeated until nothing changes, so successors start in
   the same cycle in which their predecessors complete.
2. **Routing**: every active routed job has its own router instance, reset
   with the job when it started.  Its routing zone is
   ``hazard_bounds(start, goal, W, H, 3)`` (``MEDAEnv.setState``).  All jobs
   observe the chip as it was at the start of the cycle, plan a move and
   sample its outcome with the shared physics of :mod:`meda_routing.core.dynamics`
   against one snapshot of the effective degradation.
3. **Actuation**: the actuation patterns of all jobs are OR-ed, so an MC used
   by two jobs is actuated once (``np.clip(m_pattern, 0, 1)`` in the
   reference), and the union is applied to the chip's wear counters.
4. **Dispensing** is deterministic and not routed by the policy: the droplet
   is pushed in from the off-chip reservoir by ``dispense_step`` MCs per cycle,
   first along ``x`` then along ``y``, and actuates no on-chip MCs (the
   reference adds no dispense pattern).

As in the reference, an MO picks its droplets up at the start locations
listed in the sequence graph (where the preceding MO is supposed to have left
them).  Droplet-droplet interference is not modelled (neither is it in the
reference).

Deviations from the reference, all deliberate:

* the reference gives each routing job its own copy of the wear counters
  (``setState(m_actcount=...)``) and never feeds the union back into it; here
  every job acts on, and wears, the same :class:`MEDABiochip`;
* a routed job times out after ``k_max = ceil(kmax_alpha * (W_h + H_h))``
  cycles (the convention of :func:`meda_routing.routers.base.run_job`; the
  reference ends an episode after ``W_h + H_h + 1`` steps).  What happens then
  is selectable with ``on_timeout``; the reference behaviour is ``"skip"``;
* the reported cycle count is the number of executed control cycles; the
  reference's ``k + 1`` additionally counts the tick that only detects
  completion, and records failed trials as ``k_max`` instead of failure.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.actions import max_reliable_step
from ..core.biochip import MEDABiochip
from ..core.dynamics import execute_move, plan_move
from ..core.geometry import Droplet, Rect, chip_rect
from ..core.jobs import RoutingJob, hazard_bounds
from ..routers.base import Router, RoutingState
from .library import Bioassay, JobSpec, MicrofluidicOperation

#: ``router_factory()`` or ``router_factory(job_key)`` with ``job_key`` such as
#: ``"n08.J1"``; must return a fresh :class:`Router` for every routing job.
RouterFactory = Union[Callable[[], Router], Callable[[str], Router]]

TIMEOUT_MODES = ("continue", "skip", "fail")


class MOState(IntEnum):
    """MO states (values as ``State`` in the reference ``meda_utils.py``)."""

    READY = 2
    BUSY = 3
    DONE = 5


@dataclass
class JobRecord:
    """Bookkeeping of one routing job of a bioassay run."""

    mo: str
    name: str
    start: Droplet
    goal: Droplet
    dispense: bool
    hazard: Optional[Rect] = None
    #: Cycle budget of a routed job (``None`` for dispensing).
    k_max: Optional[int] = None
    #: Number of control cycles executed before the job started.
    start_cycle: int = 0
    #: Cycle count at which the job ended (``None`` while running).
    end_cycle: Optional[int] = None
    #: Control cycles spent on the job.
    cycles: int = 0
    #: The droplet reached its goal.
    success: bool = False
    #: The job used up its ``k_max`` cycles without reaching the goal.
    timed_out: bool = False
    invalid_actions: int = 0

    @property
    def key(self) -> str:
        return f"{self.mo}.{self.name}"

    @property
    def finished(self) -> bool:
        return self.end_cycle is not None


@dataclass
class BioassayResult:
    success: bool
    #: Control cycles executed (until completion, or until the run was aborted).
    cycles: int
    job_records: List[JobRecord]
    #: Sum over cycles of the number of actuated MCs (OR-union per cycle).
    total_actuations: int
    #: ``None`` on success, else ``"timeout"``, ``"max_cycles"`` or ``"deadlock"``.
    failure: Optional[str] = None
    #: Cycle count at which each MO started / was marked done.
    mo_start: Dict[str, int] = field(default_factory=dict)
    mo_done: Dict[str, int] = field(default_factory=dict)

    @property
    def n_timeouts(self) -> int:
        return sum(r.timed_out for r in self.job_records)

    @property
    def routed_records(self) -> List[JobRecord]:
        return [r for r in self.job_records if not r.dispense]


@dataclass
class _ActiveJob:
    record: JobRecord
    droplet: Droplet
    router: Optional[Router] = None
    collision: Tuple[bool, bool, bool, bool] = (False, False, False, False)


def _wants_key(factory: Callable) -> bool:
    """True if ``factory`` has a required positional parameter (the job key)."""
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    return any(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
        for p in sig.parameters.values()
    )


class BioassayExecutor:
    """Executes a :class:`Bioassay` on a :class:`MEDABiochip` with a router.

    ``chip`` is mutated (its wear counters age with every actuation); pass a
    copy to keep the original.  ``rng`` drives the stochastic droplet movement.
    """

    def __init__(
        self,
        bioassay: Bioassay,
        router_factory: RouterFactory,
        chip: MEDABiochip,
        rng: Optional[np.random.Generator] = None,
        hazard_margin: Optional[int] = 3,
        kmax_alpha: float = 1.0,
        on_timeout: str = "continue",
        max_cycles: int = 2000,
        dispense_step: Optional[int] = 2,
    ) -> None:
        if on_timeout not in TIMEOUT_MODES:
            raise ValueError(f"on_timeout must be one of {TIMEOUT_MODES}, got {on_timeout!r}")
        if dispense_step is not None and dispense_step < 1:
            raise ValueError("dispense_step must be positive (or None for half the droplet size)")
        self.bioassay = bioassay
        self.router_factory = router_factory
        self._factory_wants_key = _wants_key(router_factory)
        self.chip = chip
        self.rng = rng if rng is not None else np.random.default_rng()
        self.hazard_margin = hazard_margin
        self.kmax_alpha = float(kmax_alpha)
        self.on_timeout = on_timeout
        self.max_cycles = int(max_cycles)
        #: MCs per cycle a dispensed droplet advances; ``None``: half its size.
        self.dispense_step = dispense_step
        self._validate_locations()
        self.reset()

    # ------------------------------------------------------------- set-up
    def _validate_locations(self) -> None:
        area = chip_rect(self.chip.width, self.chip.height)
        for spec in self.bioassay.job_specs():
            if not area.contains(spec.goal):
                raise ValueError(f"{spec.key}: goal {spec.goal.as_tuple()} lies off the chip")
            if not spec.dispense and not area.contains(spec.start):
                raise ValueError(f"{spec.key}: start {spec.start.as_tuple()} lies off the chip")

    def reset(self) -> None:
        """Forget all progress (the chip's wear is *not* reset)."""
        self._mos: List[MicrofluidicOperation] = list(self.bioassay.operations)
        self.mo_state: Dict[str, MOState] = {mo.name: MOState.READY for mo in self._mos}
        self._mo_jobs: Dict[str, List[JobRecord]] = {}
        self._active: List[_ActiveJob] = []
        self.records: List[JobRecord] = []
        self.mo_start: Dict[str, int] = {}
        self.mo_done: Dict[str, int] = {}
        self.cycle = 0
        self.total_actuations = 0
        self.failure: Optional[str] = None
        self._finished = False

    # --------------------------------------------------------- properties
    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def success(self) -> bool:
        return self._finished and self.failure is None

    @property
    def all_done(self) -> bool:
        return all(s == MOState.DONE for s in self.mo_state.values())

    def active_droplets(self) -> Dict[str, Droplet]:
        """Current location of every droplet being moved (may lie off-chip)."""
        return {job.record.key: job.droplet for job in self._active}

    # ------------------------------------------------------ state machine
    def process_states(self) -> None:
        """``MedaScheduler.processStates``: advance MOs until a fixpoint."""
        changed = True
        while changed:
            changed = False
            for mo in self._mos:
                state = self.mo_state[mo.name]
                if state == MOState.READY:
                    if all(self.mo_state[d] == MOState.DONE for d in mo.dependencies):
                        self._start_mo(mo)
                        changed = True
                elif state == MOState.BUSY:
                    if all(r.finished for r in self._mo_jobs[mo.name]):
                        self.mo_state[mo.name] = MOState.DONE
                        self.mo_done[mo.name] = self.cycle
                        changed = True

    def _start_mo(self, mo: MicrofluidicOperation) -> None:
        self.mo_state[mo.name] = MOState.BUSY
        self.mo_start[mo.name] = self.cycle
        records = [self._start_job(spec) for spec in mo.job_specs()]
        self._mo_jobs[mo.name] = records

    def _start_job(self, spec: JobSpec) -> JobRecord:
        record = JobRecord(spec.mo, spec.name, spec.start, spec.goal, spec.dispense)
        record.start_cycle = self.cycle
        self.records.append(record)
        job = _ActiveJob(record, spec.start)
        if not spec.dispense:
            hazard = hazard_bounds(
                spec.start, spec.goal, self.chip.width, self.chip.height, self.hazard_margin
            )
            record.hazard = hazard
            record.k_max = max(1, int(np.ceil(self.kmax_alpha * (hazard.width + hazard.height))))
            router = (
                self.router_factory(spec.key) if self._factory_wants_key else self.router_factory()
            )
            router.reset(RoutingJob(spec.start, spec.goal, hazard), self.chip)
            job.router = router
        if spec.start == spec.goal:  # nothing to move
            record.success = True
            record.end_cycle = self.cycle
        else:
            self._active.append(job)
        return record

    # ------------------------------------------------------ control cycle
    def _dispense_move(self, job: _ActiveJob) -> None:
        """Deterministic dispensing step (x first, then y), no on-chip actuation."""
        d, g = job.droplet, job.record.goal
        if self.dispense_step is None:
            step_x, step_y = (max(1, s) for s in max_reliable_step(d))
        else:
            step_x = step_y = int(self.dispense_step)
        dx, dy = g.xa - d.xa, g.ya - d.ya
        if dx != 0:
            job.droplet = d.shift(int(np.sign(dx)) * min(abs(dx), step_x), 0)
        elif dy != 0:
            job.droplet = d.shift(0, int(np.sign(dy)) * min(abs(dy), step_y))

    def step(self) -> bool:
        """Run one control cycle; returns :attr:`finished`."""
        if self._finished:
            return True
        self.process_states()
        if self.all_done:
            self._finished = True
            return True
        if not self._active:  # nothing runs and nothing can start
            return self._fail("deadlock")
        if self.cycle >= self.max_cycles:
            return self._fail("max_cycles")

        # Plan: every router observes the chip as it is at the start of the cycle.
        routed = [job for job in self._active if job.router is not None]
        plans = []
        for job in routed:
            rec = job.record
            assert rec.hazard is not None and rec.k_max is not None and job.router is not None
            state = RoutingState(
                self.chip, job.droplet, rec.goal, rec.hazard, rec.cycles, rec.k_max, job.collision
            )
            action = job.router.act(state)
            plan = plan_move(
                job.droplet, rec.goal, rec.hazard, action,
                job.router.adaptive_step, job.router.fixed_step,
            )
            rec.invalid_actions += int(not plan.valid)
            job.collision = plan.collision
            plans.append(plan)

        # Execute: sample every outcome against one degradation snapshot and
        # actuate the OR-union of all patterns (each MC counted once per cycle).
        degradation = self.chip.effective_degradation()
        union = np.zeros(self.chip.shape, dtype=bool)
        for job, plan in zip(routed, plans):
            assert job.record.hazard is not None
            job.droplet, pattern = execute_move(
                plan, job.droplet, job.record.hazard, degradation, self.rng
            )
            union |= pattern
        for job in self._active:
            if job.router is None:
                self._dispense_move(job)
        self.chip.actuate(union)
        self.total_actuations += int(np.count_nonzero(union))
        self.cycle += 1

        # Book-keeping: arrivals and timeouts.
        still_active: List[_ActiveJob] = []
        failed = False
        for job in self._active:
            rec = job.record
            rec.cycles += 1
            if job.droplet == rec.goal:
                rec.success = True
                rec.end_cycle = self.cycle
                continue
            if rec.k_max is not None and rec.cycles >= rec.k_max and not rec.timed_out:
                rec.timed_out = True
                if self.on_timeout == "skip":
                    # Reference behaviour: the job counts as done although the
                    # droplet did not arrive; successors use the listed starts.
                    rec.end_cycle = self.cycle
                    continue
                if self.on_timeout == "fail":
                    failed = True
            still_active.append(job)
        self._active = still_active
        if failed:
            return self._fail("timeout")
        return False

    def _fail(self, reason: str) -> bool:
        self.failure = reason
        self._finished = True
        return True

    def run(
        self, callback: Optional[Callable[["BioassayExecutor"], None]] = None
    ) -> BioassayResult:
        """Execute until completion or failure; ``callback`` runs after every cycle."""
        while not self._finished:
            before = self.cycle
            self.step()
            if callback is not None and self.cycle != before:
                callback(self)
        return self.result()

    def result(self) -> BioassayResult:
        return BioassayResult(
            success=self.success,
            cycles=self.cycle,
            job_records=list(self.records),
            total_actuations=self.total_actuations,
            failure=self.failure,
            mo_start=dict(self.mo_start),
            mo_done=dict(self.mo_done),
        )


def execute_bioassay(
    bioassay: Bioassay,
    router_factory: RouterFactory,
    chip: MEDABiochip,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> BioassayResult:
    """Convenience wrapper: ``BioassayExecutor(...).run()``."""
    return BioassayExecutor(bioassay, router_factory, chip, rng, **kwargs).run()
