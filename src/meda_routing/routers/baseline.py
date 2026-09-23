"""Health-agnostic shortest-path routing baseline (Sec. V-B and VI-C).

The paper compares the DRL policy against "health-agnostic policies that aim
to minimize the time to reach the target without knowledge of the MC health
levels" (Sec. V-B, the *baseline* of Fig. 9) and, on the PCB prototype,
against "a baseline routing approach, which adopts the shortest-path
algorithm" (Sec. VI-C).  The reference implementation's only health-agnostic
mode runs the formal synthesizer on an all-healthy health matrix
(``bShortest`` in ``matlab/MedaSchedulerClass.m``, ``mdSynthesizeJobStr``).
With that matrix every strategy that arrives within the bound ``K`` is
optimal, so the explicit shortest path used here is the choice that matches
the paper's description ("minimize the time to reach the target").

Without obstacles a shortest path on the 8-connected MC grid is obtained by
moving diagonally while the droplet is off the goal in both axes and
cardinally afterwards (``max(|dx|, |dy|)`` single-step cycles).  The
direction is recomputed from the sensed droplet location every cycle, so a
failed movement is simply retried; a droplet whose frontier is fully
degraded therefore stays stuck, as in Fig. 14(b)-(c).
"""

from __future__ import annotations

from typing import Dict, Tuple

from ..core.actions import DIRECTIONS, Action
from ..core.geometry import Rect
from .base import Router, RoutingState

#: ``step_mode`` -> ``(adaptive_step, fixed_step)`` of :class:`Router`.
#: ``"single"``/``"double"`` move 1/2 MCs per axis and cycle (the MEDAY/MEDAX
#: models of [18]); ``"adaptive"`` uses Algorithm 1.
STEP_MODES: Dict[str, Tuple[bool, int]] = {
    "single": (False, 1),
    "double": (False, 2),
    "adaptive": (True, 1),
}

#: Inverse of :data:`DIRECTIONS`: unit vector ``(ux, uy)`` -> action.
_ACTION_OF: Dict[Tuple[int, int], Action] = {vec: act for act, vec in DIRECTIONS.items()}


def parse_step_mode(step_mode: str) -> Tuple[bool, int]:
    """``(adaptive_step, fixed_step)`` for a step-mode name, see :data:`STEP_MODES`."""
    try:
        return STEP_MODES[step_mode]
    except KeyError:
        raise ValueError(
            f"unknown step_mode {step_mode!r}; expected one of {sorted(STEP_MODES)}"
        ) from None


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


def shortest_path_action(droplet: Rect, goal: Rect) -> Action:
    """Direction of a shortest 8-connected path from ``droplet`` to ``goal``.

    Diagonal while both offsets are non-zero, cardinal otherwise.  Because the
    goal lies inside the routing zone, this action is always valid (the
    droplet cannot touch the zone boundary on the side of the goal).  At the
    goal there is nothing to do and ``Action.N`` is returned by convention;
    callers stop routing once the goal is reached (see ``run_job``).
    """
    ux = _sign(goal.xa - droplet.xa)
    uy = _sign(goal.ya - droplet.ya)
    if ux == 0 and uy == 0:
        return Action.N
    return _ACTION_OF[(ux, uy)]


class ShortestPathRouter(Router):
    """Health-agnostic shortest-path router ("baseline" in Fig. 9 and 13-14).

    Parameters
    ----------
    step_mode:
        ``"single"`` (default, one MC per axis and cycle), ``"double"`` or
        ``"adaptive"`` (Algorithm 1).  The adaptive variant isolates the
        effect of health awareness from that of the adaptive step size.
    """

    name = "baseline"

    def __init__(self, step_mode: str = "single") -> None:
        self.step_mode = step_mode
        self.adaptive_step, self.fixed_step = parse_step_mode(step_mode)

    def act(self, state: RoutingState) -> Action:
        return shortest_path_action(state.droplet, state.goal)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(step_mode={self.step_mode!r})"
