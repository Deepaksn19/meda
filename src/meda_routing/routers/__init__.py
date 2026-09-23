"""Routing policies behind a common interface.

* :class:`DRLRouter` — a trained PPO agent (the paper's method);
* :class:`ShortestPathRouter` — health-agnostic shortest path ("baseline");
* :class:`FormalRouter` — optimal finite-horizon strategies on the sensed
  health map, standing in for the PRISM-games synthesizer ("formal").
"""

from .base import JobResult, Router, RoutingState, run_job
from .baseline import ShortestPathRouter
from .formal import FormalRouter

__all__ = [
    "DRLRouter",
    "FormalRouter",
    "JobResult",
    "Router",
    "RoutingState",
    "ShortestPathRouter",
    "compare_routers",
    "run_job",
    "summarize_comparison",
]


def __getattr__(name: str):
    # DRLRouter pulls in torch and Stable-Baselines3, the comparison harness
    # pandas and gymnasium; import them on first use so the baselines and the
    # bioassay scheduler stay lightweight.
    if name == "DRLRouter":
        from .drl import DRLRouter

        return DRLRouter
    if name in ("compare_routers", "summarize_comparison"):
        from . import compare

        return getattr(compare, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
