"""Head-to-head comparison of routers on random routing jobs (Sec. VI).

Section VI compares the DRL policy with a shortest-path baseline on chips
with 0% and 10% injected (sensed) faults plus ~5% inherent defects that the
health sensors cannot see.  :func:`compare_routers` reproduces this protocol
in simulation: every router is run on *identical* routing jobs, chips and
random draws (common random numbers), so differences reflect the policies
only.  Jobs and chips are drawn exactly like training episodes of
:class:`~meda_routing.envs.MEDARoutingEnv` with the given configuration.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from ..envs.meda_env import EnvConfig, MEDARoutingEnv
from .base import Router, run_job

RouterFactory = Callable[[], Router]


def compare_routers(
    routers: Dict[str, RouterFactory],
    env_config: EnvConfig,
    n_jobs: int = 500,
    seed: int = 0,
    k_max: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Run every router on the same ``n_jobs`` random jobs.

    Returns one row per (job, router) with columns ``job, router, success,
    cycles, invalid_actions, distance, droplet_w, droplet_h, seconds``.
    """
    env = MEDARoutingEnv(env_config)
    rows = []
    for i in range(n_jobs):
        env.reset(seed=seed + i)  # samples job, degradation parameters and faults
        job, chip0 = env.job, env.chip
        kmax = k_max if k_max is not None else env.k_max
        for name, factory in routers.items():
            chip = chip0.copy()
            chip.rng = np.random.default_rng(seed + i)
            rng = np.random.default_rng([seed, i])  # identical movement draws per router
            t0 = time.perf_counter()
            result = run_job(factory(), job, chip, rng, k_max=kmax)
            rows.append(
                {
                    "job": i,
                    "router": name,
                    "success": result.success,
                    "cycles": result.cycles,
                    "invalid_actions": result.invalid_actions,
                    "distance": job.start.manhattan(job.goal),
                    "droplet_w": job.start.width,
                    "droplet_h": job.start.height,
                    "seconds": time.perf_counter() - t0,
                }
            )
        if progress is not None:
            progress(i + 1, n_jobs)
    return pd.DataFrame(rows)


def summarize_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Success rate and cycle statistics per router."""

    grouped = df.groupby("router", sort=False)
    summary = pd.DataFrame(
        {
            "jobs": grouped.size(),
            "success_rate": grouped["success"].mean(),
            "mean_cycles": grouped["cycles"].mean(),
            "mean_cycles_success": df[df["success"]].groupby("router", sort=False)["cycles"].mean(),
            "invalid_actions_per_job": grouped["invalid_actions"].mean(),
            "ms_per_job": 1e3 * grouped["seconds"].mean(),
        }
    )
    return summary.reindex(grouped.size().index)
