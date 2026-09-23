"""Router comparison harness (Sec. VI protocol in simulation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meda_routing.core.dynamics import step_size
from meda_routing.core.actions import Action
from meda_routing.core.geometry import Droplet
from meda_routing.envs import EnvConfig
from meda_routing.routers import FormalRouter, ShortestPathRouter, compare_routers, summarize_comparison


def _cfg(**kw):
    return EnvConfig.from_dict({"width": 16, "height": 16, "jobs": {"droplet_sizes": [[2, 2], [3, 3]]}, **kw})


def test_all_routers_see_identical_jobs_and_chips():
    df = compare_routers(
        {"a": ShortestPathRouter, "b": lambda: ShortestPathRouter("adaptive"), "c": FormalRouter},
        _cfg(fault_fraction=0.1),
        n_jobs=8,
        seed=3,
    )
    assert len(df) == 24
    per_job = df.pivot(index="job", columns="router", values="distance")
    assert (per_job.nunique(axis=1) == 1).all()
    sizes = df.pivot(index="job", columns="router", values="droplet_w")
    assert (sizes.nunique(axis=1) == 1).all()


def test_comparison_is_deterministic():
    routers = {"base": ShortestPathRouter, "formal": FormalRouter}
    a = compare_routers(routers, _cfg(fault_fraction=0.2, hidden_defect_fraction=0.05), 6, seed=11)
    b = compare_routers(routers, _cfg(fault_fraction=0.2, hidden_defect_fraction=0.05), 6, seed=11)
    cols = ["job", "router", "success", "cycles", "invalid_actions", "distance"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_summary_columns_and_latency():
    df = compare_routers({"base": ShortestPathRouter}, _cfg(), 5, seed=0)
    summary = summarize_comparison(df)
    for col in ("success_rate", "mean_cycles", "mean_cycles_success", "ms_per_job", "ms_per_cycle"):
        assert col in summary.columns
    row = summary.loc["base"]
    assert row["success_rate"] == 1.0  # healthy chip, shortest path always arrives
    assert 0 < row["ms_per_cycle"] <= row["ms_per_job"]
    assert row["ms_per_cycle"] < 200  # the paper's per-cycle budget (Sec. II-D)


def test_double_step_mode_matches_the_medax_action_set():
    """[18]: aNN/aSS/aEE/aWW exist, double-diagonal moves do not."""
    d, far = Droplet.at(5, 5, 4, 4), Droplet.at(20, 20, 4, 4)
    assert step_size(d, far, Action.E, adaptive=False, fixed_step=2, diagonal_step=1) == (2, 0)
    assert step_size(d, far, Action.NE, adaptive=False, fixed_step=2, diagonal_step=1) == (1, 1)
    assert step_size(d, far, Action.NE, adaptive=False, fixed_step=2) == (2, 2)  # default unchanged
    assert ShortestPathRouter("double").diagonal_step == 1
    assert FormalRouter("double").diagonal_step == 1
    assert ShortestPathRouter("single").diagonal_step is None
    assert np.isnan(FormalRouter().success_probability)
