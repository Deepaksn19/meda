"""End-to-end checks of the ``meda`` command-line interface on tiny problems."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from meda_routing.cli import build_parser, main
from meda_routing.training.config import load_config

TINY = [
    "env.width=10",
    "env.height=10",
    "env.obs_size=[10,10]",
    "env.jobs.droplet_sizes=[[2,2],[3,3]]",
    "agent.extractor_kwargs={channels: [4, 8, 8], hidden_dim: 16}",
    "ppo.n_envs=2",
    "ppo.n_steps=32",
    "ppo.batch_size=32",
    "ppo.device=cpu",
    "schedule.epochs=1",
    "schedule.steps_per_epoch=64",
    "eval.episodes=4",
    "eval.n_envs=2",
    "verbose=0",
]


def _set(items):
    out = []
    for item in items:
        out += ["--set", item]
    return out


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    main(["train", "--output-dir", str(out)] + _set(TINY + ["name=cli"]))
    return out / "cli" / "seed_0"


def test_parser_lists_all_commands():
    parser = build_parser()
    for cmd in ("train", "curriculum", "evaluate", "compare", "bioassay", "plot-training", "render"):
        assert parser.parse_args(_minimal_args(cmd)).command == cmd


def _minimal_args(cmd):
    return {
        "train": ["train"],
        "curriculum": ["curriculum", "-c", "x.yaml"],
        "evaluate": ["evaluate", "-m", "m.zip"],
        "compare": ["compare"],
        "bioassay": ["bioassay"],
        "plot-training": ["plot-training", "run"],
        "render": ["render", "-m", "m.zip"],
    }[cmd]


def test_train_writes_run(trained_run):
    assert (trained_run / "model.zip").exists()
    assert load_config(trained_run / "config.yaml").env.width == 10


def test_evaluate(trained_run, tmp_path, capsys):
    out = tmp_path / "metrics.json"
    main(["evaluate", "-m", str(trained_run), "--episodes", "4", "--n-envs", "2", "--out", str(out)])
    metrics = json.loads(out.read_text())
    assert metrics["episodes"] == 4 and 0.0 <= metrics["success_rate"] <= 1.0


def test_compare_same_jobs_for_all_routers(trained_run, tmp_path):
    out = tmp_path / "cmp.csv"
    main(
        ["compare", "-m", str(trained_run), "--routers", "drl", "baseline", "formal", "--jobs", "5",
         "--out", str(out)] + _set(["env.fault_fraction=0.1"])
    )
    df = pd.read_csv(out)
    assert set(df["router"]) == {"DRL", "Baseline", "Formal"} and len(df) == 15
    # identical jobs for every router
    per_router = df.pivot(index="job", columns="router", values="distance")
    assert (per_router.nunique(axis=1) == 1).all()
    assert (tmp_path / "cmp_summary.csv").exists()


def test_bioassay_simple_baseline(tmp_path):
    main(["bioassay", "--assay", "simple", "--routers", "baseline", "--trials", "2", "--out", str(tmp_path)])
    cycles = np.loadtxt(tmp_path / "simple_baseline_cycles.txt")
    assert cycles.shape == (2,) and np.isfinite(cycles).all()
    assert (tmp_path / "simple_summary.csv").exists()


def test_plot_training_and_render(trained_run, tmp_path):
    fig = tmp_path / "curves.png"
    main(["plot-training", str(trained_run), "--out", str(fig)])
    assert fig.exists() and fig.stat().st_size > 0
    gif = tmp_path / "ep.gif"
    main(["render", "-m", str(trained_run), "--out", str(gif)])
    assert gif.exists() and gif.stat().st_size > 0


def test_drl_requires_model():
    with pytest.raises(SystemExit):
        main(["compare", "--routers", "drl", "--jobs", "1"])
