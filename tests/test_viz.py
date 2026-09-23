"""Tests for the paper-style figures (Figs. 4, 7-9, 14) and episode GIFs in ``viz/``."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

import meda_routing.viz.animation as animation
from meda_routing.core.actions import Action
from meda_routing.core.biochip import MEDABiochip
from meda_routing.core.geometry import Droplet, Rect
from meda_routing.core.jobs import RoutingJob, hazard_bounds
from meda_routing.envs.meda_env import MEDARoutingEnv
from meda_routing.routers.base import Router, RoutingState, run_job
from meda_routing.viz import (
    aggregate_histories,
    draw_completion_cdf,
    draw_routing_path,
    draw_training,
    frame_scale,
    load_history,
    plot_completion_cdf,
    plot_routing_path,
    plot_training_comparison,
    plot_training_curves,
    plot_training_panels,
    record_episode,
    router_styles,
)
from meda_routing.viz.plots import BLUE, GREEN, INK, RED, _ecdf_steps

_TOWARD = {
    (0, 1): Action.N, (0, -1): Action.S, (1, 0): Action.E, (-1, 0): Action.W,
    (1, 1): Action.NE, (-1, 1): Action.NW, (1, -1): Action.SE, (-1, -1): Action.SW,
}


# --------------------------------------------------------------------- helpers
def toward(droplet: Droplet, goal: Rect) -> Action:
    """Direction of the goal (health-agnostic greedy move)."""
    return _TOWARD[(int(np.sign(goal.xa - droplet.xa)), int(np.sign(goal.ya - droplet.ya)))]


def away_policy(obs, env) -> Action:
    """Never decreases the distance to the goal, so the episode always times out."""
    base = env.unwrapped
    dx, dy = base.goal.xa - base.droplet.xa, base.goal.ya - base.droplet.ya
    return _TOWARD[(-int(np.sign(dx)) or 1, -int(np.sign(dy)) or 1)]


class GreedyRouter(Router):
    name = "Baseline"

    def act(self, state: RoutingState) -> Action:
        return toward(state.droplet, state.goal)


def greedy_policy(obs, env) -> Action:
    base = env.unwrapped
    return toward(base.droplet, base.goal)


def history(n_epochs: int = 8, seed: int = 0) -> pd.DataFrame:
    """Synthetic ``progress.csv`` contents with the trainer's columns."""
    rng = np.random.default_rng(seed)
    epoch = np.arange(1, n_epochs + 1)
    progress = 1.0 - np.exp(-epoch / 3.0)
    cycles = 60.0 - 35.0 * progress
    return pd.DataFrame(
        {
            "epoch": epoch,
            "timesteps": epoch * 16384,
            "learning_rate": 3.5e-4,
            "mean_score": -90.0 + 85.0 * progress + rng.normal(0.0, 3.0, n_epochs),
            "std_score": 5.0,
            "success_rate": np.clip(0.5 + 0.5 * progress, 0.0, 1.0),
            "mean_cycles": cycles,
            "mean_cycles_success": cycles - 2.0,
            "epoch_seconds": 1.0,
        }
    )


def nonempty(path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ------------------------------------------------------------ training curves
def test_aggregate_histories_mean_min_max_over_runs_of_different_length():
    a = pd.DataFrame({"epoch": [1, 2, 3], "mean_score": [-10.0, 0.0, 10.0],
                      "success_rate": [0.5, 1.0, 1.0], "mean_cycles": [30.0, 20.0, 10.0]})
    b = pd.DataFrame({"epoch": [1, 2], "mean_score": [-20.0, 10.0],
                      "success_rate": [0.0, 0.5], "mean_cycles": [40.0, 30.0]})
    stats = aggregate_histories([a, b])
    assert list(stats.index) == [1, 2, 3]
    np.testing.assert_allclose(stats["score_mean"], [-15.0, 5.0, 10.0])
    np.testing.assert_allclose(stats["score_min"], [-20.0, 0.0, 10.0])
    np.testing.assert_allclose(stats["score_max"], [-10.0, 10.0, 10.0])
    np.testing.assert_allclose(stats["success_pct"], [25.0, 75.0, 100.0])  # percent, as in the paper
    np.testing.assert_allclose(stats["cycles_mean"], [35.0, 25.0, 10.0])
    assert list(stats["n_runs"]) == [2, 2, 1]


def test_load_history_accepts_frames_csv_paths_and_run_dirs(tmp_path):
    frame = history()
    run_dir = tmp_path / "seed_0"
    run_dir.mkdir()
    frame.to_csv(run_dir / "progress.csv", index=False)
    for source in (frame, run_dir / "progress.csv", str(run_dir / "progress.csv"), run_dir):
        pd.testing.assert_frame_equal(load_history(source), frame, check_dtype=False)
    with pytest.raises(ValueError, match="success_rate"):
        load_history(frame.drop(columns="success_rate"))
    with pytest.raises(ValueError):
        aggregate_histories([])
    with pytest.raises(TypeError):
        plot_training_curves({"a": [frame]}, "t", tmp_path / "x.png")


def test_draw_training_puts_the_three_metrics_on_one_axis_in_units_of_100():
    runs = [history(seed=0), history(seed=1)]
    stats = aggregate_histories(runs)
    fig = Figure()
    ax = fig.add_subplot()
    draw_training(ax, runs)
    lines = {line.get_label(): line for line in ax.get_lines()}
    np.testing.assert_allclose(lines["Score"].get_ydata(), stats["score_mean"])
    np.testing.assert_allclose(lines["Succ. Rate"].get_ydata(), 100.0 * runs[0]["success_rate"])
    np.testing.assert_allclose(lines["No. Cycles"].get_ydata(), stats["cycles_mean"])
    assert lines["Score"].get_color() == RED and lines["No. Cycles"].get_linestyle() == "-."
    assert lines["Succ. Rate"].get_marker() == "x" and lines["Succ. Rate"].get_linestyle() == "None"
    assert sum(isinstance(c, PolyCollection) for c in ax.collections) == 1  # min-max band
    fig.draw_without_rendering()
    assert "10^{2}" in ax.yaxis.get_offset_text().get_text()  # the paper's "10^2" axis


def test_draw_training_composes_without_clipping_later_data():
    fig = Figure()
    ax = fig.add_subplot()
    draw_training(ax, [history(n_epochs=8)])
    draw_training(ax, {"longer run": [history(n_epochs=30, seed=1)]})
    fig.draw_without_rendering()
    left, right = ax.get_xlim()
    assert left == 0.0 and right >= 30.0  # the first call must not freeze the x-limits
    with pytest.raises(TypeError, match="string"):
        draw_training(ax, [history()], labels="abc")  # not three one-letter labels


def test_plot_training_curves_writes_png_and_pdf(tmp_path):
    paths = []
    for seed in range(3):
        p = tmp_path / f"seed_{seed}" / "progress.csv"
        p.parent.mkdir()
        history(seed=seed).to_csv(p, index=False)
        paths.append(p)
    out = tmp_path / "figs" / "curves.png"
    result = plot_training_curves(paths, "Size 30x30", out, labels=("Score", "Success", "Cycles"), pdf=True)
    assert result == out and nonempty(out) and nonempty(out.with_suffix(".pdf"))


def test_plot_training_curves_handles_a_single_run_with_one_epoch(tmp_path):
    # e.g. ``meda plot-training`` on a one-epoch smoke-test run
    result = plot_training_curves(history(n_epochs=1), "tiny", tmp_path / "tiny")
    assert result == tmp_path / "tiny.png" and nonempty(result)


def test_plot_training_comparison_and_panels(tmp_path):
    random_init = [history(seed=s) for s in range(3)]
    transfer = [history(seed=10 + s) for s in range(3)]
    groups = {"random init": random_init, "transfer": transfer}
    out = plot_training_comparison(groups, "Size 60x60", tmp_path / "fig4.png")
    assert nonempty(out)

    fig = Figure()
    ax = fig.add_subplot()
    draw_training(ax, groups)
    colors = {line.get_label(): line.get_color() for line in ax.get_lines()}
    assert colors["Score (random init)"] == RED and colors["Score (transfer)"] == BLUE

    panels = {"Size 60x60": groups, "Size 90x90": {"transfer": transfer}, "Size 30x30": random_init}
    out = plot_training_panels(panels, tmp_path / "panels.png", title="Fig. 4", ncols=2, pdf=True)
    assert nonempty(out) and nonempty(out.with_suffix(".pdf"))


# -------------------------------------------------------- completion (Fig. 9)
def test_ecdf_counts_failed_trials_in_the_denominator():
    x, y = _ecdf_steps(np.array([5.0, 7.0, np.inf, 7.0]), 4.0, 8.0)
    np.testing.assert_allclose(x, [4.0, 5.0, 7.0, 8.0])
    np.testing.assert_allclose(y, [0.0, 0.25, 0.75, 0.75])  # levels off at the success rate
    x, y = _ecdf_steps(np.array([np.inf, np.nan]), 0.0, 1.0)
    np.testing.assert_allclose(y, [0.0, 0.0])


def test_router_styles_follow_the_paper_and_keep_other_names_distinct():
    styles = router_styles(["DRL", "gnn", "baseline", "Formal", "other", "drl"])
    assert styles["DRL"] == (RED, "-")
    assert styles["baseline"] == (INK, ":")
    assert styles["Formal"] == (BLUE, "--")
    extra = [styles[n] for n in ("gnn", "other", "drl")]
    assert len({(c, str(ls)) for c, ls in extra}) == 3
    assert all(c not in (RED, BLUE, INK) for c, _ in extra)


def test_draw_completion_cdf_steps_and_styles():
    results = {
        "Baseline": np.array([210.0, 220.0, 230.0, 240.0]),
        "Formal": [200.0, 205.0, 210.0, 215.0],
        "DRL": np.array([180.0, 185.0, np.inf, 190.0]),
        "GNN": np.full(3, np.inf),
    }
    fig = Figure()
    ax = fig.add_subplot()
    draw_completion_cdf(ax, results, "COVID-RAT")
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert (lines["Baseline"].get_color(), lines["Baseline"].get_linestyle()) == (INK, ":")
    assert (lines["Formal"].get_color(), lines["Formal"].get_linestyle()) == (BLUE, "--")
    assert (lines["DRL"].get_color(), lines["DRL"].get_linestyle()) == (RED, "-")
    assert lines["DRL"].get_drawstyle() == "steps-post"
    assert lines["DRL"].get_ydata()[-1] == pytest.approx(0.75)
    assert np.all(lines["GNN"].get_ydata() == 0.0)
    assert ax.get_xlim() == (179.0, 241.0)  # [min K - 1, max K + 1] over all routers
    with pytest.raises(ValueError):
        draw_completion_cdf(ax, {})
    with pytest.raises(ValueError):
        draw_completion_cdf(ax, {"DRL": []})


def test_plot_completion_cdf_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    results = {
        "Baseline": rng.normal(212.0, 8.0, 200).round(),
        "Formal": rng.normal(205.0, 7.0, 200).round(),
        "DRL": np.where(rng.random(200) < 0.95, rng.normal(188.0, 9.0, 200).round(), np.inf),
    }
    out = plot_completion_cdf(results, "COVID-RAT", tmp_path / "rat_cdf.png")
    assert out == tmp_path / "rat_cdf.png" and nonempty(out)


# ------------------------------------------------------ routing path (Fig. 14)
@pytest.fixture(scope="module")
def faulty_job():
    chip = MEDABiochip(20, 16, rng=np.random.default_rng(0))
    chip.reset()
    start, goal = Droplet.at(2, 6, 3, 3), Droplet.at(14, 7, 3, 3)
    chip.set_faults([(8, 6), (9, 6), (8, 7), (9, 7)])
    chip.inject_faults(0.03, protect=[start, goal], hidden=True)
    return chip, RoutingJob(start, goal, hazard_bounds(start, goal, 20, 16))


def test_plot_routing_path_single_and_compared(faulty_job, tmp_path):
    chip, job = faulty_job
    result = run_job(GreedyRouter(), job, chip.copy(), np.random.default_rng(0), record_path=True)
    health = chip.health()
    out = plot_routing_path(health, chip.health_levels, result.path, job.goal, job.hazard,
                            chip.faults, tmp_path / "path.png", title="T1")
    assert out == tmp_path / "path.png" and nonempty(out)

    stuck = [job.start, job.start.shift(2, 0), job.start.shift(2, 0), job.start.shift(4, 0)]
    paths = {"Baseline": stuck, "DRL": [d.as_tuple() for d in result.path]}  # tuples work too
    out = plot_routing_path(health, chip.health_levels, paths, job.goal, job.hazard, chip.faults,
                            tmp_path / "compare.png", hidden_defects=chip.hidden_defects, pdf=True)
    assert nonempty(out) and nonempty(out.with_suffix(".pdf"))

    fig = Figure()
    ax = fig.add_subplot()
    draw_routing_path(ax, health, chip.health_levels, paths, job.goal, job.hazard, chip.faults)
    squares = [c for c in ax.collections if isinstance(c, PatchCollection)]
    assert len(squares) == 1 and len(squares[0].get_paths()) == int(chip.faults.sum())
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert f"DRL (k = {len(result.path) - 1})" in labels
    assert "Baseline (k = 3, goal missed)" in labels and "Stalled cycles" in labels


def test_plot_routing_path_validates_inputs(faulty_job, tmp_path):
    chip, job = faulty_job
    health = chip.health()
    with pytest.raises(ValueError, match="outside"):
        plot_routing_path(health, 4, [job.start], Droplet.at(19, 15, 3, 3), job.hazard, None, tmp_path / "x.png")
    with pytest.raises(ValueError, match="shape"):
        plot_routing_path(health, 4, [job.start], job.goal, job.hazard, np.zeros((3, 3), bool), tmp_path / "x.png")
    with pytest.raises(TypeError, match="out_path"):
        plot_routing_path(health, 4, [job.start], job.goal, job.hazard)
    with pytest.raises(TypeError, match="given for 'faults'"):  # output path in the faults slot
        plot_routing_path(health, 4, [job.start], job.goal, job.hazard, tmp_path / "x.png")
    assert not (tmp_path / "x.png").exists()


def test_draw_routing_path_geometry_is_north_up_and_zero_based():
    # Non-square chip, [x, y]-indexed inputs and 0-based inclusive rectangles.
    width, height = 12, 7
    health = np.arange(width * height).reshape(width, height) % 4
    faults = np.zeros((width, height), dtype=bool)
    faults[9, 5] = faults[2, 1] = True
    start, goal = Droplet(0, 4, 1, 6), Droplet(9, 0, 10, 2)  # north-west -> south-east
    hazard = Rect(0, 0, 10, 6)  # column x = 11 lies outside the routing zone
    path = [start, start.shift(2, 0), start.shift(2, 0), start.shift(5, -2), goal]
    fig = Figure()
    ax = fig.add_subplot()
    draw_routing_path(ax, health, 4, path, goal, hazard, faults, colorbar=False)

    heat, veil = ax.images
    assert heat.origin == "lower" and list(heat.get_extent()) == [-0.5, width - 0.5, -0.5, height - 0.5]
    np.testing.assert_array_equal(np.asarray(heat.get_array()), health.T)  # rows = y (north up)
    alpha = np.asarray(veil.get_array())[:, :, 3]  # [y, x]
    assert (alpha[:, 11] > 0).all() and (alpha[:, :11] == 0).all()
    (squares,) = [c for c in ax.collections if isinstance(c, PatchCollection)]
    centers = {tuple(np.round(p.vertices[:4].mean(axis=0), 6)) for p in squares.get_paths()}
    assert centers == {(9.0, 5.0), (2.0, 1.0)}  # the faulty MCs (x, y)
    boxes = {(p.get_xy(), p.get_width(), p.get_height(), to_rgba(p.get_edgecolor(), 1.0))
             for p in ax.patches if isinstance(p, Rectangle)}
    assert ((-0.5, 3.5), 2, 3, to_rgba(INK)) in boxes  # start: MCs x 0..1, y 4..6
    assert ((8.5, -0.5), 2, 3, to_rgba(GREEN)) in boxes  # goal: MCs x 9..10, y 0..2
    assert ((-0.5, -0.5), 11, 7, to_rgba(INK)) in boxes  # hazard bounds
    track = next(line for line in ax.get_lines() if line.get_label().startswith("Droplet path"))
    np.testing.assert_allclose(track.get_xydata(), [(0.5, 5.0), (2.5, 5.0), (2.5, 5.0), (5.5, 3.0), (9.5, 1.0)])
    assert track.get_label() == "Droplet path (k = 4)"


def test_draw_routing_path_with_many_health_levels(faulty_job):
    chip, job = faulty_job
    fig = Figure()
    ax = fig.add_subplot()
    draw_routing_path(ax, chip.health() * 10, 32, [job.start, job.start.shift(1, 0)], job.goal, job.hazard,
                      colorbar=True, legend=False)  # continuous gray ramp beyond 16 levels
    assert ax.get_legend() is None and ax.images[0].norm.vmax == 31


# ------------------------------------------------------------- episode GIFs
def test_frame_scale():
    assert frame_scale(30, 30) == 13
    assert frame_scale(10, 10) == 16  # capped
    assert frame_scale(60, 30) == 6
    assert frame_scale(1000, 10) == 1


def test_record_episode_random_policy(tmp_path):
    env = MEDARoutingEnv(width=10, height=10, obs_size=None, jobs={"droplet_sizes": [(2, 2)]})
    rng = np.random.default_rng(0)
    info = record_episode(env, lambda obs, env: np.array([rng.integers(8)]), tmp_path / "rand.gif",
                          max_steps=5, fps=4, seed=1)
    assert set(info) == {"success", "cycles", "frames", "reward", "out_path"}
    assert 1 <= info["cycles"] <= 5 and info["frames"] == info["cycles"] + 1
    assert nonempty(info["out_path"])
    with Image.open(info["out_path"]) as gif:
        assert gif.format == "GIF" and gif.size == (240, 186)  # 160 px chip, widened for the caption
        assert 2 <= gif.n_frames <= info["frames"] and gif.info["duration"] == 250


def test_record_episode_greedy_policy_reaches_goal(tmp_path):
    # a fresh chip starts unworn (D = 1 before any actuation, "zero" initial wear),
    # so the greedy droplet reaches the goal on this seeded episode
    env = MEDARoutingEnv(width=16, height=16, obs_size=None, jobs={"droplet_sizes": [(3, 3)]})
    info = record_episode(env, greedy_policy, tmp_path / "greedy", seed=3, record_path=True)
    assert info["out_path"] == tmp_path / "greedy.gif" and nonempty(info["out_path"])
    assert info["success"] and info["reward"] > 0
    path = info["path"]
    assert path[0] == env.job.start and path[-1] == env.job.goal
    assert info["cycles"] == len(path) - 1 and info["frames"] == len(path)


def test_episode_frames_show_the_chip_pixel_for_pixel_under_the_trail():
    width, height, scale = 14, 9, 5
    env = MEDARoutingEnv(width=width, height=height, obs_size=None)
    start, goal = Droplet.at(1, 6, 2, 2), Droplet.at(10, 0, 2, 2)  # north-west -> south-east
    env.reset(seed=0, options={"job": RoutingJob(start, goal, hazard_bounds(start, goal, width, height))})
    frame = env.render_frame(scale=scale)
    fig, image, *_ = animation._episode_figure(frame, width, height)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[:, :, :3]
    ax = image.axes
    left, top = ax.transAxes.transform((0.0, 1.0))
    col0, row0 = int(round(left)), int(round(rgb.shape[0] - top))
    # the frame is shown unscaled and unflipped below the caption
    np.testing.assert_array_equal(rgb[row0:row0 + height * scale, col0:col0 + width * scale], frame)
    # the trail's MC coordinates land on the droplet (blue) and goal (green) the env draws
    for rect, channel in ((start, 2), (goal, 1)):
        x, y = ax.transData.transform(((rect.xa + rect.xb) / 2.0, (rect.ya + rect.yb) / 2.0))
        row, col = int(rgb.shape[0] - y), int(x)
        assert row0 + (height - 1 - rect.yb) * scale <= row < row0 + (height - rect.ya) * scale  # north up
        assert col0 + rect.xa * scale <= col < col0 + (rect.xb + 1) * scale
        assert int(np.argmax(rgb[row, col])) == channel


def test_record_episode_captions_timeouts_and_caps(tmp_path, monkeypatch):
    captions = []
    make_figure = animation._episode_figure

    def spy(*args, **kwargs):
        fig, image, trail, left, right = make_figure(*args, **kwargs)
        set_text = right.set_text
        right.set_text = lambda text: (captions.append(text), set_text(text))
        return fig, image, trail, left, right

    monkeypatch.setattr(animation, "_episode_figure", spy)
    # a timeout reported as ``terminated`` must not be captioned as a success
    env = MEDARoutingEnv(width=10, height=10, obs_size=None, timeout_terminal=True, jobs={"droplet_sizes": [(2, 2)]})
    info = record_episode(env, away_policy, tmp_path / "timeout.gif", seed=0, end_pause=0.0)
    assert not info["success"] and info["cycles"] == env.k_max
    assert captions[-1].endswith("(k_max)") and "goal reached" not in captions[-1]
    info = record_episode(env, away_policy, tmp_path / "capped.gif", max_steps=2, seed=0, end_pause=0.0)
    assert info["cycles"] == 2 and captions[-1].endswith("(max_steps)")
    info = record_episode(env, greedy_policy, tmp_path / "done.gif", seed=1, end_pause=0.0)
    assert info["success"] and captions[-1].endswith("goal reached")


def test_record_episode_output_formats(tmp_path):
    env = MEDARoutingEnv(width=8, height=8, obs_size=None, jobs={"droplet_sizes": [(2, 2)]})
    for name, expected, fmt in (("ep.mp4", "ep.mp4.gif", "GIF"), ("ep.png", "ep.png", "PNG")):
        # Pillow cannot write .mp4: fall back to GIF before the episode is recorded
        info = record_episode(env, away_policy, tmp_path / name, max_steps=2, seed=0, end_pause=0.0)
        assert info["out_path"] == tmp_path / expected
        with Image.open(info["out_path"]) as im:
            assert im.format == fmt and im.n_frames == 3


def test_record_episode_through_gym_make_wrappers(tmp_path):
    import gymnasium as gym

    from meda_routing.envs import ENV_ID

    env = gym.make(ENV_ID, width=12, height=12, obs_size=None)
    info = record_episode(env, greedy_policy, tmp_path / "wrapped.gif", seed=0, fps=8, end_pause=0.0)
    assert info["success"] and nonempty(info["out_path"])
    env.close()


def test_record_episode_writes_nothing_when_the_policy_fails(tmp_path):
    def broken(obs, env):
        raise RuntimeError("policy failed")

    env = MEDARoutingEnv(width=10, height=10, obs_size=None)
    with pytest.raises(RuntimeError, match="policy failed"):
        record_episode(env, broken, tmp_path / "broken.gif", seed=0)
    assert not (tmp_path / "broken.gif").exists()
    with pytest.raises(ValueError):
        record_episode(env, greedy_policy, tmp_path / "x.gif", fps=0)


def test_viz_renders_headless_without_pyplot(tmp_path):
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from meda_routing.viz import plot_completion_cdf\n"
        f"plot_completion_cdf({{'DRL': np.array([3.0, 4.0])}}, 't', {str(tmp_path / 'x.png')!r})\n"
        "print('matplotlib.pyplot' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
    assert nonempty(tmp_path / "x.png")
