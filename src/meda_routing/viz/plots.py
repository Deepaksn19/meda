"""Figures in the style of the paper's evaluation (Sec. V-B and VI-D).

* **Training curves** (Figs. 4, 7 and 8).  After every training epoch the
  agent is evaluated on 500 random routing jobs and every experiment is
  repeated five times (Sec. V-B).  :func:`plot_training_curves` draws one
  configuration: the mean score over the repeats with its min-max band (red),
  the success rate (markers) and the average number of cycles (dash-dot).
  :func:`plot_training_comparison` overlays configurations, e.g. random
  initialization (red) against transfer learning (blue) as in Fig. 4, and
  :func:`plot_training_panels` lays out one panel per chip size or fault level.
* **Bioassay completion** (Fig. 9).  :func:`plot_completion_cdf` draws, per
  router, the empirical probability ``P[K <= k]`` that the bioassay completes
  within ``k`` control cycles.
* **Routing paths** (Fig. 14).  :func:`plot_routing_path` draws droplet
  trajectories over the chip's health map with the routing zone, the start
  and goal locations and the faulty MCs.

Units of the training plots: the paper puts the three metrics on one y-axis
whose tick labels are factored by ``10^2`` (the "·10^2" of Figs. 4, 7, 8).
The reference code (``train.py``, ``plotAgentPerformance``) plots the raw
episode score, the raw number of cycles and the success rate in percent (its
environment reports ``b_at_goal = 100`` per success).  Our trainer logs the
success rate as a fraction, so it is multiplied by 100 here and the y-axis is
labelled in units of ``10^2`` like the paper's: 100 % success reads as 1.

Every ``plot_*`` function builds its figure with the object-oriented
:class:`matplotlib.figure.Figure` API and saves it through the Agg (raster)
or PDF canvas, then returns the path of the saved file.  pyplot is never
imported and its backend never switched, so the functions run headless and
have no side effects in notebooks or GUI sessions.  The ``draw_*`` functions
draw into a caller-supplied :class:`~matplotlib.axes.Axes` for composing
custom layouts (e.g. both assays of Fig. 9 side by side).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize, to_rgb, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from ..core.geometry import Rect

PathLike = Union[str, os.PathLike]
#: One training run: a DataFrame, a ``progress.csv`` path or a run directory.
HistoryLike = Union[pd.DataFrame, str, os.PathLike]
#: A rectangle as :class:`Rect` / :class:`Droplet` or ``(xa, ya, xb, yb)``.
RectLike = Union[Rect, Sequence[int]]

# ------------------------------------------------------------------- styles
# The paper's colors (red / blue / green / black), taken from a palette
# checked for color-vision-deficiency separation.  Every series also differs
# in line style or marker, so identity never rests on hue alone.
RED = "#e34948"
BLUE = "#2a78d6"
GREEN = "#008300"
INK = "#0b0b0b"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
MUTED = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
#: Further series, in a fixed order that stays distinguishable from red and blue.
EXTRA_COLORS: Tuple[str, ...] = ("#1baf7a", VIOLET, "#eda100", GREEN)
_EXTRA_LINESTYLES = ("-.", (0, (6, 2)), (0, (3, 1.5, 1, 1.5, 1, 1.5)), (0, (1, 1.5, 4, 1.5)))

#: Router styles of Fig. 9 (names are matched case-insensitively).
PAPER_ROUTER_STYLES: Dict[str, Tuple[str, str]] = {
    "baseline": (INK, ":"),
    "formal": (BLUE, "--"),
    "drl": (RED, "-"),
}

#: Legend labels of the three training metrics (Figs. 4 and 7).
TRAINING_LABELS: Tuple[str, str, str] = ("Score", "Succ. Rate", "No. Cycles")
#: Columns a training history must provide (see :func:`load_history`).
REQUIRED_COLUMNS: Tuple[str, ...] = ("epoch", "mean_score", "success_rate", "mean_cycles")
# Configuration i of a comparison: color and success-rate marker (Fig. 4:
# red crosses for random initialization, blue circles for transfer learning).
_GROUP_COLORS = (RED, BLUE) + EXTRA_COLORS[:3]
_GROUP_MARKERS = ("x", "o", "^", "s", "D", "v", "P", "*")
_UNFILLED_MARKERS = ("x", "+", "1", "2", "3", "4")

_RC = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def router_styles(names: Sequence[str]) -> Dict[str, Tuple[str, object]]:
    """``name -> (color, linestyle)`` for the routers of a figure.

    ``Baseline`` is dotted black, ``Formal`` dashed blue and ``DRL`` solid red
    as in Fig. 9; other names get distinct color / line-style pairs in order.
    """
    styles: Dict[str, Tuple[str, object]] = {}
    used = set()
    extra = 0
    for name in names:
        key = str(name).strip().lower()
        if key in PAPER_ROUTER_STYLES and key not in used:
            styles[name] = PAPER_ROUTER_STYLES[key]
            used.add(key)
            continue
        n = len(EXTRA_COLORS)
        linestyle = _EXTRA_LINESTYLES[(extra + extra // n) % len(_EXTRA_LINESTYLES)]
        styles[name] = (EXTRA_COLORS[extra % n], linestyle)
        extra += 1
    return styles


# --------------------------------------------------------------- helpers
def _save(fig: Figure, out_path: PathLike, pdf: bool, dpi: int) -> Path:
    """Save ``fig`` (PNG unless the suffix names another format); optional PDF copy."""
    path = Path(out_path)
    if path.suffix.lower().lstrip(".") not in fig.canvas.get_supported_filetypes():
        path = path.with_name(path.name + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if pdf and path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return path


def _nice_ticks(integer: bool = False) -> MaxNLocator:
    """Ticks on multiples of 1, 2 or 5 (the paper's 0, 5, 10, ... epochs).

    ``nbins="auto"`` scales the tick count with the axis length, so labels
    do not collide on short axes (small panels, chips only a few MCs high).
    """
    return MaxNLocator(nbins="auto", integer=integer, steps=[1, 2, 5, 10])


def _style_axes(ax: Axes) -> None:
    """Recessive hairline grid and axes."""
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _figure_legend(fig: Figure, axes: Sequence[Axes]) -> None:
    """One legend for all ``axes`` (duplicates removed), right of the figure."""
    handles: List[object] = []
    labels: List[str] = []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))


def _as_rect(rect: RectLike) -> Rect:
    if isinstance(rect, Rect):
        return rect
    xa, ya, xb, yb = (int(v) for v in rect)
    return Rect(xa, ya, xb, yb)


# ------------------------------------------------------ training curves
def load_history(history: HistoryLike) -> pd.DataFrame:
    """One training history: a DataFrame, a ``progress.csv`` path or a run directory.

    The trainer writes one row per epoch with the columns ``epoch``,
    ``timesteps``, ``learning_rate``, ``mean_score``, ``std_score``,
    ``success_rate`` (a fraction in [0, 1]), ``mean_cycles``,
    ``mean_cycles_success`` and ``epoch_seconds``.  Only
    :data:`REQUIRED_COLUMNS` are needed here.
    """
    if isinstance(history, pd.DataFrame):
        frame = history
    else:
        path = Path(history)
        if path.is_dir():
            path = path / "progress.csv"
        frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"training history lacks column(s) {missing}")
    return frame


def _as_history_list(histories: object) -> List[HistoryLike]:
    if isinstance(histories, (pd.DataFrame, str, os.PathLike)):
        return [histories]  # a single run
    if isinstance(histories, Mapping):
        raise TypeError("expected the runs of one configuration; use plot_training_comparison for a mapping")
    return list(histories)  # type: ignore[call-overload]


def aggregate_histories(
    histories: Union[HistoryLike, Sequence[HistoryLike]], cycles_column: str = "mean_cycles"
) -> pd.DataFrame:
    """Per-epoch statistics over the repeated training runs of one configuration.

    Returns a frame indexed by ``epoch`` with the columns ``score_mean``,
    ``score_min`` and ``score_max`` (over the runs' per-epoch mean score, as
    in ``plotAgentPerformance``), ``success_pct`` (mean success rate in
    percent), ``cycles_mean`` (mean of ``cycles_column``; use
    ``"mean_cycles_success"`` to count successful episodes only) and
    ``n_runs``.  Runs may differ in length: every epoch is aggregated over the
    runs that reached it.
    """
    runs = _as_history_list(histories)
    if not runs:
        raise ValueError("no training histories given")
    columns = list(dict.fromkeys(["epoch", "mean_score", "success_rate", cycles_column]))
    frames = []
    for i, run in enumerate(runs):
        frame = load_history(run)
        if cycles_column not in frame.columns:
            raise ValueError(f"training history lacks column {cycles_column!r}")
        frames.append(frame[columns].assign(_run=i))
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        raise ValueError("training histories contain no epochs")
    grouped = data.groupby("epoch", sort=True)
    return pd.DataFrame(
        {
            "score_mean": grouped["mean_score"].mean(),
            "score_min": grouped["mean_score"].min(),
            "score_max": grouped["mean_score"].max(),
            "success_pct": 100.0 * grouped["success_rate"].mean(),
            "cycles_mean": grouped[cycles_column].mean(),
            "n_runs": grouped["_run"].nunique(),
        }
    )


def _metric_labels(labels: Optional[Sequence[str]]) -> Tuple[str, str, str]:
    if labels is None:
        return TRAINING_LABELS
    if isinstance(labels, str):  # would otherwise be split into characters
        raise TypeError("labels must be a sequence of three strings, not a string")
    labels = tuple(str(label) for label in labels)
    if len(labels) != 3:
        raise ValueError("labels must name the score, success-rate and cycles series")
    return labels  # type: ignore[return-value]


def _group_styles(names: Sequence[str], colors: Optional[Mapping[str, str]] = None) -> Dict[str, Tuple[str, str]]:
    """``group -> (color, success marker)`` by position; ``colors`` overrides the hue."""
    colors = colors or {}
    return {
        name: (
            colors.get(name, _GROUP_COLORS[i % len(_GROUP_COLORS)]),
            _GROUP_MARKERS[i % len(_GROUP_MARKERS)],
        )
        for i, name in enumerate(names)
    }


def _draw_series(
    ax: Axes,
    stats: pd.DataFrame,
    colors: Tuple[str, str, str],
    marker: str,
    labels: Sequence[str],
) -> None:
    """Score line + min-max band, success-rate markers and dash-dot cycles."""
    score_color, success_color, cycles_color = colors
    epochs = stats.index.to_numpy(dtype=float)
    point = "o" if len(epochs) == 1 else None  # a lone epoch has no line to draw
    if int(stats["n_runs"].max()) > 1:
        ax.fill_between(
            epochs,
            stats["score_min"].to_numpy(dtype=float),
            stats["score_max"].to_numpy(dtype=float),
            color=score_color,
            alpha=0.2,
            linewidth=0,
            zorder=1,
        )
    ax.plot(epochs, stats["score_mean"].to_numpy(dtype=float), color=score_color, linewidth=1.6,
            marker=point, markersize=4, label=labels[0], zorder=3)
    ax.plot(
        epochs,
        stats["success_pct"].to_numpy(dtype=float),
        linestyle="none",
        marker=marker,
        markersize=5,
        markeredgewidth=1.1,
        color=success_color,
        markerfacecolor=success_color if marker in _UNFILLED_MARKERS else "none",
        markevery=max(1, len(epochs) // 40),
        label=labels[1],
        zorder=4,
    )
    ax.plot(epochs, stats["cycles_mean"].to_numpy(dtype=float), color=cycles_color, linewidth=1.2,
            linestyle="-.", marker=point, markersize=4, label=labels[2], zorder=2)


def _draw_training(
    ax: Axes,
    histories: object,
    labels: Optional[Sequence[str]],
    styles: Mapping[str, Tuple[str, str]],
    cycles_column: str,
) -> None:
    names = _metric_labels(labels)
    if isinstance(histories, Mapping):
        for group, runs in histories.items():
            color, marker = styles[group]
            stats = aggregate_histories(runs, cycles_column)
            _draw_series(ax, stats, (color, color, color), marker, [f"{n} ({group})" for n in names])
    else:
        stats = aggregate_histories(_as_history_list(histories), cycles_column)
        _draw_series(ax, stats, (RED, GREEN, INK), "x", names)
    # One axis for the three metrics, tick labels in units of 10^2 as in the paper.
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(_nice_ticks())
    ax.xaxis.set_major_locator(_nice_ticks(integer=True))
    # Epochs from 0 as in the paper.  auto=None keeps x-autoscaling on, so data
    # drawn later into the same axes (another draw_training call) is not clipped.
    ax.set_xlim(left=min(0.0, float(ax.dataLim.x0)), auto=None)
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Score, success rate (%), cycles")
    _style_axes(ax)


def draw_training(
    ax: Axes,
    histories: Union[HistoryLike, Sequence[HistoryLike], Mapping[str, Sequence[HistoryLike]]],
    *,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Mapping[str, str]] = None,
    cycles_column: str = "mean_cycles",
) -> Axes:
    """Draw training metrics into ``ax`` (no legend; see :func:`plot_training_curves`).

    ``histories`` is either the runs of one configuration (Fig. 7/8 style:
    red score, green success crosses, black cycles) or a mapping
    ``label -> runs`` of configurations (Fig. 4 style: one color per
    configuration, in order red, blue, ...; ``colors`` overrides them).
    ``labels`` renames the three metrics (default :data:`TRAINING_LABELS`).
    """
    styles = _group_styles(list(histories), colors) if isinstance(histories, Mapping) else {}
    _draw_training(ax, histories, labels, styles, cycles_column)
    return ax


def plot_training_curves(
    histories: Union[HistoryLike, Sequence[HistoryLike]],
    title: str,
    out_path: PathLike,
    labels: Optional[Sequence[str]] = None,
    *,
    cycles_column: str = "mean_cycles",
    pdf: bool = False,
    dpi: int = 150,
) -> Path:
    """Training curves of one configuration over its repeated runs (Figs. 7 and 8).

    Args:
        histories: the repeats of one configuration, each a DataFrame, a
            ``progress.csv`` path or a run directory (:func:`load_history`).
        title: axes title, e.g. ``"Size 30x30"``.
        out_path: output file; PNG unless the suffix names another format.
        labels: legend labels of the score, success-rate and cycles series
            (default ``("Score", "Succ. Rate", "No. Cycles")`` as in the paper).
        cycles_column: ``"mean_cycles"`` (all episodes, as the reference code
            averages them) or ``"mean_cycles_success"``.
        pdf: also save a PDF next to the PNG.

    Returns:
        The path of the saved figure.
    """
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(5.2, 3.2), layout="constrained")
        ax = fig.add_subplot()
        _draw_training(ax, _as_history_list(histories), labels, {}, cycles_column)
        ax.set_title(title)
        _figure_legend(fig, [ax])
        return _save(fig, out_path, pdf, dpi)


def plot_training_comparison(
    groups: Mapping[str, Sequence[HistoryLike]],
    title: str,
    out_path: PathLike,
    labels: Optional[Sequence[str]] = None,
    *,
    colors: Optional[Mapping[str, str]] = None,
    cycles_column: str = "mean_cycles",
    pdf: bool = False,
    dpi: int = 150,
) -> Path:
    """Overlay the training curves of several configurations (Fig. 4).

    ``groups`` maps a label to the runs of that configuration and is drawn in
    order: the first in red (random initialization in Fig. 4), the second in
    blue (transfer learning), then further colors; success rates use crosses,
    circles, triangles, ...  Returns the path of the saved figure.
    """
    if not groups:
        raise ValueError("no configurations given")
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(5.2, 3.2), layout="constrained")
        ax = fig.add_subplot()
        _draw_training(ax, groups, labels, _group_styles(list(groups), colors), cycles_column)
        ax.set_title(title)
        _figure_legend(fig, [ax])
        return _save(fig, out_path, pdf, dpi)


def plot_training_panels(
    panels: Mapping[str, Union[Sequence[HistoryLike], Mapping[str, Sequence[HistoryLike]]]],
    out_path: PathLike,
    *,
    title: Optional[str] = None,
    ncols: Optional[int] = None,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Mapping[str, str]] = None,
    cycles_column: str = "mean_cycles",
    pdf: bool = False,
    dpi: int = 150,
) -> Path:
    """One panel per chip size / fault level with a shared legend (Figs. 4, 7, 8).

    ``panels`` maps a panel title (e.g. ``"Size 60x60"``) to either the runs
    of one configuration (Figs. 7, 8) or a mapping ``label -> runs`` to
    compare (Fig. 4).  A configuration label keeps its color in every panel.
    Panels fill a single row unless ``ncols`` is given.
    """
    items = list(panels.items())
    if not items:
        raise ValueError("no panels given")
    ncols = len(items) if ncols is None else max(1, min(int(ncols), len(items)))
    nrows = -(-len(items) // ncols)
    names: List[str] = []
    for _, data in items:
        if isinstance(data, Mapping):
            names.extend(g for g in data if g not in names)
    styles = _group_styles(names, colors)
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(2.9 * ncols, 2.5 * nrows), layout="constrained")
        axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
        for i, (ax, (panel_title, data)) in enumerate(zip(axes, items)):
            _draw_training(ax, data, labels, styles, cycles_column)
            ax.set_title(str(panel_title))
            if i % ncols:
                ax.set_ylabel("")
            if i + ncols < len(items):
                ax.set_xlabel("")
        for ax in axes[len(items):]:
            fig.delaxes(ax)
        if title:
            fig.suptitle(title)
        _figure_legend(fig, axes[: len(items)])
        return _save(fig, out_path, pdf, dpi)


# --------------------------------------------------- bioassay completion
def _ecdf_steps(cycles: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Vertices of the step curve ``P[K <= k]`` on ``[lo, hi]`` (``where="post"``).

    Non-finite entries are failed trials: they never complete but count in
    the denominator, so the curve levels off at the success rate.
    """
    finite = cycles[np.isfinite(cycles)]
    ks, counts = np.unique(finite, return_counts=True)
    probs = np.cumsum(counts) / cycles.size
    end = probs[-1] if probs.size else 0.0
    return np.concatenate([[lo], ks, [hi]]), np.concatenate([[0.0], probs, [end]])


def draw_completion_cdf(
    ax: Axes,
    results: Mapping[str, Sequence[float]],
    title: Optional[str] = None,
    *,
    legend: bool = True,
) -> Axes:
    """Draw ``P[K <= k]`` per router into ``ax`` (Fig. 9).

    ``results`` maps a router name to the per-trial numbers of cycles ``K``
    (``np.inf`` or NaN for a failed trial).  As in the reference code
    (``meda_utils.plotProbVsCycles``) the curves span ``[min K - 1,
    max K + 1]`` over all routers.
    """
    if not results:
        raise ValueError("no router results given")
    data: Dict[str, np.ndarray] = {}
    for name, cycles in results.items():
        values = np.asarray(cycles, dtype=float).ravel()
        if values.size == 0:
            raise ValueError(f"no trials for router {name!r}")
        data[name] = values
    finite = np.concatenate([v[np.isfinite(v)] for v in data.values()])
    lo, hi = (float(finite.min()) - 1.0, float(finite.max()) + 1.0) if finite.size else (0.0, 1.0)
    styles = router_styles(list(data))
    for name, values in data.items():
        color, linestyle = styles[name]
        x, y = _ecdf_steps(values, lo, hi)
        ax.step(x, y, where="post", color=color, linestyle=linestyle,
                linewidth=1.8 if linestyle == ":" else 1.6, label=str(name))
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(_nice_ticks(integer=True))
    ax.set_xlabel("Number of Cycles ($k$)")
    ax.set_ylabel("Probability of Success")
    _style_axes(ax)
    if title:
        ax.set_title(title, fontweight="bold")
    if legend:
        ax.legend(loc="best")  # Fig. 9 has it upper left, where extra routers may run
    return ax


def plot_completion_cdf(
    results: Mapping[str, Sequence[float]],
    title: str,
    out_path: PathLike,
    *,
    pdf: bool = False,
    dpi: int = 150,
) -> Path:
    """Probability of successful bioassay completion versus cycles (Fig. 9).

    ``results`` maps a router name to its per-trial cycle counts, ``np.inf``
    marking failed trials.  ``Baseline`` is drawn dotted black, ``Formal``
    dashed blue and ``DRL`` solid red; other names get distinct styles.
    Returns the path of the saved figure.
    """
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(4.4, 3.2), layout="constrained")
        draw_completion_cdf(fig.add_subplot(), results, title)
        return _save(fig, out_path, pdf, dpi)


# ------------------------------------------------------------ routing path
Track = Tuple[Optional[str], List[Tuple[int, int, int, int]]]


def _as_tracks(path: object) -> List[Track]:
    if isinstance(path, Mapping):
        return [(str(name), [_as_rect(d).as_tuple() for d in track]) for name, track in path.items()]
    if isinstance(path, Rect):
        path = [path]
    return [(None, [_as_rect(d).as_tuple() for d in path])]  # type: ignore[attr-defined]


def _health_colormap(levels: int) -> Tuple[object, Normalize, Optional[np.ndarray]]:
    """Gray ramp from dark (``H = 0``, fully degraded) to light (healthy)."""
    dark, light = np.array(to_rgb("#3d3c3a")), np.array(to_rgb("#f4f3f0"))
    if levels <= 16:
        steps = [dark + (light - dark) * i / max(levels - 1, 1) for i in range(levels)]
        norm = BoundaryNorm(np.arange(-0.5, levels), levels)
        return ListedColormap(steps), norm, np.arange(levels)
    cmap = LinearSegmentedColormap.from_list("health", [dark, light])
    return cmap, Normalize(0, levels - 1), None


def _cell_marks(ax: Axes, mask: np.ndarray, kind: str, color: str) -> None:
    """Outline (``"square"``) or cross (``"cross"``) every MC set in ``mask``."""
    xs, ys = np.nonzero(mask)
    if kind == "square":
        inset = 0.12
        squares = [Rectangle((x - 0.5 + inset, y - 0.5 + inset), 1 - 2 * inset, 1 - 2 * inset)
                   for x, y in zip(xs, ys)]
        ax.add_collection(PatchCollection(squares, facecolors="none", edgecolors=color,
                                          linewidths=1.0, zorder=4))
    else:
        r = 0.3
        segments = [seg for x, y in zip(xs, ys)
                    for seg in (((x - r, y - r), (x + r, y + r)), ((x - r, y + r), (x + r, y - r)))]
        ax.add_collection(LineCollection(segments, colors=color, linewidths=1.0, zorder=4))


def _draw_track(ax: Axes, track: List[Tuple[int, int, int, int]], goal: Tuple[int, int, int, int],
                color: str, linestyle: object, label: str) -> Tuple[object, bool]:
    """Droplet centers joined by a line; stalls, heading and a missed goal marked.

    Returns the legend handle of the track and whether the droplet stalled.
    """
    centers = np.array([((xa + xb) / 2.0, (ya + yb) / 2.0) for xa, ya, xb, yb in track])
    # A light halo keeps dark tracks (e.g. the black Baseline) visible over
    # fully degraded MCs, which are drawn dark.
    halo = [patheffects.withStroke(linewidth=3.6, foreground="white", alpha=0.85)]
    (line,) = ax.plot(centers[:, 0], centers[:, 1], color=color, linestyle=linestyle, linewidth=1.8,
                      marker="o", markersize=3, solid_capstyle="round", dash_capstyle="round",
                      label=label, zorder=6, path_effects=halo)
    # Cycles spent without moving: circle area grows with the stall length.
    stalled = False
    i = 0
    while i < len(track):
        j = i
        while j + 1 < len(track) and track[j + 1] == track[i]:
            j += 1
        if j > i:
            stalled = True
            ax.scatter(*centers[i], s=min(25.0 + 20.0 * (j - i), 400.0), facecolors="none",
                       edgecolors=color, linewidths=1.0, zorder=6, path_effects=halo)
        i = j + 1
    # Arrow head on the last move.
    moves = [k for k in range(1, len(centers)) if not np.array_equal(centers[k], centers[k - 1])]
    if moves:
        end, prev = centers[moves[-1]], centers[moves[-1] - 1]
        tail = end - 0.01 * (end - prev) / np.linalg.norm(end - prev)
        ax.annotate("", xy=tuple(end), xytext=tuple(tail), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=12,
                                    shrinkA=0, shrinkB=0))
    if track[-1] == goal:
        return line, stalled
    # Missed the goal: mark where the droplet ended (and show the cross in the legend).
    ax.plot(*centers[-1], marker="X", markersize=9, color=color, markeredgecolor="white",
            markeredgewidth=0.8, linestyle="none", zorder=8)
    cross = Line2D([], [], linestyle="none", marker="X", markersize=7, color=color,
                   markeredgecolor="white")
    return (line, cross), stalled


def draw_routing_path(
    ax: Axes,
    chip_health: np.ndarray,
    health_levels: int,
    path: Union[Sequence[RectLike], Mapping[str, Sequence[RectLike]]],
    goal: RectLike,
    hazard: RectLike,
    faults: Optional[np.ndarray] = None,
    *,
    hidden_defects: Optional[np.ndarray] = None,
    colorbar: bool = True,
    legend: bool = True,
) -> Axes:
    """Draw droplet trajectories over the health map into ``ax`` (Fig. 14).

    Args:
        chip_health: integer health readings ``H`` of shape ``(W, H)``,
            indexed ``[x, y]`` (:meth:`MEDABiochip.health`).
        health_levels: ``2**b``; readings run from 0 to ``health_levels - 1``.
        path: droplet locations, start first (e.g. ``JobResult.path`` with
            ``record_path=True``), or a mapping ``router name -> locations``
            to compare routers on the same job (styled as in Fig. 9).
        goal, hazard: goal droplet and routing zone (``delta_g``, ``delta_h``).
        faults: optional boolean ``(W, H)`` mask of faulty MCs, outlined with
            red squares as the degraded electrodes of Fig. 14.
        hidden_defects: optional mask of defects the sensors cannot see
            (Sec. VI-C), drawn as violet crosses.

    North is up and ``x`` grows to the east; MCs outside the routing zone are
    washed out.  A trajectory joins the droplet centers; open circles mark
    cycles in which the droplet did not move (area grows with their number)
    and a cross marks the final location of a droplet that missed the goal.
    """
    health = np.asarray(chip_health)
    if health.ndim != 2:
        raise ValueError("chip_health must be a W x H array indexed [x, y]")
    width, height = health.shape
    levels = int(health_levels)
    if levels < 1:
        raise ValueError("health_levels must be positive")
    goal_rect, zone = _as_rect(goal), _as_rect(hazard)
    tracks = _as_tracks(path)
    chip = Rect(0, 0, width - 1, height - 1)
    for rect in [goal_rect, zone] + [Rect(*d) for _, track in tracks for d in track]:
        if not chip.contains(rect):
            raise ValueError(f"{rect.as_tuple()} lies outside the {width} x {height} chip")

    extent = (-0.5, width - 0.5, -0.5, height - 0.5)
    cmap, norm, ticks = _health_colormap(levels)
    image = ax.imshow(health.T, origin="lower", extent=extent, cmap=cmap, norm=norm,
                      interpolation="nearest", zorder=0)
    outside = np.ones((height, width), dtype=bool)  # [y, x]
    zx, zy = zone.slices()
    outside[zy, zx] = False
    if outside.any():
        veil = np.zeros((height, width, 4))
        veil[outside] = (1.0, 1.0, 1.0, 0.6)
        ax.imshow(veil, origin="lower", extent=extent, interpolation="nearest", zorder=0.5)
    if max(width, height) <= 40:  # draw the MC array
        ax.set_xticks(np.arange(-0.5, width, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1.0), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", length=0)

    entries: List[Tuple[object, str]] = []  # legend (handle, label)
    ax.add_patch(Rectangle((zone.xa - 0.5, zone.ya - 0.5), zone.width, zone.height, fill=False,
                           edgecolor=INK, linewidth=1.2, linestyle="--", zorder=3))
    entries.append((Patch(facecolor="none", edgecolor=INK, linestyle="--"), "Hazard bounds"))
    for mask, kind, color, label in ((faults, "square", RED, "Faulty MC"),
                                     (hidden_defects, "cross", VIOLET, "Hidden defect")):
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != health.shape:
            raise ValueError(f"{label.lower()} mask must have shape {health.shape}")
        if mask.any():
            _cell_marks(ax, mask, kind, color)
            marker = "s" if kind == "square" else "x"
            entries.append((Line2D([], [], linestyle="none", color=color, markersize=7, marker=marker,
                                   markerfacecolor="none"), label))

    start = next((track[0] for _, track in tracks if track), None)
    ends = [(Rect(*start), INK, "Start")] if start is not None else []
    for rect, color, label in ends + [(goal_rect, GREEN, "Goal")]:
        face = to_rgba(color, 0.08 if label == "Start" else 0.25)
        ax.add_patch(Rectangle((rect.xa - 0.5, rect.ya - 0.5), rect.width, rect.height,
                               facecolor=face, edgecolor=color, linewidth=1.6, zorder=5))
        entries.append((Patch(facecolor=face, edgecolor=color, linewidth=1.4), label))

    named = [name for name, _ in tracks if name is not None]
    styles = router_styles(named)
    stalled = False
    for name, track in tracks:
        if not track:
            continue
        color, linestyle = styles[name] if name is not None else (ORANGE, "-")
        reached = track[-1] == goal_rect.as_tuple()
        label = f"{name or 'Droplet path'} (k = {len(track) - 1}{'' if reached else ', goal missed'})"
        handle, track_stalled = _draw_track(ax, track, goal_rect.as_tuple(), color, linestyle, label)
        entries.append((handle, label))
        stalled |= track_stalled
    if stalled:
        entries.append((Line2D([], [], linestyle="none", marker="o", markersize=7,
                               markerfacecolor="none", markeredgecolor=MUTED), "Stalled cycles"))

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(_nice_ticks(integer=True))
    ax.yaxis.set_major_locator(_nice_ticks(integer=True))
    ax.set_xlabel("$x$ (east)")
    ax.set_ylabel("$y$ (north)")
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, ticks=ticks, location="bottom", shrink=0.6, aspect=30)
        bar.set_label("Health $H$ (0 = fully degraded)")
        bar.outline.set_linewidth(0.5)
    if legend:
        handles, labels = zip(*entries)
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return ax


def plot_routing_path(
    chip_health: np.ndarray,
    health_levels: int,
    path: Union[Sequence[RectLike], Mapping[str, Sequence[RectLike]]],
    goal: RectLike,
    hazard: RectLike,
    faults: Optional[np.ndarray] = None,
    out_path: Optional[PathLike] = None,
    *,
    hidden_defects: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    pdf: bool = False,
    dpi: int = 150,
) -> Path:
    """Droplet trajectory on the chip's health map, like Fig. 14; see :func:`draw_routing_path`.

    ``out_path`` is required; it comes after the optional ``faults`` so that
    the arguments read ``(health, levels, path, goal, hazard, faults, out)``.
    Returns the path of the saved figure.
    """
    if out_path is None:
        hint = ""
        if isinstance(faults, (str, os.PathLike)):
            hint = " (a path was given for 'faults'; pass faults=None or a mask before out_path)"
        raise TypeError(f"plot_routing_path() missing required argument: 'out_path'{hint}")
    width, height = np.shape(chip_health)
    aspect = height / width
    ax_width = 4.8 if aspect <= 1.2 else max(2.4, 5.8 / aspect)
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(ax_width + 2.4, ax_width * aspect + 1.4), layout="constrained")
        ax = fig.add_subplot()
        draw_routing_path(ax, chip_health, health_levels, path, goal, hazard, faults,
                          hidden_defects=hidden_defects)
        if title:
            ax.set_title(title)
        return _save(fig, out_path, pdf, dpi)
