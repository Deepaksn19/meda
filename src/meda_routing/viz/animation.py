"""GIF recordings of single-droplet routing episodes.

:func:`record_episode` resets a :class:`~meda_routing.envs.meda_env.MEDARoutingEnv`,
lets a policy route the droplet until the episode terminates (droplet at the
goal) or is truncated after ``k_max`` control cycles (Sec. IV-B), and writes
the frames of :meth:`MEDARoutingEnv.render_frame` (sensed degradation in
gray, routing zone lit, faults red, hidden defects purple, goal green, droplet
blue; north up) as an animated GIF: the simulated counterpart of the routing
progressions of Fig. 14.  A caption shows the control cycle ``k / k_max``,
the chosen direction and the Manhattan distance ``D(delta, delta_g)`` to the
goal, and the trail of droplet centers is drawn on top of the chip.

Frames are grabbed from a pyplot-free :class:`matplotlib.figure.Figure` by
:class:`matplotlib.animation.PillowWriter` (Pillow ships with matplotlib), so
recording works headless and leaves pyplot's backend alone.  The writer keeps
every frame in memory until the GIF is written; bound very long episodes with
``max_steps`` or a smaller ``scale``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import features

from ..core.actions import Action

#: ``policy_fn(obs, env) -> action`` (an int, :class:`Action` or 1-element array).
PolicyFn = Callable[[np.ndarray, Any], Any]

#: The longer chip side spans about this many pixels (see :func:`frame_scale`).
TARGET_PIXELS = 400
_MIN_WIDTH_PX = 240  # room for the caption on small chips
_CAPTION_PX = 26
_DPI = 100
_TRAIL_COLOR = "#eb6834"
_INK, _MUTED = "#0b0b0b", "#52514e"
# Frames must not depend on the user's savefig rc (e.g. transparent PNG output).
_FRAME_RC = {"savefig.bbox": None, "savefig.transparent": False, "savefig.facecolor": "white"}


def frame_scale(width: int, height: int, target: int = TARGET_PIXELS) -> int:
    """Pixels per MC so that the longer chip side spans about ``target`` pixels (1-16)."""
    return int(np.clip(target // max(int(width), int(height), 1), 1, 16))


def _animation_path(out_path: Union[str, os.PathLike]) -> Path:
    """``out_path``, with ``.gif`` appended unless Pillow can animate its suffix.

    Pillow writes animated GIF, APNG (``.png``/``.apng``) and WebP; any other
    suffix (e.g. ``.mp4``) would only fail when the finished episode is saved.
    """
    path = Path(out_path)
    suffixes = {".gif", ".png", ".apng"} | ({".webp"} if features.check("webp") else set())
    if path.suffix.lower() not in suffixes:
        path = path.with_name(path.name + ".gif")
    return path


def _as_action(action: Any) -> int:
    values = np.asarray(action)
    if values.size != 1:
        raise ValueError(f"policy_fn must return a single action, got {action!r}")
    return int(values.reshape(-1)[0])


def _action_name(action: int) -> str:
    try:
        return Action(action).name
    except ValueError:
        return str(action)


def _center(droplet: Any) -> Tuple[float, float]:
    xa, ya, xb, yb = droplet
    return (xa + xb) / 2.0, (ya + yb) / 2.0


def _episode_figure(frame: np.ndarray, width: int, height: int) -> Tuple[Figure, Any, Line2D, Text, Text]:
    """Figure showing ``frame`` pixel for pixel below a one-line caption."""
    img_h, img_w = frame.shape[:2]
    fig_w, fig_h = max(img_w, _MIN_WIDTH_PX), img_h + _CAPTION_PX
    # +0.01 px keeps int(size * dpi) from rounding a pixel away
    fig = Figure(figsize=((fig_w + 0.01) / _DPI, (fig_h + 0.01) / _DPI), dpi=_DPI, facecolor="white")
    ax = fig.add_axes((0.5 * (fig_w - img_w) / fig_w, 0.0, img_w / fig_w, img_h / fig_h))
    ax.set_axis_off()
    # frame rows run north to south; extent maps MC (x, y) to data coordinates (x, y)
    image = ax.imshow(frame, origin="upper", extent=(-0.5, width - 0.5, -0.5, height - 0.5),
                      interpolation="nearest", aspect="auto")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    (trail,) = ax.plot([], [], color=_TRAIL_COLOR, linewidth=1.5, alpha=0.9,
                       solid_capstyle="round", solid_joinstyle="round")
    y = 1.0 - 0.5 * _CAPTION_PX / fig_h
    left = fig.text(6.0 / fig_w, y, "", ha="left", va="center", fontsize=9, color=_INK)
    right = fig.text(1.0 - 6.0 / fig_w, y, "", ha="right", va="center", fontsize=9, color=_MUTED)
    return fig, image, trail, left, right


def record_episode(
    env: Any,
    policy_fn: PolicyFn,
    out_path: Union[str, os.PathLike],
    max_steps: Optional[int] = None,
    fps: int = 4,
    seed: Optional[int] = None,
    *,
    options: Optional[Dict[str, Any]] = None,
    scale: Optional[int] = None,
    end_pause: float = 1.0,
    record_path: bool = False,
) -> Dict[str, Any]:
    """Run one routing episode with ``policy_fn`` and save it as a GIF.

    Args:
        env: a :class:`MEDARoutingEnv` (or a wrapper around one).
        policy_fn: ``policy_fn(obs, env) -> action``, called once per cycle.
        out_path: GIF file; ``.gif`` is appended unless the suffix names an
            animated format Pillow writes (``.gif``, ``.png``/``.apng``, ``.webp``).
        max_steps: optional cap on the number of control cycles; by default
            the episode runs until it terminates or is truncated at ``k_max``.
        fps: frames (control cycles) per second.
        seed: passed to ``env.reset`` (samples the job, chip and faults).
        options: passed to ``env.reset``, e.g. ``{"job": job, "chip": chip}``
            to record a given routing job.
        scale: pixels per MC; default :func:`frame_scale` of the chip size.
        end_pause: seconds the final frame is held before the GIF loops.
        record_path: also return the droplet locations (start first) under
            ``"path"``, ready for :func:`~meda_routing.viz.plots.plot_routing_path`.

    Returns:
        ``{"success", "cycles", "frames", "reward", "out_path"}`` (plus
        ``"path"``): whether the droplet reached the goal, the number of
        control cycles, the number of recorded frames (initial state plus one
        per cycle; the held copies of the final frame are not counted), the
        episode return and the path of the GIF.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    base = getattr(env, "unwrapped", env)
    render = getattr(env, "render_frame", None) or getattr(base, "render_frame", None)
    if render is None:
        raise TypeError("env must provide render_frame(scale), like MEDARoutingEnv")
    path = _animation_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    obs, info = env.reset(seed=seed, options=options)
    width, height = int(base.width), int(base.height)
    scale = frame_scale(width, height) if scale is None else max(1, int(scale))
    fig, image, trail, left, right = _episode_figure(np.asarray(render(scale=scale)), width, height)

    droplets: List[Any] = [getattr(base, "droplet", None)]
    has_droplet = droplets[0] is not None
    k_max = info.get("k_max", getattr(base, "k_max", None))

    def caption(k: int, status: str) -> None:
        left.set_text(f"k = {k}/{k_max}" if k_max else f"k = {k}")
        right.set_text(status)

    def distance(info: Dict[str, Any]) -> str:
        return f"d = {info['distance']}" if "distance" in info else ""

    def trail_update() -> None:
        if has_droplet:
            centers = np.array([_center(d) for d in droplets])
            trail.set_data(centers[:, 0], centers[:, 1])

    writer = PillowWriter(fps=fps)
    steps, total_reward, frames = 0, 0.0, 0
    terminated = truncated = False
    # every frame must be rendered at the figure's own size
    with mpl.rc_context(_FRAME_RC):
        writer.setup(fig, str(path), dpi=_DPI)
        caption(0, f"start  {distance(info)}".strip())
        trail_update()
        writer.grab_frame()
        frames = 1
        while not (terminated or truncated) and (max_steps is None or steps < max_steps):
            action = _as_action(policy_fn(obs, env))
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_reward += float(reward)
            if has_droplet:
                droplets.append(base.droplet)
            image.set_data(np.asarray(render(scale=scale)))
            trail_update()
            k = int(info.get("num_cycles", steps))
            status = f"{_action_name(action)}  {distance(info)}".strip()
            # A timeout may be reported as ``terminated`` (EnvConfig.timeout_terminal),
            # so success is read from the info dict, not from the done flags.
            if info.get("is_success", terminated):
                status = f"{_action_name(action)}  goal reached"
            elif terminated or truncated:
                status += "  (k_max)" if k_max and k >= k_max else "  (ended)"
            elif max_steps is not None and steps >= max_steps:
                status += "  (max_steps)"
            caption(k, status)
            writer.grab_frame()
            frames += 1
        for _ in range(int(round(end_pause * fps))):
            writer.grab_frame()  # identical frames: Pillow merges them into one longer frame
        writer.finish()

    result: Dict[str, Any] = {
        "success": bool(info.get("is_success", terminated)),
        "cycles": int(info.get("num_cycles", steps)),
        "frames": frames,
        "reward": total_reward,
        "out_path": path,
    }
    if record_path:
        result["path"] = droplets if has_droplet else []
    return result
