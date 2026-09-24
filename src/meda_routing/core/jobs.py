"""Routing jobs and their sampling distribution (Sec. II-D and IV-A).

A routing job is characterized by the droplet shape/size, its start and goal
locations and the biochip area within which routing is allowed (the *hazard
bounds* ``delta_h``, also called the routing zone).

Where each default comes from is tagged next to it:

* ``[PAPER ...]`` -- the value is stated in the paper (section, figure, table).
* ``[REF-CODE]`` -- not in the paper; taken from the first author's public
  code ``melfar87/MEDA`` (incl. the Stable-Baselines PPO2 defaults, saved
  model and training log of that code).
* ``[ASSUMED]`` -- not fixed by the paper or the reference code; our choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import Droplet, Rect, chip_rect

#: [PAPER Sec. IV-A] Droplet sizes used for training in the paper: ``w, h in {2,...,6}`` with
#: aspect ratio ``w / h in [0.8, 1.25]`` (Sec. IV-A).
PAPER_DROPLET_SIZES: Tuple[Tuple[int, int], ...] = tuple(
    (w, h)
    for w in range(2, 7)
    for h in range(2, 7)
    if 0.8 <= w / h <= 1.25
)


@dataclass(frozen=True)
class RoutingJob:
    start: Droplet
    goal: Droplet
    hazard: Rect

    def __post_init__(self) -> None:
        # Rect and Droplet never compare equal, so store both ends as Droplets
        # to make "droplet == goal" checks reliable whatever the caller passed.
        for name in ("start", "goal"):
            value = getattr(self, name)
            if type(value) is not Droplet:
                object.__setattr__(self, name, Droplet(*value.as_tuple()))
        if self.start.size != self.goal.size:
            raise ValueError("start and goal droplets must have the same size")
        if not self.hazard.contains(self.start) or not self.hazard.contains(self.goal):
            raise ValueError("start and goal must lie inside the hazard bounds")

    @property
    def droplet_size(self) -> Tuple[int, int]:
        return self.start.size


def hazard_bounds(
    start: Rect, goal: Rect, width: int, height: int, margin: Optional[int] = 3
) -> Rect:
    """Routing zone: bounding box of start and goal grown by ``margin`` MCs.

    Clipped to the chip.  ``margin=None`` makes the whole chip the routing
    zone.  The reference implementation uses a margin of 3 MCs.
    """
    if margin is None:
        return chip_rect(width, height)
    return Rect(
        max(min(start.xa, goal.xa) - margin, 0),
        max(min(start.ya, goal.ya) - margin, 0),
        min(max(start.xb, goal.xb) + margin, width - 1),
        min(max(start.yb, goal.yb) + margin, height - 1),
    )


@dataclass
class JobSamplerConfig:
    droplet_sizes: Sequence[Tuple[int, int]] = field(  # [PAPER Sec. IV-A] w, h in {2..6}, w/h in [0.8, 1.25]
        default_factory=lambda: list(PAPER_DROPLET_SIZES)
    )
    #: ``"stratified"`` over-samples droplets adjacent to the chip edges, where
    #: dispensers and reservoirs sit (Sec. IV-A: 20-40% of benchmark routing
    #: jobs touch an edge).  With the reference weights ~60% of the sampled
    #: droplets touch an edge.  ``"uniform"`` samples ``xa ~ U{m, W - w - m}``.
    sampling: str = "stratified"  # [PAPER Sec. IV-A] "stratified"; its exact rule is [REF-CODE]
    #: Fraction of the extra edge mass for stratified sampling; the reference
    #: implementation adds ``W // 5`` copies of each edge coordinate.
    edge_weight: float = 0.2  # [REF-CODE] W // 5 extra copies of each edge coordinate
    #: Margin (in MCs) for ``"uniform"`` sampling: ``xa ~ U{m, W-w-1-m}``
    #: (0-based).  The default ``m = 1`` is exactly the paper's
    #: ``xa ~ U{2, W-w-1}`` in 1-based coordinates (Sec. IV-A).
    uniform_margin: int = 1  # [PAPER Sec. IV-A] x_a ~ U{2, W-w-1} (1-based)
    #: Routing-zone margin around start and goal; ``None`` = whole chip.
    hazard_margin: Optional[int] = 3  # [REF-CODE] routing zone = bbox(start, goal) +- 3 MCs


class JobSampler:
    """Samples random routing jobs on a ``W x H`` biochip."""

    def __init__(
        self,
        width: int,
        height: int,
        config: Optional[JobSamplerConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.config = config or JobSamplerConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        sizes = [tuple(s) for s in self.config.droplet_sizes]
        for w, h in sizes:
            if w > self.width or h > self.height:
                raise ValueError(f"droplet size {(w, h)} does not fit a {width}x{height} chip")
        self.sizes: List[Tuple[int, int]] = sizes
        if self.config.sampling not in ("stratified", "uniform"):
            raise ValueError(f"unknown sampling mode {self.config.sampling!r}")
        self._x_set = self._stratified_axis(self.width)
        self._y_set = self._stratified_axis(self.height)
        self._x_buf: List[int] = []
        self._y_buf: List[int] = []
        self._size_buf: List[Tuple[int, int]] = []

    def reseed(self, rng: np.random.Generator) -> None:
        """Use ``rng`` from now on and forget partially consumed sample pools."""
        self.rng = rng
        self._x_buf.clear()
        self._y_buf.clear()
        self._size_buf.clear()

    # ------------------------------------------------------------ helpers
    def _stratified_axis(self, n: int) -> List[int]:
        """Coordinate multiset with extra mass on both edges.

        Mirrors the reference implementation:
        ``[0]*(n//5) + [1, ..., n-2] + [n-1]*(n//5)``.
        """
        extra = int(math.floor(self.config.edge_weight * n + 1e-9))  # n // 5 for 0.2
        return [0] * extra + list(range(1, n - 1)) + [n - 1] * extra

    def _pop(self, buf: List, source: Sequence) -> object:
        """Draw without replacement from a reshuffled pool (stratified sampling)."""
        if not buf:
            buf.extend(source)
            order = self.rng.permutation(len(buf))
            buf[:] = [buf[i] for i in order]
        return buf.pop()

    def _center_to_droplet(self, cx: int, cy: int, w: int, h: int) -> Droplet:
        x = min(max(0, cx - w // 2), self.width - w)
        y = min(max(0, cy - h // 2), self.height - h)
        return Droplet.at(x, y, w, h)

    def _uniform_droplet(self, w: int, h: int) -> Droplet:
        m = self.config.uniform_margin
        # xa ~ U{m, W-w-1-m}; clamp for chips too small for the margin
        x_hi = max(self.width - w - 1 - m, 0)
        y_hi = max(self.height - h - 1 - m, 0)
        x = int(self.rng.integers(min(m, x_hi), x_hi + 1))
        y = int(self.rng.integers(min(m, y_hi), y_hi + 1))
        return Droplet.at(x, y, w, h)

    # ------------------------------------------------------------- public
    def sample(self, size: Optional[Tuple[int, int]] = None) -> RoutingJob:
        """Sample a routing job; ``size`` fixes the droplet size ``(w, h)``.

        Jobs whose start equals the goal are redrawn (the reference sampler
        keeps them, ~0.5% of its jobs, which then end after one cycle).
        """
        for _ in range(1000):
            if self.config.sampling == "stratified":
                if size is None:
                    w, h = self._pop(self._size_buf, self.sizes)  # type: ignore[misc]
                else:
                    w, h = size
                xg, xs = self._pop(self._x_buf, self._x_set), self._pop(self._x_buf, self._x_set)
                yg, ys = self._pop(self._y_buf, self._y_set), self._pop(self._y_buf, self._y_set)
                goal = self._center_to_droplet(int(xg), int(yg), w, h)
                start = self._center_to_droplet(int(xs), int(ys), w, h)
            else:
                if size is None:
                    w, h = self.sizes[int(self.rng.integers(len(self.sizes)))]
                else:
                    w, h = size
                goal = self._uniform_droplet(w, h)
                start = self._uniform_droplet(w, h)
            if start != goal:
                hazard = hazard_bounds(
                    start, goal, self.width, self.height, self.config.hazard_margin
                )
                return RoutingJob(start=start, goal=goal, hazard=hazard)
        raise RuntimeError("could not sample a routing job with start != goal")
