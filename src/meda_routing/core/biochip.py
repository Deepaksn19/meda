"""MEDA biochip state: per-MC degradation, health sensing and fault injection.

Implements the stochastic degradation model of Sec. III-A and Eq. (1)::

    D_ij^(n) = tau_ij ** (n_ij / c_ij)              in [0, 1]
    H_ij^(n) = floor(2**b * D_ij^(n))               b-bit health measurement

where ``n_ij`` is the number of control cycles in which MC ``(i, j)`` was
actuated and ``tau_ij ~ U(tau_min, tau_max)``, ``c_ij ~ U(c_min, c_max)`` are
per-MC degradation parameters that are *not* observable by the agent.
Sec. V-A estimates ``tau in [0.5, 0.7]`` and ``c in [500, 800]``.

Two kinds of faults are supported:

* **injected faults** (Sec. V-B): fully degraded MCs (``D = 0``) that the
  health sensors report (``H = 0``), placed in ``2 x 2`` clusters;
* **hidden defects** (Sec. VI-C): MCs that cannot actuate (e.g. imperfect
  hydrophobic coating) but are *not* visible to the health sensors.

All arrays are indexed ``[x, y]`` with shape ``(W, H)``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .geometry import Rect


@dataclass
class DegradationConfig:
    """Parameters of the degradation model (Sec. III-A, IV-A, V-A)."""

    tau_range: Tuple[float, float] = (0.5, 0.7)
    c_range: Tuple[float, float] = (500.0, 800.0)
    #: Number of bits ``b`` of the on-chip health measurement unit.
    health_bits: int = 2
    #: How the initial actuation counts ``N`` are sampled at episode start
    #: (Algorithm 2, line 6).  ``"zero"``: healthy chip; ``"uniform"``:
    #: ``n_ij ~ U{0, max_initial_actuations}`` independently per MC.
    initial_actuations: str = "zero"
    max_initial_actuations: int = 0


class MEDABiochip:
    """State of a ``W x H`` MEDA biochip."""

    def __init__(
        self,
        width: int,
        height: int,
        config: Optional[DegradationConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("biochip dimensions must be positive")
        self.width = int(width)
        self.height = int(height)
        self.config = config or DegradationConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        shape = (self.width, self.height)
        self.tau = np.full(shape, np.mean(self.config.tau_range), dtype=np.float64)
        self.c = np.full(shape, np.mean(self.config.c_range), dtype=np.float64)
        self.actuations = np.zeros(shape, dtype=np.int64)
        self.faults = np.zeros(shape, dtype=bool)
        self.hidden_defects = np.zeros(shape, dtype=bool)

    # ---------------------------------------------------------------- shapes
    @property
    def shape(self) -> Tuple[int, int]:
        return self.width, self.height

    @property
    def health_levels(self) -> int:
        """Number of distinct health readings, ``2**b``."""
        return 2 ** self.config.health_bits

    # -------------------------------------------------------------- sampling
    def sample_parameters(self) -> None:
        """Resample ``tau_ij`` and ``c_ij`` for every MC (Sec. IV-A)."""
        lo, hi = self.config.tau_range
        self.tau = self.rng.uniform(lo, hi, size=self.shape)
        lo, hi = self.config.c_range
        self.c = self.rng.uniform(lo, hi, size=self.shape)

    def sample_initial_actuations(self) -> None:
        """Sample the initial actuation matrix ``N`` (Algorithm 2, line 6)."""
        mode = self.config.initial_actuations
        if mode == "zero":
            self.actuations[:] = 0
        elif mode == "uniform":
            hi = int(self.config.max_initial_actuations)
            self.actuations = self.rng.integers(0, hi + 1, size=self.shape, dtype=np.int64)
        else:
            raise ValueError(f"unknown initial_actuations mode {mode!r}")

    def reset(self, resample_parameters: bool = True) -> None:
        """Start a fresh episode on a new chip instance (faults cleared)."""
        if resample_parameters:
            self.sample_parameters()
        self.sample_initial_actuations()
        self.faults[:] = False
        self.hidden_defects[:] = False

    # ------------------------------------------------------------ dynamics
    def actuate(self, pattern: np.ndarray) -> None:
        """Apply an actuation pattern ``U in {0,1}^(W x H)``: ``n_ij += U_ij``."""
        self.actuations += pattern.astype(np.int64, copy=False)

    def actuate_rect(self, rect: Rect) -> None:
        sx, sy = rect.slices()
        self.actuations[sx, sy] += 1

    # ---------------------------------------------------------- degradation
    def degradation(self) -> np.ndarray:
        """Degradation level ``D`` as seen by the health sensors (1 = healthy)."""
        d = np.power(self.tau, self.actuations / self.c)
        d[self.faults] = 0.0
        return d

    def effective_degradation(self) -> np.ndarray:
        """Degradation that governs the physics, including hidden defects."""
        d = self.degradation()
        d[self.hidden_defects] = 0.0
        return d

    def health(self) -> np.ndarray:
        """Measured health ``H = floor(2**b * D)`` in ``{0, ..., 2**b - 1}`` (Eq. 1).

        ``D = 1`` would give ``2**b``, which a ``b``-bit sensor cannot report,
        so the reading saturates at ``2**b - 1``.
        """
        levels = self.health_levels
        h = np.floor(levels * self.degradation()).astype(np.int64)
        return np.minimum(h, levels - 1)

    # ------------------------------------------------------ fault injection
    def inject_faults(
        self,
        fraction: float,
        cluster: int = 2,
        region: Optional[Rect] = None,
        protect: Sequence[Rect] = (),
        hidden: bool = False,
        max_attempts: int = 100_000,
    ) -> int:
        """Randomly place fully degraded MCs in ``cluster x cluster`` blocks.

        Blocks are added until at least ``fraction`` of the MCs inside
        ``region`` (default: whole chip) are faulty (Sec. V-B: "a fixed
        percentage of fully degraded MCs are randomly placed in clusters of
        size 2 x 2").  MCs inside any ``protect`` rectangle (e.g. the start
        and goal droplets) are never made faulty.

        Returns the number of faulty MCs added.
        """
        if fraction <= 0:
            return 0
        region = region or Rect(0, 0, self.width - 1, self.height - 1)
        target_mask = self.hidden_defects if hidden else self.faults
        allowed = np.zeros(self.shape, dtype=bool)
        sx, sy = region.slices()
        allowed[sx, sy] = True
        for r in protect:
            px, py = r.slices()
            allowed[px, py] = False
        total = region.area
        goal = int(np.ceil(fraction * total))
        added = 0
        existing = int(np.count_nonzero(target_mask[sx, sy]))
        need = max(goal - existing, 0)
        attempts = 0
        cw = min(cluster, region.width)
        ch = min(cluster, region.height)
        while added < need and attempts < max_attempts:
            attempts += 1
            x = int(self.rng.integers(region.xa, region.xb - cw + 2))
            y = int(self.rng.integers(region.ya, region.yb - ch + 2))
            block = (slice(x, x + cw), slice(y, y + ch))
            new = allowed[block] & ~target_mask[block]
            if not new.any():
                continue
            target_mask[block] |= allowed[block]
            added += int(np.count_nonzero(new))
        return added

    def set_faults(self, cells: Iterable[Tuple[int, int]], hidden: bool = False) -> None:
        mask = self.hidden_defects if hidden else self.faults
        for x, y in cells:
            mask[x, y] = True

    # --------------------------------------------------------------- copying
    def copy(self) -> "MEDABiochip":
        """Independent copy; its random generator starts in the same state."""
        other = MEDABiochip(self.width, self.height, self.config, copy.deepcopy(self.rng))
        other.tau = self.tau.copy()
        other.c = self.c.copy()
        other.actuations = self.actuations.copy()
        other.faults = self.faults.copy()
        other.hidden_defects = self.hidden_defects.copy()
        return other
