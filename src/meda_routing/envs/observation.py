"""Image-based observation (Sec. III-C, Fig. 2).

The observation is a 3-channel image:

* channel 0: measured health ``H / 2**b`` inside the hazard bounds and 0
  outside (the routing zone is encoded by masking the health matrix);
* channel 1: current droplet location (1 inside the droplet, 0 elsewhere);
* channel 2: goal location.

For transfer learning (Sec. IV-C) the image is resized to a unified size
(``30 x 30`` in the paper) using OpenCV's pixel-area-relation resampling
(``cv2.INTER_AREA``).

Arrays are returned channels-first with shape ``(3, rows, cols)`` where rows
index ``y`` (north) and columns index ``x`` (east), ready for PyTorch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..core.geometry import Rect

HEALTH, DROPLET, GOAL = 0, 1, 2
NUM_CHANNELS = 3  # [PAPER Fig. 2] health (masked to the routing zone), droplet, goal


def observation_shape(width: int, height: int, obs_size: Optional[Tuple[int, int]]) -> Tuple[int, int, int]:
    """Shape of the observation for a ``W x H`` chip (``obs_size = (w, h)``)."""
    if obs_size is None:
        return NUM_CHANNELS, height, width
    return NUM_CHANNELS, int(obs_size[1]), int(obs_size[0])


def build_observation(
    health: np.ndarray,
    health_levels: int,
    droplet: Rect,
    goal: Rect,
    hazard: Rect,
    obs_size: Optional[Tuple[int, int]] = None,
    collision: Optional[Tuple[bool, bool, bool, bool]] = None,
) -> np.ndarray:
    """Build the observation from an ``[x, y]``-indexed health matrix.

    Args:
        health: integer health readings ``H`` of shape ``(W, H)``.
        health_levels: ``2**b``; health is scaled to ``[0, 1)`` by ``H / 2**b``.
        droplet, goal, hazard: droplet, goal and routing-zone rectangles.
        obs_size: ``(w, h)`` of the unified observation, or ``None`` to keep
            the chip resolution.
        collision: optional ``(west, south, east, north)`` flags.  When set,
            the corresponding droplet edge is drawn with value 0.5 in the
            droplet channel, as the authors' reference implementation does
            after an invalid action.  Not described in the paper; disabled
            unless ``mark_collisions`` is enabled in the environment.
    """
    width, height = health.shape
    img = np.zeros((height, width, NUM_CHANNELS), dtype=np.float32)
    # [x, y] -> [row=y, col=x]
    hx, hy = hazard.slices()
    img[hy, hx, HEALTH] = health[hx, hy].T / float(health_levels)
    dx, dy = droplet.slices()
    img[dy, dx, DROPLET] = 1.0
    if collision is not None:
        west, south, east, north = collision
        if west:
            img[dy, droplet.xa, DROPLET] = 0.5
        if south:
            img[droplet.ya, dx, DROPLET] = 0.5
        if east:
            img[dy, droplet.xb, DROPLET] = 0.5
        if north:
            img[droplet.yb, dx, DROPLET] = 0.5
    gx, gy = goal.slices()
    img[gy, gx, GOAL] = 1.0
    if obs_size is not None and (obs_size[0] != width or obs_size[1] != height):
        img = cv2.resize(img, (int(obs_size[0]), int(obs_size[1])), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img.transpose(2, 0, 1))
