"""Droplet and rectangle geometry on a MEDA biochip.

Coordinates follow the paper (Elfar et al., TCAD 2023, Sec. III-A and Fig. 1):
``x`` indexes the columns of the ``W x H`` microelectrode (MC) array and grows
to the east, ``y`` indexes the rows and grows to the north.  A droplet is the
quadruple ``(xa, ya, xb, yb)`` of its south-west and north-east corners.
Unlike the 1-based figures of the paper, all coordinates here are 0-based and
bounds are inclusive, so a droplet occupies ``xb - xa + 1`` columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple


@dataclass(frozen=True)
class Rect:
    """Inclusive, axis-aligned rectangle ``[xa, xb] x [ya, yb]`` of MCs.

    Equality follows dataclass semantics: a :class:`Rect` never equals a
    :class:`Droplet` with the same coordinates, so compare droplets with
    droplets (or use :meth:`as_tuple`).
    """

    xa: int
    ya: int
    xb: int
    yb: int

    def __post_init__(self) -> None:
        if self.xb < self.xa or self.yb < self.ya:
            raise ValueError(f"degenerate rectangle {self.as_tuple()}")

    # ------------------------------------------------------------------ sizes
    @property
    def width(self) -> int:
        return self.xb - self.xa + 1

    @property
    def height(self) -> int:
        return self.yb - self.ya + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    # ------------------------------------------------------------- operations
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.xa, self.ya, self.xb, self.yb

    def shift(self, dx: int, dy: int) -> "Rect":
        return type(self)(self.xa + dx, self.ya + dy, self.xb + dx, self.yb + dy)

    def contains(self, other: "Rect") -> bool:
        """True if ``other`` lies completely inside ``self``."""
        return (
            self.xa <= other.xa
            and self.ya <= other.ya
            and other.xb <= self.xb
            and other.yb <= self.yb
        )

    def contains_point(self, x: int, y: int) -> bool:
        return self.xa <= x <= self.xb and self.ya <= y <= self.yb

    def intersects(self, other: "Rect") -> bool:
        return not (
            other.xb < self.xa
            or self.xb < other.xa
            or other.yb < self.ya
            or self.yb < other.ya
        )

    def intersection(self, other: "Rect") -> "Rect | None":
        if not self.intersects(other):
            return None
        return Rect(
            max(self.xa, other.xa),
            max(self.ya, other.ya),
            min(self.xb, other.xb),
            min(self.yb, other.yb),
        )

    def cells(self) -> Iterator[Tuple[int, int]]:
        for x in range(self.xa, self.xb + 1):
            for y in range(self.ya, self.yb + 1):
                yield x, y

    def slices(self) -> Tuple[slice, slice]:
        """Numpy slices selecting this rectangle from an ``[x, y]``-indexed array."""
        return slice(self.xa, self.xb + 1), slice(self.ya, self.yb + 1)

    def __iter__(self):  # allows ``xa, ya, xb, yb = rect``
        return iter(self.as_tuple())


@dataclass(frozen=True)
class Droplet(Rect):
    """A droplet ``delta = (xa, ya, xb, yb)`` (paper notation)."""

    def manhattan(self, other: "Rect") -> int:
        """Manhattan distance ``D(delta, delta_g)`` between two droplet locations.

        Routing jobs preserve the droplet size (Sec. IV-A), so the distance
        between the south-west corners equals the distance between the
        north-east corners.
        """
        return abs(self.xa - other.xa) + abs(self.ya - other.ya)

    @classmethod
    def at(cls, x: int, y: int, w: int, h: int) -> "Droplet":
        """Droplet of size ``w x h`` whose south-west corner is ``(x, y)``."""
        return cls(x, y, x + w - 1, y + h - 1)


def chip_rect(width: int, height: int) -> Rect:
    """Rectangle covering the full ``W x H`` biochip."""
    return Rect(0, 0, width - 1, height - 1)
