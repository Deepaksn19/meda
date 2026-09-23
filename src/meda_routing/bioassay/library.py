"""Benchmark bioassays as sequence graphs of microfluidic operations (Sec. V-B).

Fig. 9 of the paper evaluates the routers by executing two COVID-19 testing
bioassays [34], a PCR-based assay (COVID-PCR) and a rapid antigen test
(COVID-RAT), on a ``60 x 30`` MEDA biochip.  The paper does not list the
assays; they are transcribed here from the authors' reference implementation
(``melfar87/MEDA``, ``meda_sgs.py``: ``Bioassays.sg_CPCR``, ``sg_CRAT`` and the
toy ``sg_Simple``; chip size from ``test_bioassay.py``).

A bioassay is a sequence graph of microfluidic operations (MOs).  Every MO
entry of the reference is ``[name, type, pre, cond, starts, goals]``:

* ``type`` is one of ``Dis`` (dispense from an off-chip reservoir), ``Out``
  (move to the output port), ``Dsc`` (discard to the waste reservoir),
  ``Mix``, ``Dlt`` (dilute), ``Spt`` (split), ``Mag`` (magnetic sensing),
  ``Was`` (wash) and ``Thm`` (thermal cycling);
* the MO may start once every MO named in ``pre`` *and* ``cond`` is done;
* ``starts`` are the droplet locations where the MO picks its droplets up (the
  locations where preceding MOs left them), ``goals`` the locations it routes
  them to.

Coordinates in ``meda_sgs.py`` are 1-based and inclusive (``[8, 1, 11, 4]`` is
a ``4 x 4`` droplet); they are converted to the 0-based inclusive
:class:`~meda_routing.core.geometry.Droplet` of this package by subtracting 1
from all four values (the reference subtracts 1 from ``xa, ya`` and treats
``xb, yb`` as exclusive, which selects the same MCs).  Dispense MOs start off
the chip (e.g. ``y < 0``); such locations are kept as plain droplets and are
never validated against the chip.

Every MO expands into routing jobs as in ``Mo.createRoutingJobs`` of the
reference ``meda_scheduler.py`` (see :meth:`MicrofluidicOperation.job_specs`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

from ..core.geometry import Droplet, Rect

#: MO types of the reference scheduler (``MoTypes`` in ``meda_utils.py``).
MO_TYPES: Tuple[str, ...] = ("Dis", "Out", "Dsc", "Mix", "Dlt", "Mag", "Was", "Thm", "Spt")
#: Dispensing is not routed by the policy: the droplet is pushed in from the
#: reservoir deterministically (``MedaScheduler.updateControlActions``).
DISPENSE_TYPES = frozenset({"Dis"})
_SINGLE_JOB = frozenset({"Dis", "Out", "Dsc", "Mag", "Was", "Thm"})
_MERGE = frozenset({"Mix", "Dlt"})
_SPLIT = frozenset({"Spt"})

#: Chip size used for the reference bioassay experiments (``test_bioassay.py``).
PAPER_CHIP_SIZE: Tuple[int, int] = (60, 30)

Coords = Tuple[int, int, int, int]
#: ``(name, type, pre, cond, starts, goals)`` with 1-based inclusive coordinates.
SequenceGraphEntry = Tuple[
    str, str, Sequence[str], Sequence[str], Sequence[Coords], Sequence[Coords]
]


def droplet_from_reference(coords: Sequence[int]) -> Droplet:
    """1-based inclusive ``[xa, ya, xb, yb]`` of ``meda_sgs.py`` -> 0-based droplet."""
    xa, ya, xb, yb = (int(v) - 1 for v in coords)
    return Droplet(xa, ya, xb, yb)


@dataclass(frozen=True)
class JobSpec:
    """One routing job of an MO: move a droplet from ``start`` to ``goal``."""

    mo: str
    name: str
    start: Droplet
    goal: Droplet
    #: Dispense jobs are executed deterministically, not by the router.
    dispense: bool = False

    @property
    def key(self) -> str:
        return f"{self.mo}.{self.name}"


@dataclass(frozen=True)
class MicrofluidicOperation:
    """A node ``[name, type, pre, cond, starts, goals]`` of a sequence graph."""

    name: str
    type: str
    pre: Tuple[str, ...] = ()
    cond: Tuple[str, ...] = ()
    #: Pick-up locations (may lie off the chip for dispensing).
    starts: Tuple[Droplet, ...] = ()
    goals: Tuple[Droplet, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "pre", tuple(self.pre))
        object.__setattr__(self, "cond", tuple(self.cond))
        object.__setattr__(self, "starts", tuple(_as_droplet(r) for r in self.starts))
        object.__setattr__(self, "goals", tuple(_as_droplet(r) for r in self.goals))
        if self.type not in MO_TYPES:
            raise ValueError(f"MO {self.name}: unknown type {self.type!r}")
        need_starts = 2 if self.type in _MERGE | _SPLIT else 1
        need_goals = 2 if self.type in _SPLIT else 1
        if len(self.starts) < need_starts or len(self.goals) < need_goals:
            raise ValueError(
                f"MO {self.name} ({self.type}) needs {need_starts} start(s) and "
                f"{need_goals} goal(s), got {len(self.starts)} and {len(self.goals)}"
            )
        for spec in self.job_specs():
            if spec.start.size != spec.goal.size:
                raise ValueError(f"MO {self.name}: job {spec.name} changes the droplet size")

    @property
    def dependencies(self) -> Tuple[str, ...]:
        """MOs that must be done before this MO may start (``pre`` and ``cond``)."""
        return tuple(dict.fromkeys(self.pre + self.cond))

    @property
    def is_dispense(self) -> bool:
        return self.type in DISPENSE_TYPES

    def job_specs(self) -> List[JobSpec]:
        """Routing jobs of this MO (reference ``Mo.createRoutingJobs``).

        * ``Dis/Out/Dsc/Mag/Was/Thm``: one job ``starts[0] -> goals[0]``;
        * ``Mix/Dlt``: two jobs ``starts[0] -> goals[0]`` and
          ``starts[1] -> goals[0]`` (both droplets meet at the mixer);
        * ``Spt``: two jobs ``starts[0] -> goals[0]`` and ``starts[1] -> goals[1]``.
        """
        dispense = self.is_dispense
        s, g = self.starts, self.goals
        if self.type in _SINGLE_JOB:
            pairs = [(s[0], g[0])]
        elif self.type in _MERGE:
            pairs = [(s[0], g[0]), (s[1], g[0])]
        else:
            # NOTE: the Python reference (meda_scheduler.py) sends the second
            # split droplet to locations[0] as well, a copy-paste bug: both
            # halves would end on the same MCs and later MOs (e.g. COVID-PCR
            # n13 picking up at goals[1]) would start where no droplet was left.
            # We deliberately use goals[1], as the MATLAB reference
            # (MedaSchedulerClass.mdPreprocessMo: locations{jobId}) does.
            pairs = [(s[0], g[0]), (s[1], g[1])]
        return [
            JobSpec(self.name, f"J{i}", start, goal, dispense)
            for i, (start, goal) in enumerate(pairs)
        ]


def _as_droplet(r: Rect | Sequence[int]) -> Droplet:
    if isinstance(r, Droplet):
        return r
    xa, ya, xb, yb = tuple(r)
    return Droplet(int(xa), int(ya), int(xb), int(yb))


@dataclass(frozen=True)
class Bioassay:
    """A sequence graph of MOs designed for a ``width x height`` biochip."""

    name: str
    width: int
    height: int
    operations: Tuple[MicrofluidicOperation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        names = [mo.name for mo in self.operations]
        if len(set(names)) != len(names):
            raise ValueError(f"bioassay {self.name}: duplicate MO names")
        known = set(names)
        for mo in self.operations:
            missing = [d for d in mo.dependencies if d not in known]
            if missing:
                raise ValueError(f"bioassay {self.name}: MO {mo.name} depends on unknown {missing}")

    def __len__(self) -> int:
        return len(self.operations)

    def __iter__(self):
        return iter(self.operations)

    @property
    def names(self) -> List[str]:
        return [mo.name for mo in self.operations]

    def operation(self, name: str) -> MicrofluidicOperation:
        for mo in self.operations:
            if mo.name == name:
                return mo
        raise KeyError(name)

    def job_specs(self) -> List[JobSpec]:
        return [spec for mo in self.operations for spec in mo.job_specs()]

    def routed_job_specs(self) -> List[JobSpec]:
        """Jobs executed by a router (everything except dispensing)."""
        return [spec for spec in self.job_specs() if not spec.dispense]

    def on_chip_locations(self) -> List[Droplet]:
        """Distinct start/goal locations that lie on the biochip.

        Useful to keep reservoirs, mixers and ports free of injected faults.
        """
        chip = Rect(0, 0, self.width - 1, self.height - 1)
        seen: Dict[Droplet, None] = {}
        for spec in self.job_specs():
            for r in (spec.start, spec.goal):
                if chip.contains(r):
                    seen.setdefault(r, None)
        return list(seen)


def bioassay_from_reference(
    name: str,
    graph: Sequence[SequenceGraphEntry],
    width: int = PAPER_CHIP_SIZE[0],
    height: int = PAPER_CHIP_SIZE[1],
) -> Bioassay:
    """Convert a ``meda_sgs.py`` sequence graph (1-based coordinates)."""
    ops = [
        MicrofluidicOperation(
            name=str(entry[0]),
            type=str(entry[1]),
            pre=tuple(entry[2]),
            cond=tuple(entry[3]),
            starts=tuple(droplet_from_reference(c) for c in entry[4]),
            goals=tuple(droplet_from_reference(c) for c in entry[5]),
        )
        for entry in graph
    ]
    return Bioassay(name, int(width), int(height), tuple(ops))


# --------------------------------------------------------------------------
# Sequence graphs transcribed from meda_sgs.py (1-based inclusive coordinates;
# checked against the reference file by tests/test_bioassay.py).
# fmt: off
_SG_SIMPLE = (
    ('n00', 'Dis', (), (), ((8, -3, 11, 0),), ((8, 1, 11, 4),)),
    ('n01', 'Dis', (), (), ((21, -3, 24, 0),), ((21, 1, 24, 4),)),
    ('n08', 'Mix', ('n00', 'n01'), (), ((8, 1, 11, 4), (21, 1, 24, 4)), ((14, 19, 17, 22), (14, 19, 17, 22))),
)

_SG_CRAT = (
    ('n00', 'Dis', (), (), ((8, -3, 11, 0),), ((8, 1, 11, 4),)),
    ('n01', 'Dis', (), (), ((48, -3, 51, 0),), ((48, 1, 51, 4),)),
    ('n02', 'Dis', (), (), ((14, 31, 17, 34),), ((14, 26, 17, 29),)),
    ('n03', 'Dis', (), (), ((44, 31, 47, 34),), ((44, 26, 47, 29),)),
    ('n04', 'Dis', (), (), ((28, -3, 31, 0),), ((28, 1, 31, 4),)),
    ('n05', 'Dis', (), ('n08',), ((48, -3, 51, 0),), ((48, 1, 51, 4),)),
    ('n06', 'Dis', (), ('n09',), ((14, 31, 17, 34),), ((14, 26, 17, 29),)),
    ('n07', 'Dis', (), ('n10',), ((44, 31, 47, 34),), ((44, 26, 47, 29),)),
    ('n08', 'Mix', ('n00', 'n01'), (), ((8, 1, 11, 4), (48, 1, 51, 4)), ((14, 19, 17, 22), (14, 19, 17, 22))),
    ('n09', 'Mix', ('n02', 'n14'), ('n14',), ((14, 26, 17, 29), (14, 9, 17, 12)), ((14, 19, 17, 22), (14, 19, 17, 22))),
    ('n10', 'Mix', ('n03', 'n15'), ('n15',), ((44, 26, 47, 29), (14, 9, 17, 12)), ((14, 19, 17, 22), (14, 19, 17, 22))),
    ('n11', 'Mix', ('n04', 'n05'), (), ((28, 1, 31, 4), (48, 1, 51, 4)), ((44, 9, 47, 12), (44, 9, 47, 12))),
    ('n12', 'Mix', ('n06', 'n17'), ('n17',), ((14, 26, 17, 29), (44, 19, 47, 22)), ((44, 9, 47, 12), (44, 9, 47, 12))),
    ('n13', 'Mix', ('n07', 'n18'), ('n18',), ((44, 26, 47, 29), (44, 19, 47, 22)), ((44, 9, 47, 12), (44, 9, 47, 12))),
    ('n14', 'Spt', ('n08',), (), ((14, 19, 17, 22), (14, 19, 17, 22)), ((10, 9, 13, 12), (18, 9, 21, 12))),
    ('n15', 'Spt', ('n09',), ('n20', 'n09'), ((14, 19, 17, 22), (14, 19, 17, 22)), ((10, 9, 13, 12), (18, 9, 21, 12))),
    ('n16', 'Spt', ('n10',), ('n21', 'n10'), ((14, 19, 17, 22), (14, 19, 17, 22)), ((10, 9, 13, 12), (18, 9, 21, 12))),
    ('n17', 'Spt', ('n11',), (), ((44, 9, 47, 12), (44, 9, 47, 12)), ((40, 19, 43, 22), (48, 19, 51, 22))),
    ('n18', 'Spt', ('n12',), ('n24', 'n12'), ((44, 9, 47, 12), (44, 9, 47, 12)), ((40, 19, 43, 22), (48, 19, 51, 22))),
    ('n19', 'Spt', ('n13',), ('n25', 'n13'), ((44, 9, 47, 12), (44, 9, 47, 12)), ((40, 19, 43, 22), (48, 19, 51, 22))),
    ('n20', 'Dsc', ('n14',), (), ((10, 9, 13, 12),), ((1, 14, 4, 17),)),
    ('n21', 'Dsc', ('n15',), (), ((10, 9, 13, 12),), ((1, 14, 4, 17),)),
    ('n22', 'Dsc', ('n16',), (), ((10, 9, 13, 12),), ((1, 14, 4, 17),)),
    ('n23', 'Out', ('n16',), (), ((10, 9, 13, 12),), ((55, 14, 58, 17),)),
    ('n24', 'Dsc', ('n17',), (), ((40, 19, 43, 22),), ((1, 14, 4, 17),)),
    ('n25', 'Dsc', ('n18',), (), ((40, 19, 43, 22),), ((1, 14, 4, 17),)),
    ('n26', 'Dsc', ('n19',), (), ((40, 19, 43, 22),), ((1, 14, 4, 17),)),
    ('n27', 'Out', ('n19',), (), ((40, 19, 43, 22),), ((55, 14, 58, 17),)),
)

_SG_CPCR = (
    ('n00', 'Dis', (), (), ((14, -3, 17, 0),), ((14, 1, 17, 4),)),
    ('n01', 'Dis', (), (), ((44, -3, 47, 0),), ((44, 1, 47, 4),)),
    ('n02', 'Dis', (), (), ((14, 31, 17, 34),), ((14, 26, 17, 29),)),
    ('n03', 'Dis', (), (), ((44, 31, 47, 34),), ((44, 26, 47, 29),)),
    ('n04', 'Mix', ('n00', 'n01'), (), ((14, 1, 17, 4), (44, 1, 47, 4)), ((14, 19, 17, 22), (14, 19, 17, 22))),
    ('n05', 'Mix', ('n02', 'n03'), (), ((14, 26, 17, 29), (44, 26, 47, 29)), ((44, 9, 47, 12), (44, 9, 47, 12))),
    ('n06', 'Spt', ('n04',), (), ((14, 19, 17, 22), (14, 19, 17, 22)), ((20, 9, 23, 12), (28, 9, 31, 12))),
    ('n07', 'Spt', ('n05',), ('n08',), ((44, 9, 47, 12), (44, 9, 47, 12)), ((20, 9, 23, 12), (28, 9, 31, 12))),
    ('n08', 'Dsc', ('n06',), (), ((20, 9, 23, 12),), ((1, 14, 4, 17),)),
    ('n09', 'Mix', ('n06', 'n07'), ('n06',), ((20, 9, 23, 12), (24, 9, 27, 12)), ((14, 19, 17, 22), (14, 19, 17, 22))),
    ('n10', 'Dsc', ('n07',), (), ((20, 9, 23, 12),), ((1, 14, 4, 17),)),
    ('n11', 'Spt', ('n09',), ('n10', 'n09'), ((14, 19, 17, 22), (14, 19, 17, 22)), ((20, 9, 23, 12), (28, 9, 31, 12))),
    ('n12', 'Dsc', ('n11',), (), ((20, 9, 23, 12),), ((1, 14, 4, 17),)),
    ('n13', 'Thm', ('n11',), (), ((28, 9, 31, 12),), ((14, 9, 17, 12),)),
    ('n14', 'Thm', ('n13',), (), ((14, 9, 17, 12),), ((44, 19, 47, 22),)),
    ('n15', 'Thm', ('n14',), ('n14',), ((44, 19, 47, 22),), ((14, 9, 17, 12),)),
    ('n16', 'Thm', ('n15',), ('n15',), ((14, 9, 17, 12),), ((44, 19, 47, 22),)),
    ('n17', 'Thm', ('n16',), ('n16',), ((44, 19, 47, 22),), ((14, 9, 17, 12),)),
    ('n18', 'Thm', ('n17',), ('n17',), ((14, 9, 17, 12),), ((44, 19, 47, 22),)),
    ('n19', 'Thm', ('n18',), ('n18',), ((44, 19, 47, 22),), ((14, 9, 17, 12),)),
    ('n20', 'Thm', ('n19',), ('n19',), ((14, 9, 17, 12),), ((44, 19, 47, 22),)),
    ('n21', 'Thm', ('n20',), ('n20',), ((44, 19, 47, 22),), ((14, 9, 17, 12),)),
    ('n22', 'Thm', ('n21',), ('n21',), ((14, 9, 17, 12),), ((44, 19, 47, 22),)),
    ('n23', 'Thm', ('n22',), ('n22',), ((44, 19, 47, 22),), ((14, 9, 17, 12),)),
    ('n24', 'Thm', ('n23',), ('n23',), ((14, 9, 17, 12),), ((44, 19, 47, 22),)),
    ('n25', 'Thm', ('n24',), ('n24',), ((44, 19, 47, 22),), ((14, 9, 17, 12),)),
    ('n26', 'Thm', ('n25',), ('n25',), ((14, 9, 17, 12),), ((44, 19, 47, 22),)),
    ('n27', 'Thm', ('n26',), ('n26',), ((44, 19, 47, 22),), ((14, 9, 17, 12),)),
    ('n28', 'Thm', ('n27',), ('n27',), ((14, 9, 17, 12),), ((44, 19, 47, 22),)),
    ('n29', 'Thm', ('n28',), ('n28',), ((44, 19, 47, 22),), ((14, 9, 17, 12),)),
    ('n30', 'Thm', ('n29',), ('n29',), ((14, 9, 17, 12),), ((44, 19, 47, 22),)),
    ('n31', 'Thm', ('n30',), ('n30',), ((44, 19, 47, 22),), ((14, 9, 17, 12),)),
    ('n32', 'Thm', ('n31',), ('n31',), ((14, 9, 17, 12),), ((44, 19, 47, 22),)),
    ('n33', 'Out', ('n32',), (), ((44, 19, 47, 22),), ((55, 14, 58, 17),)),
)

# fmt: on


def simple() -> Bioassay:
    """Toy assay ``sg_Simple``: dispense two droplets and mix them."""
    return bioassay_from_reference("simple", _SG_SIMPLE)


def covid_rat() -> Bioassay:
    """COVID-19 rapid antigen test (``sg_CRAT``, 28 MOs)."""
    return bioassay_from_reference("covid-rat", _SG_CRAT)


def covid_pcr() -> Bioassay:
    """COVID-19 PCR test (``sg_CPCR``, 34 MOs, 20 thermal-cycling moves)."""
    return bioassay_from_reference("covid-pcr", _SG_CPCR)


BIOASSAYS: Dict[str, Callable[[], Bioassay]] = {
    "covid-rat": covid_rat,
    "covid-pcr": covid_pcr,
    "simple": simple,
}
_ALIASES = {"crat": "covid-rat", "rat": "covid-rat", "cpcr": "covid-pcr", "pcr": "covid-pcr"}


def available_bioassays() -> List[str]:
    return list(BIOASSAYS)


def get_bioassay(name: str) -> Bioassay:
    """Look a bioassay up by name (``"covid-rat"``, ``"covid-pcr"``, ``"simple"``)."""
    key = name.strip().lower().replace("_", "-")
    key = _ALIASES.get(key, key)
    try:
        return BIOASSAYS[key]()
    except KeyError:
        raise KeyError(f"unknown bioassay {name!r}; available: {available_bioassays()}") from None
