"""Monte-Carlo bioassay benchmark: probability of completion vs. cycles (Fig. 9).

Fig. 9 of the paper plots, for COVID-RAT and COVID-PCR, the probability that
the whole bioassay is completed within ``k`` control cycles when routed by the
baseline, the formal and the DRL policies.  The reference experiment
(``melfar87/MEDA``, ``test_bioassay.py``) executes 1000 trials on a ``60 x 30``
chip with a limit of 2000 cycles.  Before every trial the scheduler
pre-ages the chip with ``n_ij ~ U{0, 399}`` actuations
(``MedaScheduler.reset``: ``np.random.randint(0, 400)``); the degradation
parameters are ``tau ~ U(0.5, 0.7)``, ``c ~ U(500, 800)`` (Sec. V-A).  The
curve is the empirical CDF of the cycle counts (``meda_utils.plotProbVsCycles``).

Note that both reference implementations actually ran their bioassays with
the fixed parameters ``tau = 0.7``, ``c = 200`` (the ``MEDAEnv`` defaults,
which ``setState`` never overrides; ``C2Range = 200`` in the MATLAB
``TestBiochipV13.m``), i.e. much faster aging than Sec. V-A.  To reproduce
that setting pass ``degradation=DegradationConfig(tau_range=(0.7, 0.7),
c_range=(200.0, 200.0))`` to :func:`run_trials`.

Seeding: trial ``i`` derives its chip and movement random streams from
``SeedSequence(seed).spawn(n)[i]``, so different routers evaluated with the
same seed face exactly the same chips (common random numbers) and results do
not depend on how the trials are split across processes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.biochip import DegradationConfig, MEDABiochip
from ..core.geometry import chip_rect
from .library import Bioassay, get_bioassay
from .scheduler import BioassayExecutor, BioassayResult, RouterFactory

ChipFactory = Callable[[Bioassay, np.random.Generator], MEDABiochip]
ProgressCallback = Callable[[int, int, BioassayResult], None]


@dataclass
class AgedChipFactory:
    """Default chip for a bioassay trial: fresh parameters, pre-aged MCs.

    ``fault_fraction`` places fully degraded, sensed MCs and
    ``hidden_defect_fraction`` invisible defects in ``fault_cluster`` squares
    (Sec. V-B / VI-C); the on-chip start and goal locations of the bioassay
    (reservoir ports, mixers, waste, output) are kept fault-free when
    ``protect_locations`` is set.
    """

    max_initial_actuations: int = 399  # [REF-CODE] pre-aged chips, n ~ U{0, 399}
    fault_fraction: float = 0.0  # [ASSUMED] Fig. 9 uses no injected faults
    hidden_defect_fraction: float = 0.0  # [ASSUMED]
    fault_cluster: int = 2  # [PAPER Sec. V-B] 2x2 clusters
    protect_locations: bool = True  # [ASSUMED]
    degradation: Optional[DegradationConfig] = None  # paper ranges; the authors' Fig. 9 used tau=0.7, c=200 [REF-CODE]

    def __call__(self, bioassay: Bioassay, rng: np.random.Generator) -> MEDABiochip:
        base = self.degradation or DegradationConfig()
        n_max = int(self.max_initial_actuations)
        config = dataclasses.replace(
            base,
            initial_actuations="uniform" if n_max > 0 else "zero",
            max_initial_actuations=max(n_max, 0),
        )
        chip = MEDABiochip(bioassay.width, bioassay.height, config, rng)
        chip.reset(resample_parameters=True)  # tau, c ~ U(...), N ~ U{0, n_max}
        protect = bioassay.on_chip_locations() if self.protect_locations else []
        area = chip_rect(chip.width, chip.height)
        protect = [r for r in protect if area.contains(r)]
        if self.fault_fraction > 0:
            chip.inject_faults(self.fault_fraction, self.fault_cluster, protect=protect)
        if self.hidden_defect_fraction > 0:
            chip.inject_faults(
                self.hidden_defect_fraction, self.fault_cluster, protect=protect, hidden=True
            )
        return chip


@dataclass
class BenchmarkSummary:
    n_trials: int
    n_success: int
    success_rate: float
    #: Statistics over the successful trials (NaN if there are none).
    mean_cycles: float
    median_cycles: float
    p90_cycles: float
    #: Smallest ``k`` with ``P[K <= k] >= 0.9`` over *all* trials (inf if never).
    k_at_p90: float

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _trial_cycles(cycles: Sequence[float]) -> np.ndarray:
    """Per-trial cycle counts as floats, failed trials (``inf`` or NaN) as ``inf``."""
    data = np.asarray(cycles, dtype=np.float64).ravel()
    return np.where(np.isnan(data), np.inf, data)


def completion_cdf(
    cycles: Sequence[float], k_grid: Optional[Sequence[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical ``P[K <= k]`` of the bioassay completion time ``K``.

    Matches ``meda_utils.plotProbVsCycles``: by default ``k`` runs over the
    integers from ``min - 1`` to ``max + 1`` of the observed cycle counts.
    Failed trials (``inf`` or NaN) never count as completed but stay in the
    denominator, so the curve saturates at the success rate.
    """
    data = np.sort(_trial_cycles(cycles))
    if data.size == 0:
        raise ValueError("no trials")
    if k_grid is None:
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            k = np.zeros(1)
        else:
            k = np.arange(int(finite.min()) - 1, int(finite.max()) + 2, dtype=np.float64)
    else:
        k = np.asarray(k_grid, dtype=np.float64)
    p = np.searchsorted(data, k, side="right") / data.size
    return k, p


def cycles_at_probability(cycles: Sequence[float], p: float = 0.9) -> float:  # p = 0.9 [ASSUMED]
    """Smallest ``k`` with ``P[K <= k] >= p`` (inf if never reached)."""
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must lie in (0, 1], got {p}")
    data = np.sort(_trial_cycles(cycles))
    if data.size == 0:
        raise ValueError("no trials")
    idx = int(np.ceil(p * data.size - 1e-9)) - 1
    return float(data[min(max(idx, 0), data.size - 1)])


def summarize(cycles: Sequence[float], p: float = 0.9) -> BenchmarkSummary:
    """Summary statistics of per-trial cycle counts (``inf`` = failed trial).

    ``p`` is the probability level of ``k_at_p90`` (0.9 as quoted in Sec. V-B:
    "COVID-PCR within k = 762 with probability p > 0.9").
    """
    data = _trial_cycles(cycles)
    ok = data[np.isfinite(data)]
    n = int(data.size)
    has = ok.size > 0
    return BenchmarkSummary(
        n_trials=n,
        n_success=int(ok.size),
        success_rate=float(ok.size / n) if n else float("nan"),
        mean_cycles=float(ok.mean()) if has else float("nan"),
        median_cycles=float(np.median(ok)) if has else float("nan"),
        p90_cycles=float(np.percentile(ok, 90)) if has else float("nan"),
        k_at_p90=cycles_at_probability(data, p) if n else float("inf"),
    )


def trial_rngs(
    seed: Optional[int], n_trials: int, start_index: int = 0
) -> List[Tuple[np.random.Generator, np.random.Generator]]:
    """``(chip_rng, move_rng)`` of trials ``start_index, ..., start_index + n - 1``.

    Trial ``i`` uses child ``i`` of ``SeedSequence(seed)`` (identical to
    ``SeedSequence(seed).spawn(i + 1)[i]``), split into a chip and a movement
    stream.
    """
    root = np.random.SeedSequence(seed)
    out = []
    for i in range(int(start_index), int(start_index) + int(n_trials)):
        child = np.random.SeedSequence(root.entropy, spawn_key=tuple(root.spawn_key) + (i,))
        chip_ss, move_ss = child.spawn(2)
        out.append((np.random.default_rng(chip_ss), np.random.default_rng(move_ss)))
    return out


def run_trials(
    bioassay: Union[Bioassay, str],
    router_factory: RouterFactory,
    n_trials: int,
    seed: Optional[int] = 0,
    chip_factory: Optional[ChipFactory] = None,
    *,
    max_initial_actuations: Optional[int] = None,
    fault_fraction: float = 0.0,
    hidden_defect_fraction: float = 0.0,
    fault_cluster: Optional[int] = None,
    degradation: Optional[DegradationConfig] = None,
    progress: Optional[ProgressCallback] = None,
    start_index: int = 0,
    **executor_kwargs: Any,
) -> Tuple[np.ndarray, BenchmarkSummary]:
    """Execute ``n_trials`` independent runs of ``bioassay``.

    Returns the per-trial cycle counts (``np.inf`` for failed trials) as a
    float array and their :func:`summarize`.  The chip keyword arguments
    configure the default :class:`AgedChipFactory` (``None``: its defaults,
    ``max_initial_actuations=399``, ``fault_cluster=2``) and are rejected
    together with a custom ``chip_factory``; ``executor_kwargs`` are passed
    on to :class:`BioassayExecutor` (``kmax_alpha``, ``on_timeout``,
    ``max_cycles``, ...).  ``progress(i, n_trials, result)`` is called after
    trial ``i`` (1-based) with its :class:`BioassayResult`.  ``start_index``
    selects the first trial, so ``n`` trials can be split into chunks (e.g.
    across processes) that reproduce a single run exactly.
    """
    assay = get_bioassay(bioassay) if isinstance(bioassay, str) else bioassay
    chip_options = {
        "max_initial_actuations": max_initial_actuations,
        "fault_fraction": fault_fraction or None,
        "hidden_defect_fraction": hidden_defect_fraction or None,
        "fault_cluster": fault_cluster,
        "degradation": degradation,
    }
    given = {k: v for k, v in chip_options.items() if v is not None}
    if chip_factory is None:
        chip_factory = AgedChipFactory(**given)
    elif given:
        raise ValueError(
            f"{sorted(given)} only apply to the default chip factory, not to a custom chip_factory"
        )
    cycles = np.full(int(n_trials), np.inf)
    for i, (chip_rng, move_rng) in enumerate(trial_rngs(seed, int(n_trials), start_index)):
        chip = chip_factory(assay, chip_rng)
        result = BioassayExecutor(assay, router_factory, chip, move_rng, **executor_kwargs).run()
        if result.success:
            cycles[i] = result.cycles
        if progress is not None:
            progress(i + 1, int(n_trials), result)
    return cycles, summarize(cycles)
