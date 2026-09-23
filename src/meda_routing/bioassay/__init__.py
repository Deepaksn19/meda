"""Bioassay execution on MEDA biochips (Sec. V-B, Fig. 9).

* :mod:`.library` — the COVID-RAT / COVID-PCR sequence graphs of the
  reference implementation and their expansion into routing jobs;
* :mod:`.scheduler` — :class:`BioassayExecutor`, which runs all concurrent
  routing jobs of a bioassay on one aging biochip;
* :mod:`.benchmark` — Monte-Carlo trials and the completion-probability curve
  ``P[K <= k]`` of Fig. 9.
"""

from .benchmark import (
    AgedChipFactory,
    BenchmarkSummary,
    completion_cdf,
    cycles_at_probability,
    run_trials,
    summarize,
    trial_rngs,
)
from .library import (
    BIOASSAYS,
    MO_TYPES,
    PAPER_CHIP_SIZE,
    Bioassay,
    JobSpec,
    MicrofluidicOperation,
    available_bioassays,
    bioassay_from_reference,
    covid_pcr,
    covid_rat,
    droplet_from_reference,
    get_bioassay,
    simple,
)
from .scheduler import (
    TIMEOUT_MODES,
    BioassayExecutor,
    BioassayResult,
    JobRecord,
    MOState,
    execute_bioassay,
)

__all__ = [
    "AgedChipFactory",
    "BIOASSAYS",
    "BenchmarkSummary",
    "Bioassay",
    "BioassayExecutor",
    "BioassayResult",
    "JobRecord",
    "JobSpec",
    "MOState",
    "MO_TYPES",
    "MicrofluidicOperation",
    "PAPER_CHIP_SIZE",
    "TIMEOUT_MODES",
    "available_bioassays",
    "bioassay_from_reference",
    "completion_cdf",
    "covid_pcr",
    "covid_rat",
    "cycles_at_probability",
    "droplet_from_reference",
    "execute_bioassay",
    "get_bioassay",
    "run_trials",
    "simple",
    "summarize",
    "trial_rngs",
]
