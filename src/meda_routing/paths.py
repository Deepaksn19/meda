"""Where results go: everything a run produces sits in one folder of the project.

A training run writes, under ``<project>/runs/<name>/seed_<s>/``:

* ``model.zip`` (latest), ``best_model.zip`` and ``checkpoints/epoch_XXX.zip``,
* ``progress.csv``, ``summary.json``, ``config.yaml``,
* ``training_curves.png`` (redrawn after every epoch),

and the commands that use a trained model (``evaluate``, ``compare``,
``bioassay``, ``render``, ``plot-training``) put their tables, plots and GIFs
into ``<run>/eval/``, ``<run>/bioassay/`` etc. next to that model by default.

``<project>`` is the folder holding this repository's ``pyproject.toml``
(the folder with the RL code), whatever the current working directory is.
Set ``MEDA_RUNS_DIR`` to put the runs somewhere else.  Paths given
explicitly on the command line keep their usual meaning (relative to the
current directory).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, "os.PathLike[str]"]

#: Environment variable overriding the default runs folder.
RUNS_ENV_VAR = "MEDA_RUNS_DIR"


def project_root() -> Path:
    """The repository folder (with ``pyproject.toml``); the CWD if not found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.exists() and "meda-gnn-routing" in pyproject.read_text(encoding="utf-8"):
            return parent
    return Path.cwd()


def runs_dir() -> Path:
    """Default folder for all runs: ``$MEDA_RUNS_DIR`` or ``<project>/runs``."""
    override = os.environ.get(RUNS_ENV_VAR)
    return Path(override).expanduser() if override else project_root() / "runs"


def resolve_output_dir(path: Optional[PathLike]) -> Path:
    """Output folder of a config: ``runs`` (the default) means :func:`runs_dir`;
    other relative paths are taken relative to the project folder."""
    if path is None or str(path) in ("", "runs"):
        return runs_dir()
    p = Path(path).expanduser()
    return p if p.is_absolute() else project_root() / p


def find_existing(path: PathLike) -> Path:
    """``path`` as given if it exists, else relative to the project folder."""
    p = Path(path).expanduser()
    if p.exists() or p.is_absolute():
        return p
    candidate = project_root() / p
    return candidate if candidate.exists() else p


def run_dir_of(model_file: PathLike) -> Path:
    """The run folder (``seed_*``) of a resolved model file, also for checkpoints."""
    parent = Path(model_file).parent
    return parent.parent if parent.name == "checkpoints" else parent
