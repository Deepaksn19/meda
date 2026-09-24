"""Shared fixtures for the core-model and environment tests.

Chips are built directly from :class:`MEDABiochip` so that every test controls
the degradation state it needs (Sec. III-A, Eq. 1: ``D = tau ** (n / c)``):

* an *ideal* chip has ``tau = 1`` everywhere, so ``D = 1`` however often an
  MC is actuated and every droplet movement succeeds deterministically;
* a *worn* chip uses the paper's parameter ranges (Sec. V-A) and random
  initial wear, so all four 2-bit health levels occur.

Every fixture that draws random numbers is seeded.  Environments from
``make_env`` start unseeded; every test that steps or samples one seeds it
through ``reset(seed=...)`` first, so the suite is reproducible.  The only
``autouse`` fixture sends default outputs (``runs/``) to a temporary folder,
so tests never write into the repository.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, List

import numpy as np
import pytest

from meda_routing.core.biochip import DegradationConfig, MEDABiochip
from meda_routing.envs.meda_env import MEDARoutingEnv


@pytest.fixture(autouse=True)
def _runs_in_tmp(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("MEDA_RUNS_DIR", str(tmp_path_factory.mktemp("runs")))
    monkeypatch.delenv("MEDA_DEVICE", raising=False)


def make_ideal_chip(width: int, height: int, seed: int = 0) -> MEDABiochip:
    """Chip whose MCs never wear out (``tau = 1`` => ``D = 1``)."""
    chip = MEDABiochip(width, height, rng=np.random.default_rng(seed))
    chip.tau[:] = 1.0
    return chip


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed generator, fresh for every test."""
    return np.random.default_rng(20230401)


@pytest.fixture
def ideal_chip_factory() -> Callable[..., MEDABiochip]:
    """``factory(width, height, seed=0)`` -> chip with ``D = 1`` forever."""
    return make_ideal_chip


@pytest.fixture
def ideal_chip() -> MEDABiochip:
    """16 x 12 chip with ``D = 1`` forever."""
    return make_ideal_chip(16, 12)


@pytest.fixture
def worn_chip(rng: np.random.Generator) -> MEDABiochip:
    """14 x 10 chip, paper degradation parameters and ``n_ij ~ U{0, 2500}``."""
    config = DegradationConfig(initial_actuations="uniform", max_initial_actuations=2500)
    chip = MEDABiochip(14, 10, config, rng=rng)
    chip.reset()
    return chip


@pytest.fixture
def random_degradation(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    """``make(width, height, low=0, high=1, p_zero=0, p_one=0)`` -> ``D`` matrix.

    ``D ~ U(low, high)`` per MC; a fraction ``p_zero`` (``p_one``) of the MCs
    is then set to fully degraded (fully healthy).
    """

    def make(
        width: int,
        height: int,
        low: float = 0.0,
        high: float = 1.0,
        p_zero: float = 0.0,
        p_one: float = 0.0,
    ) -> np.ndarray:
        deg = rng.uniform(low, high, size=(width, height))
        deg[rng.random((width, height)) < p_zero] = 0.0
        deg[rng.random((width, height)) < p_one] = 1.0
        return deg

    return make


@pytest.fixture
def make_env() -> Iterator[Callable[..., MEDARoutingEnv]]:
    """``factory(config=None, **overrides)`` -> env; all envs are closed afterwards."""
    envs: List[MEDARoutingEnv] = []

    def factory(config: Any = None, **overrides: Any) -> MEDARoutingEnv:
        env = MEDARoutingEnv(config, **overrides)
        envs.append(env)
        return env

    yield factory
    for env in envs:
        env.close()
