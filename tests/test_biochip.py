"""Degradation model, health sensing and fault injection (Sec. III-A Eq. 1, V-A, V-B, VI-C).

* Eq. (1): ``D = tau ** (n / c)`` and the ``b``-bit reading
  ``H = floor(2**b * D)``; the paper also states ``H in {0, ..., 2**b - 1}``
  (Sec. III-C), so the fully healthy reading saturates at ``2**b - 1``.
* Sec. V-A: ``tau in [0.5, 0.7]``, ``c in [500, 800]`` sampled per MC.
* Sec. V-B: "a fixed percentage of fully degraded MCs are randomly placed in
  clusters of size 2 x 2" (visible to the sensors: ``H = 0``).
* Sec. VI-C: inherent defects that the health sensors cannot see.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from meda_routing.core.biochip import DegradationConfig, MEDABiochip
from meda_routing.core.geometry import Droplet, Rect


def exact_degradation_chip(values, bits: int = 2) -> MEDABiochip:
    """``len(values) x 1`` chip with ``D`` exactly equal to ``values`` (``n = c = 1``)."""
    chip = MEDABiochip(len(values), 1, DegradationConfig(health_bits=bits))
    chip.tau[:, 0] = values
    chip.c[:] = 1.0
    chip.actuations[:] = 1  # D = tau ** 1.0 == tau exactly
    return chip


def covered_by_full_blocks(mask: np.ndarray, allowed: np.ndarray, size: int) -> bool:
    """Every set MC lies in a ``size x size`` window whose allowed MCs are all set."""
    width, height = mask.shape
    covered = np.zeros_like(mask)
    for x in range(width - size + 1):
        for y in range(height - size + 1):
            block = (slice(x, x + size), slice(y, y + size))
            if mask[block].any() and np.all(mask[block] | ~allowed[block]):
                covered[block] |= mask[block]
    return bool(np.array_equal(covered, mask))


# ================================================================== parameters
def test_default_config_matches_paper():
    cfg = DegradationConfig()
    assert cfg.tau_range == (0.5, 0.7) and cfg.c_range == (500.0, 800.0)  # Sec. V-A
    assert cfg.health_bits == 2  # reference n_bits = 2
    assert cfg.initial_actuations == "zero"  # healthy-chip training (Sec. V-B)
    chip = MEDABiochip(5, 4)
    assert chip.shape == (5, 4) and chip.health_levels == 4
    assert chip.tau.shape == chip.c.shape == chip.actuations.shape == (5, 4)
    assert not chip.faults.any() and not chip.hidden_defects.any()
    assert np.all(chip.degradation() == 1.0)  # n = 0


@pytest.mark.parametrize("width, height", [(0, 3), (3, 0), (-1, 5)])
def test_invalid_dimensions(width, height):
    with pytest.raises(ValueError):
        MEDABiochip(width, height)


def test_sample_parameters_ranges_and_uniformity(rng):
    chip = MEDABiochip(40, 40, rng=rng)
    chip.sample_parameters()
    tau, c = chip.tau, chip.c
    assert tau.min() >= 0.5 and tau.max() <= 0.7
    assert c.min() >= 500.0 and c.max() <= 800.0
    # per-MC (not per-chip) sampling, spread over the whole range
    assert len(np.unique(tau)) == tau.size and len(np.unique(c)) == c.size
    assert tau.min() < 0.51 and tau.max() > 0.69 and c.min() < 510 and c.max() > 790
    # U(a, b): four equal-width bins hold 1/4 of the 1600 MCs each (5 sigma)
    sigma = math.sqrt(1600 * 0.25 * 0.75)
    for values, (lo, hi) in [(tau, (0.5, 0.7)), (c, (500.0, 800.0))]:
        counts, _ = np.histogram(values, bins=4, range=(lo, hi))
        assert np.all(np.abs(counts - 400) < 5 * sigma), counts
    # custom ranges are honoured
    chip = MEDABiochip(10, 10, DegradationConfig(tau_range=(0.2, 0.3), c_range=(10, 20)), rng=rng)
    chip.sample_parameters()
    assert 0.2 <= chip.tau.min() and chip.tau.max() <= 0.3
    assert 10 <= chip.c.min() and chip.c.max() <= 20


def test_initial_actuation_modes(rng):
    zero = MEDABiochip(20, 20, DegradationConfig(initial_actuations="zero"), rng=rng)
    zero.actuations[:] = 5
    zero.sample_initial_actuations()
    assert not zero.actuations.any()

    uniform = MEDABiochip(
        20, 20, DegradationConfig(initial_actuations="uniform", max_initial_actuations=4), rng=rng
    )
    uniform.sample_initial_actuations()
    assert uniform.actuations.dtype == np.int64
    assert set(np.unique(uniform.actuations).tolist()) == {0, 1, 2, 3, 4}  # U{0, max}, inclusive
    other = uniform.actuations.copy()
    uniform.sample_initial_actuations()
    assert not np.array_equal(other, uniform.actuations)  # independent per episode

    bad = MEDABiochip(4, 4, DegradationConfig(initial_actuations="gaussian"), rng=rng)
    with pytest.raises(ValueError):
        bad.sample_initial_actuations()
    with pytest.raises(ValueError):
        bad.reset()


def test_reset_resamples_and_clears_faults(rng):
    cfg = DegradationConfig(initial_actuations="uniform", max_initial_actuations=50)
    chip = MEDABiochip(8, 6, cfg, rng=rng)
    chip.reset()
    tau, c = chip.tau.copy(), chip.c.copy()
    chip.set_faults([(0, 0), (3, 2)])
    chip.set_faults([(1, 1)], hidden=True)
    chip.actuations[:] = 10_000
    chip.reset(resample_parameters=False)
    assert np.array_equal(chip.tau, tau) and np.array_equal(chip.c, c)
    assert not chip.faults.any() and not chip.hidden_defects.any()
    assert chip.actuations.max() <= 50
    chip.reset()
    assert not np.array_equal(chip.tau, tau) and not np.array_equal(chip.c, c)


# ============================================================ Eq. (1): D and H
def test_degradation_formula(rng):
    chip = MEDABiochip(9, 7, rng=rng)
    chip.sample_parameters()
    chip.actuations = rng.integers(0, 3000, size=chip.shape).astype(np.int64)
    expected = np.exp(chip.actuations / chip.c * np.log(chip.tau))  # tau ** (n / c)
    np.testing.assert_allclose(chip.degradation(), expected, rtol=1e-12, atol=0.0)
    x, y = 4, 5
    n, c, tau = chip.actuations[x, y], chip.c[x, y], chip.tau[x, y]
    assert chip.degradation()[x, y] == pytest.approx(float(tau) ** (float(n) / float(c)), rel=1e-12)
    d = chip.degradation()
    assert np.all((d > 0.0) & (d <= 1.0))


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_health_is_floor_of_scaled_degradation_saturated(bits, rng):
    levels = 2**bits
    values = {0.0, 1.0, 0.999999, 1e-9}
    for k in range(levels + 1):
        values |= {k / levels, k / levels - 1e-9, k / levels + 1e-9}
    values |= set(rng.uniform(0, 1, size=50).tolist())
    values = sorted(v for v in values if 0.0 <= v <= 1.0)
    chip = exact_degradation_chip(values, bits)
    np.testing.assert_array_equal(chip.degradation()[:, 0], values)
    expected = [min(math.floor(levels * v), levels - 1) for v in values]
    health = chip.health()
    assert health[:, 0].tolist() == expected
    assert health.dtype.kind == "i"
    assert set(health[:, 0].tolist()) == set(range(levels))  # every b-bit reading occurs
    assert chip.health_levels == levels


def test_health_readings_for_two_bits():
    chip = exact_degradation_chip([1.0, 0.8, 0.75, 0.7499, 0.5, 0.26, 0.25, 0.2499, 0.0])
    assert chip.health()[:, 0].tolist() == [3, 3, 3, 2, 2, 1, 1, 0, 0]


def test_wear_lowers_health_at_predicted_actuation_counts():
    """``H`` drops below ``k`` once ``n > c * ln(k / 2**b) / ln(tau)``."""
    tau, c = 0.6, 650.0
    chip = MEDABiochip(1, 1)
    chip.tau[:] = tau
    chip.c[:] = c
    last_d = 1.0
    for k in (3, 2, 1):
        n_k = math.floor(c * math.log(k / 4) / math.log(tau)) + 1  # first n with H <= k - 1
        chip.actuations[:] = n_k - 1
        assert chip.health()[0, 0] >= k
        chip.actuations[:] = n_k
        assert chip.health()[0, 0] == k - 1
        d = float(chip.degradation()[0, 0])
        assert d < last_d
        last_d = d
    assert (367, 882, 1764) == tuple(
        math.floor(c * math.log(k / 4) / math.log(tau)) + 1 for k in (3, 2, 1)
    )


def test_visible_faults_and_hidden_defects():
    chip = MEDABiochip(6, 5)  # n = 0: D = 1 everywhere
    chip.set_faults([(1, 1), (2, 3)])
    chip.set_faults([(4, 0), (1, 1)], hidden=True)
    d, h, e = chip.degradation(), chip.health(), chip.effective_degradation()
    for x, y in [(1, 1), (2, 3)]:  # injected faults: fully degraded and sensed
        assert d[x, y] == 0.0 and h[x, y] == 0 and e[x, y] == 0.0
    # hidden defects: invisible to the sensors, but they do not actuate
    assert d[4, 0] == 1.0 and h[4, 0] == 3 and e[4, 0] == 0.0
    mask = chip.hidden_defects | chip.faults
    assert np.array_equal(e[~mask], d[~mask])
    assert np.count_nonzero(h == 0) == 2


# ============================================================== actuation, copy
def test_actuate_adds_pattern():
    chip = MEDABiochip(6, 5)
    pattern = np.zeros((6, 5), dtype=bool)
    pattern[1:3, 2:5] = True
    chip.actuate(pattern)
    chip.actuate(pattern)
    chip.actuate_rect(Rect(2, 2, 4, 2))
    expected = np.zeros((6, 5), dtype=np.int64)
    expected[1:3, 2:5] += 2
    expected[2:5, 2] += 1
    assert np.array_equal(chip.actuations, expected)


def test_copy_is_independent(rng):
    chip = MEDABiochip(
        8, 6, DegradationConfig(initial_actuations="uniform", max_initial_actuations=9), rng=rng
    )
    chip.reset()
    chip.set_faults([(0, 0)])
    chip.set_faults([(1, 0)], hidden=True)
    other = chip.copy()
    for name in ("tau", "c", "actuations", "faults", "hidden_defects"):
        assert np.array_equal(getattr(other, name), getattr(chip, name))
        assert getattr(other, name) is not getattr(chip, name)
    other.actuate_rect(Rect(0, 0, 7, 5))
    other.set_faults([(5, 5)])
    other.tau[:] = 1.0
    assert not chip.faults[5, 5] and chip.tau.max() <= 0.7
    assert not np.array_equal(other.actuations, chip.actuations)
    # "its random generator starts in the same state" but is not shared
    assert other.rng is not chip.rng
    assert other.rng.random() == chip.rng.random()


# ============================================================ fault injection
@pytest.mark.parametrize("fraction", [0.05, 0.1, 0.2, 0.37])
def test_inject_faults_reaches_fraction_in_2x2_clusters(fraction, rng):
    chip = MEDABiochip(30, 30, rng=rng)
    added = chip.inject_faults(fraction)
    n = int(chip.faults.sum())
    target = math.ceil(fraction * 900)
    assert added == n
    assert target <= n <= target + 3  # at most one partial 2x2 block beyond the target
    assert covered_by_full_blocks(chip.faults, np.ones_like(chip.faults), 2)
    assert not chip.hidden_defects.any()
    assert np.all(chip.health()[chip.faults] == 0)  # visible to the sensors
    # the target is already met: a second call adds nothing
    assert chip.inject_faults(fraction) == 0 and int(chip.faults.sum()) == n
    # a higher fraction only adds the difference
    more = chip.inject_faults(fraction + 0.05)
    assert int(chip.faults.sum()) == n + more >= math.ceil((fraction + 0.05) * 900)


@pytest.mark.parametrize("cluster", [1, 3])
def test_inject_faults_cluster_size(cluster, rng):
    chip = MEDABiochip(24, 24, rng=rng)
    chip.inject_faults(0.15, cluster=cluster)
    target = math.ceil(0.15 * 576)
    assert target <= chip.faults.sum() <= target + cluster**2 - 1
    assert covered_by_full_blocks(chip.faults, np.ones_like(chip.faults), cluster)


def test_inject_faults_respects_protect_and_region(rng):
    for trial in range(20):
        chip = MEDABiochip(20, 16, rng=rng)
        start = Droplet.at(int(rng.integers(0, 15)), int(rng.integers(0, 11)), 5, 5)
        goal = Droplet.at(int(rng.integers(0, 15)), int(rng.integers(0, 11)), 4, 5)
        region = Rect(2, 1, 17, 14) if trial % 2 else None
        chip.inject_faults(0.3, protect=(start, goal), region=region)
        allowed = np.zeros(chip.shape, dtype=bool)
        zone = region or Rect(0, 0, 19, 15)
        allowed[zone.slices()] = True
        for r in (start, goal):
            assert not chip.faults[r.slices()].any()
            allowed[r.slices()] = False
        assert not chip.faults[~allowed].any()
        assert chip.faults[zone.slices()].sum() >= math.ceil(0.3 * zone.area)
        # blocks are clipped by the protected MCs, never shifted onto them
        assert covered_by_full_blocks(chip.faults, allowed, 2)


def test_inject_faults_cannot_exceed_allowed_area(rng):
    chip = MEDABiochip(8, 8, rng=rng)
    protect = Rect(0, 0, 7, 3)  # half the chip
    added = chip.inject_faults(1.0, protect=[protect], max_attempts=3000)
    assert added == int(chip.faults.sum()) == 32  # every unprotected MC, nothing more
    assert not chip.faults[protect.slices()].any()


def test_inject_hidden_defects_only_touch_hidden_mask(rng):
    chip = MEDABiochip(16, 16, rng=rng)
    added = chip.inject_faults(0.1, hidden=True)
    assert added == int(chip.hidden_defects.sum()) >= math.ceil(0.1 * 256)
    assert not chip.faults.any()
    assert np.all(chip.health() == 3)  # the sensors see a healthy chip
    assert np.all(chip.effective_degradation()[chip.hidden_defects] == 0.0)


def test_inject_zero_fraction_is_noop(rng):
    chip = MEDABiochip(10, 10, rng=rng)
    state = rng.bit_generator.state
    assert chip.inject_faults(0.0) == 0 and chip.inject_faults(-1.0) == 0
    assert not chip.faults.any()
    assert rng.bit_generator.state == state  # no random numbers consumed
