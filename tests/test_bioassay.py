"""Tests for bioassay sequence graphs, execution and the Fig. 9 benchmark."""

from __future__ import annotations

import ast
import os
import urllib.request
from typing import Dict, List

import numpy as np
import pytest

from meda_routing.bioassay import (
    AgedChipFactory,
    Bioassay,
    BioassayExecutor,
    MicrofluidicOperation,
    MOState,
    available_bioassays,
    bioassay_from_reference,
    completion_cdf,
    covid_pcr,
    covid_rat,
    cycles_at_probability,
    droplet_from_reference,
    get_bioassay,
    run_trials,
    simple,
    summarize,
    trial_rngs,
)
from meda_routing.core.actions import DIRECTIONS, Action
from meda_routing.core.biochip import DegradationConfig, MEDABiochip
from meda_routing.core.geometry import Droplet, chip_rect
from meda_routing.core.jobs import RoutingJob, hazard_bounds
from meda_routing.routers.base import Router, RoutingState, run_job

#: The reference bioassay definitions, pinned to the commit the transcription
#: was checked against.  Set MEDA_REFERENCE_SGS to a local copy to run offline.
REFERENCE_SGS_URL = (
    "https://raw.githubusercontent.com/melfar87/MEDA/"
    "1667016da1abe7d8339e9b277e10dd4881b6886a/meda_sgs.py"
)

_ACTION_OF = {vec: act for act, vec in DIRECTIONS.items()}


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


class GreedyRouter(Router):
    """Moves diagonally towards the goal, cardinally once aligned."""

    name = "greedy"

    def act(self, state: RoutingState) -> Action:
        ux = _sign(state.goal.xa - state.droplet.xa)
        uy = _sign(state.goal.ya - state.droplet.ya)
        return _ACTION_OF.get((ux, uy), Action.N)


class SingleStepGreedyRouter(GreedyRouter):
    """Greedy directions with one MC per axis and cycle (no Algorithm 1)."""

    name = "greedy-single"
    adaptive_step = False
    fixed_step = 1


class StallRouter(GreedyRouter):
    """Pushes south (invalid at the zone boundary) for ``n_stall`` cycles, then greedy."""

    name = "stall"

    def __init__(self, n_stall: int) -> None:
        self.n_stall = n_stall
        self.seen_k: List[int] = []

    def act(self, state: RoutingState) -> Action:
        self.seen_k.append(state.k)
        if state.k < self.n_stall:
            return Action.S
        return super().act(state)


def healthy_chip(width: int = 60, height: int = 30) -> MEDABiochip:
    """All MCs fresh (D = 1): every movement succeeds deterministically."""
    return MEDABiochip(width, height, rng=np.random.default_rng(0))


def aged_chip(seed: int, assay: Bioassay) -> MEDABiochip:
    return AgedChipFactory()(assay, np.random.default_rng(seed))


# ---------------------------------------------------------------- library
def _parse_reference(path: str) -> Dict[str, list]:
    tree = ast.parse(open(path).read())
    graphs: Dict[str, list] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Bioassays":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                    graphs[stmt.targets[0].id] = ast.literal_eval(stmt.value)
    return graphs


@pytest.fixture(scope="module")
def reference_sgs(tmp_path_factory) -> str:
    """``$MEDA_REFERENCE_SGS``, or the pinned file downloaded (skipped when offline)."""
    local = os.environ.get("MEDA_REFERENCE_SGS")
    if local:
        return local
    path = tmp_path_factory.mktemp("reference") / "meda_sgs.py"
    try:
        with urllib.request.urlopen(REFERENCE_SGS_URL, timeout=30) as response:
            path.write_bytes(response.read())
    except OSError as err:  # URLError and timeouts are OSErrors
        pytest.skip(f"reference meda_sgs.py not available: {err}")
    return str(path)


def test_transcription_matches_reference_file(reference_sgs):
    graphs = _parse_reference(reference_sgs)
    for attr, factory in (("sg_CRAT", covid_rat), ("sg_CPCR", covid_pcr), ("sg_Simple", simple)):
        ours = factory()
        ref = bioassay_from_reference(ours.name, graphs[attr])
        assert ref == ours, attr
        # independent check of the coordinate convention: the reference makes
        # (xa, ya) 0-based and treats (xb, yb) as exclusive -> same MC set
        for entry, mo in zip(graphs[attr], ours.operations):
            assert (mo.name, mo.type) == (entry[0], entry[1])
            assert list(mo.pre) == entry[2] and list(mo.cond) == entry[3]
            for raw, dr in zip(entry[4] + entry[5], mo.starts + mo.goals):
                xa, ya, xb_excl, yb_excl = raw[0] - 1, raw[1] - 1, raw[2], raw[3]
                assert set(dr.cells()) == {
                    (x, y) for x in range(xa, xb_excl) for y in range(ya, yb_excl)
                }


def test_coordinate_conversion():
    assert droplet_from_reference([8, 1, 11, 4]) == Droplet(7, 0, 10, 3)
    n00 = covid_rat().operation("n00")
    assert n00.type == "Dis"
    assert n00.starts[0] == Droplet(7, -4, 10, -1)  # off-chip reservoir
    assert n00.goals[0] == Droplet(7, 0, 10, 3)


@pytest.mark.parametrize("name,n_mos", [("covid-rat", 28), ("covid-pcr", 34), ("simple", 3)])
def test_graph_structure(name, n_mos):
    assay = get_bioassay(name)
    assert len(assay) == n_mos
    assert (assay.width, assay.height) == (60, 30)
    names = set(assay.names)
    area = chip_rect(assay.width, assay.height)
    for mo in assay:
        assert set(mo.dependencies) <= names
        for dr in mo.starts + mo.goals:
            assert dr.size == (4, 4)
        for spec in mo.job_specs():
            assert area.contains(spec.goal)
            assert spec.dispense == (mo.type == "Dis")
            if not spec.dispense:
                assert area.contains(spec.start)
    # the dependency graph is acyclic (topological order exists)
    done: set = set()
    remaining = list(assay)
    while remaining:
        ready = [mo for mo in remaining if set(mo.dependencies) <= done]
        assert ready, "cyclic dependencies"
        done |= {mo.name for mo in ready}
        remaining = [mo for mo in remaining if mo.name not in done]


def test_type_counts_and_off_chip_dispense():
    rat = covid_rat()
    types = [mo.type for mo in rat]
    assert types.count("Dis") == 8 and types.count("Mix") == 6 and types.count("Spt") == 6
    assert types.count("Dsc") == 6 and types.count("Out") == 2
    pcr = covid_pcr()
    assert [mo.type for mo in pcr].count("Thm") == 20
    area = chip_rect(60, 30)
    for assay in (rat, pcr):
        for mo in assay:
            if mo.type == "Dis":
                assert not area.intersects(mo.starts[0])


def test_job_expansion_per_type():
    rat = covid_rat()
    mix = rat.operation("n08").job_specs()
    assert [(s.start, s.goal) for s in mix] == [
        (Droplet(7, 0, 10, 3), Droplet(13, 18, 16, 21)),
        (Droplet(47, 0, 50, 3), Droplet(13, 18, 16, 21)),
    ]
    spt = rat.operation("n14").job_specs()
    # deliberate fix of the reference copy-paste bug: second half -> goals[1]
    assert [s.goal for s in spt] == [Droplet(9, 8, 12, 11), Droplet(17, 8, 20, 11)]
    assert [s.key for s in spt] == ["n14.J0", "n14.J1"]
    dsc = rat.operation("n20").job_specs()
    assert len(dsc) == 1 and not dsc[0].dispense
    thm = covid_pcr().operation("n13").job_specs()
    assert len(thm) == 1 and thm[0].start == Droplet(27, 8, 30, 11)


def test_registry_and_validation():
    assert set(available_bioassays()) == {"covid-rat", "covid-pcr", "simple"}
    assert get_bioassay("COVID_PCR") == covid_pcr()
    assert get_bioassay("crat").name == "covid-rat"
    with pytest.raises(KeyError):
        get_bioassay("nope")
    dr = [(0, 0, 3, 3)]
    with pytest.raises(ValueError):
        MicrofluidicOperation("a", "Foo", starts=dr, goals=dr)
    with pytest.raises(ValueError):  # split needs two goals
        MicrofluidicOperation("a", "Spt", starts=dr * 2, goals=dr)
    with pytest.raises(ValueError):  # size change
        MicrofluidicOperation("a", "Out", starts=dr, goals=[(0, 0, 4, 3)])
    with pytest.raises(ValueError):  # unknown dependency
        Bioassay("x", 10, 10, (MicrofluidicOperation("a", "Out", pre=("b",), starts=dr, goals=dr),))


# -------------------------------------------------------------- execution
def _single(mo_type: str, starts, goals, width=30, height=30) -> Bioassay:
    mo = MicrofluidicOperation("m", mo_type, starts=starts, goals=goals)
    return Bioassay("t", width, height, (mo,))


def test_dispense_is_deterministic_and_does_not_actuate():
    chip = healthy_chip()
    ex = BioassayExecutor(covid_rat(), GreedyRouter, chip, np.random.default_rng(0))
    ex.step()  # first cycle: the five unconditional dispensers start
    drops = ex.active_droplets()
    assert drops["n00.J0"] == Droplet(7, -2, 10, 1)  # partly off-chip
    assert drops["n02.J0"] == Droplet(13, 28, 16, 31)
    assert ex.total_actuations == 0 and chip.actuations.sum() == 0
    ex.step()
    assert "n00.J0" not in ex.active_droplets()  # bottom: 4 MCs in 2 cycles
    assert ex.active_droplets()["n02.J0"] == Droplet(13, 26, 16, 29)
    ex.step()  # top: 5 MCs in 2 + 2 + 1
    rec = {r.key: r for r in ex.records}
    assert rec["n00.J0"].cycles == 2 and rec["n02.J0"].cycles == 3
    assert rec["n02.J0"].success and rec["n02.J0"].k_max is None


def test_dispense_moves_x_first_then_y():
    assay = _single("Dis", [(-4, 5, -1, 8)], [(1, 8, 4, 11)])
    ex = BioassayExecutor(
        assay, GreedyRouter, healthy_chip(), np.random.default_rng(0), dispense_step=2
    )
    path = []
    result = ex.run(callback=lambda e: path.append(e.active_droplets().get("m.J0")))
    assert result.success and result.cycles == 5
    assert path[:4] == [
        Droplet(-2, 5, 1, 8), Droplet(0, 5, 3, 8), Droplet(1, 5, 4, 8), Droplet(1, 7, 4, 10)
    ]
    assert result.total_actuations == 0


def test_simple_assay_on_healthy_chip_exact_cycles():
    chip = healthy_chip()
    result = BioassayExecutor(simple(), GreedyRouter, chip, np.random.default_rng(0)).run()
    # dispense: 2 cycles; mix: J0 3 NE + 6 N, J1 4 NW + 5 N (Algorithm 1 steps)
    assert result.success and result.failure is None
    assert result.cycles == 11
    rec = {r.key: r for r in result.job_records}
    assert rec["n08.J0"].cycles == 9 and rec["n08.J1"].cycles == 9
    assert rec["n08.J0"].start_cycle == 2
    assert result.mo_done == {"n00": 2, "n01": 2, "n08": 11}
    assert result.total_actuations == int(chip.actuations.sum()) > 0
    assert result.n_timeouts == 0


def test_router_state_and_factory_key():
    keys: List[str] = []
    routers: List[StallRouter] = []

    def factory(key: str) -> Router:
        keys.append(key)
        routers.append(StallRouter(0))
        return routers[-1]

    chip = healthy_chip()
    ex = BioassayExecutor(simple(), factory, chip, np.random.default_rng(0))
    result = ex.run()
    assert result.success
    assert keys == ["n08.J0", "n08.J1"]  # one fresh router per routed job
    for rec, router in zip(result.routed_records, routers):
        assert router.seen_k == list(range(rec.cycles))
        hz = hazard_bounds(rec.start, rec.goal, 60, 30, 3)
        assert rec.hazard == hz and rec.k_max == hz.width + hz.height


def test_routers_see_chip_state_from_start_of_cycle():
    seen: Dict[str, List[int]] = {}

    class Spy(GreedyRouter):
        def reset(self, job, chip):
            self.log = seen.setdefault(str(job.start), [])

        def act(self, state):
            self.log.append(int(state.chip.actuations.sum()))
            return super().act(state)

    BioassayExecutor(simple(), Spy, healthy_chip(), np.random.default_rng(0)).run()
    a, b = seen.values()
    assert a == b  # both jobs of the mix observed identical wear in every cycle


def test_actuation_union_counts_each_mc_once():
    # both jobs of a mix start on the same MCs and move to the same goal:
    # identical patterns, so every MC is actuated once per cycle
    start, goal = (5, 5, 8, 8), (7, 5, 10, 8)
    chip = healthy_chip()
    result = BioassayExecutor(
        _single("Mix", [start, start], [goal, goal]), GreedyRouter, chip, np.random.default_rng(0)
    ).run()
    assert result.success and result.cycles == 1
    assert result.total_actuations == 16 and chip.actuations.max() == 1
    # disjoint jobs: patterns add up
    chip = healthy_chip()
    result = BioassayExecutor(
        _single("Spt", [start, (15, 5, 18, 8)], [goal, (17, 5, 20, 8)]),
        GreedyRouter, chip, np.random.default_rng(0),
    ).run()
    assert result.cycles == 1 and result.total_actuations == 32 == chip.actuations.sum()


def test_zero_length_job_completes_immediately():
    dr = (3, 3, 6, 6)
    made: List[str] = []

    def factory(key: str) -> Router:
        made.append(key)
        return GreedyRouter()

    result = BioassayExecutor(_single("Mag", [dr], [dr]), factory, healthy_chip()).run()
    assert result.success and result.cycles == 0 and result.job_records[0].success
    assert made == []  # like run_job: no router is created (or synthesized) for it


def test_actuation_union_partial_overlap():
    # two split halves whose target footprints share 2 x 4 MCs: 16 + 16 - 8
    chip = healthy_chip(30, 30)
    assay = _single("Spt", [(5, 5, 8, 8), (7, 5, 10, 8)], [(7, 5, 10, 8), (9, 5, 12, 8)])
    result = BioassayExecutor(assay, GreedyRouter, chip, np.random.default_rng(0)).run()
    assert result.success and result.cycles == 1
    assert result.total_actuations == 24 == int(chip.actuations.sum())
    assert chip.actuations.max() == 1
    assert chip.actuations[9:11, 5:9].all()  # the shared MCs, actuated once


@pytest.mark.parametrize("router_cls", [GreedyRouter, SingleStepGreedyRouter])
def test_single_job_matches_run_job(router_cls):
    """One routed job evolves exactly as under ``run_job``: same physics, RNG use and wear."""
    start, goal = Droplet(27, 8, 30, 11), Droplet(13, 8, 16, 11)  # COVID-PCR n13
    assay = Bioassay("t", 60, 30, (MicrofluidicOperation("m", "Thm", starts=[start], goals=[goal]),))
    job = RoutingJob(start, goal, hazard_bounds(start, goal, 60, 30, 3))
    outcomes = set()
    for seed in range(6):
        chip = aged_chip(seed, assay)
        chip_ref = chip.copy()
        rec = BioassayExecutor(
            assay, router_cls, chip, np.random.default_rng(seed), on_timeout="fail"
        ).run().job_records[0]
        ref = run_job(router_cls(), job, chip_ref, np.random.default_rng(seed))
        assert (rec.success, rec.cycles, rec.invalid_actions) == (
            ref.success, ref.cycles, ref.invalid_actions
        )
        np.testing.assert_array_equal(chip.actuations, chip_ref.actuations)
        outcomes.add(rec.cycles)
    assert len(outcomes) > 1  # the aged chips make the movement stochastic


def test_collision_marks_persist_like_run_job():
    """The marks of the last invalid action persist through valid ones (env / run_job)."""

    class Recorder(Router):
        name = "recorder"

        def __init__(self) -> None:
            self.marks: List[tuple] = []

        def act(self, state: RoutingState) -> Action:
            self.marks.append(tuple(state.collision))
            if state.k == 0:
                return Action.S  # invalid: the droplet touches the zone's south edge
            return GreedyRouter.act(self, state)

    start, goal = Droplet(5, 0, 8, 3), Droplet(9, 6, 12, 9)
    assay = Bioassay("t", 30, 30, (MicrofluidicOperation("m", "Mag", starts=[start], goals=[goal]),))
    routers: List[Recorder] = []

    def factory() -> Router:
        routers.append(Recorder())
        return routers[-1]

    BioassayExecutor(assay, factory, healthy_chip(30, 30), np.random.default_rng(0)).run()
    ref = Recorder()
    run_job(ref, RoutingJob(start, goal, hazard_bounds(start, goal, 30, 30, 3)),
            healthy_chip(30, 30), np.random.default_rng(0))
    south = (False, True, False, False)
    assert ref.marks[0] == (False,) * 4 and ref.marks[1:] == [south] * (len(ref.marks) - 1)
    assert routers[0].marks == ref.marks and len(ref.marks) >= 3


def test_router_factory_contract():
    # a Router class is called without arguments, never with the job key
    class NeedsArg(GreedyRouter):
        def __init__(self, model) -> None:
            self.model = model

    with pytest.raises(TypeError):
        BioassayExecutor(simple(), NeedsArg, healthy_chip()).run()
    # the two jobs of the mix must not share one (stateful) router instance
    shared = GreedyRouter()
    with pytest.raises(ValueError, match="fresh instance"):
        BioassayExecutor(simple(), lambda: shared, healthy_chip()).run()
    # sequential jobs may reuse an instance once the previous job is over
    seq = Bioassay("seq", 30, 30, (
        MicrofluidicOperation("a", "Mag", starts=[(2, 2, 5, 5)], goals=[(8, 2, 11, 5)]),
        MicrofluidicOperation("b", "Mag", pre=("a",), starts=[(8, 2, 11, 5)], goals=[(8, 9, 11, 12)]),
    ))
    assert BioassayExecutor(seq, lambda: shared, healthy_chip(30, 30)).run().success


def test_on_timeout_skip_successor_starts_at_listed_location():
    """Reference behaviour: a skipped job's successor picks its droplet up where the graph says."""
    seq = Bioassay("seq", 30, 30, (
        MicrofluidicOperation("a", "Mag", starts=[(2, 0, 5, 3)], goals=[(12, 0, 15, 3)]),
        MicrofluidicOperation("b", "Mag", pre=("a",), starts=[(12, 0, 15, 3)], goals=[(12, 8, 15, 11)]),
    ))

    def factory(key: str) -> Router:
        return StallRouter(10**6) if key == "a.J0" else GreedyRouter()

    result = BioassayExecutor(seq, factory, healthy_chip(30, 30), on_timeout="skip").run()
    a, b = result.job_records
    assert a.timed_out and not a.success and a.cycles == a.k_max == 19 + 7
    # b starts at its listed location although droplet a never left (2, 0, 5, 3)
    assert b.start == Droplet(12, 0, 15, 3) and b.success and b.start_cycle == a.k_max
    assert result.success and result.cycles == a.k_max + b.cycles
    # the same graph under "continue" must physically finish job a first
    result = BioassayExecutor(seq, factory, healthy_chip(30, 30), on_timeout="continue",
                              max_cycles=200).run()
    assert result.failure == "max_cycles"


def test_executor_deterministic_under_seed():
    assay = covid_rat()

    def run(move_seed: int):
        chip = aged_chip(11, assay)
        res = BioassayExecutor(assay, GreedyRouter, chip, np.random.default_rng(move_seed)).run()
        return res, chip

    (r1, c1), (r2, c2), (r3, _) = run(5), run(5), run(6)
    assert r1.success and r1 == r2
    np.testing.assert_array_equal(c1.actuations, c2.actuations)
    assert [r.cycles for r in r1.job_records] != [r.cycles for r in r3.job_records]


# J0 of the simple assay: k_max = 16 + 25 = 41, J1: 17 + 25 = 42.
def test_on_timeout_continue():
    result = BioassayExecutor(
        simple(), lambda: StallRouter(45), healthy_chip(), np.random.default_rng(0),
        on_timeout="continue",
    ).run()
    assert result.success and result.cycles == 2 + 45 + 9
    routed = result.routed_records
    assert [r.k_max for r in routed] == [41, 42]
    assert all(r.timed_out and r.success for r in routed)
    assert result.n_timeouts == 2


def test_on_timeout_skip_reference_behaviour():
    result = BioassayExecutor(
        simple(), lambda: StallRouter(10**6), healthy_chip(), np.random.default_rng(0),
        on_timeout="skip",
    ).run()
    assert result.success and result.cycles == 2 + 42
    routed = result.routed_records
    assert [r.cycles for r in routed] == [41, 42]
    assert all(r.timed_out and not r.success for r in routed)


def test_on_timeout_fail_and_max_cycles_and_deadlock():
    result = BioassayExecutor(
        simple(), lambda: StallRouter(10**6), healthy_chip(), np.random.default_rng(0),
        on_timeout="fail",
    ).run()
    assert not result.success and result.failure == "timeout" and result.cycles == 2 + 41

    result = BioassayExecutor(simple(), GreedyRouter, healthy_chip(), max_cycles=5).run()
    assert not result.success and result.failure == "max_cycles" and result.cycles == 5
    # the simple assay needs exactly 11 cycles on a healthy chip
    assert BioassayExecutor(simple(), GreedyRouter, healthy_chip(), max_cycles=11).run().success
    assert not BioassayExecutor(simple(), GreedyRouter, healthy_chip(), max_cycles=10).run().success

    dr = [(0, 0, 3, 3)]
    cyclic = Bioassay("c", 10, 10, (
        MicrofluidicOperation("a", "Out", pre=("b",), starts=dr, goals=dr),
        MicrofluidicOperation("b", "Out", pre=("a",), starts=dr, goals=dr),
    ))
    result = BioassayExecutor(cyclic, GreedyRouter, healthy_chip(10, 10)).run()
    assert result.failure == "deadlock" and result.cycles == 0

    with pytest.raises(ValueError):
        BioassayExecutor(simple(), GreedyRouter, healthy_chip(), on_timeout="retry")
    with pytest.raises(ValueError):  # COVID-RAT does not fit a 30 x 30 chip
        BioassayExecutor(covid_rat(), GreedyRouter, healthy_chip(30, 30))


def test_state_machine_respects_pre_and_cond():
    ex = BioassayExecutor(covid_rat(), GreedyRouter, healthy_chip(), np.random.default_rng(0))
    ex.step()
    busy = {n for n, s in ex.mo_state.items() if s == MOState.BUSY}
    assert busy == {"n00", "n01", "n02", "n03", "n04"}  # n05-n07 wait for their cond
    result = ex.run()
    assert result.success
    for mo in covid_rat():
        for dep in mo.dependencies:
            assert result.mo_done[dep] <= result.mo_start[mo.name]


def test_covid_assays_complete_on_aged_chip():
    for assay in (covid_rat(), covid_pcr()):
        result = BioassayExecutor(
            assay, GreedyRouter, aged_chip(3, assay), np.random.default_rng(3)
        ).run()
        assert result.success, assay.name
        assert len(result.job_records) == 40
        assert 50 < result.cycles < 2000


# -------------------------------------------------------------- benchmark
def test_aged_chip_factory():
    assay = covid_rat()
    chip = AgedChipFactory()(assay, np.random.default_rng(0))
    assert chip.shape == (60, 30)
    assert chip.actuations.min() >= 0 and chip.actuations.max() <= 399
    assert chip.actuations.max() > 300
    assert 0.5 <= chip.tau.min() and chip.tau.max() <= 0.7
    assert 500 <= chip.c.min() and chip.c.max() <= 800
    faulty = AgedChipFactory(fault_fraction=0.1, hidden_defect_fraction=0.05)(
        assay, np.random.default_rng(0)
    )
    assert faulty.faults.mean() >= 0.1 and faulty.hidden_defects.mean() >= 0.05
    for loc in assay.on_chip_locations():
        sx, sy = loc.slices()
        assert not faulty.faults[sx, sy].any() and not faulty.hidden_defects[sx, sy].any()
    fresh = AgedChipFactory(max_initial_actuations=0)(assay, np.random.default_rng(0))
    assert fresh.actuations.sum() == 0
    # the fixed parameters the reference bioassay runs actually used
    ref = AgedChipFactory(
        degradation=DegradationConfig(tau_range=(0.7, 0.7), c_range=(200.0, 200.0))
    )(assay, np.random.default_rng(0))
    assert np.all(ref.tau == 0.7) and np.all(ref.c == 200.0)
    assert ref.actuations.max() <= 399 and ref.actuations.max() > 300


def test_completion_cdf_and_summary():
    k, p = completion_cdf([5, 7, 7, 9])
    assert list(k) == [4, 5, 6, 7, 8, 9, 10]  # min - 1 .. max + 1
    assert list(p) == [0, 0.25, 0.25, 0.75, 0.75, 1.0, 1.0]
    assert np.all(np.diff(p) >= 0)
    k, p = completion_cdf([5, np.inf, 7, np.inf])
    assert p[-1] == 0.5  # failures stay in the denominator
    k, p = completion_cdf([np.inf, np.inf])
    assert list(p) == [0.0]
    _, p = completion_cdf([1, 2, 3], k_grid=[0, 2, 100])
    assert list(p) == [0, 2 / 3, 1]

    data = list(range(1, 11))
    assert cycles_at_probability(data, 0.9) == 9
    assert cycles_at_probability(data, 0.95) == 10
    assert cycles_at_probability([1, 2, np.inf], 0.9) == np.inf
    s = summarize([10, 20, 30, np.inf])
    assert s.n_trials == 4 and s.n_success == 3 and s.success_rate == 0.75
    assert s.mean_cycles == 20 and s.median_cycles == 20 and s.k_at_p90 == np.inf
    assert s.as_dict()["n_success"] == 3


def test_run_trials_deterministic_and_progress():
    calls = []
    cycles, summary = run_trials(
        "simple", GreedyRouter, 6, seed=7, progress=lambda i, n, r: calls.append((i, n, r.success))
    )
    assert calls == [(i, 6, True) for i in range(1, 7)]
    assert cycles.shape == (6,) and np.all(np.isfinite(cycles))
    assert summary.success_rate == 1.0
    k, p = completion_cdf(cycles)
    assert p[-1] == 1.0 and np.all(np.diff(p) >= 0)

    again, _ = run_trials(simple(), GreedyRouter, 6, seed=7)
    np.testing.assert_array_equal(cycles, again)
    # trial i does not depend on the number of trials (common random numbers)
    prefix, _ = run_trials(simple(), GreedyRouter, 3, seed=7)
    np.testing.assert_array_equal(cycles[:3], prefix)
    other, _ = run_trials(covid_rat(), GreedyRouter, 3, seed=8)
    rat, _ = run_trials(covid_rat(), GreedyRouter, 3, seed=7)
    assert not np.array_equal(rat, other)

    c0 = AgedChipFactory()(simple(), trial_rngs(7, 3)[1][0])
    c1 = AgedChipFactory()(simple(), trial_rngs(7, 5)[1][0])
    c2 = AgedChipFactory()(simple(), trial_rngs(7, 1, start_index=1)[0][0])
    np.testing.assert_array_equal(c0.actuations, c1.actuations)
    np.testing.assert_array_equal(c0.actuations, c2.actuations)
    # chunks started at ``start_index`` reproduce a single run
    tail, _ = run_trials(simple(), GreedyRouter, 3, seed=7, start_index=3)
    np.testing.assert_array_equal(cycles[3:], tail)


def test_failed_trials_as_nan_and_probability_bounds():
    # NaN marks a failed trial exactly like inf (as in the Fig. 9 plot helper)
    k, p = completion_cdf([5, np.nan, 7, np.inf])
    assert (k[0], k[-1], p[-1]) == (4, 8, 0.5)
    assert cycles_at_probability([1, np.nan], 0.5) == 1
    assert cycles_at_probability([1, np.nan], 0.9) == np.inf
    s = summarize([10, np.nan])
    assert (s.n_success, s.mean_cycles, s.k_at_p90) == (1, 10, np.inf)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            cycles_at_probability([1, 2], bad)


def test_run_trials_failures_and_options():
    cycles, summary = run_trials("simple", GreedyRouter, 3, seed=0, max_cycles=3)
    assert np.all(np.isinf(cycles)) and summary.success_rate == 0.0
    assert np.isnan(summary.mean_cycles)
    # chip options configure the default factory and are never silently ignored
    for option in (
        {"fault_fraction": 0.1},
        {"max_initial_actuations": 0},
        {"fault_cluster": 3},
        {"degradation": DegradationConfig()},
    ):
        with pytest.raises(ValueError):
            run_trials("simple", GreedyRouter, 1, chip_factory=AgedChipFactory(), **option)
    cycles, _ = run_trials("simple", GreedyRouter, 2, seed=0, max_initial_actuations=0)
    assert list(cycles) == [11, 11]  # never actuated MCs: D = 1, deterministic movement
    cycles, _ = run_trials(
        "simple", GreedyRouter, 2, seed=0, fault_fraction=0.1, hidden_defect_fraction=0.02,
        on_timeout="continue",
    )
    assert cycles.shape == (2,)
