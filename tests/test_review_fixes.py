"""Regression tests for issues found by the final adversarial review."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from meda_routing.core import MEDABiochip
from meda_routing.envs import MEDARoutingEnv
from meda_routing.training.config import TrainConfig, coerce_numbers, load_config
from meda_routing.training.curriculum import _seed_number, load_curriculum, run_curriculum


def _job_sequence(policy_seed: int, n: int = 15):
    env = MEDARoutingEnv({"width": 14, "height": 14})
    env.reset(seed=3)
    rng = np.random.default_rng(policy_seed)
    jobs = []
    for _ in range(n):
        done = False
        while not done:
            *_, terminated, truncated, _ = env.step(int(rng.integers(8)))
            done = terminated or truncated
        jobs.append((env.job, env.chip.tau.copy()))
        env.reset()
    return jobs


def test_job_sequence_does_not_depend_on_the_policy():
    """Evaluation sees the same jobs every epoch: sampling has its own random stream."""
    a, b = _job_sequence(1), _job_sequence(2)
    assert [j for j, _ in a] == [j for j, _ in b]
    assert all(np.array_equal(ta, tb) for (_, ta), (_, tb) in zip(a, b))


def test_ppo_value_loss_weight_matches_ppo2():
    assert TrainConfig().ppo.vf_coef == 0.25  # PPO2's 0.5 * (0.5 * mean) in SB3 terms


def test_scientific_notation_overrides_are_numbers():
    cfg = load_config(None, ["schedule.lr0=1e-3", "env.fault_fraction=1E-1"])
    assert cfg.schedule.lr0 == pytest.approx(1e-3) and cfg.env.fault_fraction == pytest.approx(0.1)
    assert coerce_numbers({"a": ["2e5", "x", "1e"]}) == {"a": [2e5, "x", "1e"]}


def test_seed_directories_sort_numerically():
    paths = [Path("seed_10"), Path("seed_2"), Path("seed_1")]
    assert [p.name for p in sorted(paths, key=_seed_number)] == ["seed_1", "seed_2", "seed_10"]


def test_curriculum_cli_overrides_beat_stage_settings(tmp_path):
    base = tmp_path / "base.yaml"
    load_config(
        None,
        [
            "env.width=8",
            "env.height=8",
            "env.obs_size=[8,8]",
            "env.jobs.droplet_sizes=[[2,2]]",
            "agent.extractor_kwargs={channels: [4], hidden_dim: 8}",
            "ppo.n_envs=2",
            "ppo.n_steps=32",
            "ppo.device=cpu",
            "schedule.steps_per_epoch=64",
            "eval.episodes=2",
            "eval.n_envs=2",
            "verbose=0",
        ],
    ).save_yaml(base)
    cur = tmp_path / "cur.yaml"
    cur.write_text(f"name: c\nbase: {base}\nstages:\n  - name: a\n    schedule: {{epochs: 5}}\n")
    spec = load_curriculum(cur, ["schedule.epochs=1"])
    assert spec["overrides"] == ["schedule.epochs=1"]
    done = run_curriculum(cur, output_dir=tmp_path / "runs", overrides=["schedule.epochs=1"])
    assert load_config(done["a"][0] / "config.yaml").schedule.epochs == 1


def test_hidden_defects_never_overlap_sensed_faults():
    chip = MEDABiochip(30, 30, rng=np.random.default_rng(0))
    chip.reset()
    chip.inject_faults(0.2)
    chip.inject_faults(0.05, hidden=True)
    assert not (chip.faults & chip.hidden_defects).any()
    assert chip.hidden_defects.mean() >= 0.05 and chip.faults.mean() >= 0.2


def test_persistent_chip_keeps_droplet_sizes_balanced():
    env = MEDARoutingEnv(
        {"width": 30, "height": 30, "persistent_chip": True, "chip_seed": 1, "fault_fraction": 0.1}
    )
    env.reset(seed=0)
    counts = Counter()
    for _ in range(450):
        env.reset()
        counts[env.job.droplet_size] += 1
    assert max(counts.values()) < 2 * min(counts.values())


def test_kmax_basis():
    chip_based = MEDARoutingEnv({"width": 30, "height": 30, "kmax_basis": "chip"})
    chip_based.reset(seed=0)
    assert chip_based.k_max == 60
    zone = MEDARoutingEnv({"width": 30, "height": 30})
    zone.reset(seed=0)
    assert zone.k_max == zone.hazard.width + zone.hazard.height
    with pytest.raises(ValueError):
        MEDARoutingEnv({"kmax_basis": "nope"}).reset(seed=0)


def test_online_mode_evaluates_on_a_snapshot_of_the_training_chip(tmp_path, monkeypatch):
    from meda_routing.training import trainer as trainer_module

    seen = {}
    share = trainer_module._share_chip_snapshot
    evaluate = trainer_module.evaluate_model

    def share_and_record(train_env, eval_env):
        share(train_env, eval_env)
        seen["train_chip"] = train_env.envs[0].unwrapped.chip
        seen["wear"] = seen["train_chip"].actuations.copy()

    def evaluate_and_check(model, eval_env, *args, **kwargs):
        eval_env.reset()
        for env in eval_env.envs:
            chip = env.unwrapped.chip
            assert chip is not seen["train_chip"]  # a copy: evaluation wear stays out of training
            assert np.array_equal(chip.actuations, seen["wear"])
            assert np.array_equal(chip.tau, seen["train_chip"].tau)
        metrics = evaluate(model, eval_env, *args, **kwargs)
        assert np.array_equal(seen["train_chip"].actuations, seen["wear"])
        seen["checked"] = seen.get("checked", 0) + 1
        return metrics

    monkeypatch.setattr(trainer_module, "_share_chip_snapshot", share_and_record)
    monkeypatch.setattr(trainer_module, "evaluate_model", evaluate_and_check)
    config = load_config(
        None,
        [
            "env.width=8",
            "env.height=8",
            "env.obs_size=[8,8]",
            "env.jobs.droplet_sizes=[[2,2]]",
            "env.persistent_chip=true",
            "env.chip_seed=3",
            "agent.extractor_kwargs={channels: [4], hidden_dim: 8}",
            "ppo.n_envs=2",
            "ppo.n_steps=32",
            "ppo.device=cpu",
            "schedule.epochs=2",
            "schedule.steps_per_epoch=64",
            "eval.episodes=2",
            "eval.n_envs=2",
            "verbose=0",
        ],
    )
    trainer_module.Trainer(config, tmp_path / "run").run()
    assert seen["checked"] == 2 and seen["wear"].sum() > 0


def test_env_only_commands_reject_other_overrides(capsys):
    from meda_routing.cli import main

    with pytest.raises(SystemExit, match="only takes env"):
        main(["compare", "--routers", "baseline", "--jobs", "1", "--set", "eval.episodes=4"])


def test_json_safe_maps_nan_and_inf_to_null():
    import json

    from meda_routing.training.trainer import json_safe

    value = {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": np.float64("nan"), "e": 2}}
    assert json_safe(value) == {"a": None, "b": [1.0, None], "c": {"d": None, "e": 2}}
    json.dumps(json_safe(value), allow_nan=False)


def _two_stage_curriculum(tmp_path):
    base = tmp_path / "base.yaml"
    load_config(
        None,
        [
            "env.width=8",
            "env.height=8",
            "env.obs_size=[8,8]",
            "env.jobs.droplet_sizes=[[2,2]]",
            "agent.extractor_kwargs={channels: [4], hidden_dim: 8}",
            "ppo.n_envs=2",
            "ppo.n_steps=32",
            "ppo.device=cpu",
            "schedule.epochs=1",
            "schedule.steps_per_epoch=64",
            "eval.episodes=2",
            "eval.n_envs=2",
            "verbose=0",
        ],
    ).save_yaml(base)
    cur = tmp_path / "cur.yaml"
    cur.write_text(
        f"name: c\nbase: {base}\nstages:\n  - name: a\n  - name: b\n    init_from: a\n"
    )
    return cur


def test_curriculum_only_names_the_untrained_parent(tmp_path):
    from meda_routing.training.curriculum import CurriculumError

    cur = _two_stage_curriculum(tmp_path)
    runs = tmp_path / "runs"
    (runs / "c" / "a" / "seed_0").mkdir(parents=True)  # a crashed run without model.zip
    with pytest.raises(CurriculumError, match="train a first"):
        run_curriculum(cur, output_dir=runs, only=["b"])
    assert not (runs / "c" / "b").exists()  # nothing written for the failed stage
    with pytest.raises(CurriculumError, match="unknown stage"):
        run_curriculum(cur, output_dir=runs, only=["nope"])
    done = run_curriculum(cur, output_dir=runs, only=["a"])
    assert (done["a"][0] / "model.zip").exists()
    done = run_curriculum(cur, output_dir=runs, only=["b"])  # now transfers from a
    assert load_config(done["b"][0] / "config.yaml").init_from == str(runs / "c" / "a" / "seed_0")


def test_curriculum_only_starts_each_seed_from_the_parent_with_that_seed(tmp_path, monkeypatch):
    from meda_routing.training import curriculum as curriculum_module

    calls = []

    class RecordingTrainer:
        def __init__(self, config, run_dir, seed):
            self.config, self.run_dir = config, run_dir

        def run(self):
            calls.append((self.run_dir.name, Path(self.config.init_from).name))

    monkeypatch.setattr(curriculum_module, "Trainer", RecordingTrainer)
    cur = _two_stage_curriculum(tmp_path)
    runs = tmp_path / "runs"
    for seed in (0, 1, 2, 10):
        (runs / "c" / "a" / f"seed_{seed}").mkdir(parents=True)
        (runs / "c" / "a" / f"seed_{seed}" / "model.zip").touch()
    run_curriculum(cur, output_dir=runs, only=["b"], overrides=["seed=2", "repeats=1"])
    run_curriculum(cur, output_dir=runs, only=["b"], overrides=["seed=10", "repeats=1"])
    assert calls == [("seed_2", "seed_2"), ("seed_10", "seed_10")]


def test_env_config_files_accept_scientific_notation(tmp_path):
    from meda_routing.cli import main

    config = tmp_path / "sci.yaml"
    config.write_text("env: {width: 10, height: 10, fault_fraction: 1e-1}\n")
    main(["compare", "--routers", "baseline", "--jobs", "1", "--config", str(config)])


def test_device_resolution(monkeypatch):
    from meda_routing.devices import describe_devices, resolve_device

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "mps"} or resolve_device("auto").startswith("cuda:")
    with pytest.warns(UserWarning):
        assert resolve_device("cuda:99") == "cpu"  # a missing GPU falls back to the CPU
    monkeypatch.setenv("MEDA_DEVICE", "cpu")
    assert resolve_device("cuda") == "cpu"  # the environment variable wins
    with pytest.raises(ValueError):
        monkeypatch.delenv("MEDA_DEVICE")
        resolve_device("tpu")
    assert "auto ->" in describe_devices()


def test_outputs_land_in_the_runs_folder_next_to_the_model(tmp_path, monkeypatch):
    from meda_routing import paths
    from meda_routing.cli import main

    runs = tmp_path / "all_runs"
    monkeypatch.setenv("MEDA_RUNS_DIR", str(runs))
    assert paths.resolve_output_dir("runs") == runs and paths.resolve_output_dir(None) == runs
    config = load_config(
        None,
        [
            "name=tiny",
            "env.width=8",
            "env.height=8",
            "env.obs_size=[8,8]",
            "env.jobs.droplet_sizes=[[2,2]]",
            "agent.extractor_kwargs={channels: [4], hidden_dim: 8}",
            "ppo.n_envs=2",
            "ppo.n_steps=32",
            "schedule.epochs=2",
            "schedule.steps_per_epoch=64",
            "schedule.checkpoint_every=1",
            "eval.episodes=2",
            "eval.n_envs=2",
            "verbose=0",
        ],
    )
    from meda_routing.training.trainer import train

    (run,) = train(config)
    assert run == runs / "tiny" / "seed_0"
    for name in ("model.zip", "best_model.zip", "progress.csv", "training_curves.png",
                 "checkpoints/epoch_001.zip", "checkpoints/epoch_002.zip"):
        assert (run / name).exists(), name
    main(["compare", "-m", str(run / "checkpoints" / "epoch_002.zip"), "--routers", "drl", "baseline",
          "--jobs", "2"])
    assert (run / "eval" / "compare_drl_baseline_seed0.png").exists()
