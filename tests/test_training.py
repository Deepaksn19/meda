"""Training stack: configs, dynamic LR scheduler, CNN, trainer and curricula."""

from __future__ import annotations

import glob
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch

from meda_routing.agents import MedaCNN, get_extractor, policy_kwargs_for, register_extractor
from meda_routing.training.config import TrainConfig, apply_override, load_config
from meda_routing.training.curriculum import deep_merge, load_curriculum, run_curriculum
from meda_routing.training.lr_schedule import DynamicLearningRate
from meda_routing.training.trainer import PROGRESS_COLUMNS, Trainer, resolve_model_path

REPO = Path(__file__).resolve().parents[1]

TINY = [
    "env.width=10",
    "env.height=10",
    "env.obs_size=[10,10]",
    "env.jobs.droplet_sizes=[[2,2],[3,3]]",
    "agent.extractor_kwargs={channels: [4, 8, 8], hidden_dim: 16}",
    "ppo.n_envs=2",
    "ppo.n_steps=32",
    "ppo.batch_size=32",
    "ppo.device=cpu",
    "schedule.epochs=2",
    "schedule.steps_per_epoch=128",
    "eval.episodes=6",
    "eval.n_envs=2",
    "verbose=0",
]


# ------------------------------------------------------------------ configs
def test_defaults_match_paper():
    cfg = TrainConfig()
    assert cfg.agent.extractor_kwargs == {"channels": [64, 128, 128], "hidden_dim": 256}
    assert (cfg.schedule.lr0, cfg.schedule.lr_min, cfg.schedule.lr_decay) == (3.5e-4, 1e-6, 0.7)
    assert cfg.schedule.steps_per_epoch == 2**14 and cfg.schedule.success_threshold == 0.99
    assert cfg.eval.episodes == 500 and cfg.ppo.n_envs == 8
    # PPO2(n_steps=64, nminibatches=16) with 8 envs -> minibatch of 32
    assert cfg.ppo.n_envs * cfg.ppo.n_steps // cfg.ppo.batch_size == 16
    assert cfg.env.obs_size == (30, 30) and cfg.env.adaptive_step


@pytest.mark.parametrize("path", sorted(glob.glob(str(REPO / "configs/training/*.yaml"))))
def test_training_configs_load(path):
    cfg = load_config(path)
    assert cfg.env.width >= 30 and cfg.agent.extractor == "cnn"


@pytest.mark.parametrize("path", sorted(glob.glob(str(REPO / "configs/curricula/*.yaml"))))
def test_curricula_resolve(path):
    spec = load_curriculum(path)
    names = set()
    for stage in spec["stages"]:
        stage = dict(stage)
        name = stage.pop("name")
        parent = stage.pop("init_from", None)
        assert parent is None or parent in names, f"{name} transfers from a later stage"
        TrainConfig.from_dict({**deep_merge(spec["base"], stage), "name": name})
        names.add(name)


def test_paper_yaml_equals_dataclass_defaults():
    """The documented paper YAML must not silently diverge from the code defaults."""
    yaml_cfg = load_config(REPO / "configs/training/paper_30x30_healthy.yaml").to_dict()
    default = TrainConfig(name=yaml_cfg["name"]).to_dict()
    for section in ("env", "agent", "ppo", "schedule", "eval"):
        a, b = yaml_cfg[section], default[section]
        if section == "env":
            a = {**a, "obs_size": list(a["obs_size"])}
            b = {**b, "obs_size": list(b["obs_size"])}
            for d in (a, b):
                d["jobs"]["droplet_sizes"] = [list(s) for s in d["jobs"]["droplet_sizes"]]
                d["degradation"]["tau_range"] = list(d["degradation"]["tau_range"])
                d["degradation"]["c_range"] = list(d["degradation"]["c_range"])
        assert a == b, section


def test_overrides_and_unknown_keys():
    data = {}
    apply_override(data, "env.width=64")
    apply_override(data, "schedule.intra_epoch=constant")
    cfg = TrainConfig.from_dict(data)
    assert cfg.env.width == 64 and cfg.schedule.intra_epoch == "constant"
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"not_a_key": 1})
    with pytest.raises(ValueError):
        apply_override({}, "no_equals_sign")


def test_config_yaml_roundtrip(tmp_path):
    cfg = load_config(None, TINY)
    cfg.save_yaml(tmp_path / "c.yaml")
    again = load_config(tmp_path / "c.yaml")
    assert again.to_dict() == TrainConfig.from_dict(cfg.to_dict()).to_dict()


# ------------------------------------------------------------ LR scheduler
def test_dynamic_lr_paper_rule():
    lr = DynamicLearningRate(3.5e-4, 1e-6, 0.7, 0.99, "constant")
    assert lr(1.0) == pytest.approx(3.5e-4)
    assert not lr.end_epoch(0.99)  # strictly greater than 0.99
    assert lr.end_epoch(0.995)
    assert lr.base_rate == pytest.approx(0.7 * 3.5e-4)
    for _ in range(100):
        lr.end_epoch(1.0)
    assert lr.base_rate == pytest.approx(1e-6)  # floor eta_min


def test_dynamic_lr_sqrt_within_epoch():
    lr = DynamicLearningRate(intra_epoch="sqrt")
    steps = 1000
    for epoch in range(3):
        start = epoch * steps
        lr.start_epoch(start, steps)
        total = start + steps  # SB3 with reset_num_timesteps=False
        for done in (0, 250, 500, 999):
            progress_remaining = 1.0 - (start + done) / total
            assert lr(progress_remaining) == pytest.approx(3.5e-4 * math.sqrt(1 - done / steps))


def test_dynamic_lr_matches_ppo2_update_fractions():
    """PPO2: lr = base * sqrt(1 - (update - 1) / n_updates) for update = 1..n_updates."""
    lr = DynamicLearningRate(intra_epoch="sqrt")
    epoch_steps, rollout = 2**14, 8 * 64
    n_updates = epoch_steps // rollout
    for epoch in range(2):
        start = epoch * epoch_steps
        lr.start_epoch(start, epoch_steps, rollout)
        total = start + epoch_steps
        for update in range(1, n_updates + 1):
            # SB3 evaluates the schedule after collecting the update's rollout
            progress_remaining = 1.0 - (start + update * rollout) / total
            expected = 3.5e-4 * math.sqrt(1.0 - (update - 1) / n_updates)
            assert lr(progress_remaining) == pytest.approx(expected)


# -------------------------------------------------------------------- CNN
def test_cnn_matches_table_one_and_reference_weights():
    """Shapes of the authors' saved model 0825a: c1 (3,3,3,64) ... fc1 (115200,256)."""
    net = MedaCNN(gym.spaces.Box(0, 1, (3, 30, 30), np.float32))
    convs = [m for m in net.cnn if isinstance(m, torch.nn.Conv2d)]
    assert [tuple(c.weight.shape) for c in convs] == [(64, 3, 3, 3), (128, 64, 3, 3), (128, 128, 3, 3)]
    assert all(c.stride == (1, 1) and c.padding == (1, 1) for c in convs)
    fc = net.linear[0]
    assert tuple(fc.weight.shape) == (256, 115200)
    assert net(torch.zeros(2, 3, 30, 30)).shape == (2, 256)


def test_registry():
    assert get_extractor("cnn") is MedaCNN
    kwargs = policy_kwargs_for("cnn", {"channels": [8], "hidden_dim": 4})
    assert kwargs["net_arch"] == {"pi": [], "vf": []} and kwargs["share_features_extractor"]
    with pytest.raises(KeyError):
        get_extractor("nope")
    with pytest.raises(ValueError):
        register_extractor("cnn")(type("Other", (MedaCNN,), {}))


# ---------------------------------------------------------------- trainer
def test_trainer_smoke(tmp_path):
    cfg = load_config(None, TINY + ["name=tiny"])
    history = Trainer(cfg, tmp_path / "run", seed=3).run()
    assert list(history.columns) == PROGRESS_COLUMNS and len(history) == 2
    assert history["timesteps"].tolist() == [128, 256]
    assert ((0 <= history["success_rate"]) & (history["success_rate"] <= 1)).all()
    for name in ("model.zip", "best_model.zip", "progress.csv", "config.yaml", "summary.json"):
        assert (tmp_path / "run" / name).exists()
    assert load_config(tmp_path / "run" / "config.yaml").seed == 3
    assert resolve_model_path(tmp_path / "run") == tmp_path / "run" / "model.zip"
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "run" / "progress.csv"), history, check_dtype=False)


def test_transfer_curriculum_copies_weights(tmp_path):
    base = tmp_path / "base.yaml"
    cfg = load_config(None, TINY)
    cfg.save_yaml(base)
    cur = tmp_path / "cur.yaml"
    cur.write_text(
        f"name: tl\nbase: {base}\ndefaults:\n  schedule: {{epochs: 1}}\nstages:\n"
        "  - name: a\n    env: {width: 10, height: 10}\n"
        "  - name: b\n    init_from: a\n    env: {width: 14, height: 14, fault_fraction: 0.1}\n"
    )
    done = run_curriculum(cur, output_dir=tmp_path / "runs")
    assert set(done) == {"a", "b"}
    stage_b = load_config(done["b"][0] / "config.yaml")
    assert stage_b.init_from == str(done["a"][0]) and stage_b.env.width == 14
    assert stage_b.env.obs_size == (10, 10)  # unified observation makes weights transferable


def test_transfer_rejects_mismatched_observation(tmp_path):
    cfg = load_config(None, TINY + ["schedule.epochs=1"])
    Trainer(cfg, tmp_path / "a").run()
    bad = load_config(None, TINY + ["env.obs_size=[12,12]", f"init_from={tmp_path / 'a'}"])
    with pytest.raises(ValueError, match="observation shape"):
        Trainer(bad, tmp_path / "b").run()
