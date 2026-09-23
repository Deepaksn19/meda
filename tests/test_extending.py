"""The GNN example plugs into the unchanged training stack (docs/EXTENDING.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.gnn_extractor_example import GridGNNExtractor, grid_adjacency  # noqa: E402
from meda_routing.agents import get_extractor  # noqa: E402
from meda_routing.training.config import load_config  # noqa: E402
from meda_routing.training.trainer import Trainer  # noqa: E402


def test_grid_adjacency_is_row_normalized_8_neighbourhood():
    adj = grid_adjacency(3, 4, torch.device("cpu")).to_dense()
    assert torch.allclose(adj.sum(1), torch.ones(12))
    assert (adj[5] > 0).sum() == 8  # interior node
    assert (adj[0] > 0).sum() == 3  # corner node


def test_gnn_extractor_is_size_independent():
    gnn = GridGNNExtractor(gym.spaces.Box(0, 1, (3, 30, 30), np.float32), hidden=8, layers=2)
    assert get_extractor("gnn_example") is GridGNNExtractor
    assert gnn(torch.rand(2, 3, 30, 30)).shape == (2, 24)
    assert gnn(torch.rand(1, 3, 17, 45)).shape == (1, 24)  # same weights, other chip


def test_gnn_trains_with_the_unchanged_trainer(tmp_path):
    cfg = load_config(
        None,
        [
            "env.width=8",
            "env.height=8",
            "env.obs_size=null",
            "env.jobs.droplet_sizes=[[2,2]]",
            "agent.extractor=gnn_example",
            "agent.extractor_kwargs={hidden: 8, layers: 1}",
            "ppo.n_envs=2",
            "ppo.n_steps=32",
            "ppo.device=cpu",
            "schedule.epochs=1",
            "schedule.steps_per_epoch=64",
            "eval.episodes=2",
            "eval.n_envs=2",
            "verbose=0",
        ],
    )
    history = Trainer(cfg, tmp_path / "gnn").run()
    assert len(history) == 1 and (tmp_path / "gnn" / "model.zip").exists()
