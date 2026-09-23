"""Minimal example of plugging a graph neural network into the training stack.

This is **not** a proposed method — it only demonstrates the extension point
used by ``agents.extractor`` in the training configs.  The MEDA observation
``(3, H, W)`` is read as a grid graph: one node per microelectrode (MC) with
features ``[health, droplet, goal, x/W, y/H]`` and edges to the 8 neighbouring
MCs.  A few mean-aggregation message-passing layers are followed by a readout
that pools over the droplet's nodes, the goal's nodes and the whole chip.

All weights are independent of the chip size, so the same network can be
trained on native-resolution observations of any chip (``obs_size: null``)
and transferred between chip sizes without resizing.

Usage, from the repository root::

    meda train --config configs/training/quick_cpu_30x30.yaml \\
        --import-module examples.gnn_extractor_example \\
        --set agent.extractor=gnn_example --set "agent.extractor_kwargs={}"
    meda evaluate --model runs/quick_cpu_30x30/seed_0 \\
        --import-module examples.gnn_extractor_example

``--import-module`` imports this file, which registers ``"gnn_example"``,
before the command runs; every command that loads the trained model needs
it too.  ``agent.extractor_kwargs`` is reset because the config's defaults
are the CNN's arguments.  From Python, import this module first (see
``tests/test_extending.py``).
"""

from __future__ import annotations

from typing import Dict, Tuple

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from meda_routing.agents import register_extractor


def grid_adjacency(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Row-normalized sparse adjacency of the 8-connected ``H x W`` grid."""
    idx = torch.arange(height * width, device=device).reshape(height, width)
    rows, cols = [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            src = idx[max(0, -dy) : height - max(0, dy), max(0, -dx) : width - max(0, dx)]
            dst = idx[max(0, dy) : height - max(0, -dy), max(0, dx) : width - max(0, -dx)]
            rows.append(src.reshape(-1))
            cols.append(dst.reshape(-1))
    r, c = torch.cat(rows), torch.cat(cols)
    deg = torch.bincount(r, minlength=height * width).float()
    vals = 1.0 / deg[r]
    size = (height * width, height * width)
    return torch.sparse_coo_tensor(torch.stack([r, c]), vals, size, check_invariants=False).coalesce()


class GridSAGELayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.self_lin = nn.Linear(dim_in, dim_out)
        self.neigh_lin = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        b, n, f = x.shape
        neigh = torch.sparse.mm(adj, x.permute(1, 0, 2).reshape(n, b * f))
        neigh = neigh.reshape(n, b, f).permute(1, 0, 2)
        return torch.relu(self.self_lin(x) + self.neigh_lin(neigh))


@register_extractor("gnn_example")
class GridGNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, hidden: int = 64, layers: int = 4) -> None:
        super().__init__(observation_space, features_dim=3 * hidden)
        self.layers = nn.ModuleList(
            [GridSAGELayer(5 if i == 0 else hidden, hidden) for i in range(layers)]
        )
        self._adj: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def _adjacency(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, str(device))
        if key not in self._adj:
            self._adj[key] = grid_adjacency(h, w, device)
        return self._adj[key]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b, _, h, w = obs.shape
        ys = torch.linspace(0, 1, h, device=obs.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(0, 1, w, device=obs.device).view(1, 1, 1, w).expand(b, 1, h, w)
        x = torch.cat([obs, xs, ys], dim=1).flatten(2).transpose(1, 2)  # (B, N, 5)
        adj = self._adjacency(h, w, obs.device)
        for layer in self.layers:
            x = layer(x, adj)
        droplet = obs[:, 1].flatten(1).unsqueeze(-1)  # (B, N, 1) droplet mask
        goal = obs[:, 2].flatten(1).unsqueeze(-1)
        pool = lambda m: (x * m).sum(1) / m.sum(1).clamp_min(1e-6)  # noqa: E731
        return torch.cat([pool(droplet), pool(goal), x.mean(1)], dim=1)
