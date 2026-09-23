"""CNN agent of Table I.

=====  ===============  ==========  ======  ===========
Layer  Type             Activation  Size    Kernel/pad
=====  ===============  ==========  ======  ===========
L1     Convolution      ReLU        64      3x3 / 1
L2     Convolution      ReLU        128     3x3 / 1
L3     Convolution      ReLU        128     3x3 / 1
L4     Fully connected  ReLU        256     -
L5     Output           -           8       -
=====  ===============  ==========  ======  ===========

Table I labels the ``3`` column "Stride", but the authors' reference
implementation (``my_net.py``) uses ``filter_size=3, stride=1, pad='SAME'``
for every convolution, i.e. the spatial size is preserved.  The output layer
(8 action logits) and the value head are the linear heads of the SB3
actor-critic policy, which share this feature extractor like the PPO2
``ActorCriticPolicy`` used by the authors.  The reference code's smaller
variant (32/64/64 filters, FC 128) is available through the ``channels`` and
``hidden_dim`` arguments.
"""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MedaCNN(BaseFeaturesExtractor):
    """Convolutional feature extractor for ``(3, H, W)`` MEDA observations."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        channels: Sequence[int] = (64, 128, 128),
        hidden_dim: int = 256,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(observation_space, features_dim=hidden_dim)
        in_ch = observation_space.shape[0]
        layers = []
        for out_ch in channels:
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding), nn.ReLU()]
            in_ch = out_ch
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, hidden_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
