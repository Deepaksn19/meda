"""Gymnasium environment for MEDA droplet routing."""

import gymnasium as gym

from .meda_env import EnvConfig, MEDARoutingEnv
from .observation import build_observation, observation_shape
from .reward import RewardConfig, compute_reward

ENV_ID = "MEDA-Routing-v0"

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="meda_routing.envs.meda_env:MEDARoutingEnv")

__all__ = [
    "ENV_ID",
    "EnvConfig",
    "MEDARoutingEnv",
    "RewardConfig",
    "build_observation",
    "compute_reward",
    "observation_shape",
]
