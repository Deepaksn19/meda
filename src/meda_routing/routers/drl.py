"""Router backed by a trained DRL agent (Stable-Baselines3 PPO model).

The router rebuilds exactly the observation the agent was trained on (same
unified observation size, health scaling and routing-zone masking) from the
chip's health sensors, so a model trained with :mod:`meda_routing.training`
can drive bioassays and head-to-head comparisons against the baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import yaml
from stable_baselines3 import PPO

from ..core.actions import Action
from ..envs.observation import build_observation
from .base import Router, RoutingState


def _load_env_section(model_path: Path) -> dict:
    for candidate in (model_path.parent / "config.yaml", model_path.parent.parent / "config.yaml"):
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as fh:
                return (yaml.safe_load(fh) or {}).get("env", {}) or {}
    return {}


class DRLRouter(Router):
    name = "DRL"

    def __init__(
        self,
        model: PPO,
        obs_size: Optional[Tuple[int, int]] = (30, 30),
        adaptive_step: bool = True,
        fixed_step: int = 1,
        mark_collisions: bool = False,
        deterministic: bool = True,
    ) -> None:
        self.model = model
        self.obs_size = tuple(obs_size) if obs_size is not None else None
        self.adaptive_step = adaptive_step
        self.fixed_step = fixed_step
        self.mark_collisions = mark_collisions
        self.deterministic = deterministic
        expected = model.observation_space.shape
        if self.obs_size is not None and expected[1:] != (self.obs_size[1], self.obs_size[0]):
            raise ValueError(f"obs_size {self.obs_size} does not match model input {expected}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: str = "cpu",
        deterministic: bool = True,
    ) -> "DRLRouter":
        """Load ``model.zip`` (or a run directory) and its training config."""
        from ..training.trainer import resolve_model_path

        model_path = resolve_model_path(path)
        model = PPO.load(model_path, device=device)
        env = _load_env_section(model_path)
        obs_size = env.get("obs_size", (30, 30))
        if obs_size is None:
            obs_size = None
        return cls(
            model,
            obs_size=tuple(obs_size) if obs_size is not None else None,
            adaptive_step=env.get("adaptive_step", True),
            fixed_step=env.get("fixed_step", 1),
            mark_collisions=env.get("mark_collisions", False),
            deterministic=deterministic,
        )

    def observation(self, state: RoutingState) -> np.ndarray:
        if self.obs_size is None:
            expected = self.model.observation_space.shape
            if expected[1:] != (state.chip.height, state.chip.width):
                raise ValueError(
                    f"model expects native observations of shape {expected}, chip is "
                    f"{state.chip.width}x{state.chip.height}"
                )
        return build_observation(
            state.chip.health(),
            state.chip.health_levels,
            state.droplet,
            state.goal,
            state.hazard,
            self.obs_size,
            state.collision if self.mark_collisions else None,
        )

    def act(self, state: RoutingState) -> Action:
        obs = self.observation(state)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return Action(int(action))

    def clone(self) -> "DRLRouter":
        """A new router sharing the (read-only) model; routers are per job."""
        return DRLRouter(
            self.model,
            self.obs_size,
            self.adaptive_step,
            self.fixed_step,
            self.mark_collisions,
            self.deterministic,
        )
