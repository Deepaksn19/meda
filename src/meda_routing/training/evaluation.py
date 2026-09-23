"""Per-epoch evaluation (Sec. V-B).

"The metrics are collected after each training epoch by testing the agent for
500 random routing jobs": mean score (episode return), success rate and the
average number of cycles.  Episodes are split evenly across the vectorized
environments so that short episodes are not over-represented.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv


def evaluate_model(
    model: BaseAlgorithm, vec_env: VecEnv, n_episodes: int = 500, deterministic: bool = True
) -> Dict[str, Any]:
    n_envs = vec_env.num_envs
    targets = np.array([(n_episodes + i) // n_envs for i in range(n_envs)], dtype=int)
    counts = np.zeros(n_envs, dtype=int)
    returns = np.zeros(n_envs)
    scores: List[float] = []
    cycles: List[int] = []
    successes: List[bool] = []
    obs = vec_env.reset()
    while (counts < targets).any():
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(actions)
        returns += rewards
        for i in range(n_envs):
            if dones[i]:
                if counts[i] < targets[i]:
                    scores.append(float(returns[i]))
                    cycles.append(int(infos[i]["num_cycles"]))
                    successes.append(bool(infos[i]["is_success"]))
                    counts[i] += 1
                returns[i] = 0.0
    scores_a = np.asarray(scores)
    cycles_a = np.asarray(cycles)
    success_a = np.asarray(successes)
    return {
        "episodes": int(len(scores)),
        "mean_score": float(scores_a.mean()),
        "std_score": float(scores_a.std()),
        "success_rate": float(success_a.mean()),
        "mean_cycles": float(cycles_a.mean()),
        "mean_cycles_success": float(cycles_a[success_a].mean()) if success_a.any() else float("nan"),
    }
