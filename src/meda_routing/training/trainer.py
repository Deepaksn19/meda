"""PPO training of routing agents (Sec. IV-B, Algorithm 2).

Every epoch the agent collects ``steps_per_epoch`` environment steps with
``n_envs`` parallel actors, running PPO updates on the way; it is then
evaluated on 500 random routing jobs, and the dynamic learning-rate scheduler
is updated from the epoch's success rate.  Metrics are appended to
``progress.csv`` and the model is checkpointed after every epoch.

Run directory layout::

    <output_dir>/<name>/seed_<s>/config.yaml
                                 progress.csv
                                 model.zip        # after the last epoch
                                 best_model.zip   # best (success rate, cycles)
                                 summary.json
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from ..agents.registry import policy_kwargs_for
from ..envs.meda_env import EnvConfig, MEDARoutingEnv
from .config import TrainConfig
from .evaluation import evaluate_model
from .lr_schedule import DynamicLearningRate

#: Not stored in saved models: the learning-rate schedule object is only
#: needed during training, and pickling it would tie every checkpoint to the
#: exact definition of :class:`DynamicLearningRate`.  Loaded models fall back
#: to SB3's default (constant) learning rate.
SAVE_EXCLUDE = ["learning_rate", "lr_schedule"]

PROGRESS_COLUMNS = [
    "epoch",
    "timesteps",
    "learning_rate",
    "mean_score",
    "std_score",
    "success_rate",
    "mean_cycles",
    "mean_cycles_success",
    "epoch_seconds",
]


def make_env_fn_kwargs(env_config: EnvConfig) -> Dict[str, Any]:
    return {"config": env_config}


def make_vec(env_config: EnvConfig, n_envs: int, seed: int, kind: str = "dummy") -> VecEnv:
    vec_cls = SubprocVecEnv if kind == "subproc" else DummyVecEnv
    return make_vec_env(
        MEDARoutingEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=make_env_fn_kwargs(env_config),
        vec_env_cls=vec_cls,
    )


def _seed_number(path: Path) -> int:
    """``seed_10`` sorts after ``seed_2``."""
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 1 << 62


def json_safe(value: Any) -> Any:
    """``value`` with NaN and infinite floats replaced by ``None`` (strict JSON)."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def resolve_model_path(path: Union[str, Path]) -> Path:
    """Accept a ``.zip`` file or a run directory containing ``model.zip``."""
    p = Path(path)
    if p.is_dir():
        for candidate in ("model.zip", "best_model.zip"):
            if (p / candidate).exists():
                return p / candidate
        seeds = sorted(p.glob("seed_*/model.zip"), key=lambda m: _seed_number(m.parent))
        if seeds:
            return seeds[0]
        raise FileNotFoundError(f"no model.zip found in {p}")
    if p.suffix != ".zip" and p.with_suffix(".zip").exists():
        return p.with_suffix(".zip")
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def build_model(config: TrainConfig, env: VecEnv, lr: DynamicLearningRate, seed: int, log_dir: Optional[Path]) -> PPO:
    ppo = config.ppo
    return PPO(
        policy=ActorCriticPolicy,
        env=env,
        learning_rate=lr,
        n_steps=ppo.n_steps,
        batch_size=ppo.batch_size,
        n_epochs=ppo.n_epochs,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        clip_range=ppo.clip_range,
        clip_range_vf=ppo.clip_range_vf,
        ent_coef=ppo.ent_coef,
        vf_coef=ppo.vf_coef,
        max_grad_norm=ppo.max_grad_norm,
        policy_kwargs=policy_kwargs_for(config.agent.extractor, config.agent.extractor_kwargs),
        tensorboard_log=str(log_dir) if log_dir is not None else None,
        seed=seed,
        device=ppo.device,
        verbose=0,
    )


def load_pretrained_weights(model: PPO, path: Union[str, Path]) -> Path:
    """Initialize ``model``'s policy from a trained model (transfer learning).

    Only the network weights are transferred; the new stage keeps its own
    hyperparameters and a fresh optimizer.  Observation shapes must match,
    which is why transfer learning uses the unified 30x30 observation.
    """
    src = resolve_model_path(path)
    pretrained = PPO.load(src, device=model.device)
    if pretrained.observation_space.shape != model.observation_space.shape:
        raise ValueError(
            f"cannot transfer {src}: observation shape {pretrained.observation_space.shape} "
            f"!= {model.observation_space.shape}; use a unified obs_size"
        )
    model.policy.load_state_dict(pretrained.policy.state_dict())
    return src


def _share_chip_snapshot(train_env: VecEnv, eval_env: VecEnv) -> None:
    """Give every evaluation env a copy of the (first) training env's chip."""
    if not isinstance(train_env, DummyVecEnv) or not isinstance(eval_env, DummyVecEnv):
        raise ValueError("persistent_chip evaluation needs dummy vectorized envs")
    chip = train_env.envs[0].unwrapped.chip
    for env in eval_env.envs:
        env.unwrapped.load_chip(chip.copy())


class Trainer:
    """Trains one agent (one seed) for a :class:`TrainConfig`."""

    def __init__(self, config: TrainConfig, run_dir: Union[str, Path], seed: Optional[int] = None) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.seed = config.seed if seed is None else int(seed)
        self.history: List[Dict[str, Any]] = []

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg, flush=True)

    def run(self) -> pd.DataFrame:
        cfg = self.config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        replace(cfg, seed=self.seed).save_yaml(self.run_dir / "config.yaml")

        sched = cfg.schedule
        lr = DynamicLearningRate(
            sched.lr0, sched.lr_min, sched.lr_decay, sched.success_threshold, sched.intra_epoch
        )
        train_env = make_vec(cfg.env, cfg.ppo.n_envs, self.seed, cfg.ppo.vec_env)
        eval_env = make_vec(cfg.env, cfg.eval.n_envs, cfg.eval.seed, "dummy")
        log_dir = self.run_dir / "tb" if cfg.tensorboard else None
        model = build_model(cfg, train_env, lr, self.seed, log_dir)
        if cfg.init_from:
            src = load_pretrained_weights(model, cfg.init_from)
            self._log(f"[{cfg.name}] initialized from {src}")

        n_params = sum(p.numel() for p in model.policy.parameters())
        self._log(
            f"[{cfg.name}] seed={self.seed} chip={cfg.env.width}x{cfg.env.height} "
            f"obs={model.observation_space.shape} faults={cfg.env.fault_fraction:.0%} "
            f"params={n_params:,} device={model.device}"
        )
        progress_csv = self.run_dir / "progress.csv"
        best_key = (-1.0, -np.inf)
        try:
            for epoch in range(1, sched.epochs + 1):
                t0 = time.time()
                epoch_lr = lr.base_rate
                lr.start_epoch(
                    model.num_timesteps, sched.steps_per_epoch, cfg.ppo.n_envs * cfg.ppo.n_steps
                )
                # Algorithm 2, line 2 (resample <- True): every epoch starts
                # with fresh routing jobs, as PPO2.learn() did; SB3 resets the
                # environments when it has no last observation.
                model._last_obs = None
                model.learn(
                    total_timesteps=sched.steps_per_epoch,
                    reset_num_timesteps=False,
                    tb_log_name="ppo",
                    progress_bar=False,
                )
                if cfg.env.persistent_chip:
                    # online mode: evaluate on a snapshot of the chip being adapted to
                    _share_chip_snapshot(train_env, eval_env)
                eval_env.seed(cfg.eval.seed)  # same 500 jobs every epoch
                metrics = evaluate_model(model, eval_env, cfg.eval.episodes, cfg.eval.deterministic)
                decayed = lr.end_epoch(metrics["success_rate"])
                row = {
                    "epoch": epoch,
                    "timesteps": int(model.num_timesteps),
                    "learning_rate": epoch_lr,
                    **{k: metrics[k] for k in PROGRESS_COLUMNS if k in metrics},
                    "epoch_seconds": time.time() - t0,
                }
                self.history.append(row)
                pd.DataFrame(self.history, columns=PROGRESS_COLUMNS).to_csv(progress_csv, index=False)
                model.save(self.run_dir / "model.zip", exclude=SAVE_EXCLUDE)
                key = (metrics["success_rate"], -metrics["mean_cycles"])
                if key > best_key:
                    best_key = key
                    model.save(self.run_dir / "best_model.zip", exclude=SAVE_EXCLUDE)
                self._log(
                    f"[{cfg.name}] epoch {epoch:3d}/{sched.epochs}  "
                    f"success {metrics['success_rate']:6.1%}  score {metrics['mean_score']:8.2f}  "
                    f"cycles {metrics['mean_cycles']:6.2f}  lr {epoch_lr:.2e}"
                    f"{' -> decay' if decayed else ''}  ({row['epoch_seconds']:.0f}s)"
                )
        finally:
            train_env.close()
            eval_env.close()

        history = pd.DataFrame(self.history, columns=PROGRESS_COLUMNS)
        summary = {
            "name": cfg.name,
            "seed": self.seed,
            "epochs": len(self.history),
            "final": self.history[-1] if self.history else None,
            "best_success_rate": float(history["success_rate"].max()) if len(history) else None,
        }
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as fh:
            # e.g. mean_cycles_success is NaN while no evaluation job succeeds
            json.dump(json_safe(summary), fh, indent=2, allow_nan=False)
        return history


def train(config: TrainConfig, output_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    """Train ``config.repeats`` agents with seeds ``seed, seed+1, ...``."""
    root = Path(output_dir or config.output_dir) / config.name
    root.mkdir(parents=True, exist_ok=True)
    config.save_yaml(root / "config.yaml")
    run_dirs = []
    for r in range(config.repeats):
        seed = config.seed + r
        run_dir = root / f"seed_{seed}"
        Trainer(config, run_dir, seed).run()
        run_dirs.append(run_dir)
    return run_dirs
