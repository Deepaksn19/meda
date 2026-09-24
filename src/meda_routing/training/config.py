"""Training configuration (Sec. IV) loaded from YAML.

Defaults reproduce the paper's setup; values the paper leaves open come from
the authors' reference implementation (``melfar87/MEDA``: ``train.py``,
``my_net.py``) or from Stable-Baselines ``PPO2`` defaults, which the authors
used unchanged.

Where each default comes from is tagged next to it:

* ``[PAPER ...]`` -- the value is stated in the paper (section, figure, table).
* ``[REF-CODE]`` -- not in the paper; taken from the first author's public
  code ``melfar87/MEDA`` (incl. the Stable-Baselines PPO2 defaults, saved
  model and training log of that code).
* ``[ASSUMED]`` -- not fixed by the paper or the reference code; our choice.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..envs.meda_env import EnvConfig


@dataclass
class AgentConfig:
    #: Registered feature extractor (``meda_routing.agents.registry``).
    extractor: str = "cnn"  # [PAPER Table I] the CNN; other names = your own extractor (GNN)
    #: Table I: 3x3 convolutions with 64/128/128 filters and a 256-unit FC layer.
    extractor_kwargs: Dict[str, Any] = field(
        # [PAPER Table I] 3x3 conv 64/128/128 + FC 256; stride 1, SAME padding is [REF-CODE]
        default_factory=lambda: {"channels": [64, 128, 128], "hidden_dim": 256}
    )


@dataclass
class PPOConfig:
    """PPO hyperparameters in Stable-Baselines3 terms.

    The reference code uses ``PPO2(n_steps=64, nminibatches=16)`` with 8
    environments: ``8 * 64 = 512`` samples per update, split into 16
    minibatches of 32 (SB3 ``batch_size`` is the minibatch size).  The other
    values are the PPO2 defaults, including value-function clipping with the
    same range as the policy (PPO2 ``cliprange_vf=None``).

    ``vf_coef``: PPO2 used 0.5, but its value loss is ``0.5 * mean(...)`` while
    SB3's is a plain mean, so 0.25 in SB3 reproduces PPO2's effective weight.
    (PPO2 also took the element-wise maximum of the clipped and unclipped
    value losses; SB3 uses the clipped prediction only.)
    """

    n_envs: int = 8  # [PAPER Sec. V-B] "eight parallel environments"
    n_steps: int = 64  # [REF-CODE] PPO2(n_steps=64)
    batch_size: int = 32  # [REF-CODE] 8*64 samples / nminibatches=16
    n_epochs: int = 4  # [REF-CODE] PPO2 noptepochs default
    gamma: float = 0.99  # [REF-CODE] PPO2 default
    gae_lambda: float = 0.95  # [REF-CODE] PPO2 default
    ent_coef: float = 0.01  # [REF-CODE] PPO2 default
    vf_coef: float = 0.25  # [REF-CODE] PPO2's 0.5, translated to SB3 (see above)
    max_grad_norm: float = 0.5  # [REF-CODE] PPO2 default
    clip_range: float = 0.2  # [REF-CODE] PPO2 default
    clip_range_vf: Optional[float] = 0.2  # [REF-CODE] PPO2 clips the value like the policy
    #: ``"dummy"`` (single process) or ``"subproc"`` vectorized environments.
    vec_env: str = "dummy"  # [ASSUMED]
    #: ``auto`` = local GPU if any, else CPU (``devices.py``).
    device: str = "auto"  # [ASSUMED] (the paper trained on an RTX 6000 GPU, Sec. V-B)


@dataclass
class ScheduleConfig:
    """Epochs and the dynamic learning-rate scheduler of Sec. IV-B."""

    epochs: int = 25  # [PAPER Sec. V-B, Figs. 4/7] 10-40 epochs; 25 for 30x30 [ASSUMED]
    #: Environment steps per training epoch (``2**14`` in Sec. V-B).
    steps_per_epoch: int = 2**14  # [PAPER Sec. V-B] 2^14 steps
    #: ``eta_0``, ``eta_min`` and ``beta_eta`` (Sec. IV-B).
    lr0: float = 3.5e-4  # [PAPER Sec. IV-B] eta_0
    lr_min: float = 1.0e-6  # [PAPER Sec. IV-B] eta_min
    lr_decay: float = 0.7  # [PAPER Sec. IV-B] beta_eta
    #: The base rate is decayed only if the epoch's success rate exceeds this.
    success_threshold: float = 0.99  # [PAPER Sec. IV-B] decay if success > 99%
    #: Learning rate within an epoch: ``"constant"`` or ``"sqrt"``
    #: (``eta_i * sqrt(remaining fraction of the epoch)``, as in the reference
    #: ``LearningRateSchedule``; the paper only specifies the per-epoch base rate).
    intra_epoch: str = "sqrt"  # [REF-CODE] LearningRateSchedule
    #: Save a full model checkpoint ``checkpoints/epoch_XXX.zip`` every this
    #: many epochs and after the last one (0: only ``model.zip`` and
    #: ``best_model.zip``).
    checkpoint_every: int = 5  # [ASSUMED]


@dataclass
class EvalConfig:
    #: "tested ... for 500 random routing jobs" after every epoch (Sec. V-B).
    episodes: int = 500  # [PAPER Sec. V-B] 500 random routing jobs per epoch
    deterministic: bool = True  # [REF-CODE] greedy actions during evaluation
    n_envs: int = 8  # [ASSUMED] only affects speed
    seed: int = 10_000  # [ASSUMED] same 500 jobs every epoch


@dataclass
class TrainConfig:
    name: str = "meda"  # [ASSUMED]
    seed: int = 0  # [ASSUMED]
    #: Independent repetitions with different seeds (the paper uses 5).
    repeats: int = 1  # [ASSUMED] to save compute; the paper repeats 5 times [PAPER Sec. V-B]
    output_dir: str = "runs"  # [ASSUMED] <project>/runs (paths.py)
    #: Initialize from a trained model (transfer learning, Sec. IV-C): path to a
    #: ``model.zip`` or a run directory containing one.
    init_from: Optional[str] = None  # [PAPER Sec. IV-C] transfer learning when set
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tensorboard: bool = False  # [ASSUMED]
    #: Draw ``training_curves.png`` into the run folder after every epoch.
    save_plots: bool = True  # [ASSUMED]
    verbose: int = 1  # [ASSUMED]

    # ------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        data = copy.deepcopy(data)
        kwargs: Dict[str, Any] = {}
        sections = {
            "agent": AgentConfig,
            "ppo": PPOConfig,
            "schedule": ScheduleConfig,
            "eval": EvalConfig,
        }
        for key, value in data.items():
            if key == "env":
                kwargs[key] = EnvConfig.from_dict(value or {})
            elif key in sections:
                kwargs[key] = sections[key](**(value or {}))
            else:
                kwargs[key] = value
        unknown = set(kwargs) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown training config keys: {sorted(unknown)}")
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Union[str, Path], overrides: Optional[List[str]] = None) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = coerce_numbers(yaml.safe_load(fh) or {})
        for item in overrides or []:
            apply_override(data, item)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save_yaml(self, path: Union[str, Path]) -> None:
        data = _plain(self.to_dict())
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)


_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+")


def coerce_numbers(obj: Any) -> Any:
    """Turn strings such as ``"1e-3"`` into floats.

    YAML 1.1 (PyYAML) only reads scientific notation with a decimal point as
    a number (``1.0e-3``); ``1e-3`` would otherwise reach the code as a string.
    """
    if isinstance(obj, dict):
        return {k: coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_numbers(v) for v in obj]
    if isinstance(obj, str) and _NUMBER.fullmatch(obj.strip()):
        return float(obj)
    return obj


def _plain(obj: Any) -> Any:
    """Convert tuples to lists recursively so the YAML stays portable."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def apply_override(data: Dict[str, Any], item: str) -> None:
    """Apply a ``dotted.key=value`` override (value parsed as YAML)."""
    if "=" not in item:
        raise ValueError(f"override {item!r} must look like key.subkey=value")
    key, raw = item.split("=", 1)
    value = coerce_numbers(yaml.safe_load(raw))
    node = data
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot override {key!r}: {part!r} is not a mapping")
    node[parts[-1]] = value


def load_config(
    path: Optional[Union[str, Path]] = None, overrides: Optional[List[str]] = None
) -> TrainConfig:
    if path is None:
        data: Dict[str, Any] = {}
        for item in overrides or []:
            apply_override(data, item)
        return TrainConfig.from_dict(data)
    return TrainConfig.from_yaml(path, overrides)
