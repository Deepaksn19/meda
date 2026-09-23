#!/usr/bin/env python3
"""Run the authors' trained agent inside this repository's environment.

The first author's repository (melfar87/MEDA) ships trained PPO2 models, e.g.
``policy/0825a_030x030_E100_NPS64_00.zip`` (the Table I CNN, trained on
healthy 30x30 chips with 4x4..6x6 droplets; git-LFS, ~119 MB).  Stable-Baselines
v2 stores the weights as plain numpy arrays, so they can be loaded into
PyTorch without TensorFlow.

If this environment reproduces the original one (movement model, adaptive
steps, action semantics, routing zones, job distribution), the authors'
agent should perform here as it did in its own training log: ~100% success
and ~10.5 cycles per job (0825a) with a score of ~111.

The script replicates the *original* observation encoding for this check:
channels (goal, droplet, health), x-major image axes, health rounded to the
nearest ``1/2**b`` (so a healthy MC reads 1.0), and persistent collision marks.

Usage::

    # download the model (git-LFS) first, e.g.
    #   curl -L -o 0825a.zip https://media.githubusercontent.com/media/melfar87/MEDA/master/policy/0825a_030x030_E100_NPS64_00.zip
    python scripts/evaluate_reference_model.py 0825a.zip --episodes 500
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile

import cv2
import numpy as np
import torch
from torch import nn

from meda_routing.core.actions import Action
from meda_routing.envs import MEDARoutingEnv
from meda_routing.training.config import load_config

#: Original ``Direction`` enum order: NN, NE, EE, SE, SS, SW, WW, NW.
REFERENCE_ACTIONS = [Action.N, Action.NE, Action.E, Action.SE, Action.S, Action.SW, Action.W, Action.NW]


class ReferenceCNN(nn.Module):
    """``myCnn`` + policy/value heads of the original ``MyCnnPolicy`` (NHWC semantics)."""

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        for name in ("c1", "c2", "c3"):
            w = params[f"model/{name}/w:0"]  # (kh, kw, in, out)
            conv = nn.Conv2d(w.shape[2], w.shape[3], 3, 1, 1)
            conv.weight.data = torch.from_numpy(w.transpose(3, 2, 0, 1).copy())
            conv.bias.data = torch.from_numpy(params[f"model/{name}/b:0"].reshape(-1).copy())
            self.convs.append(conv)
        self.fc = self._linear(params, "fc1")
        self.pi = self._linear(params, "pi")
        self.vf = self._linear(params, "vf")

    @staticmethod
    def _linear(params: dict, name: str) -> nn.Linear:
        w, b = params[f"model/{name}/w:0"], params[f"model/{name}/b:0"]
        layer = nn.Linear(w.shape[0], w.shape[1])
        layer.weight.data = torch.from_numpy(w.T.copy())
        layer.bias.data = torch.from_numpy(b.copy())
        return layer

    def forward(self, x: torch.Tensor):
        # x: (N, C, H, W) where TF's NHWC image was (N, H, W, C)
        for conv in self.convs:
            x = torch.relu(conv(x))
        flat = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)  # TF conv_to_fc order (h, w, c)
        z = torch.relu(self.fc(flat))
        return self.pi(z), self.vf(z)


def load_sb2_params(path: str) -> dict:
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("data"))
        params = np.load(io.BytesIO(z.read("parameters")), allow_pickle=False)
        arrays = {k: params[k] for k in params.files}
    print(f"loaded {path}: n_steps={meta.get('n_steps')} nminibatches={meta.get('nminibatches')} "
          f"conv1={arrays['model/c1/w:0'].shape} fc1={arrays['model/fc1/w:0'].shape}")
    return arrays


def reference_observation(env: MEDARoutingEnv, collision: np.ndarray, bits: int = 2) -> np.ndarray:
    """The original ``MEDAEnv._getObs``: (W, H, 3) = goal, droplet, health; resized to 30x30."""
    chip, d, g, hz = env.chip, env.droplet, env.goal, env.hazard
    obs = np.zeros((chip.width, chip.height, 3), dtype=np.float64)
    obs[g.xa : g.xb + 1, g.ya : g.yb + 1, 0] = 1.0
    obs[d.xa : d.xb + 1, d.ya : d.yb + 1, 1] = 1.0
    west, south, east, north = collision
    if west:
        obs[d.xa, d.ya : d.yb + 1, 1] = 0.5
    if south:
        obs[d.xa : d.xb + 1, d.ya, 1] = 0.5
    if east:
        obs[d.xb, d.ya : d.yb + 1, 1] = 0.5
    if north:
        obs[d.xa : d.xb + 1, d.yb, 1] = 0.5
    health = np.ceil(chip.degradation() * 2**bits - 0.5) / 2**bits
    obs[hz.xa : hz.xb + 1, hz.ya : hz.yb + 1, 2] = health[hz.xa : hz.xb + 1, hz.ya : hz.yb + 1]
    obs = cv2.resize(obs, (30, 30), interpolation=cv2.INTER_AREA)
    return obs.transpose(2, 0, 1).astype(np.float32)  # (C, H=x, W=y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="SB2 model zip from melfar87/MEDA/policy")
    ap.add_argument("--config", default="configs/training/reference_0825a_30x30.yaml")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--set", action="append", default=[], help="env override, e.g. env.fault_fraction=0.1")
    args = ap.parse_args()

    net = ReferenceCNN(load_sb2_params(args.model)).eval()
    env = MEDARoutingEnv(load_config(args.config, args.set).env)
    scores, cycles, success = [], [], []
    with torch.no_grad():
        for i in range(args.episodes):
            env.reset(seed=args.seed + i)
            done, total = False, 0.0
            while not done:
                # env._collision follows the original: flags of the last invalid action
                obs = torch.from_numpy(reference_observation(env, env._collision))[None]
                logits, _ = net(obs)
                action = REFERENCE_ACTIONS[int(logits.argmax())]
                _, reward, term, trunc, info = env.step(action)
                total += reward
                done = term or trunc
            scores.append(total)
            cycles.append(info["num_cycles"])
            success.append(info["is_success"])
    print(json.dumps({
        "episodes": args.episodes,
        "success_rate": float(np.mean(success)),
        "mean_cycles": float(np.mean(cycles)),
        "mean_score": float(np.mean(scores)),
    }, indent=2))


if __name__ == "__main__":
    main()
