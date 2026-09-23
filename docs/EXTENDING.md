# Extending the base: towards GNN-based routing

The base paper's agent is a CNN over a fixed-size image. This repository is
meant as the starting point for a **graph neural network (GNN)** approach.
This page describes the extension points and the shortcomings of the base
method a GNN can address.

## Why a GNN? Weaknesses of the base method

1. **Resolution loss.** Every chip is resampled to a 30×30 observation
   (Sec. IV-C). On a 180×180 chip one observation pixel covers 6×6 MCs, so
   a 2×2 droplet or a single degraded MC blurs away. The MC grid is a graph,
   and a message-passing network with size-independent weights can consume
   the native grid.
2. **Parameter growth.** Without resizing, the Table I FC layer grows with
   chip area: 128·W·H·256 weights, about 1.06 B at 180×180. GNN weights do
   not depend on the number of nodes.
3. **Transfer across chip sizes.** The paper needs a resize to share weights
   between sizes (Fig. 3b). With a GNN, the same weights apply to any grid.
4. **Multiple droplets.** Bioassays route many droplets concurrently
   (`bioassay/scheduler.py`), but the base agent sees one droplet and knows
   nothing about the others. Droplet-level graphs, with droplets and modules
   as nodes and interference as edges, extend naturally to multi-droplet
   and multi-agent routing. For a MEDA multi-agent RL environment, see
   T.-C. Liang et al., ICML 2021 (`tcliang-tw/meda-env`).

## Extension point 1: a new feature extractor

Policies are Stable-Baselines3 actor-critic policies whose feature extractor
is looked up by name (`agents/registry.py`):

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from meda_routing.agents import register_extractor

@register_extractor("my_gnn")
class MyGNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden=64):
        super().__init__(observation_space, features_dim=hidden)
        ...
    def forward(self, obs):          # obs: (batch, 3, H, W) float tensor
        ...                          # channels: health (masked to zone), droplet, goal
        return features              # (batch, features_dim)
```

Select it in a config with `agent.extractor: my_gnn`, and set
`agent.extractor_kwargs` to its own arguments; the defaults are the CNN's
`channels` and `hidden_dim`. The actor (8 logits) and critic heads are
linear layers on top of the features, as in the paper.

The `meda` command must import the module that registers the extractor.
This applies to training and to every command that loads the trained model
(`evaluate`, `compare`, `bioassay`, `render`). Pass
`--import-module <module>` to do this. The current directory is put on
`sys.path` first, so run from the repository root to use a module that
lives in the repository:

```bash
meda train -c configs/training/quick_cpu_30x30.yaml \
    --import-module examples.gnn_extractor_example \
    --set agent.extractor=gnn_example --set "agent.extractor_kwargs={}"
meda compare -m runs/quick_cpu_30x30/seed_0 --routers drl baseline formal \
    --import-module examples.gnn_extractor_example --set env.fault_fraction=0.1
```

[`examples/gnn_extractor_example.py`](../examples/gnn_extractor_example.py)
is a small, working (not tuned) example. It treats the observation as an
8-connected grid graph with node features `[health, droplet, goal, x, y]`,
applies GraphSAGE-style mean aggregation, and pools over the droplet, the
goal and the whole chip. `tests/test_extending.py` trains it for a few PPO
updates to show the plumbing works.

Because such a network's weights do not depend on the grid size, you can
train with native observations (`env.obs_size: null`) and move a model
between chip sizes by building a new model for the new size and loading the
state dict (see `training/trainer.py::load_pretrained_weights`; the obs-shape
check there is only needed for CNNs).

## Extension point 2: a different observation

`envs/observation.py::build_observation` produces the image. For a graph
observation (e.g. a coarsened region-adjacency graph around the routing
zone, or droplet/module graphs), add a new observation mode to
`MEDARoutingEnv` that returns a `gymnasium.spaces.Dict` or `Graph` space,
and write a matching extractor. Keep the physics (`core/`) untouched, so
results stay comparable with the paper's CNN.

## Extension point 3: routers

Everything that evaluates policies (`routers/compare.py`, the bioassay
benchmark, the Fig. 9 CDFs) works on the `Router` interface
(`routers/base.py`): implement `act(state) -> Action` and optionally
`reset(job, chip)`. `DRLRouter` wraps any trained SB3 model, whatever
feature extractor it uses. Before inference it rebuilds the observation
from the chip's health sensors, so a GNN agent can be benchmarked against
the CNN, the shortest-path baseline and the MDP-optimal "formal" router
without other changes.

## Suggested evaluation protocol

To compare a new agent with the base paper, use the paper's own metrics:

* training curves: success rate, mean score and cycles per epoch on 500
  fixed random jobs (Figs. 4, 7, 8), across chip sizes and 0/10/20% faults;
* robustness: `meda compare` with `env.fault_fraction` 0.1–0.2 and
  `env.hidden_defect_fraction` 0.05 (Sec. VI);
* bioassays: `meda bioassay` CDFs for COVID-RAT and COVID-PCR (Fig. 9);
* decision latency per cycle (the paper requires < 200 ms; the CNN takes
  < 0.1 s).
