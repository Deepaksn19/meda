# Implementation notes

This document maps the base paper to the code. It also records every
decision taken where the paper is silent, ambiguous or internally
inconsistent.

> M. Elfar, Y.-C. Chang, H. H.-Y. Ku, T.-C. Liang, K. Chakrabarty, M. Pajic,
> "Deep Reinforcement Learning-Based Approach for Efficient and Reliable
> Droplet Routing on MEDA Biochips", *IEEE TCAD* 42(4):1212–1222, 2023,
> doi:[10.1109/TCAD.2022.3194808](https://doi.org/10.1109/TCAD.2022.3194808).

**Guiding rule.** Where the paper is explicit, we follow the paper. Where it
is silent, we follow the first author's public reference implementation,
[`melfar87/MEDA`](https://github.com/melfar87/MEDA). That code uses TF 1.14
and stable-baselines 2.10 PPO2, and its README lists this paper. Anything
that comes from neither source is marked **(ours)** and is configurable.

One artifact is the strongest evidence for Table I and the PPO settings: the
authors' saved PPO2 model `policy/0825a_030x030_E100_NPS64_00.zip` in that
repository, together with its training log `...NPS64.pickle`. It dates from
August 2021, the time of the paper's experiments.

## 1. Paper → code map

| Paper | What | Code |
|---|---|---|
| Sec. III-A, Fig. 1 | droplet quadruple `δ = (xa, ya, xb, yb)` | `core/geometry.py` (`Droplet`, `Rect`) |
| Sec. III-A, Eq. (1) | degradation `D = τ^(n/c)`, `b`-bit health `H = ⌊2^b D⌋` | `core/biochip.py` (`MEDABiochip`) |
| Sec. III-A, [18] | frontier-set probabilistic movement | `core/movement.py`, `core/dynamics.py` |
| Sec. III-B, Alg. 1 | parameterized action space, adaptive step | `core/actions.py` (`adaptive_step`) |
| Sec. III-C, Fig. 2 | 3-channel observation (health ⊙ hazard mask, droplet, goal) | `envs/observation.py` |
| Sec. III-D | reward `α_dis r_dis + α_ter r_ter + α_act r_act` | `envs/reward.py` |
| Sec. IV-A | chip / droplet sizes, degradation sampling, start/goal distribution | `core/jobs.py`, `envs/meda_env.py` |
| Table I | CNN (64/128/128 conv, FC 256, 8 outputs) | `agents/cnn.py` |
| Sec. IV-B, Alg. 2 | PPO, `k_max`, episode resampling | `training/trainer.py`, `envs/meda_env.py` |
| Sec. IV-B | dynamic LR scheduler (`η0 = 3.5e-4`, `βη = 0.7`, `ηmin = 1e-6`) | `training/lr_schedule.py` |
| Sec. IV-C, Fig. 3 | traditional vs. transfer learning, 30×30 `INTER_AREA` resize | `training/curriculum.py`, `configs/curricula/` |
| Sec. I | offline training in simulation, then "online DRL training" on a real chip | `env.persistent_chip` + `env.chip_seed` (one chip whose wear accumulates across episodes), `configs/training/online_adaptation_30x30.yaml` |
| Sec. V-B, Figs. 4, 7, 8 | per-epoch eval on 500 jobs: score, success rate, cycles | `training/evaluation.py`, `viz/plots.py` |
| Sec. V-B, Fig. 9 | COVID-RAT / COVID-PCR, P(completion ≤ k) vs. baseline & formal | `bioassay/`, `routers/baseline.py`, `routers/formal.py` |
| Sec. VI | shortest-path baseline vs. DRL, 0%/10% faults + ~5% unsensed defects | `routers/compare.py` (simulation analogue) |

## 2. Decisions where the paper is silent or ambiguous

| # | Topic | Paper | Reference code | Our choice |
|---|---|---|---|---|
| 1 | Coordinates | 1-based in the figures | 0-based, half-open `[x0, x1)` | 0-based, **inclusive** `[xa, xb]`. Only the representation differs; the MCs covered are the same. |
| 2 | Movement model | "we employ the probabilistic transitions modeling from [18]", not spelled out | `_updatePattern`: the droplet advances **one MC at a time** toward the actuated target footprint. Each axis succeeds with `p = mean(D)` over the actuated frontier MCs; the frontier is the line beyond the leading edge, extended by 1 MC on each side. A failed axis stops for that cycle. | Same as the code (`core/movement.py`). For unit cardinal/diagonal moves this reproduces the MATLAB model of [18] (`BiochipClass.m`), including partial diagonal outcomes. Probabilities use the true `D`, not the quantized `H`. |
| 3 | Health quantization | `H = ⌊2^b·D⌋ ∈ {0..2^b−1}`, observed as `H/2^b` | rounds to nearest: `ceil(2^b·D − 0.5)/2^b ∈ {0, …, 1}` | **Paper** (floor, saturated at `2^b−1`). |
| 4 | Health bits `b` | not given | `n_bits = 2` | `b = 2` |
| 5 | Degradation parameters | `τij ~ U(0.5, 0.7)`, `cij ~ U(500, 800)` per MC | fixed `τ = 0.7` (0.8 before Oct 2021), `c = 200` | **Paper** (per-MC sampling every episode). Note that the authors' bioassay runs (Fig. 9) used the fixed code values `τ = 0.7`, `c = 200`, which age the chip much faster; `meda bioassay --tau-range 0.7 0.7 --c-range 200 200` reproduces that setting. |
| 6 | Initial actuations `N` | sampled per episode (Alg. 2); healthy training resets them | zeros during training; `U{0, 399}` in the bioassay scheduler | zeros for training; `U{0, 399}` for bioassays (both configurable). |
| 7 | Hazard bounds `δh` | a routing zone per job; sampling not described | bbox(start ∪ goal) ± 3 MCs, clipped to the chip | Same (`hazard_margin: 3`). The MATLAB formal flow uses ±2. |
| 8 | `k_max` | `α(Wh + Hh)` with `α ∈ [1, 2]`; the text also says `2(W + H)` | `W + H` of the chip during training; `1·(Wh + Hh)` for bioassays | `α = 1` on the hazard zone (`kmax_alpha`). An episode ends after `k_max` cycles, as in Alg. 2 (`k ≥ k_max`); the code's `step_count > max_step` allows one more. Timeouts are *truncations*, not terminal failures, so PPO bootstraps them. This avoids the state aliasing the paper warns about. `timeout_terminal: true` reproduces PPO2's cut-off instead. |
| 9 | Invalid action | "causes the droplet to exit the routing job area" | invalid if the droplet already touches the zone boundary in a requested direction; otherwise an over-long step is clamped | Same as the code. An invalid action holds the droplet, and the holding pattern is still actuated. |
| 10 | Reward coefficients | not given ("discussed in Section IV", but they are not) | `_getRewardH`: +100 at the goal; `+0.5·Δd` if closer, else `0.8·Δd − 1`; −1 for an invalid action | Same as the code. The authors' 30×30 training log converges to a mean return of ≈111, which matches `100 + 0.5·E[D]`. The plain linear form of the paper is available by setting `alpha_dis_away = alpha_dis` and `stall_penalty = 0`. |
| 11 | Channel order | health, droplet, goal | goal, droplet, health | **Paper** order. This is irrelevant to learning. |
| 12 | Collision cue | — | after an invalid action, the droplet edge on the blocked side is drawn as 0.5 | Optional (`mark_collisions`, off by default). |
| 13 | Droplet sizes | `w, h ∈ {2..6}`, `w/h ∈ [0.8, 1.25]` (9 sizes) | the 0825a run used 4×4, 5×4, 5×5, 6×5, 6×6 | **Paper** (all 9 sizes). |
| 14 | Start/goal distribution | "stratified"; `xa ~ U{2, W−w−1}` | stratified: droplet centres drawn without replacement from `[0]·W/5 + [1..W−2] + [W−1]·W/5` | Code's stratified sampler. The paper motivates stratification with benchmark statistics: 20–40% of real routing jobs start or end next to a chip edge. The code's sampler over-weights edges much more strongly. On a 30×30 chip, 62% of sampled droplets touch an edge, and 86% of jobs have an endpoint on one. `jobs.edge_weight` tunes this. The code builds the y-pool from the chip *width*, a bug that only matters for non-square chips; we use the height. A uniform sampler (`jobs.sampling: uniform`) is also available. It draws exactly the paper's `xa ~ U{2, W−w−1}` (1-based) and therefore avoids edges. |
| 15 | CNN (Table I) | the "Stride 3 / Padding 1" columns; ReLU listed on the output | saved 0825a weights: `c1 (3,3,3,64)`, `c2 (3,3,64,128)`, `c3 (3,3,128,128)`, `fc1 (115200,256)`, `pi (256,8)`, `vf (256,1)` | 3×3 kernels, **stride 1**, SAME padding: the 115200 = 128·30·30 FC input proves stride 1. The policy logits are linear, as in PPO2's `ActorCriticPolicy`. The actor and critic share the extractor. |
| 16 | PPO settings | PPO2, 8 environments | saved model: `n_steps=64`, `nminibatches=16`, `noptepochs=4`, `γ=0.99`, `λ=0.95`, `ent=0.01`, `vf=0.5`, `max_grad_norm=0.5`, `clip=0.2`, `clip_vf=0.2` | Same values, translated to SB3 (`batch_size = 8·64/16 = 32`, `clip_range_vf = 0.2`). |
| 17 | LR schedule | per-epoch rule, `η0 = 3.5e-4`, decay 0.7 if success > 0.99, floor 1e-6 | same rule, but decays only at exactly 100% success. Within each epoch the rate is `η_i·sqrt(remaining fraction)`. | **Paper** threshold, plus the code's within-epoch sqrt decay (`intra_epoch: sqrt`, or `constant`). |
| 18 | Epoch length | 2^14 steps | 16384 in the 0825a run | 2^14 |
| 19 | Per-epoch evaluation | 500 random jobs | 500 deterministic episodes on a separate environment | Same, but the eval environment is **re-seeded every epoch** (ours), so every epoch sees the same 500 jobs and the curves are less noisy. |
| 20 | Fault injection | fixed % of fully degraded MCs in 2×2 clusters | `int(W·H·p/2)` random 2×2 blocks (≈ 2p coverage) | Exact coverage `p` (ours). Start/goal droplets are never covered (`protect_endpoints`, ours). |
| 21 | Transfer learning | initialize from the pretrained agent | loads the whole PPO2 model | Weights only, with a fresh optimizer and the new stage's hyperparameters (ours). |
| 22 | Formal baseline | PRISM-games strategies from [18] | MATLAB + PRISM-games (the model generator is not public) | Exact finite-horizon value iteration on the same MDP (`routers/formal.py`). It uses a health snapshot at job start with `D̂ = H/(2^b−1)`, maximizes `P[F≤K goal]`, and sets `K = ⌈1.5·dist⌉ + 1` as in `pfcnGetKmax`. The action set follows [18] (`BiochipClass.m`): single steps, or the MEDAX double-step model with `aNN/aSS/aEE/aWW` but no double-diagonal move (`step_mode: double`). The reference Fig. 9 driver (`TestBiochipV13.m`, `bDStep = 1`) uses the latter, so `meda bioassay` defaults to it. |
| 23 | Health-agnostic baseline | minimizes time, ignores health (Sec. V-B); shortest path (Sec. VI) | PRISM with an all-healthy matrix | Greedy shortest path on the grid, replanned every cycle (`routers/baseline.py`). It uses the same step modes as the formal router: double for bioassays (Fig. 9) and single for the Sec. VI comparison, where the PCB baseline moves one electrode at a time. |
| 24 | Bioassays | COVID-RAT, COVID-PCR; no job lists | `meda_sgs.py`: `sg_CRAT` (28 operations), `sg_CPCR` (34 operations), 60×30 chip, 4×4 droplets, 1000 trials, `k_max = 2000` | Transcribed from the code (`bioassay/library.py`), and checked against the reference file in the tests. Deviations:<br>• The second split droplet goes to `goals[1]`; the code sends it to `goals[0]`, a copy-paste bug.<br>• Timed-out jobs keep routing (`on_timeout: continue`); the code treats them as done (`skip`).<br>• All jobs sense and wear one shared chip; the code gives each job a snapshot copy.<br>• `cycles` counts executed control cycles and failed trials are `inf`; the code adds one detection tick and records failures as 2000 cycles.<br>Like the code, each operation starts from its *listed* coordinates, which are not always where the previous operation left the droplet (e.g. COVID-PCR `n09`, COVID-RAT `n09`/`n10`). |
| 25 | Hardware (Sec. V-A, VI) | PCB DMFB measurements and experiments | — | Not reproducible without hardware. `routers/compare.py` runs the Sec. VI protocol in simulation, with sensed faults plus unsensed `hidden_defect_fraction`. |

## 3. Known limitations

* **Training cost.** The Table I network has 29.7 M parameters, almost all
  of them in the FC layer: 128·30·30·256 weights. One paper epoch (2^14
  steps plus evaluation) takes minutes on a GPU but much longer on a CPU. For
  CPU experiments use `configs/training/quick_cpu_30x30.yaml`, the
  reference code's smaller 32/64/64/128 variant.
* **Traditional learning at native resolution.** The FC layer grows with the
  chip area. That is exactly the scalability problem transfer learning with a
  30×30 input avoids, and it motivates graph-based agents
  (see [EXTENDING.md](EXTENDING.md)).
* **Resizing large chips to 30×30** makes an observation pixel cover
  `(W/30)²` MCs. On a 180×180 chip a 2×2 droplet is smaller than a pixel.
  This is a genuine weakness of the base method and a natural target for a
  resolution-independent GNN.
* **Section V-A** estimates `τ` and `c` from electrode measurements. Note
  that `τ^(n/c) = exp(n·ln τ / c)`, so only the ratio `ln τ / c` can be
  identified from such data.
