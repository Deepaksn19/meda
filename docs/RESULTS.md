# Validation results

These results check that this reimplementation reproduces the base paper's
setup. A full-scale replication of the training curves (Figs. 4, 7, 8) and
of Fig. 9 needs GPU time. The authors trained on an RTX 6000. On the 4-core
CPU used here, one PPO minibatch step of the Table I CNN takes about 1 s, so
a single 2^14-step epoch takes about 35 min. The checks below are the ones
that fit on a CPU, and they are designed to be as informative as possible.

## 1. The authors' own trained agent in our environment

The first author's repository contains the trained model of the paper's
30×30 run (`policy/0825a_030x030_E100_NPS64_00.zip`, Table I CNN, August
2021) and its training log. `scripts/evaluate_reference_model.py` loads
those TensorFlow weights into PyTorch and runs the agent in
`MEDARoutingEnv` on 300 random jobs from the same job distribution
(`configs/training/reference_0825a_30x30.yaml`: healthy 30×30 chips,
4×4 … 6×6 droplets). The agent sees observations in its original encoding.

| | success rate | cycles / job | score |
|---|---|---|---|
| authors' log, converged (epochs 30–100) | 100% | ≈10.5 | ≈111.1–111.5 |
| **authors' agent in this environment** | **100%** | **10.09** | **110.85** |

The agent was never trained on our code. It succeeds on every job with the
same efficiency as in its own training log. So our environment reproduces
the original movement model, adaptive step (Algorithm 1), action semantics,
routing zones, job distribution and reward. A mismatch in any of these, such
as swapped axes, a different frontier definition or wrong step sizes, would
have broken the agent.

```bash
curl -L -o 0825a.zip https://media.githubusercontent.com/media/melfar87/MEDA/master/policy/0825a_030x030_E100_NPS64_00.zip
python scripts/evaluate_reference_model.py 0825a.zip --episodes 300
```

## 2. Optimal-policy check of the reward and job distribution

On a healthy chip the adaptive shortest-path router is cycle-optimal. On
the same 500 jobs it scores **111.02** (authors' log: ≈111.1–111.5) in
**7.16** cycles per job. The score depends only on the start–goal distance
distribution when every step makes progress, and it matches. The authors'
converged agent needs ≈10.5 cycles, not 7.2. The reward pays for progress,
not for speed, so their converged policy is not cycle-optimal. Only the
discount factor favours shorter paths.

## 3. Health-aware vs. health-agnostic routing (Sec. V-B / VI)

Setup: 30×30 chips, 20% sensed faults in 2×2 clusters, 5% defects the
health sensors cannot see, and 60 identical jobs per router
(`routers.compare_routers`):

| router | success rate | cycles (successful jobs) |
|---|---|---|
| Baseline: health-agnostic shortest path, single step | 83.3% | 19.4 |
| Baseline, adaptive step (Algorithm 1) | 81.7% | 11.7 |
| Formal: MDP-optimal on the sensed health map | 100% | 19.2 |

Health-agnostic routing gets stuck in front of fully degraded frontiers, as
in Fig. 14(b)–(c). The health-aware formal strategy avoids them. The DRL
agent combines health awareness with the adaptive step of the second row.

## 4. Bioassays (Fig. 9 setting)

COVID-RAT on a pre-aged 60×30 chip (`meda bioassay --assay covid-rat
--routers baseline formal`, 12 trials): baseline 241 cycles on average,
formal 237 (paper, Fig. 9: roughly 190–235 for both). The DRL curve
requires the `covid_60x30` agent (see the README).

## 5. Runtime (Sec. II-D, V-B)

| | this repository (1 CPU thread) | paper |
|---|---|---|
| DRL decision per control cycle (observation + Table I CNN) | 14.8 ms | < 0.1 s; required < 200 ms |
| formal strategy per routing job (60×30 chip, 4×4 droplet) | 0.08–0.10 s on average, 0.37 s max | 5–48 s with PRISM-games |

Our formal router solves the same single-droplet MDP with vectorized value
iteration. PRISM-games builds and solves a stochastic-game model from
scratch, which explains the gap. The paper's scalability argument against
formal synthesis concerns much larger chips and full bioassays.

## 6. Learning on a reduced problem (CPU)

See the curves produced by the 16×16 validation run below (added when the run
completes).
