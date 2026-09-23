"""Command-line interface: ``meda <command> ...``.

Commands
--------
train          train one agent from a YAML config (Sec. IV)
curriculum     train a chain of agents: traditional or transfer learning (Fig. 3)
evaluate       evaluate a trained agent on random routing jobs (Sec. V-B metrics)
compare        DRL vs. baseline vs. formal routers on identical jobs (Sec. VI)
bioassay       COVID-RAT / COVID-PCR completion-time benchmark (Fig. 9)
plot-training  training curves from run directories (Figs. 4, 7, 8)
render         record a GIF of the agent routing a droplet

``train`` and ``curriculum`` accept ``--set key=value`` overrides of any
training-config value, e.g. ``--set schedule.epochs=40``.  ``evaluate``,
``compare`` and ``render`` accept overrides of the environment only
(``--set env.fault_fraction=0.1``); their other settings are flags.
``bioassay`` builds its chips from its own flags.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import yaml


# --------------------------------------------------------------- helpers
def _import_modules(modules: List[str]) -> None:
    """Import modules that register custom feature extractors (e.g. a GNN).

    The current directory is put on ``sys.path`` first, so that modules of
    the repository such as ``examples.gnn_extractor_example`` can be found
    when ``meda`` runs as an installed console script.
    """
    if not modules:
        return
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    for module in modules:
        importlib.import_module(module)


def _env_config_from(model: Optional[str], config: Optional[str], overrides: List[str]):
    """Env config of a trained model's run (or a training YAML) plus overrides."""
    from .training.config import TrainConfig, apply_override, coerce_numbers
    from .training.trainer import resolve_model_path

    data: Dict = {}
    if config:
        with open(config, "r", encoding="utf-8") as fh:
            data = coerce_numbers(yaml.safe_load(fh) or {})
    elif model:
        model_path = resolve_model_path(model)
        for candidate in (model_path.parent / "config.yaml", model_path.parent.parent / "config.yaml"):
            if candidate.exists():
                with open(candidate, "r", encoding="utf-8") as fh:
                    data = coerce_numbers(yaml.safe_load(fh) or {})
                break
    for item in overrides:
        if not item.split("=", 1)[0].strip().startswith("env."):
            raise SystemExit(
                f"--set {item}: this command only takes env.* overrides "
                f"(e.g. env.fault_fraction=0.1); see --help for its other settings"
            )
        apply_override(data, item)
    return TrainConfig.from_dict({k: v for k, v in data.items() if k in {"env"}}).env


def _router_factories(names: List[str], model: Optional[str], step_mode: str) -> Dict[str, Callable]:
    from .routers.baseline import ShortestPathRouter
    from .routers.formal import FormalRouter

    factories: Dict[str, Callable] = {}
    drl = None
    for name in names:
        key = name.lower()
        if key == "baseline":
            factories["Baseline"] = lambda: ShortestPathRouter(step_mode=step_mode)
        elif key == "formal":
            factories["Formal"] = lambda: FormalRouter(step_mode=step_mode)
        elif key == "drl":
            if model is None:
                raise SystemExit("--model is required for the 'drl' router")
            from .routers.drl import DRLRouter

            drl = drl or DRLRouter.load(model)
            factories["DRL"] = drl.clone
        else:
            raise SystemExit(f"unknown router {name!r} (choose from drl, baseline, formal)")
    return factories


def _print_table(df) -> None:
    import pandas as pd

    with pd.option_context("display.max_columns", None, "display.width", 160, "display.precision", 3):
        print(df)


# --------------------------------------------------------------- commands
def cmd_train(args: argparse.Namespace) -> None:
    from .training.config import load_config
    from .training.trainer import train

    config = load_config(args.config, args.set)
    run_dirs = train(config, args.output_dir)
    print("\n".join(str(d) for d in run_dirs))


def cmd_curriculum(args: argparse.Namespace) -> None:
    from .training.curriculum import CurriculumError, run_curriculum

    try:
        done = run_curriculum(args.config, args.output_dir, args.only, args.set)
    except CurriculumError as err:
        raise SystemExit(f"meda curriculum: {err}") from None
    for name, dirs in done.items():
        print(f"{name}: {', '.join(str(d) for d in dirs)}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO

    from .training.evaluation import evaluate_model
    from .training.trainer import json_safe, make_vec, resolve_model_path

    env_config = _env_config_from(args.model, args.config, args.set)
    model = PPO.load(resolve_model_path(args.model), device=args.device)
    vec = make_vec(env_config, args.n_envs, args.seed)
    metrics = evaluate_model(model, vec, args.episodes, deterministic=not args.stochastic)
    vec.close()
    # strict JSON: NaN (e.g. no successful episode) becomes null
    text = json.dumps(json_safe(metrics), indent=2, allow_nan=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


def cmd_compare(args: argparse.Namespace) -> None:
    from .routers.compare import compare_routers, summarize_comparison

    env_config = _env_config_from(args.model, args.config, args.set)
    factories = _router_factories(args.routers, args.model, args.step_mode)

    def progress(i: int, n: int) -> None:
        if i % max(1, n // 10) == 0 or i == n:
            print(f"  {i}/{n} jobs", file=sys.stderr, flush=True)

    df = compare_routers(factories, env_config, args.jobs, args.seed, k_max=args.k_max, progress=progress)
    summary = summarize_comparison(df)
    _print_table(summary)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        summary.to_csv(out.with_name(out.stem + "_summary.csv"))


def cmd_bioassay(args: argparse.Namespace) -> None:
    from .bioassay.benchmark import run_trials
    from .bioassay.library import get_bioassay

    assay = get_bioassay(args.assay)
    factories = _router_factories(args.routers, args.model, args.step_mode)
    degradation = None
    # the chips' health sensors must match what a DRL model was trained with
    uses_drl = args.model is not None and any(r.lower() == "drl" for r in args.routers)
    health_bits = _env_config_from(args.model, None, []).degradation.health_bits if uses_drl else None
    if args.tau_range or args.c_range or health_bits is not None:
        from .core.biochip import DegradationConfig

        defaults = DegradationConfig()
        degradation = DegradationConfig(
            tau_range=tuple(args.tau_range or defaults.tau_range),
            c_range=tuple(args.c_range or defaults.c_range),
            health_bits=health_bits if health_bits is not None else defaults.health_bits,
        )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, np.ndarray] = {}
    rows = []
    for name, factory in factories.items():

        def progress(i: int, n: int, result, name=name) -> None:
            if i % max(1, n // 10) == 0 or i == n:
                print(f"  [{name}] trial {i}/{n}: {result.cycles} cycles", file=sys.stderr, flush=True)

        cycles, summary = run_trials(
            assay,
            factory,
            args.trials,
            seed=args.seed,
            max_initial_actuations=args.max_initial_actuations,
            fault_fraction=args.fault_fraction,
            hidden_defect_fraction=args.hidden_defect_fraction,
            degradation=degradation,
            progress=progress,
            on_timeout=args.on_timeout,
            max_cycles=args.max_cycles,
        )
        results[name] = cycles
        rows.append({"router": name, **summary.as_dict()})
        np.savetxt(out_dir / f"{args.assay}_{name.lower()}_cycles.txt", cycles)
    import pandas as pd

    table = pd.DataFrame(rows).set_index("router")
    _print_table(table)
    table.to_csv(out_dir / f"{args.assay}_summary.csv")
    try:
        from .viz.plots import plot_completion_cdf

        fig = plot_completion_cdf(results, assay.name, out_dir / f"{args.assay}_cdf.png")
        print(f"figure: {fig}")
    except ImportError:  # plotting is optional
        pass


def cmd_plot_training(args: argparse.Namespace) -> None:
    from .viz.plots import plot_training_comparison, plot_training_curves

    def histories(run: str) -> List[Path]:
        p = Path(run)
        if p.is_file():
            return [p]
        found = sorted(p.glob("progress.csv")) or sorted(p.glob("seed_*/progress.csv"))
        if not found:
            raise SystemExit(f"no progress.csv under {p}")
        return found

    if len(args.runs) == 1:
        title = args.title or Path(args.runs[0]).name
        fig = plot_training_curves(histories(args.runs[0]), title, args.out)
    else:
        labels = args.labels or [Path(r).name for r in args.runs]
        groups = {label: histories(run) for label, run in zip(labels, args.runs)}
        fig = plot_training_comparison(groups, args.title or "training", args.out)
    print(f"figure: {fig}")


def cmd_render(args: argparse.Namespace) -> None:
    from .envs.meda_env import MEDARoutingEnv
    from .routers.drl import DRLRouter
    from .viz.animation import record_episode

    env_config = _env_config_from(args.model, args.config, args.set)
    router = DRLRouter.load(args.model)
    env = MEDARoutingEnv(env_config)

    def policy(obs, env):
        action, _ = router.model.predict(obs, deterministic=True)
        return int(action)

    info = record_episode(env, policy, args.out, seed=args.seed)
    print(json.dumps({k: v for k, v in info.items() if k != "frames"}, indent=2, default=str))


# --------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meda", description="DRL droplet routing on MEDA biochips (Elfar et al., TCAD 2023)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_set(p: argparse.ArgumentParser, env_only: bool = False) -> None:
        what = "env override, e.g. env.fault_fraction=0.1" if env_only else \
            "config override, e.g. schedule.epochs=40 or env.fault_fraction=0.1"
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help=f"{what} (repeatable)")

    def add_import(p: argparse.ArgumentParser) -> None:
        p.add_argument("--import-module", action="append", default=[], metavar="MODULE",
                       help="module to import first, e.g. one that registers a custom "
                            "feature extractor (repeatable)")

    p = sub.add_parser("train", help="train one agent")
    p.add_argument("--config", "-c", help="training YAML (default: paper defaults)")
    p.add_argument("--output-dir", "-o")
    add_import(p)
    add_set(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("curriculum", help="traditional / transfer learning chains (Fig. 3)")
    p.add_argument("--config", "-c", required=True, help="curriculum YAML")
    p.add_argument("--output-dir", "-o")
    p.add_argument("--only", nargs="*", help="train only these stages (others are reused)")
    add_import(p)
    add_set(p)
    p.set_defaults(func=cmd_curriculum)

    p = sub.add_parser("evaluate", help="evaluate a trained agent on random jobs")
    p.add_argument("--model", "-m", required=True, help="model.zip or run directory")
    p.add_argument("--config", "-c", help="training YAML for the env (default: the run's config)")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", help="write metrics JSON here")
    add_import(p)
    add_set(p, env_only=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("compare", help="compare routers on identical random jobs (Sec. VI)")
    p.add_argument("--model", "-m", help="trained agent (needed for the drl router)")
    p.add_argument("--config", "-c", help="training YAML for the env (default: the run's config)")
    p.add_argument("--routers", nargs="+", default=["drl", "baseline"])
    p.add_argument("--step-mode", default="single", choices=["single", "double", "adaptive"],
                   help="step mode of the baseline and formal routers")
    p.add_argument("--jobs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-max", type=int, default=None,
                   help="fixed cycle budget per job (Sec. VI uses 40); default: the env's k_max")
    p.add_argument("--out", help="CSV with one row per (job, router)")
    add_import(p)
    add_set(p, env_only=True)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("bioassay", help="bioassay completion benchmark (Fig. 9)")
    p.add_argument("--assay", default="covid-rat", help="covid-rat | covid-pcr | simple")
    p.add_argument("--model", "-m", help="trained agent for the drl router (60x30 chip)")
    p.add_argument("--routers", nargs="+", default=["baseline", "formal", "drl"])
    p.add_argument("--step-mode", default="double", choices=["single", "double", "adaptive"],
                   help="step mode of the baseline and formal routers (default: double, the "
                        "MEDAX model the reference Fig. 9 driver uses)")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-initial-actuations", type=int, default=399)
    p.add_argument("--tau-range", type=float, nargs=2, metavar=("LO", "HI"),
                   help="degradation tau range (default: paper, 0.5 0.7; the authors' "
                        "bioassay runs used 0.7 0.7)")
    p.add_argument("--c-range", type=float, nargs=2, metavar=("LO", "HI"),
                   help="degradation c range (default: paper, 500 800; the authors' "
                        "bioassay runs used 200 200)")
    p.add_argument("--fault-fraction", type=float, default=0.0)
    p.add_argument("--hidden-defect-fraction", type=float, default=0.0)
    p.add_argument("--on-timeout", default="continue", choices=["continue", "skip", "fail"])
    p.add_argument("--max-cycles", type=int, default=2000)
    p.add_argument("--out", default="results/bioassay")
    add_import(p)
    p.set_defaults(func=cmd_bioassay)

    p = sub.add_parser("plot-training", help="plot training curves (Figs. 4, 7, 8)")
    p.add_argument("runs", nargs="+", help="run dirs (<name> or <name>/seed_<s>) or progress.csv files")
    p.add_argument("--labels", nargs="*")
    p.add_argument("--title")
    p.add_argument("--out", default="training_curves.png")
    p.set_defaults(func=cmd_plot_training)

    p = sub.add_parser("render", help="record a GIF of one routing episode")
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--config", "-c")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="episode.gif")
    add_import(p)
    add_set(p, env_only=True)
    p.set_defaults(func=cmd_render)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _import_modules(getattr(args, "import_module", None) or [])
    args.func(args)


if __name__ == "__main__":
    main()
