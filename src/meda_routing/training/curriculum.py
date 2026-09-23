"""Training multiple CNNs: traditional vs. transfer learning (Sec. IV-C, Fig. 3).

* **Traditional learning** trains one randomly initialized agent per biochip
  size and fault-injection level, each on observations at the chip's native
  resolution.
* **Transfer learning** first trains ``H(30, 0%)`` on healthy 30x30 chips and
  then initializes every following agent from an already trained one, e.g.
  ``H(60, 0%) <- H*(30, 0%)`` and ``H(30, 10%) <- H*(30, 0%)``.  All agents
  share the unified 30x30 observation (``cv2.INTER_AREA`` resampling), so the
  weights transfer as-is.

A curriculum YAML lists stages; each stage deep-merges its overrides onto a
base training config and may name an earlier stage in ``init_from``::

    name: transfer
    base: configs/training/paper_defaults.yaml
    stages:
      - name: s030_f00
        env: {width: 30, height: 30}
      - name: s060_f00
        init_from: s030_f00
        env: {width: 60, height: 60}
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .config import TrainConfig, apply_override, coerce_numbers
from .trainer import Trainer


def _seed_number(path: Path) -> int:
    """``seed_10`` sorts after ``seed_2``."""
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 1 << 62


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_curriculum(path: Union[str, Path], overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    base: Dict[str, Any] = {}
    if data.get("base"):
        base_path = Path(data["base"])
        if not base_path.is_absolute() and not base_path.exists():
            base_path = path.parent / base_path
        with open(base_path, "r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
    base = deep_merge(base, data.get("defaults", {}))
    return {
        "name": data.get("name", path.stem),
        "base": coerce_numbers(base),
        "stages": coerce_numbers(data.get("stages", [])),
        # command-line overrides win over the base *and* the stage settings
        "overrides": list(overrides or []),
    }


def run_curriculum(
    path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    only: Optional[List[str]] = None,
    overrides: Optional[List[str]] = None,
) -> Dict[str, List[Path]]:
    """Train all stages in order; returns ``{stage name: [run dirs]}``."""
    spec = load_curriculum(path, overrides)
    base = deep_merge(spec["base"], {})
    for item in spec["overrides"]:
        apply_override(base, item)
    root = Path(output_dir or base.get("output_dir", "runs")) / spec["name"]
    done: Dict[str, List[Path]] = {}
    for stage in spec["stages"]:
        stage = dict(stage)
        name = stage.pop("name")
        init_from = stage.pop("init_from", None)
        data = deep_merge(spec["base"], stage)
        for item in spec["overrides"]:
            apply_override(data, item)
        data["name"] = name
        data.pop("init_from", None)
        config = TrainConfig.from_dict(data)
        if only and name not in only:
            # still register existing results so later stages can transfer from them
            existing = sorted((root / name).glob("seed_*"), key=_seed_number)
            if existing:
                done[name] = existing
            continue
        run_dirs = []
        for r in range(config.repeats):
            seed = config.seed + r
            if init_from is not None:
                if init_from in done:
                    parents = done[init_from]
                    parent = parents[r] if r < len(parents) else parents[0]
                else:
                    parent = Path(init_from)  # explicit path
                config.init_from = str(parent)
            run_dir = root / name / f"seed_{seed}"
            Trainer(config, run_dir, seed).run()
            run_dirs.append(run_dir)
        done[name] = run_dirs
    return done
