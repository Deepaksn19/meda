#!/usr/bin/env python3
"""Overlay our training curve on the authors' logged training run.

The first author's repository (melfar87/MEDA) contains the training log of
the 30x30 agent trained in August 2021, ``policy/0825a_030x030_E100_NPS64.pickle``
(success rate, score and cycles after each of 100 epochs, evaluated on 500
random jobs).  ``configs/training/reference_0825a_30x30.yaml`` mirrors that
run.  This script plots both curves and prints when each first reaches
90 / 99 / 100 % success.

The pickle is loaded with a restricted unpickler that only admits the few
harmless classes the file needs (argparse.Namespace and numpy arrays), so no
code from the downloaded file can run.

Usage::

    python scripts/compare_with_reference_log.py runs/reference_0825a_30x30 \
        --out figures/reference_vs_ours.png
"""

from __future__ import annotations

import argparse
import io
import pickle
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REFERENCE_URL = (
    "https://raw.githubusercontent.com/melfar87/MEDA/master/policy/"
    "0825a_030x030_E100_NPS64.pickle"
)
_ALLOWED = {
    ("argparse", "Namespace"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        if module.startswith("numpy.core") and not hasattr(np, "core"):
            module = module.replace("numpy.core", "numpy._core")
        return super().find_class(module, name)


def load_reference(source: str) -> pd.DataFrame:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as resp:  # noqa: S310
            raw = resp.read()
    else:
        raw = Path(source).read_bytes()
    data = _SafeUnpickler(io.BytesIO(raw)).load()
    success = np.asarray(data["a_goals"][0], dtype=float) / 100.0
    return pd.DataFrame(
        {
            "epoch": np.arange(1, len(success) + 1),
            "success_rate": success,
            "mean_score": np.asarray(data["a_rewards"][0], dtype=float),
            "mean_cycles": np.asarray(data["a_cycles"][0], dtype=float),
        }
    )


def load_ours(paths: List[str]) -> List[pd.DataFrame]:
    frames = []
    for p in map(Path, paths):
        files = [p] if p.is_file() else sorted(p.glob("progress.csv")) or sorted(p.glob("seed_*/progress.csv"))
        frames += [pd.read_csv(f) for f in files]
    if not frames:
        raise SystemExit(f"no progress.csv found in {paths}")
    return frames


def milestones(df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for thr in (0.9, 0.99, 1.0):
        hit = df.index[df["success_rate"] >= thr - 1e-9]
        out[f">={thr:.0%}"] = int(df.loc[hit[0], "epoch"]) if len(hit) else None
    tail = df.tail(5)
    out["final success"] = round(float(tail["success_rate"].iloc[-1]), 3)
    out["final cycles (last 5 mean)"] = round(float(tail["mean_cycles"].mean()), 2)
    out["final score (last 5 mean)"] = round(float(tail["mean_score"].mean()), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ours", nargs="+", help="run dir(s) or progress.csv file(s)")
    ap.add_argument("--reference", default=REFERENCE_URL, help="path or URL of the authors' pickle")
    ap.add_argument("--out", default="reference_vs_ours.png")
    args = ap.parse_args()

    ref = load_reference(args.reference)
    ours = load_ours(args.ours)
    print("authors (0825a):", milestones(ref))
    for i, df in enumerate(ours):
        print(f"ours [{i}]:", milestones(df))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = int(max(df["epoch"].max() for df in ours))
    ref = ref[ref["epoch"] <= max(n, 30)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col, label in zip(
        axes, ("success_rate", "mean_cycles", "mean_score"), ("success rate", "cycles per job", "score")
    ):
        ax.plot(ref["epoch"], ref[col], "k--", label="authors' log (0825a)")
        for i, df in enumerate(ours):
            ax.plot(df["epoch"], df[col], "r-", alpha=0.8, label="this repo" if i == 0 else None)
        ax.set_xlabel("training epoch (2^14 steps)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"figure: {args.out}")


if __name__ == "__main__":
    main()
