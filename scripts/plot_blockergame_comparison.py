#!/usr/bin/env python3
"""
Plot blockergame training curves: average cost, return, peak violations.

Each curve is read from disk (one float per line per epoch). The band is either rolling mean ± rolling
std (``--shade rolling``) or mean ± std across multiple log paths per model (``--shade across``).

Optional: Gaussian smoothing on the displayed mean and band (``--line-gauss-sigma``), and a global
return floor (``--return-floor``).

Usage:
  python3 scripts/plot_blockergame_comparison.py --output figures/metrics.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d
except ImportError:
    gaussian_filter1d = None
    uniform_filter1d = None

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "matplotlib is required: pip install matplotlib"
    ) from e


class PlotCfg:
    """Runtime options set from CLI before plotting."""

    return_floor: float = 0.0


DEFAULT_MODELS: Dict[str, List[str]] = {
    "IQL": ["blockergame_baseline_iql"],
    "VDN": ["blockergame_baseline_vdn"],
    "DSW (full)": ["blockergame_-DSW"],
    "DSW-QMIX-L": ["blockergame_-DSW-wo_ALM"],
    "DSW-QMIX-S": ["blockergame_ablation_dsw_qmix_s"],
    "DSW-QMIX-H": ["blockergame_ablation_dsw_qmix_h"],
}


def load_log(path: str) -> np.ndarray:
    vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals.append(float(line))
    return np.asarray(vals, dtype=np.float64)


def _apply_return_floor_array(y: np.ndarray, metric_key: str) -> np.ndarray:
    if metric_key != "return" or PlotCfg.return_floor <= 0.0:
        return y
    return np.maximum(np.asarray(y, dtype=np.float64), PlotCfg.return_floor)


def mean_std_across_runs(stacked: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = stacked.mean(axis=0)
    sd = stacked.std(axis=0, ddof=0)
    return mu, sd


def rolling_mean_std(y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    w = max(1, int(window))
    if w == 1:
        return y.copy(), np.zeros_like(y)
    if uniform_filter1d is None:
        n = len(y)
        half = w // 2
        mu = np.empty(n)
        sd = np.empty(n)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            seg = y[lo:hi]
            mu[i] = float(seg.mean())
            sd[i] = float(seg.std(ddof=0))
        return mu, sd
    mu = uniform_filter1d(y, size=w, mode="nearest")
    m2 = uniform_filter1d(y * y, size=w, mode="nearest")
    var = np.maximum(m2 - mu * mu, 0.0)
    return mu, np.sqrt(var)


def smooth_for_plot(
    y: np.ndarray,
    gauss_sigma: float,
) -> np.ndarray:
    """Extra smoothing for displayed mean and band (Gaussian, edge-safe)."""
    y = np.asarray(y, dtype=np.float64)
    if gauss_sigma <= 0 or len(y) < 2:
        return y.copy()
    if gaussian_filter1d is not None:
        return gaussian_filter1d(y, sigma=float(gauss_sigma), mode="nearest")
    w = max(3, int(gauss_sigma * 4) | 1)
    if uniform_filter1d is None:
        return y.copy()
    return uniform_filter1d(y, size=w, mode="nearest")


def build_series(
    paths: List[str],
    rolling_window: int,
    shade_mode: str,
    metric_key: str,
    gauss_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    runs = [load_log(p) for p in paths]

    if len(runs) == 1:
        y = runs[0]
        if shade_mode == "across":
            raise ValueError("Need multiple --run paths for shade_mode=across")
        y = _apply_return_floor_array(y, metric_key)
        mu, sd = rolling_mean_std(y, rolling_window)
        mu = smooth_for_plot(mu, gauss_sigma)
        sd = smooth_for_plot(sd, gauss_sigma * 0.85)
        mu = _apply_return_floor_array(mu, metric_key)
    else:
        m = min(len(r) for r in runs)
        stacked = np.stack([r[:m].copy() for r in runs], axis=0)
        stacked = _apply_return_floor_array(stacked, metric_key)
        mu, sd = mean_std_across_runs(stacked)
        mu = smooth_for_plot(mu, gauss_sigma)
        sd = smooth_for_plot(sd, gauss_sigma * 0.85)
        mu = _apply_return_floor_array(mu, metric_key)

    t = np.arange(1, len(mu) + 1, dtype=np.float64)
    lower = mu - sd
    upper = mu + sd
    return t, mu, lower, upper


def plot_all(
    log_root: str,
    models: Dict[str, List[str]],
    rolling_window: int,
    shade_mode: str,
    max_steps: Optional[int],
    title_prefix: str,
    figsize: Tuple[float, float],
    gauss_sigma: float,
) -> plt.Figure:
    metrics = [
        ("cost.log", "Average cost", "cost"),
        ("return.log", "Return (logged per epoch)", "return"),
        ("peak_violation.log", "Peak violations", "peak_violations"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)
    cmap = plt.cm.tab10(np.linspace(0, 0.9, max(len(models), 3)))

    for ax, (log_name, ylabel, mkey) in zip(axes, metrics):
        for idx, (label, rel_dirs) in enumerate(models.items()):
            paths = [os.path.join(log_root, d, log_name) for d in rel_dirs]
            for p in paths:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Missing: {p}")
            t, mu, lo, hi = build_series(
                paths,
                rolling_window,
                shade_mode,
                mkey,
                gauss_sigma,
            )
            if max_steps is not None:
                mask = t <= max_steps
                t, mu, lo, hi = t[mask], mu[mask], lo[mask], hi[mask]
            color = cmap[idx % len(cmap)]
            ax.plot(t, mu, label=label, color=color, linewidth=2.0, antialiased=True)
            ax.fill_between(t, lo, hi, color=color, alpha=0.2)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("Training epoch (index in log)")
    fig.suptitle(title_prefix)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot blockergame cost / return / peak violations (raw logs).")
    parser.add_argument("--log-root", type=str, default="./log", help="Root containing blockergame_* dirs")
    parser.add_argument(
        "--output",
        type=str,
        default="blockergame_metrics_comparison.png",
        help="Output figure path (.png / .pdf / .svg)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=2000,
        help="Rolling window for mean/std band (default 2000; larger = smoother).",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Truncate x-axis to this many epochs")
    parser.add_argument("--title", type=str, default="Blockergame — model comparison", help="Figure title")
    parser.add_argument(
        "--models-json",
        type=str,
        default=None,
        help='JSON {"Label": ["rel_dir"], ...} overrides defaults.',
    )
    parser.add_argument("--figsize", type=str, default="10,12", help="W,H inches")
    parser.add_argument(
        "--shade",
        choices=("rolling", "across"),
        default="rolling",
        help="rolling: rolling std; across: std across multiple paths per model",
    )
    parser.add_argument(
        "--line-gauss-sigma",
        type=float,
        default=3.5,
        help="Extra Gaussian smoothing (epochs scale) on plotted mean and band; 0=off.",
    )
    parser.add_argument(
        "--return-floor",
        type=float,
        default=0.0,
        help="Minimum return for all curves (0=off).",
    )
    args = parser.parse_args()

    PlotCfg.return_floor = max(0.0, float(args.return_floor))

    if args.models_json:
        with open(args.models_json, "r", encoding="utf-8") as f:
            models: Dict[str, List[str]] = json.load(f)
    else:
        models = DEFAULT_MODELS

    w, h = [float(x.strip()) for x in args.figsize.split(",")]
    fig = plot_all(
        log_root=os.path.expanduser(args.log_root),
        models=models,
        rolling_window=args.rolling_window,
        shade_mode=args.shade,
        max_steps=args.max_steps,
        title_prefix=args.title,
        figsize=(w, h),
        gauss_sigma=args.line_gauss_sigma,
    )
    out = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
