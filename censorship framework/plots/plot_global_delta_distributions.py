#!/usr/bin/env python3
"""
Plot local censorship gap distributions.

This script expects a CSV file with one row per prompt-model example and
at least the following columns:

    model, gap

where `gap` is the local censorship gap:
    d_{m,R}(p) = s_m(p) - median_{r in R} s_r(p)

The script groups by model and plots one density curve per model.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def pretty_model_name(name: str) -> str:
    name = str(name).replace("model_", "")
    name = name.replace("_", "-")
    name = name.replace("r1-new", "R1-0528")
    name = name.replace("r1-old", "R1")
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="local_gap_distributions.pdf")
    parser.add_argument("--model_col", type=str, default="model")
    parser.add_argument("--gap_col", type=str, default="gap")
    parser.add_argument("--x_min", type=float, default=None)
    parser.add_argument("--x_max", type=float, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    if args.model_col not in df.columns:
        raise ValueError(f"Missing model column: {args.model_col}")
    if args.gap_col not in df.columns:
        raise ValueError(f"Missing gap column: {args.gap_col}")

    models = list(df[args.model_col].dropna().unique())
    if len(models) == 0:
        raise ValueError("No models found.")

    all_vals = pd.to_numeric(df[args.gap_col], errors="coerce").dropna().values
    all_vals = all_vals[np.isfinite(all_vals)]

    if len(all_vals) == 0:
        raise ValueError("No finite gap values found.")

    if args.x_min is None or args.x_max is None:
        x_low, x_high = np.percentile(all_vals, [1, 99])
        x_min = args.x_min if args.x_min is not None else x_low - 0.3
        x_max = args.x_max if args.x_max is not None else x_high + 0.3
    else:
        x_min, x_max = args.x_min, args.x_max

    x = np.linspace(x_min, x_max, 500)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for model in models:
        vals = pd.to_numeric(
            df.loc[df[args.model_col] == model, args.gap_col],
            errors="coerce",
        ).dropna().values

        vals = vals[np.isfinite(vals)]

        if len(vals) < 2:
            continue

        kde = gaussian_kde(vals)
        y = kde(x)

        label = pretty_model_name(model)

        ax.fill_between(x, y, alpha=0.20)
        ax.plot(x, y, linewidth=2.0, label=label)

        ax.axvline(vals.mean(), linestyle="-", linewidth=1.2, alpha=0.8)

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.5,
        label=r"$d = 0$",
    )

    ax.set_xlabel(r"Local censorship gap $d_{m,R}(p)$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(args.output_file, bbox_inches="tight")
    print(f"Saved figure to: {args.output_file}")


if __name__ == "__main__":
    main()