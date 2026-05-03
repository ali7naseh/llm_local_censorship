#!/usr/bin/env python3
"""
Plot subtopic-level local censorship results.

This script expects a CSV file containing subtopic-level results with columns:

    Category, Sub_Category, n, mean_d, pi_plus, ci90_low, ci90_high

It selects the top-k subtopics by severity = mean_d * pi_plus and plots
mean local gap with confidence intervals.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clean_label(category: str, sub_category: str) -> str:
    category = str(category)
    sub_category = str(sub_category)

    if sub_category not in ["None", "nan", "NaN", ""]:
        label = sub_category
    else:
        label = category

    label = re.sub(r"_?questions?$", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+questions?$", "", label, flags=re.IGNORECASE)

    label = label.replace("_", " ").replace("-", " ")
    label = re.sub(r"\s+", " ", label).strip()
    label = label.title()

    # Common acronym/name fixes.
    label = label.replace("Lgbtq", "LGBTQ")
    label = label.replace("Ccp", "CCP")

    return label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="subtopic_local_censorship.pdf")

    parser.add_argument("--category_col", type=str, default="Category")
    parser.add_argument("--sub_category_col", type=str, default="Sub_Category")
    parser.add_argument("--n_col", type=str, default="n")
    parser.add_argument("--mean_col", type=str, default="mean_d")
    parser.add_argument("--pi_col", type=str, default="pi_plus")
    parser.add_argument("--ci_low_col", type=str, default="ci90_low")
    parser.add_argument("--ci_high_col", type=str, default="ci90_high")

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--min_n", type=int, default=20)
    parser.add_argument("--tau_d", type=float, default=0.20)
    parser.add_argument("--title", type=str, default="Most Severe Subtopics")

    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    required_cols = [
        args.category_col,
        args.sub_category_col,
        args.n_col,
        args.mean_col,
        args.pi_col,
        args.ci_low_col,
        args.ci_high_col,
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in [args.n_col, args.mean_col, args.pi_col, args.ci_low_col, args.ci_high_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[args.n_col, args.mean_col, args.pi_col])
    df = df[df[args.n_col] >= args.min_n].copy()

    if df.empty:
        raise ValueError("No eligible subtopics after filtering.")

    df["severity"] = df[args.mean_col] * df[args.pi_col]

    plot_df = (
        df.sort_values("severity", ascending=False)
        .head(args.top_k)
        .sort_values(args.mean_col, ascending=True)
        .copy()
    )

    labels = [
        clean_label(row[args.category_col], row[args.sub_category_col])
        for _, row in plot_df.iterrows()
    ]

    y_pos = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    norm = plt.Normalize(
        vmin=max(0.0, float(plot_df[args.pi_col].min())),
        vmax=min(1.0, float(plot_df[args.pi_col].max())),
    )
    colors = plt.cm.viridis(norm(plot_df[args.pi_col].values))

    ax.barh(
        y_pos,
        plot_df[args.mean_col].values,
        color=colors,
        alpha=0.90,
    )

    xerr_low = plot_df[args.mean_col].values - plot_df[args.ci_low_col].values
    xerr_high = plot_df[args.ci_high_col].values - plot_df[args.mean_col].values

    ax.errorbar(
        plot_df[args.mean_col].values,
        y_pos,
        xerr=[xerr_low, xerr_high],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
    )

    ax.axvline(
        args.tau_d,
        linestyle="--",
        linewidth=2,
        label=rf"$\tau_d={args.tau_d}$",
    )

    xmax = max(plot_df[args.ci_high_col].max(), plot_df[args.mean_col].max()) + 0.35

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            row[args.ci_high_col] + 0.05,
            i,
            f"n={int(row[args.n_col])}",
            va="center",
            fontsize=10,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)

    ax.set_title(args.title, fontsize=14)
    ax.set_xlabel(r"Mean local gap $\bar{d}$")
    ax.set_xlim(0, xmax)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"Consistency $\pi^+$")

    plt.tight_layout()

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(args.output_file, bbox_inches="tight")
    print(f"Saved figure to: {args.output_file}")


if __name__ == "__main__":
    main()