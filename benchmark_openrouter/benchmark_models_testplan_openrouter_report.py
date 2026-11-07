#!/usr/bin/env python3
"""
Skill Discovery Benchmark -- OpenRouter Report Generator
=========================================================
MIA5150 Project: Agent Skill Discovery at Scale

Generates accuracy, latency, and dashboard plots from OpenRouter benchmark
results stored in a given directory.  Run this after (or alongside)
benchmark_models_testplan_openrouter.py to produce visual reports.

Usage
-----
  python benchmark_models_testplan_openrouter_report.py tests_openrouter
  python benchmark_models_testplan_openrouter_report.py tests_openrouter --completed-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from tabulate import tabulate

from benchmark_models import K_VALUES

# ── Colour palette (auto-assigned, colour-blind-friendly) ────────────────────

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]
_MARKERS = ["o", "s", "v", "^", "D", "P", "X", "*", "h", "<", ">", "p"]

_C_GREY = "#90A4AE"


def _model_style(idx: int) -> tuple[str, str]:
    return _PALETTE[idx % len(_PALETTE)], _MARKERS[idx % len(_MARKERS)]


def _display_name(model_slug: str, model_registry: dict) -> str:
    return model_registry.get(model_slug, {}).get("display", model_slug)


# ── Load data ────────────────────────────────────────────────────────────────

def load_results(run_dir: str) -> tuple[pd.DataFrame, dict, dict]:
    """Load details.csv, run_config.json, and build a model registry."""
    csv_path = os.path.join(run_dir, "details.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("ERROR: details.csv is empty")
        sys.exit(1)

    cfg_path = os.path.join(run_dir, "run_config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)

    # Build model registry from openrouter script
    try:
        from benchmark_models_testplan_openrouter import OPENROUTER_MODELS
        registry = dict(OPENROUTER_MODELS)
    except ImportError:
        registry = {}

    # Fill in any models from CSV not in registry
    for model in df["model"].unique():
        if model not in registry:
            slug_short = model.split("/")[-1] if "/" in model else model
            registry[model] = {"display": slug_short, "context_window": 128_000}

    return df, cfg, registry


def get_completed_combos(df: pd.DataFrame) -> set[tuple[str, int]]:
    """Identify completed combos (no rate-limit or skip errors)."""
    completed: set[tuple[str, int]] = set()
    for (model, lib_size), group in df.groupby(["model", "library_size"]):
        rate_limited = group["rate_limited"].any() if "rate_limited" in group else False
        err_col = group["error"].astype(str).replace("nan", "")
        has_skip = err_col.str.contains(
            "RATE_LIMIT_SKIPPED|HOURLY_STALL_SKIPPED|DAILY_LIMIT",
            na=False, regex=True
        ).any()
        all_errors = err_col.str.len().gt(0).all()
        if not rate_limited and not has_skip and not all_errors:
            completed.add((str(model), int(lib_size)))
    return completed


# ── Plot: Accuracy vs Library Size ───────────────────────────────────────────

def plot_accuracy(df: pd.DataFrame, registry: dict, out: str,
                  models_order: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    for idx, model_name in enumerate(models_order):
        mdf = df[df["model"] == model_name]
        if mdf.empty:
            continue

        summary = mdf.groupby("library_size").agg(
            accuracy=("exact_match", "mean"),
            ctx_limit=("context_limit", "any"),
        ).sort_index()

        sizes = summary.index.values
        acc = summary["accuracy"].values
        ctx = summary["ctx_limit"].values
        color, marker = _model_style(idx)
        display = _display_name(model_name, registry)

        ax.plot(sizes, acc, marker=marker, linestyle="-", color=color,
                lw=2.2, ms=8, label=display, alpha=0.9, zorder=4)

        ctx_idx = np.where(ctx)[0]
        if len(ctx_idx):
            ax.scatter(sizes[ctx_idx], acc[ctx_idx],
                       marker="X", s=200, color=color, zorder=10,
                       edgecolors="white", linewidths=1.5)
            first = sizes[ctx_idx[0]]
            ax.axvline(first, color=color, ls=":", alpha=0.35, lw=1.5)

        for x, y, is_ctx in zip(sizes, acc, ctx):
            if not is_ctx:
                ax.annotate(f"{y:.0%}", (x, y),
                            textcoords="offset points", xytext=(0, 11),
                            ha="center", fontsize=7, fontweight="bold", color=color)

    measured_sizes = sorted(df["library_size"].unique())
    ax.set_xlabel("Library Size (number of skills)", fontsize=12)
    ax.set_ylabel("Exact Match Accuracy", fontsize=12)
    ax.set_ylim(-0.08, 1.15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xscale("log")
    ax.set_xticks(measured_sizes)
    ax.set_xticklabels([str(s) for s in measured_sizes])
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9,
              ncol=max(1, len(models_order) // 6))
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.grid(axis="x", alpha=0.1, linestyle="--", which="both")

    ax.set_title("Skill Discovery: Accuracy vs Library Size (OpenRouter)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.text(0.5, 1.015,
            "X markers = context window limit | full descriptions (up to 1024 chars)",
            transform=ax.transAxes, ha="center", fontsize=8, color=_C_GREY)

    fig.tight_layout()
    fig.savefig(os.path.join(out, "comparison_accuracy.png"), dpi=200)
    plt.close(fig)
    print(f"  comparison_accuracy.png")


# ── Plot: Latency vs Library Size ────────────────────────────────────────────

def plot_latency(df: pd.DataFrame, registry: dict, out: str,
                 models_order: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    valid_df = df[(~df["context_limit"]) & (df["error"].fillna("") == "")]

    for idx, model_name in enumerate(models_order):
        mdf = valid_df[valid_df["model"] == model_name]
        if mdf.empty:
            continue

        summary = mdf.groupby("library_size").agg(
            avg=("latency_s", "mean"),
            p50=("latency_s", "median"),
            p95=("latency_s", lambda x: np.percentile(x, 95)),
            p05=("latency_s", lambda x: np.percentile(x, 5)),
        ).sort_index()

        sizes = summary.index.values
        avg_lat = summary["avg"].values
        p95 = summary["p95"].values
        p05 = summary["p05"].values
        color, marker = _model_style(idx)
        display = _display_name(model_name, registry)

        ax.plot(sizes, avg_lat, marker=marker, ls="-", color=color,
                lw=2.2, ms=7, label=f"{display} (avg)", alpha=0.9)
        ax.fill_between(sizes, p05, p95, color=color, alpha=0.10)

        for x, y in zip(sizes, avg_lat):
            ax.annotate(f"{y:.1f}s", (x, y),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7, color=color)

    measured_sizes = sorted(df["library_size"].unique())
    ax.set_xlabel("Library Size (number of skills)", fontsize=12)
    ax.set_ylabel("Latency (seconds)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(measured_sizes)
    ax.set_xticklabels([str(s) for s in measured_sizes])
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mtick.NullFormatter())
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9,
              ncol=max(1, len(models_order) // 6))
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.grid(axis="x", alpha=0.1, linestyle="--", which="both")

    ax.set_title("Skill Discovery: Latency vs Library Size (OpenRouter)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.text(0.5, 1.015,
            "Shaded region = p5 - p95 range | context-limited sizes excluded",
            transform=ax.transAxes, ha="center", fontsize=8, color=_C_GREY)

    fig.tight_layout()
    fig.savefig(os.path.join(out, "comparison_latency.png"), dpi=200)
    plt.close(fig)
    print(f"  comparison_latency.png")


# ── Plot: 4-panel Dashboard ──────────────────────────────────────────────────

def plot_dashboard(df: pd.DataFrame, registry: dict, out: str,
                   models_order: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    lib_sizes = sorted(df["library_size"].unique())

    # Top-left: Accuracy
    ax = axes[0, 0]
    for idx, mn in enumerate(models_order):
        mdf = df[df["model"] == mn]
        s = mdf.groupby("library_size")["exact_match"].mean().sort_index()
        color, marker = _model_style(idx)
        ax.plot(s.index, s.values, marker=marker, ls="-", color=color,
                lw=2, ms=5, label=_display_name(mn, registry))
    ax.set_xlabel("Library Size")
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("Accuracy Degradation", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(lib_sizes)
    ax.set_xticklabels([str(s) for s in lib_sizes], fontsize=7)
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(fontsize=6, loc="lower left", ncol=max(1, len(models_order) // 4))
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_ylim(-0.05, 1.1)

    # Top-right: Latency
    ax = axes[0, 1]
    valid_df = df[(~df["context_limit"]) & (df["error"].fillna("") == "")]
    for idx, mn in enumerate(models_order):
        mdf = valid_df[valid_df["model"] == mn]
        if mdf.empty:
            continue
        s = mdf.groupby("library_size")["latency_s"].mean().sort_index()
        color, marker = _model_style(idx)
        ax.plot(s.index, s.values, marker=marker, ls="-", color=color,
                lw=2, ms=5, label=_display_name(mn, registry))
    ax.set_xlabel("Library Size")
    ax.set_ylabel("Avg Latency (s)")
    ax.set_title("Latency Scaling", fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(lib_sizes)
    ax.set_xticklabels([str(s) for s in lib_sizes], fontsize=7)
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mtick.NullFormatter())
    ax.legend(fontsize=6, loc="upper left", ncol=max(1, len(models_order) // 4))
    ax.grid(alpha=0.2, linestyle="--")

    # Bottom-left: Context Feasibility heatmap
    ax = axes[1, 0]
    matrix = []
    for mn in models_order:
        row = []
        for ls in lib_sizes:
            subset = df[(df["model"] == mn) & (df["library_size"] == ls)]
            if subset.empty:
                row.append(np.nan)
            elif subset["context_limit"].any():
                row.append(0)
            else:
                row.append(1)
        matrix.append(row)

    matrix_arr = np.array(matrix, dtype=float)
    ax.imshow(matrix_arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(lib_sizes)))
    ax.set_xticklabels([str(s) for s in lib_sizes], fontsize=7, rotation=45)
    ax.set_yticks(range(len(models_order)))
    ax.set_yticklabels([_display_name(m, registry) for m in models_order],
                       fontsize=7)
    ax.set_xlabel("Library Size")
    ax.set_title("Context Window Feasibility", fontweight="bold")

    for i in range(len(models_order)):
        for j in range(len(lib_sizes)):
            val = matrix_arr[i, j]
            txt = "-" if np.isnan(val) else ("LIMIT" if val == 0 else "OK")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    fontweight="bold",
                    color="white" if (not np.isnan(val) and val == 0) else "black")

    # Bottom-right: Outcome distribution
    ax = axes[1, 1]
    err_data = []
    for mn in models_order:
        mdf = df[df["model"] == mn]
        total = len(mdf)
        if total == 0:
            continue
        ctx_n = int(mdf["context_limit"].sum())
        rate_n = int(mdf["rate_limited"].sum()) if "rate_limited" in mdf else 0
        other_n = int(((mdf["error"].fillna("") != "") & (~mdf["context_limit"])
                       & (~mdf.get("rate_limited", False))).sum())
        ok_n = total - ctx_n - rate_n - other_n
        err_data.append({
            "model": _display_name(mn, registry),
            "success": ok_n / total,
            "ctx_limit": ctx_n / total,
            "rate_limited": rate_n / total,
            "other_error": other_n / total,
        })

    if err_data:
        edf = pd.DataFrame(err_data)
        x = range(len(edf))
        w = 0.6
        bottom = np.zeros(len(edf))
        ax.bar(x, edf["success"], w, bottom=bottom,
               label="Success", color="#2E7D32")
        bottom += edf["success"].values
        ax.bar(x, edf["ctx_limit"], w, bottom=bottom,
               label="Context Limit", color="#FF7043")
        bottom += edf["ctx_limit"].values
        ax.bar(x, edf["rate_limited"], w, bottom=bottom,
               label="Rate Limited", color="#FFCA28")
        bottom += edf["rate_limited"].values
        ax.bar(x, edf["other_error"], w, bottom=bottom,
               label="Other Error", color="#90A4AE")
        ax.set_xticks(list(x))
        ax.set_xticklabels(edf["model"], fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("Proportion of Queries")
        ax.set_title("Query Outcome Distribution", fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.08)

    fig.suptitle("Skill Discovery Benchmark -- OpenRouter Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)

    fig.tight_layout()
    fig.savefig(os.path.join(out, "comparison_dashboard.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  comparison_dashboard.png")


# ── Plot: Cost breakdown ────────────────────────────────────────────────────

def plot_cost(df: pd.DataFrame, registry: dict, out: str,
              models_order: list[str]) -> None:
    """Bar chart of estimated cost per model."""
    if "est_cost_usd" not in df.columns:
        return

    valid_df = df[df["est_cost_usd"] > 0]
    if valid_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    costs = []
    for mn in models_order:
        mdf = valid_df[valid_df["model"] == mn]
        costs.append({
            "model": _display_name(mn, registry),
            "total": mdf["est_cost_usd"].sum(),
            "per_query": mdf["est_cost_usd"].mean(),
        })

    cdf = pd.DataFrame(costs)
    if cdf["total"].sum() == 0:
        plt.close(fig)
        return

    x = range(len(cdf))
    colors = [_model_style(i)[0] for i in range(len(cdf))]
    ax.bar(x, cdf["total"], color=colors, alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cdf["model"], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Estimated Cost (USD)")
    ax.set_title("Total Estimated Cost per Model", fontweight="bold")

    for i, (_, row) in enumerate(cdf.iterrows()):
        if row["total"] > 0:
            ax.text(i, row["total"], f"${row['total']:.4f}",
                    ha="center", va="bottom", fontsize=7)

    ax.grid(axis="y", alpha=0.2, linestyle="--")

    fig.tight_layout()
    fig.savefig(os.path.join(out, "comparison_cost.png"), dpi=200)
    plt.close(fig)
    print(f"  comparison_cost.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate plots from OpenRouter benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("result_dir",
                    help="Directory containing details.csv and run_config.json")
    ap.add_argument("--completed-only", action="store_true",
                    help="Only plot data from fully completed combos")
    ap.add_argument("--all", action="store_true",
                    help="Include models with incomplete data (partial query counts). "
                         "By default only models with full data (20 queries per combo) are shown.")
    ap.add_argument("--exclude", type=str, default="",
                    help="Comma-separated model slugs to exclude, "
                         "e.g. --exclude anthropic/claude-sonnet-5.5")
    args = ap.parse_args()

    run_dir = args.result_dir
    df, cfg, registry = load_results(run_dir)

    # Determine model order (preserve config order, then alphabetical extras)
    cfg_models = cfg.get("models", [])
    csv_models = sorted(df["model"].unique())
    models_order = [m for m in cfg_models if m in set(csv_models)]
    models_order += [m for m in csv_models if m not in set(models_order)]

    # Exclude explicitly listed models
    if args.exclude:
        excluded = {s.strip() for s in args.exclude.split(",") if s.strip()}
        df = df[~df["model"].isin(excluded)]
        models_order = [m for m in models_order if m not in excluded]
        if excluded:
            print(f"  Excluded: {', '.join(excluded)}")

    # By default, only keep models with full data (20 queries per combo).
    # Use --all to include partial models.
    if not args.all:
        full_models: list[str] = []
        partial_models: list[str] = []
        for m in models_order:
            mdf = df[df["model"] == m]
            sizes = mdf["library_size"].unique()
            min_queries = mdf.groupby("library_size").size().min() if len(sizes) > 0 else 0
            if min_queries >= 20:
                full_models.append(m)
            else:
                partial_models.append(m)
        if partial_models:
            names = [_display_name(m, registry) for m in partial_models]
            print(f"  Excluding partial-data models: {', '.join(names)} "
                  f"(use --all to include)")
            df = df[~df["model"].isin(set(partial_models))]
            models_order = full_models

    if args.completed_only:
        completed = get_completed_combos(df)
        df = df[df.apply(
            lambda r: (r["model"], int(r["library_size"])) in completed,
            axis=1,
        )]
        if df.empty:
            print("No completed combos -- nothing to plot.")
            return
        print(f"Plotting {len(completed)} completed combos only")

    print(f"\nGenerating plots in {run_dir}/ ...")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })

    plot_accuracy(df, registry, run_dir, models_order)
    plot_latency(df, registry, run_dir, models_order)
    plot_dashboard(df, registry, run_dir, models_order)
    plot_cost(df, registry, run_dir, models_order)

    # Print summary table
    summary = df.groupby(["model"]).agg(
        combos=("library_size", "nunique"),
        queries=("query", "count"),
        accuracy=("exact_match", "mean"),
        avg_lat=("latency_s", "mean"),
        cost=("est_cost_usd", "sum"),
    ).reset_index()
    summary["display"] = summary["model"].map(
        lambda m: _display_name(m, registry)
    )

    show = ["display", "combos", "queries", "accuracy", "avg_lat", "cost"]
    hdrs = ["Model", "Combos", "Queries", "Accuracy", "Avg Lat(s)", "Cost $"]
    print(f"\n{tabulate(summary[show], headers=hdrs, tablefmt='github', floatfmt='.3f', showindex=False)}")

    print(f"\nDone. Plots saved to {run_dir}/")


if __name__ == "__main__":
    main()
