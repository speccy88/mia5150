#!/usr/bin/env python3
"""Render audited CSV results as deterministic 1920×1080 PNG and editable SVG.

This module never calls an API, imports a historical report generator, reads raw
reported metrics, or fills unavailable measures with zero. Metrics must come from
this study's independent audit. See VISUALIZATIONS.md for interpretation limits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mia5150-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "mia5150-cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

BASE_COMMIT = "3fb7e66a1a219bc43d7c2057dca3d1c70a3c6c45"
STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
PRIMARY = "no_reported_invalid"
SENSITIVITY = "retained_ranking"
WIDTH, HEIGHT, DPI = 16, 9, 120
INK, MUTED, GRID, BLUE, ORANGE = "#172B4D", "#53667D", "#DCE3EC", "#2056A8", "#A95115"
MODEL_NAMES = {
    "deepseek/deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "moonshotai/kimi-k2.5": "Kimi K2.5",
    "qwen/qwen3.8-flash": "Qwen 3.8 Flash",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "openai/gpt-6.1-terra": "GPT-6.1 Terra",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
    "moonshotai/kimi-k3.1-pro": "Kimi K3.1 Pro",
    "deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "svg.fonttype": "none", "svg.hashsalt": "mia5150-audit",
    "axes.axisbelow": True, "lines.linewidth": 2.1,
})


def number(value):
    """Parse missing as NaN; zero remains a genuine measured zero."""
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan", "na"}:
        return math.nan
    return float(value)


def finite(value):
    return math.isfinite(number(value))


def load_summary(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"model", "library_size", "cohort", "n_expected", "n_completed", "n_scored", "top1_accuracy"}
    if not rows or not required <= rows[0].keys():
        raise ValueError(f"Audit summary missing required fields: {required - (rows[0].keys() if rows else set())}")
    keys = [(r["model"], r["library_size"], r["cohort"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate model/size/cohort rows are not chartable")
    for row in rows:
        if number(row["n_scored"]) > number(row["n_completed"]):
            raise ValueError("Scored denominator exceeds completed denominator")
        for field in ["top1_accuracy", "recall@5", "recall@10", "ndcg@5", "ndcg@10", "mrr@5", "mrr@10"]:
            if field in row and finite(row[field]) and not 0 <= number(row[field]) <= 1:
                raise ValueError(f"Metric outside [0,1]: {field}")
        if number(row["n_scored"]) == 0 and finite(row.get("top1_accuracy")):
            raise ValueError("Empty quality cohort must not be plotted as a measured score")
    return rows


def index_rows(rows):
    return {(r["model"], str(r["library_size"]), r["cohort"]): r for r in rows}


def get(index, model, size, cohort=PRIMARY):
    try:
        return index[(model, str(size), cohort)]
    except KeyError as exc:
        raise ValueError(f"Missing expected audit cell: {model}, {size}, {cohort}") from exc


def values(index, model, sizes, field, cohort=PRIMARY):
    return np.array([number(get(index, model, n, cohort).get(field)) for n in sizes])


def canvas(title, subtitle, note):
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    fig.text(.045, .948, title, size=27, weight="bold", va="top")
    fig.text(.045, .89, subtitle, size=13.5, color=MUTED, va="top")
    fig.text(.045, .076, note, size=10.4, color=MUTED, va="top", linespacing=1.5)
    fig.text(.955, .012, "MIA5150  ·  independent offline audit  ·  3fb7e66", size=9,
             color=MUTED, ha="right")
    return fig


def percentage_axis(ax):
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=.65)


def size_axis(ax, sizes, denominators=None):
    ax.set_xscale("log")
    ax.set_xlim(min(sizes) / 1.17, max(sizes) * 1.17)
    ax.set_xticks(sizes)
    labels = [str(n) for n in sizes]
    if denominators is not None:
        labels = [f"{n}\n{s}/{c}" for n, (s, c) in zip(sizes, denominators)]
    ax.set_xticklabels(labels, size=10)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis="x", length=3)
    ax.tick_params(axis="y", labelsize=10)


def facet_axes(fig, top=.805):
    return fig.subplots(3, 3, gridspec_kw={"left": .065, "right": .97, "bottom": .16,
                                        "top": top, "hspace": .66, "wspace": .23}).ravel()


def draw_interval(ax, xs, row_values, lows, highs, color, label=None, marker="o"):
    ax.plot(xs, row_values, color=color, marker=marker, markersize=4.6, label=label)
    # Retain uncertainty only when actually supplied by the auditor.
    valid = np.isfinite(row_values) & np.isfinite(lows) & np.isfinite(highs)
    ax.fill_between(xs, np.where(valid, lows, np.nan), np.where(valid, highs, np.nan),
                    color=color, alpha=.12, linewidth=0)


def plot_accuracy(index, models, sizes):
    fig = canvas("Top-1 accuracy across all nine models",
                 "Primary identifiable subset; retained-ranking sensitivity shown separately. Library size uses a logarithmic axis.",
                 "Solid / shaded: no reported invalid skills, with audit 95% intervals. Dashed: retained-ranking sensitivity, not recovered raw response order.\n"
                 "Under each size: primary n / completed n; 20 queries planned per cell. Blank line segments are unavailable, never zero. Intervals do not capture parser uncertainty.")
    for ax, model in zip(facet_axes(fig), models):
        draw_interval(ax, sizes, values(index, model, sizes, "top1_accuracy"),
                      values(index, model, sizes, "top1_accuracy_ci_low"),
                      values(index, model, sizes, "top1_accuracy_ci_high"), BLUE)
        ax.plot(sizes, values(index, model, sizes, "top1_accuracy", SENSITIVITY), color=ORANGE,
                linestyle="--", marker="s", markersize=3.5, markerfacecolor="white", linewidth=1.5)
        ax.set_title(MODEL_NAMES.get(model, model), loc="left", size=12.5, weight="bold", pad=7)
        percentage_axis(ax)
        denominator = [(int(number(get(index, model, n)["n_scored"])),
                        int(number(get(index, model, n)["n_completed"]))) for n in sizes]
        size_axis(ax, sizes, denominator)
    return fig


def plot_ranking_trends(index, models, sizes):
    fig = canvas("Recall@5 and NDCG@5 by library size",
                 "All nine models · primary subset with no reported invalid skill · higher is better for both metrics",
                 "Blue circles: Recall@5. Orange squares: NDCG@5. Shading shows audit 95% intervals, not uncertainty from lost raw response positions.\n"
                 "Under each size: primary n / completed n; 20 queries planned per cell. Metrics are conditional on this identifiable subset, not full-population estimates.")
    for ax, model in zip(facet_axes(fig), models):
        for metric, color, marker in [("recall@5", BLUE, "o"), ("ndcg@5", ORANGE, "s")]:
            draw_interval(ax, sizes, values(index, model, sizes, metric),
                          values(index, model, sizes, metric + "_ci_low"),
                          values(index, model, sizes, metric + "_ci_high"), color, marker=marker)
        ax.set_title(MODEL_NAMES.get(model, model), loc="left", size=12.5, weight="bold", pad=7)
        percentage_axis(ax)
        denominator = [(int(number(get(index, model, n)["n_scored"])),
                        int(number(get(index, model, n)["n_completed"]))) for n in sizes]
        size_axis(ax, sizes, denominator)
    return fig


def plot_latency(index, models, sizes):
    fig = canvas("Successful-response latency: median and p95",
                 "Recorded end-to-end request timing, in seconds. All panels use the same logarithmic time scale.",
                 "Median: solid circles. p95: dashed squares. Each point uses retained completed rows; context placeholders and absent requests are excluded.\n"
                 "Under each size: completed n; 20 planned per cell. With at most 20 observations per cell, p95 is a descriptive tail estimate, not a stable production SLA.")
    max_y = max([number(get(index, model, size, SENSITIVITY).get("e2e_latency_p95_s"))
                 for model in models for size in sizes if finite(get(index, model, size, SENSITIVITY).get("e2e_latency_p95_s"))] or [1])
    min_y = min([number(get(index, model, size, SENSITIVITY).get("e2e_latency_median_s"))
                 for model in models for size in sizes if finite(get(index, model, size, SENSITIVITY).get("e2e_latency_median_s")) and number(get(index, model, size, SENSITIVITY).get("e2e_latency_median_s")) > 0] or [.1])
    for ax, model in zip(facet_axes(fig), models):
        for metric, linestyle, marker in [("e2e_latency_median_s", "-", "o"), ("e2e_latency_p95_s", "--", "s")]:
            ys = values(index, model, sizes, metric, SENSITIVITY)
            ys[ys <= 0] = np.nan
            ax.plot(sizes, ys, color=BLUE if linestyle == "-" else ORANGE,
                    linestyle=linestyle, marker=marker, markersize=4)
        ax.set_title(MODEL_NAMES.get(model, model), loc="left", size=12.5, weight="bold", pad=7)
        ax.set_yscale("log")
        ax.set_ylim(min_y / 1.5, max_y * 1.5)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}s"))
        ax.grid(axis="y", color=GRID, linewidth=.65)
        size_axis(ax, sizes)
        ax.set_xticklabels([f"{n}\nn={int(number(get(index, model, n, SENSITIVITY)['n_completed']))}" for n in sizes], size=10)
    return fig


def coverage_state(row):
    expected, completed = int(number(row["n_expected"])), int(number(row["n_completed"]))
    blocked = int(number(row.get("n_context_excluded", 0)))
    absent = int(number(row.get("n_unattempted", 0)))
    if completed == expected:
        return "complete", f"{completed}/{expected}"
    if completed > 0:
        return "partial", f"{completed}/{expected}\n{absent} absent" if absent else f"{completed}/{expected}\npartial"
    if blocked == expected:
        return "blocked", f"0/{expected}\ncontext"
    return "unattempted", f"0/{expected}\nno result"


def plot_coverage(index, models, sizes):
    fig = canvas("Coverage of the planned model × size grid",
                 "Completed responses / 20 planned queries in each cell. Absence and context rejection remain distinct from measured failure.",
                 "Complete = all planned responses retained. Partial = fewer responses retained. Context = recorded placeholders, not executed zero-quality answers.\n"
                 "Absent rows cannot prove a request was never attempted; “no result” means no retained response. See the audit provenance for run-history limitations.")
    ax = fig.add_axes([.22, .2, .74, .61])
    colors = {"complete": "#D6E6F7", "partial": "#F4DFC8", "blocked": "#E6DDEB", "unattempted": "#F0F2F5"}
    hatches = {"complete": None, "partial": "/", "blocked": "xx", "unattempted": ".."}
    for r, model in enumerate(models):
        for c, size in enumerate(sizes):
            state, label = coverage_state(get(index, model, size, SENSITIVITY))
            ax.add_patch(Rectangle((c-.5, r-.5), 1, 1, facecolor=colors[state], edgecolor="white", lw=3,
                                   hatch=hatches[state]))
            ax.text(c, r, label, ha="center", va="center", size=12, weight="bold", linespacing=1.3)
    ax.set_xlim(-.5, len(sizes)-.5)
    ax.set_ylim(len(models)-.5, -.5)
    ax.set_xticks(range(len(sizes)), [f"N={n:,}" for n in sizes])
    ax.set_yticks(range(len(models)), [MODEL_NAMES.get(m, m) for m in models])
    ax.tick_params(axis="both", length=0, pad=9)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, state in enumerate(colors):
        x = .22 + i * .19
        fig.add_artist(Rectangle((x, .12), .015, .023, transform=fig.transFigure,
                                 facecolor=colors[state], edgecolor=MUTED, hatch=hatches[state]))
        fig.text(x+.021, .131, {"complete": "Complete", "partial": "Partial", "blocked": "Context blocked", "unattempted": "No retained result"}[state], va="center", size=11)
    return fig


def plot_quality_matrix(index, models):
    metrics = ["recall@5", "ndcg@5", "mrr@5", "recall@10", "ndcg@10", "mrr@10"]
    fig = canvas("Ranking quality at K=5 and K=10",
                 "Same identifiable size–query cells for all nine models. Configured model order is preserved; no claim of a statistically significant winner.",
                 "Primary common panel: no reported invalid skill for any model. n counts shared size–query cells, not independent query IDs; repeated queries remain clustered.\n"
                 "One gold skill per query: Recall = Hit rate; MRR = MAP. Precision = Recall/K. 95% intervals and sensitivity are in common_panel_summary.csv.")
    ax = fig.add_axes([.28, .18, .61, .64])
    data = np.array([[number(get(index, m, "ALL").get(metric)) for metric in metrics] for m in models])
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", vmin=0, vmax=1, cmap="Blues")
    for r in range(len(models)):
        for c in range(len(metrics)):
            v = data[r, c]
            ax.text(c, r, "NA" if not np.isfinite(v) else f"{v:.1%}", ha="center", va="center", size=13,
                    color="white" if np.isfinite(v) and v > .63 else INK, weight="bold")
    labels = [f"{MODEL_NAMES.get(m, m)}   n={int(number(get(index, m, 'ALL')['n_scored']))}" for m in models]
    ax.set_yticks(range(len(models)), labels, size=11.5)
    ax.set_xticks(range(len(metrics)), ["Recall@5", "NDCG@5", "MRR@5", "Recall@10", "NDCG@10", "MRR@10"], size=12)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=10)
    ax.axvline(2.5, color="white", lw=5)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", axis="y", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    cax = fig.add_axes([.915, .18, .012, .64])
    fig.colorbar(im, cax=cax, format=ticker.PercentFormatter(1), ticks=[0, .25, .5, .75, 1])
    return fig


def plot_cost(index, models):
    fig = canvas("Quality versus recomputed estimated cost",
                 "Estimated token-price cost only. No auditable per-request provider charge is available in the retained ledger.",
                 "Both axes use the same common identifiable size–query panel for all nine models. x: estimated US dollars per retained request. y: top-1 accuracy.\n"
                 "Vertical lines: clustered-query audit 95% intervals. This is a selected subset. Historical “actual” totals lack matched billing evidence; missing actual cost is not US$0.")
    ax = fig.add_axes([.08, .19, .57, .61])
    quality, cost = [], []
    for i, model in enumerate(models):
        primary = get(index, model, "ALL")
        retained = get(index, model, "ALL", PRIMARY)
        n = number(retained["n_completed"])
        x = number(retained.get("estimated_cost_usd")) / n if n else math.nan
        y = number(primary.get("top1_accuracy"))
        quality.append(y); cost.append(x)
        if not np.isfinite(x) or not np.isfinite(y) or x <= 0:
            continue
        lo, hi = number(primary.get("top1_accuracy_ci_low")), number(primary.get("top1_accuracy_ci_high"))
        if np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(x, y, yerr=[[max(0, y-lo)], [max(0, hi-y)]], fmt="none", color=BLUE, alpha=.35, capsize=3, linewidth=1)
        ax.scatter([x], [y], s=65, color=BLUE, edgecolors="white", linewidths=1.4, zorder=3)
        offset = {0: (9, 10), 1: (9, 10), 2: (10, 20), 3: (9, 10), 4: (9, -15), 5: (9, 10), 6: (9, 10), 7: (9, 10), 8: (-11, -18)}.get(i, (9, 10))
        ax.annotate(str(i+1), (x,y), textcoords="offset points", xytext=offset, ha="center", va="center", color=BLUE, size=12, weight="bold", arrowprops={"arrowstyle": "-", "color": BLUE, "lw": .7}, zorder=4)
    ax.set_xscale("log")
    valid_cost = [x for x in cost if np.isfinite(x) and x > 0]
    low, high = min(valid_cost) / 1.5, max(valid_cost) * 1.5
    ax.set_xlim(low, high)
    ax.set_xticks([base * 10.0**exponent for exponent in range(-8, 4) for base in [1, 2, 5] if low <= base * 10.0**exponent <= high])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Estimated cost per completed request (US$, logarithmic scale)", labelpad=12, size=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:g}"))
    percentage_axis(ax)
    ax.set_ylabel("Top-1 accuracy, primary subset", labelpad=10, size=12)
    ax.grid(axis="x", color=GRID, linewidth=.65)
    for i, model in enumerate(models):
        primary = get(index, model, "ALL")
        retained = get(index, model, "ALL", PRIMARY)
        y = .795 - i * .06
        fig.text(.68, y, f"{i+1:02d}  {MODEL_NAMES.get(model, model)}", size=12, weight="bold")
        fig.text(.708, y-.026, f"quality n={int(number(primary['n_scored']))}; cost n={int(number(retained['n_completed']))}", size=10.5, color=MUTED)
    fig.text(.68, .205, "ACTUAL BILLING\nNot independently recoverable", size=12, color=ORANGE, weight="bold", linespacing=1.7)
    return fig


def unavailable_figure(title, subtitle, heading, details, note):
    fig = canvas(title, subtitle, note)
    fig.text(.075, .76, "NOT IDENTIFIABLE FROM THIS SNAPSHOT", size=13, weight="bold", color=ORANGE)
    fig.text(.075, .68, heading, size=26, weight="bold", linespacing=1.35, va="top")
    for i, detail in enumerate(details):
        fig.text(.085, .47 - .10*i, "•  " + detail, size=17, color=MUTED, linespacing=1.4)
    return fig



RETRIEVAL_METHODS = ["bm25", "vector", "hybrid_rrf60", "domain_vector_centroid"]
METHOD_STYLES = {
    "prompt": ("Prompt", INK, "o", "-"),
    "bm25": ("BM25", BLUE, "s", "-"),
    "vector": ("BGE vector", ORANGE, "^", "-"),
    "hybrid_rrf60": ("Hybrid RRF60", "#65751C", "D", "-"),
    "domain_vector_centroid": ("Centroid domain (new)", "#A84079", "x", "--"),
}


def build_matched_rows(prompt_path, retrieval_path, models, sizes):
    """Join independently scored rows by exact archived query index and size.

    All baseline rows in a facet use the prompt model's actual primary subset.
    Name-equivalent single-target scoring matches the historical name labels;
    a strict UUID result is not silently substituted.
    """
    prompts = [json.loads(line) for line in prompt_path.read_text().splitlines() if line.strip()]
    with retrieval_path.open(newline="") as handle:
        retrieval = list(csv.DictReader(handle))
    lookup = {}
    for row in retrieval:
        if int(row["k"]) != 5 or row["label_contract"] != "name_equivalent_single_target":
            continue
        key = (row["method"], int(row["size"]), int(row["query_index"]))
        if key in lookup:
            raise ValueError(f"Duplicate retrieval key {key}")
        lookup[key] = row
    result = []
    for model in models:
        for size in sizes:
            matched = [r for r in prompts if r["model"] == model and int(r["library_size"]) == size
                       and r["status"] == "completed" and r["scores"].get(PRIMARY) is not None]
            query_indices = [int(r["query_id"][1:])-1 for r in matched]
            if len(set(query_indices)) != len(query_indices) or any(q < 0 or q >= 20 for q in query_indices):
                raise ValueError("Prompt query IDs are not a unique subset of q01..q20")
            for method in ["prompt"] + RETRIEVAL_METHODS:
                record = {"model": model, "library_size": size, "method": method,
                          "n": len(matched), "query_indices": query_indices,
                          "label_contract": "name_equivalent_single_target",
                          "historical_candidate_match": "conditional"}
                for metric in ["recall", "ndcg", "mrr"]:
                    if method == "prompt":
                        scores = [r["scores"][PRIMARY][metric + "@5"] for r in matched]
                    else:
                        missing = [(method, size, q) for q in query_indices if (method, size, q) not in lookup]
                        if missing:
                            raise ValueError(f"Retrieval matching incomplete: {missing}")
                        scores = [number(lookup[(method, size, q)][metric]) for q in query_indices]
                    if any(not 0 <= v <= 1 for v in scores):
                        raise ValueError("Invalid matched metric value")
                    record[metric + "@5"] = float(np.mean(scores)) if scores else None
                result.append(record)
    return result


def plot_matched_retrieval(rows, models, sizes, metric):
    label = {"recall@5": "Recall@5", "ndcg@5": "NDCG@5"}[metric]
    fig = canvas(f"{label}: prompt and reconstructed retrieval",
                 "Same query identities within each model–size panel. Historical candidate-catalog equivalence remains conditional.",
                 "n under each size is shared by all five methods in that panel; model subsets differ. Descriptive single-name-target scores, with no significance claim.\n"
                 "The historical domain router is unavailable. Dashed centroid-domain routing is a new label-free supplement, not a recovered historical result.")
    lookup = {(r["model"], r["library_size"], r["method"]): r for r in rows}
    handles = []
    for method, (name, color, marker, style) in METHOD_STYLES.items():
        handles.append(Line2D([], [], color=color, marker=marker, linestyle=style, markersize=5, label=name))
    fig.legend(handles=handles, ncol=5, loc="center", bbox_to_anchor=(.51, .839), frameon=False, fontsize=11)
    for ax, model in zip(facet_axes(fig, top=.785), models):
        for method, (name, color, marker, style) in METHOD_STYLES.items():
            ys = [number(lookup[(model, n, method)][metric]) for n in sizes]
            ax.plot(sizes, ys, color=color, marker=marker, linestyle=style, markersize=4,
                    linewidth=2.3 if method == "prompt" else 1.5, alpha=1 if method == "prompt" else .85,
                    zorder=4 if method == "prompt" else 3)
        ax.set_title(MODEL_NAMES.get(model, model), loc="left", size=12.5, weight="bold", pad=7)
        percentage_axis(ax)
        size_axis(ax, sizes)
        ax.set_xticklabels([f"{n}\nn={lookup[(model, n, 'prompt')]['n']}" for n in sizes], size=10)
    return fig


def plot_jev_gap(evidence):
    if evidence.get("matching_completed_jev_cells") != 0:
        raise ValueError("Non-empty Jev evidence needs an explicit matched-pair implementation; refusing an invented gain")
    return unavailable_figure(
        "Jev: no matched gain can be estimated",
        "Bounded repository evidence search for the exact 20-query OpenRouter panel.",
        "No completed Jev cells satisfy\nthe matched-evidence contract.",
        ["Historical scores are not reused across different query panels.",
         "Text fragments in requests do not establish completed matching results.",
         "Fresh paid reranking was not run; a new budget would be required."],
        "Source: matched-retrieval results/jev-audit.json. No bar at zero: unavailable evidence is not a measured zero gain.\n"
        "The bounded search does not establish that unarchived results never existed; candidate identity and schema would also need verification.")


def save_figure(fig, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    # Never use bbox_inches='tight': it changes the promised 16:9 canvas.
    png, svg = output / f"{stem}.png", output / f"{stem}.svg"
    fig.savefig(png, dpi=DPI, metadata={"Software": "MIA5150 independent audit visualizations"})
    fig.savefig(svg, metadata={"Date": None, "Creator": "MIA5150 independent audit visualizations"})
    # Matplotlib emits harmless trailing spaces in SVG path data; normalize them
    # while retaining newline separators, then hash the actual delivered bytes.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    # Renderer-aware canvas-bound check for every visible text element.
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    outside = []
    hidden_ticks = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if tick.get_loc() < lo or tick.get_loc() > hi:
                    hidden_ticks.update([tick.label1, tick.label2])
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text() or text in hidden_ticks:
            continue
        bounds = text.get_window_extent(fig.canvas.get_renderer())
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > width+1 or bounds.y1 > height+1:
            outside.append(text.get_text())
    plt.close(fig)
    if outside:
        raise ValueError(f"Text outside 16:9 canvas in {stem}: {outside}")
    return {"name": stem, "png": png.name, "svg": svg.name,
            "width_px": int(WIDTH*DPI), "height_px": int(HEIGHT*DPI),
            "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
            "svg_sha256": hashlib.sha256(svg.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=STUDY / "results/summary.csv")
    parser.add_argument("--common-summary", type=Path, default=STUDY / "results/common_panel_summary.csv")
    parser.add_argument("--prompt-rows", type=Path, default=STUDY / "results/per_query.jsonl")
    parser.add_argument("--retrieval-results", type=Path, default=STUDY.parent / "openrouter-matched-retrieval-20261108/results")
    parser.add_argument("--config", type=Path, default=ROOT / "benchmark_openrouter/tests_openrouter/run_config.json")
    parser.add_argument("--output", type=Path, default=STUDY / "results/figures")
    args = parser.parse_args()
    rows = load_summary(args.summary)
    config = json.loads(args.config.read_text())
    models, sizes = config["models"], config["library_sizes"]
    if len(models) != 9:
        raise ValueError("This audited layout expects the nine configured models")
    index = index_rows(rows)
    common_rows = load_summary(args.common_summary)
    common_index = index_rows(common_rows)
    charts = [("01_accuracy_by_size", plot_accuracy(index, models, sizes)),
              ("02_recall_ndcg_at5", plot_ranking_trends(index, models, sizes)),
              ("04_quality_estimated_cost", plot_cost(common_index, models)),
              ("05_latency_median_p95", plot_latency(index, models, sizes)),
              ("06_coverage_grid", plot_coverage(index, models, sizes)),
              ("07_ranking_quality_k5_k10", plot_quality_matrix(common_index, models))]
    extra_sources = {}
    matched_rows = None
    retrieval_csv = args.retrieval_results / "per_query_metrics.csv"
    if retrieval_csv.exists():
        provenance_path = args.retrieval_results / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        query_path = args.config.parent / "test_queries.json"
        if provenance["query_sha256"] != hashlib.sha256(query_path.read_bytes()).hexdigest():
            raise ValueError("Retrieval query-file hash does not match the archived prompt panel")
        if provenance["snapshot"] != BASE_COMMIT:
            raise ValueError("Retrieval snapshot differs from the audited snapshot")
        matched_rows = build_matched_rows(args.prompt_rows, retrieval_csv, models, sizes)
        for row in matched_rows:
            if row["method"] != "prompt":
                continue
            summary_row = get(index, row["model"], row["library_size"])
            if row["n"] != int(number(summary_row["n_scored"])):
                raise ValueError("Matched figure denominator differs from independent prompt summary")
            for metric in ["recall@5", "ndcg@5", "mrr@5"]:
                if row["n"] and not math.isclose(number(row[metric]), number(summary_row[metric]), abs_tol=1e-12):
                    raise ValueError("Matched prompt mean differs from independent audit summary")
        charts.append(("03a_matched_recall5", plot_matched_retrieval(matched_rows, models, sizes, "recall@5")))
        charts.append(("03b_matched_ndcg5", plot_matched_retrieval(matched_rows, models, sizes, "ndcg@5")))
        extra_sources["retrieval_metrics_sha256"] = hashlib.sha256(retrieval_csv.read_bytes()).hexdigest()
        extra_sources["retrieval_provenance_sha256"] = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
        extra_sources["prompt_rows_sha256"] = hashlib.sha256(args.prompt_rows.read_bytes()).hexdigest()
    jev_path = args.retrieval_results / "jev-audit.json"
    if jev_path.exists():
        charts.append(("08_jev_evidence_gap", plot_jev_gap(json.loads(jev_path.read_text()))))
        extra_sources["jev_audit_sha256"] = hashlib.sha256(jev_path.read_bytes()).hexdigest()
    outputs = [save_figure(fig, args.output, stem) for stem, fig in charts]
    manifest = {"schema_version": 1, "base_commit": BASE_COMMIT,
                "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "common_summary_sha256": hashlib.sha256(args.common_summary.read_bytes()).hexdigest(),
                "primary_cohort": PRIMARY, "sensitivity_cohort": SENSITIVITY,
                "format": "16:9 PNG and editable-text SVG", "figures": outputs,
                "comparison_sources": extra_sources,
                "notes": ["No raw-order reconstruction is claimed.", "Missing measures remain absent, never zero.",
                          "Plot sources are independent audit outputs, not historical report aggregates."]}
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (args.output / "figure_data.json").write_text(json.dumps({"models": models, "library_sizes": sizes, "audit_summary_rows": rows, "common_panel_summary_rows": common_rows}, indent=2)+"\n")
    if matched_rows is not None:
        (args.output / "matched_figure_data.json").write_text(json.dumps(matched_rows, indent=2) + "\n")
    print(f"Generated {len(outputs)} figure pairs in {args.output}")


if __name__ == "__main__":
    main()
