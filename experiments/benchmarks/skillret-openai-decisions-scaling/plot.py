"""Render figures and export the exact plotted points from saved benchmark data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import fmean, stdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, PercentFormatter


HERE = Path(__file__).resolve().parent
SIZES = (50, 100, 250, 500, 1000, 2000, 5000)
METHODS = {
    "bm25": ("BM25", "#2379a5", "o"),
    "vector": ("Vector", "#24967e", "s"),
    "hybrid": ("Hybrid", "#c24c65", "^"),
}
PROMPT_MODELS = {
    "deepseek/deepseek-v4-flash-0731": ("DeepSeek V4 Flash", "#2379a5", "o"),
    "moonshotai/kimi-k2.5": ("Kimi K2.5", "#24967e", "s"),
    "qwen/qwen3.8-flash": ("Qwen 3.8 Flash", "#c24c65", "^"),
    "openai/gpt-6-luna": ("GPT-6 Luna", "#8f559e", "D"),
    "openai/gpt-6.1-terra": ("GPT-6.1 Terra", "#d77724", "v"),
    "google/gemini-2.5-flash": ("Gemini 2.5 Flash", "#526478", "p"),
    "anthropic/claude-sonnet-5.5": ("Claude Sonnet 5.5", "#7d6129", "X"),
    "moonshotai/kimi-k3.1-pro": ("Kimi K3.1 Pro", "#558735", "P"),
    "deepseek/deepseek-v4.1-flash-1017": ("DeepSeek V4.1 Flash", "#3555ac", "h"),
}
BENCHMARKS = Path("experiments/benchmarks")
OCTOBER_SOURCE = BENCHMARKS / "skillret-gap-closure-20261024/report/full-retrieval.csv"
PROMPT_SOURCE = BENCHMARKS / "openrouter-prompt-audit-20261108/results/summary.csv"
PROMPT_AUDIT_COMMIT = "5965356f8856f68b899ab0aa14b79de95fab506d"
JEV_ROOT = BENCHMARKS / "skillret-reranker-study/reference_run/jev"
JEV_SOURCE = JEV_ROOT / "summary-200.json"
JEV_CATALOG_SIZE = 6006
JEV_FIELDS = {"BM25 (Jev baseline)": "bm25_recall_at_5", "BM25 + Jev": "jev_recall_at_5"}
FIGURES = {
    "recall-at-5": "Recall@5 by retrieval method and catalog size",
    "recall-at-20-ceiling": "Recall@5 and candidate Recall@20",
    "reranking-improvement": "Paired change in Recall@5 after reranking",
    "latency": "Retrieval and reranking latency",
    "cost-and-token-usage": "API cost, token usage and cache effectiveness",
    "presentation-comparison": "Skill discovery Recall@5 across catalog sizes",
}


@dataclass(frozen=True)
class Point:
    figure: str
    series: str
    method: str
    catalog_size: int
    metric: str
    value: float
    uncertainty_kind: str
    lower: float | None
    upper: float | None
    source: str
    source_field: str
    population: str = "pilot_200"
    n_queries: int | None = None
    n_seeds: int | None = None
    source_sha256: str = ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(row: dict, field: str) -> float | None:
    raw = row.get(field)
    if raw is None or raw == "":
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def published_points(repository: Path, results: Path) -> list[Point]:
    figure = "presentation-comparison"
    sources = (
        OCTOBER_SOURCE, JEV_SOURCE, JEV_ROOT / "paired-results-200.csv",
        JEV_ROOT / "manifest-results-200.json", JEV_ROOT / "rows.jsonl",
    )
    prompt_path = repository / PROMPT_SOURCE
    prompt_data = prompt_path.read_bytes() if prompt_path.is_file() else b""
    if not prompt_data and repository.is_dir():
        git = subprocess.run(
            ["git", "-C", str(repository), "show", f"{PROMPT_AUDIT_COMMIT}:{PROMPT_SOURCE}"],
            capture_output=True, check=False,
        )
        if git.returncode == 0:
            prompt_data = git.stdout
    if not prompt_data or not all((repository / source).is_file() for source in sources):
        # The presentation export also preserves exact historical values for the ZIP.
        fallback = results / "presentation_plot_data.csv"
        if not fallback.is_file():
            missing_sources = [source for source in sources if not (repository / source).is_file()]
            if not prompt_data:
                missing_sources.append(PROMPT_SOURCE)
            missing = ", ".join(map(str, missing_sources))
            raise FileNotFoundError(f"Historical plot inputs unavailable: {missing}")
        points = [
            Point(
                row["figure"], row["series"], row["method"], int(row["catalog_size"]),
                row["metric"], float(row["value"]), row["uncertainty_kind"],
                number(row, "lower"), number(row, "upper"), row["source"], row["source_field"],
                row["population"], int(row["n_queries"]),
                int(row["n_seeds"]) if row["n_seeds"] else None, row["source_sha256"],
            )
            for row in read_csv(fallback)
            if row["population"] != "pilot_200"
        ]
        if {point.population for point in points} != {"october_4392", "openrouter_prompt_20", "jev_200"}:
            raise ValueError("The presentation CSV lacks a historical evaluation population")
        return points

    points = []
    october_path = repository / OCTOBER_SOURCE
    october_hash = sha256(october_path)
    october_groups = defaultdict(dict)
    for row in read_csv(october_path):
        if int(row["queries"]) != 4392 or row["method"] not in METHODS:
            raise ValueError("Unexpected October full-retrieval population or method")
        key = (row["method"], int(row["size"]))
        seed = int(row["seed"])
        if seed in october_groups[key]:
            raise ValueError("Duplicate October seed")
        october_groups[key][seed] = float(row["macro_recall5"])
    october_grid = {(method, size) for method in METHODS for size in (10, 100, 1000, 6006)}
    if set(october_groups) != october_grid:
        raise ValueError("Unexpected October full-retrieval grid")
    for (method, size), values in sorted(october_groups.items()):
        if set(values) != {42, 43, 44}:
            raise ValueError("October retrieval is missing a catalog seed")
        mean = fmean(values.values())
        sd = stdev(values.values())
        points.append(Point(
            figure, METHODS[method][0], method, size, "recall_at_5", mean, "seed_sd",
            mean - sd, mean + sd, str(OCTOBER_SOURCE), "macro_recall5",
            "october_4392", 4392, len(values), october_hash,
        ))

    prompt_hash = hashlib.sha256(prompt_data).hexdigest()
    prompt_cells = set()
    for row in csv.DictReader(prompt_data.decode("utf-8").splitlines()):
        if row["cohort"] != "retained_ranking" or row["library_size"] == "ALL":
            continue
        model, size = row["model"], int(row["library_size"])
        if int(row["n_expected"]) != 20 or model not in PROMPT_MODELS:
            raise ValueError("Unexpected OpenRouter prompt-based population or model")
        if (model, size) in prompt_cells:
            raise ValueError("Duplicate OpenRouter catalog cell")
        prompt_cells.add((model, size))
        scored = int(row["n_scored"])
        value = number(row, "recall@5")
        if scored == 0 and value is None:
            continue
        if value is None or not 0 < scored == int(row["n_completed"]) <= 20:
            raise ValueError("Unexpected scored OpenRouter prompt-based population")
        points.append(Point(
            figure, PROMPT_MODELS[model][0], model, size, "recall_at_5",
            value, "", None, None, str(PROMPT_SOURCE), "recall@5",
            "openrouter_prompt_20", scored, None, prompt_hash,
        ))
    expected_prompt = {(model, size) for model in PROMPT_MODELS for size in SIZES}
    prompt_points = [point for point in points if point.population == "openrouter_prompt_20"]
    if prompt_cells != expected_prompt or sum(point.n_queries for point in prompt_points) != 1184:
        raise ValueError("Unexpected OpenRouter prompt-based grid or scored count")

    jev_path = repository / JEV_SOURCE
    jev_hash = sha256(jev_path)
    summary = json.loads(jev_path.read_text(encoding="utf-8"))
    jev_manifest = json.loads(
        (repository / JEV_ROOT / "manifest-results-200.json").read_text(encoding="utf-8")
    )
    if jev_manifest["dataset"]["files"]["skills"]["rows"] != JEV_CATALOG_SIZE:
        raise ValueError("Unexpected historical Jev catalog size")
    paired_path = repository / JEV_ROOT / "paired-results-200.csv"
    paired = read_csv(paired_path)
    ids = [row["query_id"] for row in paired]
    if len(set(ids)) != 200 or ids != jev_manifest["query_ids"]:
        raise ValueError("Unexpected size of the historical Jev query panel")
    if summary["successful_queries"] != 200 or any(row["status"] != "ok" for row in paired):
        raise ValueError("Historical Jev paired rows include a failure")
    for field in (*JEV_FIELDS.values(), "bm25_recall_at_20"):
        recomputed = fmean(float(row[field]) for row in paired)
        if not math.isclose(recomputed, summary[field], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Historical Jev summary disagrees with paired rows for {field}")
    with (repository / JEV_ROOT / "rows.jsonl").open(encoding="utf-8") as file:
        historical_rows = [json.loads(line) for line in file]
    if [row["query_id"] for row in historical_rows] != ids:
        raise ValueError("Historical Jev raw rows and paired query IDs disagree")
    for row, scored in zip(historical_rows, paired):
        gold = set(row["evaluator_only_gold"])
        bm25 = [candidate[0] for candidate in row["bm25_top20"]]
        measures = {
            "bm25_recall_at_5": len(gold.intersection(bm25[:5])) / len(gold),
            "bm25_recall_at_20": len(gold.intersection(bm25)) / len(gold),
            "jev_recall_at_5": len(gold.intersection(row["jev_top5"])) / len(gold),
        }
        if any(value != float(scored[field]) for field, value in measures.items()):
            raise ValueError("Historical Jev paired recall disagrees with saved raw rankings")
    for series, field in JEV_FIELDS.items():
        points.append(Point(
            figure, series, "bm25", JEV_CATALOG_SIZE, "recall_at_5", summary[field], "",
            None, None, str(JEV_SOURCE), field, "jev_200", 200, None, jev_hash,
        ))
    return points


def pilot_points(path: Path) -> list[Point]:
    rows = read_csv(path)
    cells = {(row["method"].lower(), int(row["catalog_size"])) for row in rows}
    if len(cells) != len(rows) or not cells.issubset(
        {(method, size) for method in METHODS for size in SIZES}
    ):
        raise ValueError("Unexpected or duplicate pilot method/catalog cells")
    source = "plot_data.csv"
    digest = sha256(path)
    points = []
    additive_fields = {
        "input_tokens": "Input tokens",
        "cached_input_tokens": "Cached input tokens",
        "cache_write_tokens": "Cache-write tokens",
        "output_tokens": "Output tokens",
    }

    for row in rows:
        method = row["method"].lower()
        label = METHODS[method][0]
        size = int(row["catalog_size"])
        n_queries = int(row["n_queries"])
        n_seeds = int(row["n_seeds"])

        def add(figure, series, field, *, sd_field=None, ci_fields=None, scale=1):
            value = number(row, field)
            if value is None:
                return
            kind, lower, upper = "", None, None
            sd = number(row, sd_field) if sd_field else None
            if sd is not None:
                kind, lower, upper = "seed_sd", value - sd, value + sd
            if ci_fields:
                lower, upper = (number(row, column) for column in ci_fields)
                if lower is not None and upper is not None:
                    kind = "paired_ci95"
            points.append(Point(
                figure, series, method, size, field, value * scale, kind,
                lower * scale if lower is not None else None,
                upper * scale if upper is not None else None,
                source, f"{scale}*{field}" if scale != 1 else field,
                n_queries=n_queries, n_seeds=n_seeds, source_sha256=digest,
            ))

        for figure in ("recall-at-5", "presentation-comparison"):
            add(figure, label, "recall_at_5", sd_field="recall_at_5_seed_std")
            add(
                figure, f"{label} + OpenAI Decisions", "reranked_recall_at_5",
                sd_field="reranked_recall_at_5_seed_std",
            )
        for series, field in (
            ("Retrieval Recall@5", "recall_at_5"),
            ("Reranked Recall@5", "reranked_recall_at_5"),
            ("Candidate Recall@20", "recall_at_20"),
        ):
            add("recall-at-20-ceiling", series, field)
        add(
            "reranking-improvement", label, "recall_improvement",
            ci_fields=("improvement_ci_lower", "improvement_ci_upper"), scale=100,
        )
        for series, field in (
            ("Retrieval", "retrieval_latency"),
            ("Fresh API", "api_latency"),
            ("End-to-end", "end_to_end_latency"),
        ):
            for statistic in ("median", "p95"):
                add("latency", series, f"{field}_{statistic}_ms")
        attempted = number(row, "n_attempted_cases")
        http_attempts = number(row, "api_http_attempts")
        unknown_billing = number(row, "n_api_unknown_billing_attempts")
        if attempted and (not http_attempts or unknown_billing != http_attempts):
            add("cost-and-token-usage", label, "api_cost_usd")
            add("cost-and-token-usage", label, "api_cost_per_query_usd")
        for rate, numerator, denominator_name, denominator in (
            ("Local cache", "n_local_cache_hits", "n_attempted_cases", attempted),
            ("Provider prompt cache", "n_provider_cache_hits", "api_http_attempts", http_attempts),
        ):
            count = number(row, numerator)
            if count is not None and denominator:
                points.append(Point(
                    "cost-and-token-usage", rate, method, size,
                    "local_cache_hit_rate" if rate == "Local cache" else "provider_cache_hit_rate",
                    count / denominator, "", None, None,
                    source, f"{numerator}/({denominator_name})",
                    n_queries=n_queries, n_seeds=n_seeds, source_sha256=digest,
                ))

    for size in SIZES:
        group = [
            row for row in rows
            if int(row["catalog_size"]) == size and (number(row, "input_tokens") or 0) > 0
        ]
        for field, series in additive_fields.items():
            values = [number(row, field) for row in group]
            observed = [value for value in values if value is not None]
            if observed:
                points.append(Point(
                    "cost-and-token-usage", series, "all", size, field, math.fsum(observed),
                    "", None, None, source, f"sum({field})", source_sha256=digest,
                ))
    return points


def write_points(path: Path, points: list[Point]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in fields(Point)])
        writer.writeheader()
        writer.writerows(asdict(point) for point in points)


def points_hash(points: list[Point]) -> str:
    rows = [
        {key: "" if value is None else str(value) for key, value in asdict(point).items()}
        for point in points
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def draw_line(ax, points, sizes, *, color, marker, style="-", label=None, band=False, width=2):
    if not points:
        return None
    found = {point.catalog_size: point for point in points}
    y = [found[size].value if size in found else math.nan for size in sizes]
    line, = ax.plot(
        sizes, y, color=color, marker=marker, ls=style, label=label, lw=width,
        markersize=4.8, zorder=4,
    )
    intervals = [point for point in points if point.lower is not None and point.upper is not None]
    if intervals:
        x = [point.catalog_size for point in intervals]
        low = [point.lower for point in intervals]
        high = [point.upper for point in intervals]
        if band:
            bounds = {point.catalog_size: (point.lower, point.upper) for point in intervals}
            lows = [bounds.get(size, (math.nan, math.nan))[0] for size in sizes]
            highs = [bounds.get(size, (math.nan, math.nan))[1] for size in sizes]
            ax.fill_between(sizes, lows, highs, color=color, alpha=0.13, zorder=2)
        if not band or len(intervals) == 1:
            ax.vlines(x, low, high, color=color, alpha=0.73, lw=1.45, zorder=3)
            ax.scatter(x, low, color=color, marker="_", s=19, zorder=3)
            ax.scatter(x, high, color=color, marker="_", s=19, zorder=3)
    return line


def format_catalog(ax, sizes=SIZES, *, xlabel=True):
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{size:,}" for size in sizes])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(sizes[0] / 1.13, sizes[-1] * 1.13)
    if xlabel:
        ax.set_xlabel("Catalog size (skills, log scale)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def format_recall(ax):
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .2, .4, .6, .8, 1])


def annotate_missing(ax, label):
    ax.text(
        .5, .5, label, ha="center", va="center", transform=ax.transAxes,
        color="#526478", fontsize=10,
    )


def legend(ax, **kwargs):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, **kwargs)


def pilot_lines(ax, points, sizes=SIZES):
    for method, (label, color, marker) in METHODS.items():
        for field, style, series in (
            ("recall_at_5", "-", label),
            ("reranked_recall_at_5", "--", f"{label} + OpenAI Decisions"),
        ):
            draw_line(
                ax, [p for p in points if p.method == method and p.metric == field],
                sizes, color=color, marker=marker, style=style, label=series, band=True,
            )


def evidence_status(points):
    baseline = sum(point.metric == "recall_at_5" for point in points)
    reranked = sum(point.metric == "reranked_recall_at_5" for point in points)
    expected = len(SIZES) * len(METHODS)
    if baseline == expected and reranked == expected:
        return "200 exposed development queries across three catalog seeds"
    return f"{baseline}/{expected} retrieval cells and {reranked}/{expected} reranked cells measured"


def recall_figure(points):
    fig, ax = plt.subplots(figsize=(9.7, 5.6), layout="constrained")
    fig.suptitle(FIGURES["recall-at-5"], x=.02, ha="left")
    ax.set_title(evidence_status(points), loc="left", fontsize=10, pad=9)
    pilot_lines(ax, points)
    format_catalog(ax)
    format_recall(ax)
    ax.set_ylabel("Macro Recall@5")
    if not points:
        annotate_missing(ax, "No measured retrieval cells")
    legend(ax, loc="lower left", title="Bands show ±1 SD across seeds", ncols=2, handlelength=3.1)
    return fig


def ceiling_figure(points):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, layout="constrained")
    fig.suptitle(FIGURES["recall-at-20-ceiling"], x=.02, ha="left")
    for ax, (method, (label, color, marker)) in zip(axes, METHODS.items()):
        subset = [point for point in points if point.method == method]
        for series, style, series_color in (
            ("Retrieval Recall@5", "-", color),
            ("Reranked Recall@5", "--", color),
            ("Candidate Recall@20", ":", "#526478"),
        ):
            draw_line(
                ax, [point for point in subset if point.series == series], SIZES,
                color=series_color, marker=marker, style=style, label=series,
            )
        format_catalog(ax)
        format_recall(ax)
        ax.set_title(label, loc="left")
        if not subset:
            annotate_missing(ax, "No measured cells")
    axes[0].set_ylabel("Macro recall")
    legend_axis = max(axes, key=lambda axis: len(axis.get_legend_handles_labels()[0]))
    legend(legend_axis, loc="lower left", fontsize=9)
    return fig


def improvement_figure(points):
    fig, ax = plt.subplots(figsize=(9.7, 5.2), layout="constrained")
    fig.suptitle(FIGURES["reranking-improvement"], x=.02, ha="left")
    ax.set_title("95% paired bootstrap intervals, resampled by query", loc="left", fontsize=10)
    format_catalog(ax)
    ax.set_ylabel("Recall@5 change (percentage points)")
    ax.axhline(0, color="#526478", lw=1)
    for method, (label, color, marker) in METHODS.items():
        draw_line(
            ax, [point for point in points if point.method == method], SIZES,
            color=color, marker=marker, label=label,
        )
    if not points:
        ax.set_ylim(-1, 1)
        annotate_missing(ax, "No completed paired reranking cells")
    legend(ax, loc="best")
    return fig


def latency_figure(points):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.1), sharey=True, layout="constrained")
    fig.suptitle(FIGURES["latency"], x=.02, ha="left")
    components = {"Retrieval": "#2379a5", "Fresh API": "#d77724", "End-to-end": "#8f559e"}
    log_y = bool(points) and min(point.value for point in points) > 0
    for ax, (method, (label, _, _)) in zip(axes, METHODS.items()):
        subset = [point for point in points if point.method == method]
        for series, color in components.items():
            for stat, style in (("median", "-"), ("p95", "--")):
                draw_line(
                    ax, [p for p in subset if p.series == series and stat in p.metric],
                    SIZES, color=color, marker="o", style=style, label=f"{series} {stat}",
                )
        format_catalog(ax)
        ax.set_title(label, loc="left")
        if log_y:
            ax.set_yscale("log")
        if not subset:
            annotate_missing(ax, "No measured latency")
    axes[0].set_ylabel("Latency (ms, log scale)" if log_y else "Latency (ms)")
    handles = [
        Line2D([0], [0], color=color, lw=2, label=series)
        for series, color in components.items()
        if any(p.series == series for p in points)
    ]
    if handles:
        handles.extend([
            Line2D([0], [0], color="#526478", ls="-", label="Median"),
            Line2D([0], [0], color="#526478", ls="--", label="p95"),
        ])
        fig.legend(handles=handles, loc="outside lower center", frameon=False, ncols=5)
    return fig


def cost_figure(points):
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.7), layout="constrained")
    fig.suptitle(FIGURES["cost-and-token-usage"], x=.02, ha="left")
    for ax, label in zip(axes.flat, (
        "Metered API cost estimate", "Metered API cost per attempted case",
        "Reported tokens across retrieval methods", "Cache hit rates",
    )):
        ax.set_title(label, loc="left")
        format_catalog(ax)
    cost_ax, per_case_ax, tokens_ax, cache_ax = axes.flat
    for ax, field in ((cost_ax, "api_cost_usd"), (per_case_ax, "api_cost_per_query_usd")):
        ax.set_ylabel("USD")
        total = [point for point in points if point.metric == field]
        for method, (label, color, marker) in METHODS.items():
            draw_line(
                ax, [point for point in total if point.method == method],
                SIZES, color=color, marker=marker, label=label,
            )
        if not total:
            annotate_missing(ax, "No metered benchmark cost")
        legend(ax, fontsize=9)

    token_points = [point for point in points if point.method == "all"]
    tokens_ax.set_ylabel("Tokens")
    for series, color in (
        ("Input tokens", "#2379a5"), ("Cached input tokens", "#24967e"),
        ("Cache-write tokens", "#8f559e"), ("Output tokens", "#d77724"),
    ):
        draw_line(
            tokens_ax, [point for point in token_points if point.series == series],
            SIZES, color=color, marker="o", label=series,
        )
    if not token_points:
        annotate_missing(tokens_ax, "No metered benchmark tokens")
    legend(tokens_ax, fontsize=9)

    cache_points = [point for point in points if point.metric.endswith("_hit_rate")]
    cache_ax.set_ylabel("Share")
    cache_ax.set_ylim(0, 1)
    cache_ax.yaxis.set_major_formatter(PercentFormatter(1))
    for method, (label, color, marker) in METHODS.items():
        for series, style in (("Local cache", "-"), ("Provider prompt cache", "--")):
            draw_line(
                cache_ax, [p for p in cache_points if p.series == series and p.method == method],
                SIZES, color=color, marker=marker, style=style, label=f"{label}: {series}",
            )
    if not cache_points:
        annotate_missing(cache_ax, "No measured cache activity")
    legend(cache_ax, fontsize=8, ncols=2)
    return fig


def presentation_figure(points):
    fig = plt.figure(figsize=(15.5, 8.71875), layout="constrained")
    fig.suptitle(FIGURES["presentation-comparison"], x=.02, ha="left")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.35, 1))
    main_ax = fig.add_subplot(grid[0, :])
    october_ax, prompt_ax = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    pilot = [point for point in points if point.population == "pilot_200"]
    plot_sizes = (*SIZES, JEV_CATALOG_SIZE)
    pilot_lines(main_ax, pilot, plot_sizes)
    for series, color, marker in (
        ("BM25 (Jev baseline)", "#6b7480", "x"),
        ("BM25 + Jev", "#2a333c", "D"),
    ):
        draw_line(
            main_ax, [p for p in points if p.population == "jev_200" and p.series == series],
            plot_sizes, color=color, marker=marker, style="none", label=series,
        )
    format_catalog(main_ax, plot_sizes)
    main_ax.set_xlim(plot_sizes[0] / 1.13, plot_sizes[-1] * 1.28)
    main_ax.get_xticklabels()[-2].set_horizontalalignment("right")
    main_ax.get_xticklabels()[-1].set_horizontalalignment("left")
    main_ax.set_title(f"Decisions scaling pilot, {evidence_status(pilot)}", loc="left")
    main_ax.set_ylabel("Macro Recall@5")
    main_ax.text(
        .98, .96, "Jev trial, 200 development queries at 6,006 skills",
        ha="right", va="top", fontsize=9, transform=main_ax.transAxes,
    )
    legend(
        main_ax, loc="lower left", ncols=4, fontsize=9, handlelength=3.1,
        title="Pilot bands show ±1 SD across seeds",
    )

    october_sizes = (10, 100, 1000, 6006)
    format_catalog(october_ax, october_sizes)
    october_ax.set_title("October matched retrieval, 4,392 queries", loc="left")
    october_ax.set_ylabel("Macro Recall@5")
    for method, (label, color, marker) in METHODS.items():
        draw_line(
            october_ax, [p for p in points if p.population == "october_4392" and p.method == method],
            october_sizes, color=color, marker=marker, label=label, band=True,
        )
    legend(october_ax, loc="lower left", fontsize=9, ncols=3)

    format_catalog(prompt_ax)
    prompt_ax.set_title("OpenRouter prompt listing, 20 single-label queries", loc="left")
    prompt_ax.set_ylabel("Recall@5")
    for model, (label, color, marker) in PROMPT_MODELS.items():
        draw_line(
            prompt_ax, [p for p in points if p.population == "openrouter_prompt_20" and p.method == model],
            SIZES, color=color, marker=marker, label=label, width=1.5,
        )
    prompt_ax.text(
        .02, .25, "Saved parsed rankings. Points use 5–20 scored queries.",
        transform=prompt_ax.transAxes, fontsize=9,
    )
    legend(prompt_ax, loc="lower left", fontsize=9, ncols=3)
    for ax in (main_ax, october_ax, prompt_ax):
        format_recall(ax)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--repository", type=Path, default=HERE.parent.parent.parent)
    args = parser.parse_args()
    results = args.results.resolve()
    points = pilot_points(results / "plot_data.csv") + published_points(args.repository, results)
    figures = results / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.labelcolor": "#20313d", "text.color": "#20313d",
        "figure.facecolor": "white", "font.size": 10.5, "figure.titleweight": "bold",
        "figure.titlesize": 15, "savefig.facecolor": "white",
        "svg.hashsalt": "skillret-openai-decisions-scaling", "svg.fonttype": "none",
    })
    constructors = (
        recall_figure, ceiling_figure, improvement_figure,
        latency_figure, cost_figure, presentation_figure,
    )
    exports = {}
    for (slug, title), make_figure in zip(FIGURES.items(), constructors):
        figure_points = [point for point in points if point.figure == slug]
        digest = points_hash(figure_points)
        description = f"figure_points_sha256={digest}"
        figure = make_figure(figure_points)
        figure.savefig(
            figures / f"{slug}.png", dpi=320, metadata={"Title": title, "Description": description},
        )
        figure.savefig(
            figures / f"{slug}.svg", metadata={"Title": title, "Description": description, "Date": None},
        )
        figure.savefig(
            figures / f"{slug}.pdf",
            metadata={
                "Title": title, "Subject": description, "CreationDate": None, "ModDate": None,
            },
        )
        plt.close(figure)
        exports[slug] = {
            "figure_points_sha256": digest,
            "files": {f"{slug}.{fmt}": sha256(figures / f"{slug}.{fmt}") for fmt in ("png", "svg", "pdf")},
        }
    points_path = results / "figure_points.csv"
    presentation_path = results / "presentation_plot_data.csv"
    write_points(points_path, points)
    write_points(presentation_path, [point for point in points if point.figure == "presentation-comparison"])
    (results / "figure_exports.json").write_text(
        json.dumps({
            "figure_points_csv_sha256": sha256(points_path),
            "presentation_plot_data_csv_sha256": sha256(presentation_path),
            "figures": exports,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
