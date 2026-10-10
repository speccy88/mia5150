#!/usr/bin/env python3
"""
Skill Discovery Benchmark -- Listing Method
============================================
MIA5150 Project: Agent Skill Discovery at Scale

Evaluates the **Listing** discovery strategy: all skill summaries are
loaded directly into the Deep Agent's context via SkillsMiddleware.
No retrieval step -- the agent sees every skill and must pick the
correct one.

This demonstrates how the Listing method degrades as library size
grows and eventually hits context-window limits.

Outputs
-------
- Console tables (Markdown-compatible) with accuracy per library size
- results_listing/listing_accuracy.png
- results_listing/listing_scaling.png
- results_listing/metrics.json
- results_listing/details.csv

Usage
-----
  export OPENAI_API_KEY="sk-..."
  python benchmark_listing.py --dataset test.jsonl --num-queries 20

  # Custom library sizes:
  python benchmark_listing.py --num-queries 20 --lib-sizes 50,100,250,500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

# ── Defaults ─────────────────────────────────────────────────────────────────

SEED = 42
DEFAULT_NUM_QUERIES = 20
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_LIB_SIZES = [50, 100, 250, 500, 1000]

# ── Shared helpers (same as benchmark_bm25.py) ──────────────────────────────

def load_skills(path: str) -> list[dict]:
    skills: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                skills.append(json.loads(line))
    return skills


_USE_WHEN_RE = re.compile(
    r"[Uu]se\s+(?:this\s+)?(?:skill\s+)?(?:when|for)\s+(.+?)(?:\.\s|$)",
    re.DOTALL,
)


def _extract_trigger(description: str) -> str | None:
    m = _USE_WHEN_RE.search(description)
    if m:
        phrase = m.group(1).strip()
        if len(phrase) > 20:
            return phrase
    return None


def _generate_query(skill: dict, rng: random.Random) -> str:
    desc = skill.get("description", "")
    trigger = _extract_trigger(desc)
    if trigger and rng.random() < 0.6:
        return trigger

    action = skill.get("primary_action", "work on")
    obj = skill.get("primary_object", "task")
    name_words = skill["name"].replace("-", " ").replace("_", " ")

    templates = [
        f"I need to {action} a {obj} using {name_words}",
        f"Help me {action} {obj}",
        f"I want to {action} my {obj}",
        f"Can you help with {name_words}?",
        f"I'm looking for a tool to {action} a {obj}",
    ]
    return rng.choice(templates)


def create_test_queries(
    skills: list[dict], num_queries: int, seed: int = SEED
) -> list[dict]:
    rng = random.Random(seed)
    sampled = rng.sample(skills, min(num_queries, len(skills)))
    queries: list[dict] = []
    for skill in sampled:
        queries.append(
            {
                "query": _generate_query(skill, rng),
                "gold_skill": skill["name"],
                "gold_id": skill["id"],
                "gold_domain": skill.get("domain", "unknown"),
                "gold_major": skill.get("major", "unknown"),
            }
        )
    return queries


def _spec_name(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _clean_yaml_str(text: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text).replace('"', '\\"')


def _skill_to_skill_md(skill: dict) -> str:
    original = skill["name"]
    spec = _spec_name(original)
    desc = _clean_yaml_str(skill.get("description", "No description.")[:300])
    return (
        f'---\nname: "{spec}"\n'
        f'description: "{desc}"\n'
        f"---\n\nOriginal name: {original}\n"
    )


def _safe_skill_dir(name: str) -> str:
    return _spec_name(name)


# ── Deep Agent extraction ────────────────────────────────────────────────────

def _extract_selected(result: dict, name_lookup: dict[str, str]) -> str:
    messages = result.get("messages", [])

    # Strategy 1: read_file call on a SKILL.md
    for msg in messages:
        if not (hasattr(msg, "type") and msg.type == "ai"):
            continue
        content = msg.content
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("name") == "read_file":
                try:
                    args = json.loads(block.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    continue
                path = args.get("file_path", "")
                m = re.match(r"/skills/([^/]+)/SKILL\.md", path)
                if m:
                    safe = m.group(1)
                    return name_lookup.get(safe, safe)

    # Strategy 2: first short AI text block
    for msg in messages:
        if not (hasattr(msg, "type") and msg.type == "ai"):
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            text = content.strip().strip("\"'`*").split("\n")[0].strip().strip("*")
            return text
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text and len(text) < 120:
                        text = text.strip("\"'`*").split("\n")[0].strip().strip("*")
                        return text

    return ""


# ── Listing evaluation ───────────────────────────────────────────────────────

_LISTING_PROMPT = (
    "You are an assistant agent evaluating skill discovery performance.\n"
    "You have access to a set of skills. Given a user task, your ONLY job\n"
    "is to identify which of your available skills is most relevant.\n\n"
    "IMPORTANT RULES:\n"
    "- Reply with ONLY the exact skill name. Nothing else.\n"
    "- Do NOT read, activate, or execute any skill files.\n"
    "- Do NOT use any tools. Just pick the best skill from the summaries\n"
    "  you already see and return its name."
)


def evaluate_listing(
    queries: list[dict],
    all_skills: list[dict],
    model_name: str,
    library_size: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """Load ALL skills into the agent context and measure selection accuracy."""
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from deepagents.backends.utils import create_file_data
    from langgraph.checkpoint.memory import MemorySaver

    rng = random.Random(seed)
    gold_names = {q["gold_skill"] for q in queries}

    # Build library: ensure gold skills are included
    gold_skills = [s for s in all_skills if s["name"] in gold_names]
    other = [s for s in all_skills if s["name"] not in gold_names]
    n_fill = min(library_size - len(gold_skills), len(other))
    if n_fill < 0:
        print(f"  SKIP: library_size {library_size} < {len(gold_skills)} gold skills")
        return pd.DataFrame()
    library = gold_skills + rng.sample(other, n_fill)
    rng.shuffle(library)

    print(f"\n  Building {len(library)} SKILL.md files for listing ...")

    # Pre-build ALL skill files (same for every query)
    skills_files: dict = {}
    name_lookup: dict[str, str] = {}
    for s in library:
        safe = _safe_skill_dir(s["name"])
        name_lookup[safe] = s["name"]
        path = f"/skills/{safe}/SKILL.md"
        skills_files[path] = create_file_data(_skill_to_skill_md(s))

    print(f"  Creating Deep Agent with {len(library)} skills ...")

    agent = create_deep_agent(
        model=f"openai:{model_name}",
        backend=StateBackend(),
        skills=["/skills/"],
        checkpointer=MemorySaver(),
        system_prompt=_LISTING_PROMPT,
    )

    rows: list[dict] = []
    for i, q in enumerate(queries):
        t0 = time.time()
        try:
            result = agent.invoke(
                {
                    "messages": [{"role": "user", "content": q["query"]}],
                    "files": skills_files,
                },
                config={"configurable": {"thread_id": f"list-{library_size}-{i}"}},
            )
            latency = time.time() - t0
            selected = _extract_selected(result, name_lookup)
        except Exception as exc:
            latency = time.time() - t0
            selected = f"ERROR: {exc}"

        exact = int(selected == q["gold_skill"])
        fuzzy = int(selected.lower().strip() == q["gold_skill"].lower().strip())

        rows.append(
            {
                "library_size": len(library),
                "query": q["query"][:100],
                "gold_skill": q["gold_skill"],
                "selected_skill": selected,
                "exact_match": exact,
                "fuzzy_match": fuzzy,
                "latency_s": round(latency, 3),
            }
        )

        tag = "OK" if fuzzy else "MISS"
        print(
            f"  [{i + 1}/{len(queries)}] {tag:4s}  "
            f"gold={q['gold_skill'][:30]:30s}  sel={selected[:30]}"
        )

    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)

# Colour palette
_C_GREEN = "#2E7D32"
_C_GREEN_LIGHT = "#66BB6A"
_C_ORANGE = "#FF7043"
_C_GREY = "#90A4AE"
_C_BG_BAND = "#E8F5E9"


def plot_listing_accuracy(df: pd.DataFrame, out: str, model: str = "") -> None:
    """Primary graph: accuracy degradation curve with shaded zones."""
    summary = df.groupby("library_size").agg(
        exact_match=("exact_match", "mean"),
        avg_latency=("latency_s", "mean"),
    ).sort_index()

    sizes = summary.index.values
    acc = summary["exact_match"].values
    lat = summary["avg_latency"].values

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    # Shaded background zones
    if len(sizes) >= 2:
        mid = sizes[len(sizes) // 2]
        ax1.axvspan(sizes[0] - 10, mid, alpha=0.08, color=_C_GREEN,
                    label="_nolegend_")
        ax1.axvspan(mid, sizes[-1] + 10, alpha=0.06, color="#FFCCBC",
                    label="_nolegend_")
        ax1.text(
            sizes[0] + (mid - sizes[0]) * 0.5, 1.04,
            "Listing effective", ha="center", fontsize=9,
            color=_C_GREEN, fontstyle="italic",
        )
        ax1.text(
            mid + (sizes[-1] - mid) * 0.5, 1.04,
            "Listing degrades", ha="center", fontsize=9,
            color="#D84315", fontstyle="italic",
        )

    # Accuracy line
    ax1.plot(sizes, acc, "o-", color=_C_GREEN, lw=2.5, ms=8, zorder=5,
             label="Exact Match Accuracy")
    ax1.fill_between(sizes, acc, alpha=0.12, color=_C_GREEN_LIGHT)

    # Data labels on accuracy points
    for x, y in zip(sizes, acc):
        ax1.annotate(
            f"{y:.0%}", (x, y),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=10, fontweight="bold", color=_C_GREEN,
        )

    ax1.set_xlabel("Library Size (number of skills)", fontsize=12)
    ax1.set_ylabel("Exact Match Accuracy", fontsize=12, color=_C_GREEN)
    ax1.set_ylim(0, 1.15)
    ax1.tick_params(axis="y", labelcolor=_C_GREEN)

    # Latency line (secondary axis)
    ax2 = ax1.twinx()
    ax2.plot(sizes, lat, "s--", color=_C_ORANGE, lw=2, ms=6, alpha=0.8,
             label="Avg Latency")
    for x, y in zip(sizes, lat):
        ax2.annotate(
            f"{y:.1f}s", (x, y),
            textcoords="offset points", xytext=(0, -14),
            ha="center", fontsize=8, color=_C_ORANGE,
        )
    ax2.set_ylabel("Avg Latency (s)", fontsize=11, color=_C_ORANGE)
    ax2.tick_params(axis="y", labelcolor=_C_ORANGE)
    ax2.spines["right"].set_visible(True)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=10, framealpha=0.9)

    subtitle = f"model: {model}  |  " if model else ""
    subtitle += f"n={int(df.groupby('library_size').size().iloc[0])} queries per size"
    ax1.set_title(
        "Listing Method: Skill Discovery Accuracy vs Library Size",
        fontsize=13, fontweight="bold", pad=18,
    )
    ax1.text(
        0.5, 1.02, subtitle,
        transform=ax1.transAxes, ha="center", fontsize=9, color=_C_GREY,
    )

    ax1.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "listing_accuracy.png"), dpi=200)
    plt.close(fig)


def plot_listing_scaling(df: pd.DataFrame, out: str, model: str = "") -> None:
    """Secondary graph: bar chart with exact + fuzzy match side by side."""
    summary = df.groupby("library_size").agg(
        exact_match=("exact_match", "mean"),
        fuzzy_match=("fuzzy_match", "mean"),
    ).sort_index()

    sizes = summary.index.values
    x = range(len(sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(
        [i - width / 2 for i in x], summary["exact_match"],
        width, label="Exact Match", color=_C_GREEN, edgecolor="white", lw=1.2,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x], summary["fuzzy_match"],
        width, label="Fuzzy Match", color=_C_GREEN_LIGHT, edgecolor="white", lw=1.2,
    )

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.0%}", ha="center", fontsize=9, fontweight="bold",
                color=_C_GREEN)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.0%}", ha="center", fontsize=9, color=_C_GREEN_LIGHT)

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(int(s)) for s in sizes])
    ax.set_xlabel("Library Size (number of skills)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    subtitle = f"model: {model}" if model else ""
    ax.set_title(
        "Listing Method: Accuracy Breakdown by Library Size",
        fontsize=13, fontweight="bold", pad=16,
    )
    if subtitle:
        ax.text(0.5, 1.01, subtitle,
                transform=ax.transAxes, ha="center", fontsize=9, color=_C_GREY)

    fig.tight_layout()
    fig.savefig(os.path.join(out, "listing_scaling.png"), dpi=200)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Skill Discovery Benchmark -- Listing Method"
    )
    ap.add_argument("--dataset", default="test.jsonl", help="Skills JSONL file")
    ap.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model")
    ap.add_argument(
        "--lib-sizes",
        default=",".join(str(s) for s in DEFAULT_LIB_SIZES),
        help="Comma-separated library sizes to test (default: 50,100,250,500,1000)",
    )
    ap.add_argument("--output-dir", default="results_listing")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    lib_sizes = [int(x) for x in args.lib_sizes.split(",")]
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"Loading skills from {args.dataset} ...")
    skills = load_skills(args.dataset)
    print(f"  {len(skills)} skills loaded")

    # ── Queries ───────────────────────────────────────────────────────────
    print(f"\nGenerating {args.num_queries} test queries (seed={args.seed}) ...")
    queries = create_test_queries(skills, args.num_queries, args.seed)
    print(f"  {len(queries)} queries created\n")

    for q in queries[:5]:
        print(f"  [{q['gold_skill'][:28]:28s}]  \"{q['query'][:65]}\"")
    if len(queries) > 5:
        print(f"  ... and {len(queries) - 5} more")

    with open(os.path.join(out, "test_queries.json"), "w") as fh:
        json.dump(queries, fh, indent=2)

    # ── Evaluate each library size ────────────────────────────────────────
    all_dfs: list[pd.DataFrame] = []

    for lib_size in lib_sizes:
        print("\n" + "=" * 62)
        print(f"Listing Evaluation  (N={lib_size}, model={args.model})")
        print("=" * 62)

        df = evaluate_listing(
            queries, skills, args.model,
            library_size=lib_size, seed=args.seed,
        )
        if df.empty:
            continue
        all_dfs.append(df)

        acc = df["exact_match"].mean()
        lat = df["latency_s"].mean()
        errors = (df["selected_skill"].str.startswith("ERROR")).sum()
        print(f"\n  N={lib_size}: Exact Match={acc:.1%}  "
              f"Avg Latency={lat:.2f}s  Errors={errors}")

    if not all_dfs:
        print("No results collected.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)

    # ── Summary table ─────────────────────────────────────────────────────
    summary = combined.groupby("library_size").agg(
        exact_match=("exact_match", "mean"),
        fuzzy_match=("fuzzy_match", "mean"),
        avg_latency=("latency_s", "mean"),
        errors=("selected_skill", lambda s: (s.str.startswith("ERROR")).sum()),
    ).reset_index()

    print("\n" + "=" * 62)
    print("Listing Method -- Summary")
    print("=" * 62)
    print()
    print(tabulate(summary, headers="keys", tablefmt="github", floatfmt=".3f",
                   showindex=False))

    # Markdown table
    with open(os.path.join(out, "listing_table.md"), "w") as fh:
        fh.write(f"## Listing Method Accuracy ({args.model})\n\n")
        fh.write(tabulate(summary, headers="keys", tablefmt="github",
                          floatfmt=".3f", showindex=False))
        fh.write("\n")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_listing_accuracy(combined, out, model=args.model)
    plot_listing_scaling(combined, out, model=args.model)
    print(f"\n  Plots saved to {out}/")

    # ── Save metrics ──────────────────────────────────────────────────────
    combined.to_csv(os.path.join(out, "details.csv"), index=False)

    metrics = {
        "method": "listing",
        "model": args.model,
        "num_queries": len(queries),
        "seed": args.seed,
        "library_sizes": lib_sizes,
        "summary": summary.to_dict(orient="records"),
    }
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n{'=' * 62}")
    print(f"All results saved to {out}/")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
