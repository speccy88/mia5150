#!/usr/bin/env python3
"""
Skill Discovery Benchmark -- BM25 Baseline
===========================================
MIA5150 Project: Agent Skill Discovery at Scale

Evaluates BM25 lexical retrieval + LLM-based selection for skill discovery
from a large skill library (6 000+ skills).

Two evaluation layers
---------------------
1. **Retrieval** -- Does BM25 surface the gold skill in the top-k candidates?
   Measured as Candidate Recall@k across multiple library sizes.

2. **Selection** -- Given BM25 candidates, does a LangChain Deep Agent
   pick the correct skill?  Uses ``create_deep_agent`` with StateBackend
   and SkillsMiddleware for realistic skill discovery.

Outputs
-------
- Console tables (Markdown-compatible) with all metrics
- results/retrieval_recall_at_k.png
- results/scaling.png
- results/selection_metrics.png
- results/metrics.json
- results/selection_details.csv
- results/test_queries.json

Usage
-----
  # Retrieval-only evaluation (no API key needed):
  python benchmark_bm25.py --dataset test.jsonl --retrieval-only

  # Full benchmark with LLM selection:
  export OPENAI_API_KEY="sk-..."
  python benchmark_bm25.py --dataset test.jsonl

  # Custom parameters:
  python benchmark_bm25.py --dataset test.jsonl \\
      --num-queries 100 --k 10 --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rank_bm25 import BM25Okapi
from tabulate import tabulate

# ── Defaults ─────────────────────────────────────────────────────────────────

SEED = 42
K_VALUES = [1, 3, 5, 10, 20]
LIBRARY_SIZES = [50, 100, 500, 1000, 3000]  # full size appended at runtime
DEFAULT_NUM_QUERIES = 50
DEFAULT_K = 10
DEFAULT_MODEL = "gpt-4o-mini"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_skills(path: str) -> list[dict]:
    """Load skills from a JSONL file."""
    skills: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                skills.append(json.loads(line))
    return skills


# ── Document representation (controlled) ─────────────────────────────────────

def skill_to_tokens(skill: dict) -> list[str]:
    """Tokenise a skill for BM25 indexing.

    Representation: ``name + description``.
    This is a deliberate, controlled choice -- changing it changes
    BM25 behaviour and must be documented.
    """
    text = f"{skill['name']} {skill.get('description', '')}"
    return re.findall(r"\w+", text.lower())


# ── Query generation ─────────────────────────────────────────────────────────

_USE_WHEN_RE = re.compile(
    r"[Uu]se\s+(?:this\s+)?(?:skill\s+)?(?:when|for)\s+(.+?)(?:\.\s|$)",
    re.DOTALL,
)


def _extract_trigger(description: str) -> str | None:
    """Pull the 'Use when ...' phrase from a skill description."""
    m = _USE_WHEN_RE.search(description)
    if m:
        phrase = m.group(1).strip()
        if len(phrase) > 20:
            return phrase
    return None


def _generate_query(skill: dict, rng: random.Random) -> str:
    """Build a realistic user query that *should* map to ``skill``."""
    desc = skill.get("description", "")

    # 60 % chance: use the trigger phrase (realistic, keyword-overlapping)
    trigger = _extract_trigger(desc)
    if trigger and rng.random() < 0.6:
        return trigger

    # 40 % (or fallback): template from action / object
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
    """Sample skills and create test queries with gold labels."""
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


# ── BM25 retrieval ───────────────────────────────────────────────────────────

def build_bm25_index(skills: list[dict]) -> BM25Okapi:
    corpus = [skill_to_tokens(s) for s in skills]
    return BM25Okapi(corpus)


def retrieve_top_k(
    bm25: BM25Okapi, query: str, skills: list[dict], k: int
) -> list[tuple[dict, float]]:
    tokens = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(skills[i], float(scores[i])) for i in top_idx]


# ── Phase 1: retrieval evaluation ────────────────────────────────────────────

def evaluate_retrieval(
    queries: list[dict],
    all_skills: list[dict],
    k_values: list[int],
    library_sizes: list[int],
    seed: int = SEED,
) -> pd.DataFrame:
    """Candidate Recall@k across library sizes."""
    rng = random.Random(seed)
    gold_names = {q["gold_skill"] for q in queries}

    rows: list[dict] = []

    for lib_size in library_sizes:
        gold_skills = [s for s in all_skills if s["name"] in gold_names]
        other_skills = [s for s in all_skills if s["name"] not in gold_names]

        if lib_size <= len(gold_skills):
            continue

        n_fill = min(lib_size - len(gold_skills), len(other_skills))
        fillers = rng.sample(other_skills, n_fill)
        library = gold_skills + fillers
        rng.shuffle(library)
        actual = len(library)

        bm25 = build_bm25_index(library)

        for k in k_values:
            if k > actual:
                continue
            hits = sum(
                1
                for q in queries
                if q["gold_skill"]
                in {s["name"] for s, _ in retrieve_top_k(bm25, q["query"], library, k)}
            )
            rows.append(
                {
                    "library_size": actual,
                    "k": k,
                    "recall_at_k": hits / len(queries),
                    "hits": hits,
                    "total": len(queries),
                }
            )
        print(f"  Library {actual:>5d}: done")

    return pd.DataFrame(rows)


# ── Phase 2: Deep Agent selection ────────────────────────────────────────────

_SELECTOR_PROMPT = (
    "You are an assistant agent evaluating skill discovery performance.\n"
    "You have access to a set of skills. Given a user task, your ONLY job\n"
    "is to identify which of your available skills is most relevant.\n\n"
    "IMPORTANT RULES:\n"
    "- Reply with ONLY the exact skill name. Nothing else.\n"
    "- Do NOT read, activate, or execute any skill files.\n"
    "- Do NOT use any tools. Just pick the best skill from the summaries\n"
    "  you already see and return its name."
)


def _spec_name(name: str) -> str:
    """Convert a skill name to Agent Skills spec format (lowercase, hyphens)."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)   # non-alphanum → hyphens
    return s.strip("-")                  # no leading/trailing hyphens


def _clean_yaml_str(text: str) -> str:
    """Strip characters that break YAML (control chars, non-ASCII oddities)."""
    # Keep only printable ASCII + common unicode
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text).replace('"', '\\"')


def _skill_to_skill_md(skill: dict) -> str:
    """Convert a skill dict to spec-compliant SKILL.md frontmatter.

    The original skill name is preserved in the body so the agent
    can return it; the YAML ``name`` field uses the spec-compliant form.
    """
    original = skill["name"]
    spec = _spec_name(original)
    desc = _clean_yaml_str(skill.get("description", "No description.")[:300])
    return (
        f'---\nname: "{spec}"\n'
        f'description: "{desc}"\n'
        f"---\n\nOriginal name: {original}\n"
    )


def _safe_skill_dir(name: str) -> str:
    """Sanitise a skill name into a safe directory name."""
    return _spec_name(name)


def _create_deep_agent(model_name: str):
    """Build a reusable Deep Agent with StateBackend for skill selection."""
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from langgraph.checkpoint.memory import MemorySaver

    return create_deep_agent(
        model=f"openai:{model_name}",
        backend=StateBackend(),
        skills=["/skills/"],
        checkpointer=MemorySaver(),
        system_prompt=_SELECTOR_PROMPT,
    )


def _extract_selected(result: dict, name_lookup: dict[str, str]) -> str:
    """Extract the selected skill from a Deep Agent invocation.

    Two strategies, tried in order:
    1. Look for ``read_file`` tool calls on a ``SKILL.md`` -- this means the
       agent activated that skill.  Map the sanitised directory name back to
       the original skill name via *name_lookup*.
    2. Fall back to the first short text block in the first AI message
       (the agent often emits the skill name before making tool calls).
    """
    messages = result.get("messages", [])

    # Strategy 1: find a read_file call targeting a SKILL.md
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


def evaluate_selection(
    queries: list[dict],
    all_skills: list[dict],
    model_name: str,
    k: int = DEFAULT_K,
    library_size: int | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """BM25 retrieval -> Deep Agent selection -> evaluation."""
    from deepagents.backends.utils import create_file_data

    rng = random.Random(seed)
    gold_names = {q["gold_skill"] for q in queries}

    # Build library (may sub-sample)
    if library_size and library_size < len(all_skills):
        gold_skills = [s for s in all_skills if s["name"] in gold_names]
        other = [s for s in all_skills if s["name"] not in gold_names]
        n = min(library_size - len(gold_skills), len(other))
        library = gold_skills + rng.sample(other, n)
        rng.shuffle(library)
    else:
        library = all_skills[:]

    bm25 = build_bm25_index(library)
    agent = _create_deep_agent(model_name)

    rows: list[dict] = []
    for i, q in enumerate(queries):
        candidates = retrieve_top_k(bm25, q["query"], library, k)
        cand_names = {s["name"] for s, _ in candidates}
        cand_recall = int(q["gold_skill"] in cand_names)

        # Build SKILL.md files for each candidate + reverse lookup
        skills_files = {}
        name_lookup: dict[str, str] = {}
        for s, _ in candidates:
            safe = _safe_skill_dir(s["name"])
            name_lookup[safe] = s["name"]
            path = f"/skills/{safe}/SKILL.md"
            skills_files[path] = create_file_data(_skill_to_skill_md(s))

        t0 = time.time()
        try:
            result = agent.invoke(
                {
                    "messages": [{"role": "user", "content": q["query"]}],
                    "files": skills_files,
                },
                config={"configurable": {"thread_id": f"q-{i}"}},
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
                "query": q["query"][:100],
                "gold_skill": q["gold_skill"],
                "selected_skill": selected,
                "candidate_recall": cand_recall,
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
    }
)


def plot_recall_at_k(df: pd.DataFrame, out: str) -> None:
    max_sz = df["library_size"].max()
    sub = df[df["library_size"] == max_sz].sort_values("k")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sub["k"], sub["recall_at_k"], "o-", color="#1976D2", lw=2, ms=7)
    for _, r in sub.iterrows():
        ax.annotate(
            f"{r['recall_at_k']:.2f}",
            (r["k"], r["recall_at_k"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("k (candidates retrieved)")
    ax.set_ylabel("Candidate Recall")
    ax.set_title(f"BM25 Candidate Recall@k  (N={max_sz} skills)")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "retrieval_recall_at_k.png"), dpi=150)
    plt.close(fig)


def plot_scaling(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for k_val in sorted(df["k"].unique()):
        sub = df[df["k"] == k_val].sort_values("library_size")
        ax.plot(sub["library_size"], sub["recall_at_k"], "o-", lw=2, ms=5, label=f"k={k_val}")
    ax.set_xlabel("Library Size (number of skills)")
    ax.set_ylabel("Candidate Recall")
    ax.set_title("BM25 Candidate Recall vs Library Size")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.08)
    ax.legend(title="k", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scaling.png"), dpi=150)
    plt.close(fig)


def plot_selection(df: pd.DataFrame, out: str) -> None:
    labels = ["Candidate\nRecall", "Exact\nMatch", "Fuzzy\nMatch"]
    vals = [
        df["candidate_recall"].mean(),
        df["exact_match"].mean(),
        df["fuzzy_match"].mean(),
    ]
    colours = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, vals, color=colours, width=0.45, edgecolor="white", lw=1.5)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v:.1%}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel("Rate")
    ax.set_title(f"LLM Skill Selection Accuracy  (n={len(df)} queries)")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "selection_metrics.png"), dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Skill Discovery Benchmark -- BM25 Baseline"
    )
    ap.add_argument("--dataset", default="test.jsonl", help="Skills JSONL file")
    ap.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES)
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k for LLM selection")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model for selection")
    ap.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run only BM25 retrieval eval (no API key needed)",
    )
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

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
        print(f"  ... and {len(queries) - 5} more\n")

    # Save queries for reproducibility
    with open(os.path.join(out, "test_queries.json"), "w") as fh:
        json.dump(queries, fh, indent=2)

    # ── Phase 1: BM25 retrieval ───────────────────────────────────────────
    k_values = K_VALUES
    lib_sizes = sorted(set(LIBRARY_SIZES + [len(skills)]))

    print("=" * 62)
    print("Phase 1: BM25 Retrieval Evaluation")
    print(f"  k values      : {k_values}")
    print(f"  library sizes  : {lib_sizes}")
    print("=" * 62)

    ret_df = evaluate_retrieval(queries, skills, k_values, lib_sizes, args.seed)

    pivot = ret_df.pivot_table(index="library_size", columns="k", values="recall_at_k")
    print("\nCandidate Recall@k by Library Size:\n")
    print(tabulate(pivot, headers="keys", tablefmt="github", floatfmt=".3f"))
    print()

    # Save markdown table
    md_table = tabulate(pivot, headers="keys", tablefmt="github", floatfmt=".3f")
    with open(os.path.join(out, "retrieval_table.md"), "w") as fh:
        fh.write("## BM25 Candidate Recall@k\n\n")
        fh.write(md_table + "\n")

    plot_recall_at_k(ret_df, out)
    plot_scaling(ret_df, out)
    print(f"  Retrieval plots saved to {out}/\n")

    # ── Phase 2: LLM selection ────────────────────────────────────────────
    sel_metrics: dict = {}
    if not args.retrieval_only:
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "ERROR: OPENAI_API_KEY not set.\n"
                "  Set the key or run with --retrieval-only."
            )
            sys.exit(1)

        print("=" * 62)
        print(f"Phase 2: LLM Selection  (model={args.model}, k={args.k})")
        print("=" * 62)

        sel_df = evaluate_selection(
            queries, skills, args.model, k=args.k, seed=args.seed
        )

        sel_metrics = {
            "model": args.model,
            "k": args.k,
            "n_queries": len(sel_df),
            "candidate_recall": round(sel_df["candidate_recall"].mean(), 4),
            "exact_match": round(sel_df["exact_match"].mean(), 4),
            "fuzzy_match": round(sel_df["fuzzy_match"].mean(), 4),
            "avg_latency_s": round(sel_df["latency_s"].mean(), 3),
        }

        print(f"\n{'─' * 40}")
        print("Selection summary:")
        for mk, mv in sel_metrics.items():
            if isinstance(mv, float):
                print(f"  {mk:25s}: {mv:.3f}")
            else:
                print(f"  {mk:25s}: {mv}")

        # Markdown table
        sel_tbl = pd.DataFrame(
            [
                ["Candidate Recall", f"{sel_metrics['candidate_recall']:.1%}"],
                ["Exact Match", f"{sel_metrics['exact_match']:.1%}"],
                ["Fuzzy Match", f"{sel_metrics['fuzzy_match']:.1%}"],
                ["Avg Latency (s)", f"{sel_metrics['avg_latency_s']:.2f}"],
            ],
            columns=["Metric", "Value"],
        )
        print("\n" + tabulate(sel_tbl, headers="keys", tablefmt="github", showindex=False))

        with open(os.path.join(out, "selection_table.md"), "w") as fh:
            fh.write(f"## LLM Selection Metrics ({args.model}, k={args.k})\n\n")
            fh.write(tabulate(sel_tbl, headers="keys", tablefmt="github", showindex=False))
            fh.write("\n")

        plot_selection(sel_df, out)
        sel_df.to_csv(os.path.join(out, "selection_details.csv"), index=False)
        print(f"\n  Selection results saved to {out}/")

    # ── Save combined metrics ─────────────────────────────────────────────
    all_metrics = {
        "config": {
            "dataset": args.dataset,
            "num_queries": len(queries),
            "seed": args.seed,
            "k_values": k_values,
            "library_sizes": lib_sizes,
            "document_repr": "name + description",
        },
        "retrieval": ret_df.to_dict(orient="records"),
    }
    if sel_metrics:
        all_metrics["selection"] = sel_metrics

    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(all_metrics, fh, indent=2)

    print(f"\n{'=' * 62}")
    print(f"All results saved to {out}/")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
