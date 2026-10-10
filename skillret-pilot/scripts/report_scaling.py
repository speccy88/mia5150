"""Summarize the saved three-query/five-model scaling study; no API calls."""

from collections import Counter
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import ROOT, read_json, rows, save_json
from agent_runner import Budget
from handoff import mean, write_csv


def figures(summary, spec):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    metrics = [
        ("conditional_exact", "Exact-set fraction; cells show exact / valid"),
        ("mean_logical_tokens", "Mean logical tokens per returned workflow"),
        ("candidate_recall", "Mean candidate recall"),
    ]
    out = ROOT / "reports/figures"
    out.mkdir(exist_ok=True)
    for metric, title in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        values = [r[metric] for r in summary if r[metric] is not None]
        options = (
            {
                "norm": LogNorm(
                    vmin=max(1, min(values, default=1)),
                    vmax=max(2, max(values, default=2)),
                )
            }
            if metric == "mean_logical_tokens"
            else {"vmin": 0, "vmax": 1}
        )
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")
        for ax, method in zip(axes.flat, spec["methods"]):
            grid = [
                [
                    next(
                        r
                        for r in summary
                        if (r["model"], r["N"], r["method"]) == (model, n, method)
                    )
                    for n in spec["sizes"]
                ]
                for model in spec["models"]
            ]
            data = np.array(
                [
                    [r[metric] if r[metric] is not None else np.nan for r in line]
                    for line in grid
                ]
            )
            plot = ax.imshow(
                np.ma.masked_invalid(data), cmap=cmap, aspect="auto", **options
            )
            for i, line in enumerate(grid):
                for j, r in enumerate(line):
                    value = r[metric]
                    label = (
                        "—"
                        if value is None
                        else (
                            f"{int(r['exact'])}/{r['valid']}"
                            if metric == "conditional_exact"
                            else (
                                f"{value:,.0f}"
                                if metric == "mean_logical_tokens"
                                else f"{value:.2f}"
                            )
                        )
                    )
                    if value is not None and metric != "conditional_exact":
                        count = (
                            r["usage_cases"]
                            if metric == "mean_logical_tokens"
                            else r["retrieval_cases"]
                        )
                        label += f"\n(n={count})"
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=9,
                        bbox={
                            "facecolor": "white",
                            "alpha": 0.8,
                            "edgecolor": "none",
                            "pad": 1.5,
                        },
                    )
            ax.set_title(method)
            ax.set_xticks(range(3), spec["sizes"])
            ax.set_yticks(range(5), spec["models"], fontsize=8)
            ax.set_xlabel("Skills in library (N)")
        fig.colorbar(
            plot,
            ax=list(axes.flat),
            shrink=0.85,
            label="Tokens (log scale)"
            if metric == "mean_logical_tokens"
            else "Fraction",
        )
        fig.suptitle(
            title
            + "\nThree fixed pilot queries; grey cells mean this metric is unavailable",
            fontsize=12,
        )
        fig.savefig(out / f"openrouter_scaling_{metric}.png", dpi=150)
        plt.close(fig)


def report():
    spec = read_json("manifests/openrouter_scaling.json")
    raw = [
        r
        for r in rows("results/openrouter/scaling/trial.jsonl")
        if r["run_id"] == spec["run_id"]
    ]
    cases = list({r["case_key"]: r for r in raw}.values())
    summary = []
    strata = []
    for model in spec["models"]:
        for n in spec["sizes"]:
            for method in spec["methods"]:
                group = [
                    r
                    for r in cases
                    if (r["model"], r["N"], r["strategy"]) == (model, n, method)
                ]
                valid = [r for r in group if r["status"] == "valid"]
                attempted = [r for r in group if r["attempt_count"] > 0]
                usage = [r["logical_usage"] for r in attempted if "logical_usage" in r]
                row = {
                    "model": model,
                    "N": n,
                    "method": method,
                    "planned": 3,
                    "saved": len(group),
                    "attempted": len(attempted),
                    "valid": len(valid),
                    "exact": sum(r["selection"]["exact_set_match"] for r in valid),
                    "conditional_exact": mean(
                        [r["selection"]["exact_set_match"] for r in valid]
                    ),
                    "precision": mean([r["selection"]["precision"] for r in valid]),
                    "recall": mean([r["selection"]["recall"] for r in valid]),
                    "f1": mean([r["selection"]["f1"] for r in valid]),
                    "retrieval_cases": sum("candidate_recall" in r for r in group),
                    "candidate_recall": mean(
                        [
                            r["candidate_recall"]
                            for r in group
                            if "candidate_recall" in r
                        ]
                    ),
                    "candidate_completeness": mean(
                        [
                            r["candidate_completeness"]
                            for r in group
                            if "candidate_completeness" in r
                        ]
                    ),
                    "mean_logical_tokens": mean(
                        [u["input_tokens"] + u["output_tokens"] for u in usage]
                    ),
                    "usage_cases": len(usage),
                    "logical_usd": sum(
                        u["actual_usd"] for u in usage if u["actual_usd"] is not None
                    ),
                    "mean_logical_seconds": mean(
                        [
                            r["logical_workflow_seconds"]
                            for r in attempted
                            if "logical_workflow_seconds" in r
                        ]
                    ),
                    "errors": dict(
                        Counter(
                            r.get("error_category") or r["status"]
                            for r in group
                            if r["status"] != "valid"
                        )
                    ),
                }
                summary.append(row)
                for kind in ["single", "multiple"]:
                    sub = [
                        r
                        for r in group
                        if (len(r["evaluator_only_gold"]) == 1) == (kind == "single")
                    ]
                    good = [r for r in sub if r["status"] == "valid"]
                    strata.append(
                        {
                            "model": model,
                            "N": n,
                            "method": method,
                            "gold_group": kind,
                            "planned": 2 if kind == "single" else 1,
                            "saved": len(sub),
                            "valid": len(good),
                            "exact": sum(
                                r["selection"]["exact_set_match"] for r in good
                            ),
                        }
                    )
    prior = read_json("evidence/openrouter_scaling_start.json")[
        "previous_completed_budget"
    ]
    budget = Budget().totals()
    new_requests = [
        r
        for r in rows("evidence/traces/requests.jsonl")
        if r["case_id"] in {c["case_key"] for c in cases if not c.get("source_reused")}
    ]
    route_rows = [
        r
        for r in rows("results/openrouter/scaling/routes.jsonl")
        if r["run_id"] == spec["run_id"]
    ]
    ids = {r["request_id"] for r in new_requests} | {
        c["request_id"] for r in route_rows for c in r["calls"]
    }
    physical = {
        r["request_id"]: r
        for r in rows("evidence/traces/responses.jsonl")
        if r["request_id"] in ids
    }
    costs = []
    for label in spec["models"]:
        model = spec["sources"][label]["llm"]["model"]
        matched = [c for c in physical.values() if c["pin"]["model"] == model]
        costs.append(
            {
                "model": label,
                "new_returned_generations": len(matched),
                "new_actual_usd": sum(c["actual_usd"] for c in matched),
                "new_input_tokens": sum(c["usage"]["input_tokens"] for c in matched),
                "new_output_tokens": sum(c["usage"]["output_tokens"] for c in matched),
            }
        )
    result = {
        "run_id": spec["run_id"],
        "query_ids": spec["query_ids"],
        "planned": 180,
        "saved": len(cases),
        "reused_cases": sum(bool(r.get("source_reused")) for r in cases),
        "attempted": sum(r["attempt_count"] > 0 for r in cases),
        "valid": sum(r["status"] == "valid" for r in cases),
        "exact": sum(
            r.get("selection", {}).get("exact_set_match", 0)
            for r in cases
            if r["status"] == "valid"
        ),
        "new_attempts": budget["requests"] - prior["requests"],
        "incremental_actual_usd": budget["actual_usd"] - prior["actual_usd"],
        "budget": budget,
        "estimated_cad": budget["actual_usd"] * 1.5,
        "summary": summary,
        "costs": costs,
        "failures": dict(
            Counter(
                r.get("error_category") or r["status"]
                for r in cases
                if r["status"] != "valid"
            )
        ),
    }
    save_json("reports/OPENROUTER_SCALING.json", result)
    write_csv("reports/OPENROUTER_SCALING.csv", summary)
    write_csv("reports/OPENROUTER_SCALING_SINGLE_MULTI.csv", strata)
    write_csv("reports/OPENROUTER_SCALING_COSTS.csv", costs)
    figures(summary, spec)
    write_csv(
        "reports/OPENROUTER_SCALING_FAILURES.csv",
        [
            {
                k: r.get(k)
                for k in [
                    "model",
                    "query_id",
                    "N",
                    "strategy",
                    "status",
                    "error_category",
                    "source_reused",
                    "case_key",
                ]
            }
            for r in cases
            if r["status"] != "valid"
        ],
    )
    lines = []
    for label in spec["models"]:
        for n in spec["sizes"]:
            group = [r for r in summary if (r["model"], r["N"]) == (label, n)]
            cells = [
                f"{int(r['exact'])}/{r['valid']}" if r["valid"] else "— (0 valid)"
                for r in group
            ]
            lines.append(
                f"| {label} | {n} | "
                + " | ".join(cells)
                + f" | {sum(r['attempted'] for r in group)}/12 |"
            )
    table = "\n".join(lines)
    cost_table = "\n".join(
        f"| {r['model']} | {r['new_returned_generations']} | ${r['new_actual_usd']:.6f} |"
        for r in costs
    )
    failure_text = "\n".join(
        f"- {name}: {count} cases." for name, count in result["failures"].items()
    )
    largest = [r for r in cases if r["N"] == 1000 and r["status"] == "valid"]
    overlap = sum(
        bool(set(r["returned_ids"]) & set(r["evaluator_only_gold"])) for r in largest
    )
    extras = sum(
        bool(set(r["returned_ids"]) - set(r["evaluator_only_gold"])) for r in largest
    )
    large_exact = sum(r["selection"]["exact_set_match"] for r in largest)
    text = f"""# OpenRouter scaling benchmark: measured results

**{result['valid']} valid selections and {int(result['exact'])} exact matches from {result['attempted']} attempted cases; {result['saved']}/180 planned cases recorded.** Five fixed models, three existing pilot queries (two single-positive, one multi-positive), N=50/250/1000, and four methods. {result['reused_cases']} N=50 records were explicitly reused, including failures; the added sizes are new live calls through real Deep Agents SkillsMiddleware and candidate-only StateBackend.

| Model | N | Listing exact/valid | BM25 exact/valid | Vector exact/valid | Domain-vector exact/valid | Attempts/planned |
|---|---:|---:|---:|---:|---:|---:|
{table}

These are conditional exact-set counts, not accuracy on a common valid denominator. Invalid outputs, context exclusions, provider pauses, and gating reduce coverage. Missing results are never filled with invented successes or treated as successful empty selections. All denominators and precision/recall/F1 are in [CSV](OPENROUTER_SCALING.csv); [single/multiple-positive strata](OPENROUTER_SCALING_SINGLE_MULTI.csv) remain separate.

At N=1,000, **{int(large_exact)}/{len(largest)} valid decisions matched the complete annotated set**. However, {overlap}/{len(largest)} included at least one annotated skill, and {extras}/{len(largest)} included IDs outside the annotated set. Low exact-set accuracy therefore does not mean no relevant skills were found. These extra IDs are false positives under the pinned qrels, not proof that those skills are useless in practice. The sample and differing valid denominators do not support a general model ranking.

## Cost and provenance

This expansion made **{result['new_attempts']} additional reserved attempts**, costing **US${result['incremental_actual_usd']:.9f}**. Whole-project ledger: **US${budget['actual_usd']:.9f}**, estimated **CAD${budget['actual_usd']*1.5:.9f}** using the declared 1.50 planning factor. {budget['requests']}/300 attempts; unresolved reserve US${budget['unresolved_reserved_usd']:.9f}. Provider-returned costs, request IDs and raw sanitized traces are saved. Zero-charge rejections require explicit reconciliation. USD3/CAD5 ceilings remain unchanged.

| Model | New returned generations | Additional actual USD |
|---|---:|---:|
{cost_table}

The original pre-credit full-document BM25 baseline remains **Recall@5 = 0.516667**, across **200 fixed queries / 6,006 skills**, at dataset revision `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`. It is a different representation and denominator from this metadata-only selection study and is not a published-score reproduction.

## Protocol and limits

The three query IDs ({', '.join(spec['query_ids'])}) were fixed before this expansion to fit the remaining shared request cap. The full 20-query × three-size × four-method × five-model sweep would require 1,200 selectors plus routers; it was not run. No final-holdout queries were consumed. Gold-preserving nested libraries and stable display order are identical across models/methods. Listing coverage is 1 by construction. No prompt or output repair was introduced.

Each model retains its prior provider, temperature0, 512-token output cap, 96-token router cap, 90-second timeout and fixed UUID clarification. GLM requests low reasoning; the other models request none. Current public endpoint metadata was refreshed before calls; no fallback occurs. A passing native canary and earlier valid SkillRet selections establish integration. Invalid benchmark outputs are outcomes, not grounds to re-sample for correctness. Domain-vector still requires valid listing/BM25/vector outputs for that query and N, which creates selection bias in its conditional score.

Existing matching N=50 decisions and query-only domain routes are reused with explicit source case IDs. Logical domain tokens/cost include the router for each workflow; physical costs deduplicate requests and include only newly charged calls in the expansion table. Candidate recall measures retrieval coverage separately from selection. Strict byte-based context preflight can conservatively exclude listing even if a tokenizer-specific count might fit; no truncation or hidden summarization occurs.

Three synthetic queries cannot establish general superiority or a statistically reliable scaling curve. Differences in reasoning, hosted provider quantization and cache state also limit model comparisons. Actual native reads were already demonstrated separately: the preceding five-model activation study recorded 16 reads across 25 cases, including four real SkillRet candidate reads. No additional activation calls or downloaded skill execution are part of this scaling study.

![Exact-set fraction among valid selections; labels show valid query counts](figures/openrouter_scaling_conditional_exact.png)

![Logical tokens, including reused query-only routes](figures/openrouter_scaling_mean_logical_tokens.png)

![Candidate retrieval recall](figures/openrouter_scaling_candidate_recall.png)

## Failures and blocked work

{failure_text or 'No saved failures.'}

Unknown IDs and malformed selections are retained. A transport failure pauses that provider's remaining new cases in this batch; saved prior cases stay available. Ambiguous billing stops all further paid work until reconciled. Read [failure rows](OPENROUTER_SCALING_FAILURES.csv) and the raw traces before interpreting the conditional scores.

## Resume and review

```bash
.venv/bin/python scripts/benchmark_openrouter.py --live
.venv/bin/python scripts/report_scaling.py
.venv/bin/python scripts/audit_scaling.py
.venv/bin/python scripts/verify.py
.venv/bin/python pilot.py package
```

The live command requires the private key and ALLOW_PAID_RUNS=1 and reuses existing cases. Report/audit commands make no model calls. The two notebooks default to saved results/offline checks. A model pause is retained; changing provider or retry policy requires an explicitly separate protocol, not deleting the ledger.
"""
    (ROOT / "reports/OPENROUTER_SCALING.md").write_text(text)
    return result


if __name__ == "__main__":
    r = report()
    print(
        json.dumps(
            {k: v for k, v in r.items() if k not in {"summary", "costs"}}, indent=2
        )
    )
