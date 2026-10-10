"""Build the five-model readout entirely from saved evidence; no API requests."""

from collections import Counter
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import ROOT, rows, read_json, save_json, config, digest
from agent_runner import Budget
from local_runner import native_read_events
from trial import run_config
from handoff import write_csv, mean
from scripts.run_openrouter_models import MODELS


def current_records(label):
    cfg = config(str(ROOT / "manifests/configs" / f"{label}.json"))
    pin = read_json(cfg["llm"]["pin_manifest"])
    run_id = digest(run_config(cfg, pin))
    trial = [
        r
        for r in rows(f"results/openrouter/{label}/trial.jsonl")
        if r["run_id"] == run_id
    ]
    trial = list({r["case_key"]: r for r in trial}.values())
    qids = ["fixture-" + str(i) for i in range(3)] + read_json(
        "manifests/live_query_ids.json"
    )[:2]
    activation_keys = {digest([run_config(cfg, pin), "activation", q]) for q in qids}
    activation = [
        r
        for r in rows(f"results/openrouter/{label}/activation.jsonl")
        if r["case_key"] in activation_keys
    ]
    smoke = rows(f"results/openrouter/{label}/smoke.jsonl")
    routes = rows(f"results/openrouter/{label}/routes.jsonl")
    return cfg, pin, run_id, trial, activation, smoke, routes


def report():
    summaries = []
    models = []
    strata = []
    failures = []
    responses = rows("evidence/traces/responses.jsonl")
    for label in MODELS:
        cfg, pin, run_id, trial, activation, smoke, routes = current_records(label)
        # One physical request can appear in several workflow histories: dedupe.
        calls = {
            c["request_id"]: c for c in responses if c["pin"]["model"] == pin["model"]
        }
        requests = {
            c["request_id"]: c
            for c in rows("evidence/traces/requests.jsonl")
            if c["pin"]["model"] == pin["model"]
        }
        native = sum(len(native_read_events(a)) for a in activation)
        model = {
            "label": label,
            "model": pin["model"],
            "provider": pin["provider_tag"],
            "run_id": run_id,
            "canary_passed": smoke[-1]["canary_correct"] if smoke else False,
            "canary_attempts": len(smoke),
            "selector_planned": 20,
            "selector_saved": len(trial),
            "selector_attempted": sum(r["attempt_count"] > 0 for r in trial),
            "selector_valid": sum(r["status"] == "valid" for r in trial),
            "selector_exact": sum(
                r.get("selection", {}).get("exact_set_match", 0)
                for r in trial
                if r["status"] == "valid"
            ),
            "activation_saved": len(activation),
            "activation_valid": sum(r["status"] == "valid" for r in activation),
            "activation_fixtures_valid": sum(
                r["status"] == "valid" and r["query_id"].startswith("fixture-")
                for r in activation
            ),
            "activation_real_valid": sum(
                r["status"] == "valid" and not r["query_id"].startswith("fixture-")
                for r in activation
            ),
            "actual_native_reads": native,
            "physical_attempts_all_configs": len(requests),
            "returned_generations_all_configs": len(calls),
            "actual_usd_all_configs": sum(c["actual_usd"] for c in calls.values()),
            "input_tokens_all_configs": sum(
                c["usage"]["input_tokens"] for c in calls.values()
            ),
            "output_tokens_all_configs": sum(
                c["usage"]["output_tokens"] for c in calls.values()
            ),
            "reasoning_tokens_all_configs": sum(
                (c["usage"].get("output_token_details") or {}).get("reasoning", 0)
                for c in calls.values()
            ),
            "finish_reasons_all_configs": dict(
                Counter(
                    c["raw_response"]["choices"][0].get("finish_reason")
                    for c in calls.values()
                )
            ),
            "reasoning_setting": pin["reasoning"],
        }
        models.append(model)
        for method in ["listing", "bm25", "vector", "domain_vector"]:
            group = [r for r in trial if r["strategy"] == method]
            valid = [r for r in group if r["status"] == "valid"]
            attempted = [r for r in group if r["attempt_count"] > 0]
            logical = [r["logical_usage"] for r in attempted if "logical_usage" in r]
            row = {
                "model": label,
                "method": method,
                "N": 50,
                "planned": 5,
                "saved": len(group),
                "attempted": len(attempted),
                "valid": len(valid),
                "exact": sum(r["selection"]["exact_set_match"] for r in valid),
                "conditional_exact": mean(
                    [r["selection"]["exact_set_match"] for r in valid]
                ),
                "exact_per_planned": sum(
                    r["selection"]["exact_set_match"] for r in valid
                )
                / 5,
                "precision": mean([r["selection"]["precision"] for r in valid]),
                "recall": mean([r["selection"]["recall"] for r in valid]),
                "f1": mean([r["selection"]["f1"] for r in valid]),
                "retrieval_cases": sum("candidate_recall" in r for r in group),
                "candidate_recall": mean(
                    [r["candidate_recall"] for r in group if "candidate_recall" in r]
                ),
                "candidate_completeness": mean(
                    [
                        r["candidate_completeness"]
                        for r in group
                        if "candidate_completeness" in r
                    ]
                ),
                "logical_usage_cases": len(logical),
                "mean_logical_tokens": mean(
                    [u["input_tokens"] + u["output_tokens"] for u in logical]
                ),
                "logical_usd": sum(
                    u["actual_usd"] for u in logical if u["actual_usd"] is not None
                ),
                "mean_logical_seconds": mean(
                    [r["logical_workflow_seconds"] for r in attempted]
                ),
                "errors": dict(
                    Counter(
                        r.get("error_category") or r["status"]
                        for r in group
                        if r["status"] != "valid"
                    )
                ),
            }
            summaries.append(row)
            for kind in ["single", "multiple"]:
                sub = [
                    r
                    for r in group
                    if (len(r["evaluator_only_gold"]) == 1) == (kind == "single")
                ]
                sv = [r for r in sub if r["status"] == "valid"]
                strata.append(
                    {
                        "model": label,
                        "method": method,
                        "gold_group": kind,
                        "saved": len(sub),
                        "valid": len(sv),
                        "exact": sum(r["selection"]["exact_set_match"] for r in sv),
                    }
                )
        for kind, records in [
            ("selection", trial),
            ("activation", activation),
            ("canary", smoke),
        ]:
            for r in records:
                if r["status"] != "valid":
                    failures.append(
                        {
                            "model": label,
                            "phase": kind,
                            "query": r.get("query_id"),
                            "case_key": r["case_key"],
                            "status": r["status"],
                            "error": r.get("error_category"),
                            "native_reads_before_failure": len(native_read_events(r))
                            if kind == "activation"
                            else 0,
                        }
                    )
    budget = Budget().totals()
    baseline = rows("results/baseline_summary.jsonl")[-1]
    result = {
        "scope": "Five pilot queries per model; N=50 only; UUID contract v2; selection and activation separate.",
        "models": models,
        "selection": summaries,
        "failures": failures,
        "budget": budget,
        "estimated_cad_at_1_50": budget["actual_usd"] * 1.5,
        "full_document_bm25": baseline,
    }
    save_json("reports/OPENROUTER_MODELS.json", result)
    write_csv("reports/OPENROUTER_MODELS.csv", summaries)
    write_csv("reports/OPENROUTER_SINGLE_MULTI.csv", strata)
    write_csv("reports/OPENROUTER_FAILURES.csv", failures)
    total_valid = sum(m["selector_valid"] for m in models)
    total_attempted = sum(m["selector_attempted"] for m in models)
    total_saved = sum(m["selector_saved"] for m in models)
    total_exact = sum(m["selector_exact"] for m in models)
    table = "\n".join(
        f"| {m['label']} | {m['provider']} | {'pass' if m['canary_passed'] else 'FAIL'} | {m['selector_valid']}/{m['selector_attempted']}/20 | {int(m['selector_exact'])} | {m['activation_valid']}/{m['activation_saved']} | {m['actual_native_reads']} | ${m['actual_usd_all_configs']:.6f} |"
        for m in models
    )
    methods = "\n".join(
        f"| {r['model']} | {r['method']} | {r['attempted']}/5 | {r['valid']} | {int(r['exact'])}/{r['valid']} | {r['candidate_recall']:.3f} |"
        if r["candidate_recall"] is not None
        else f"| {r['model']} | {r['method']} | {r['attempted']}/5 | {r['valid']} | — | — |"
        for r in summaries
    )
    failure_text = "\n".join(
        f"- {r['model']} {r['phase']} {r['query'] or 'fixture'}: {r['status']}; {r['error'] or 'strict output validation'}."
        for r in failures
    )
    text = f"""# Five OpenRouter models: measured pilot

**{total_valid} valid selector decisions out of 100 planned** on five fixed SkillRet pilot queries, N=50. Models use real Deep Agents SkillsMiddleware and candidate-only StateBackend through native ChatOpenRouter. Valid formatting does not imply correct selection. All results below are regenerated from saved rows.

There are {total_attempted} attempted selector cases, {int(total_exact)} exact sets, {total_saved-total_attempted} gated cases and {100-total_saved} unrun cases. Qwen3.8 27B expansion stopped after two distinct listing requests hit Darkbloom shared-pool rate limits; successful short-prompt results do not complete its comparison. Unknown IDs and malformed calls count as failures, and are never repaired. Larger live libraries remain unrun.

| Model | Pinned provider | Current canary | Valid / attempted / planned selectors | Exact sets | Valid activation / attempted | Actual native reads | All-config USD |
|---|---|---|---:|---:|---:|---:|---:|
{table}

Costs include initial and revised canaries, selectors, domain routers and activation; model subtotals include the earlier credential check. Whole-project paid ledger: **US${budget['actual_usd']:.9f}**, approximately **CAD {budget['actual_usd']*1.5:.9f}** at the declared planning factor 1.50. {budget['requests']} reserved attempts; unresolved reserve **US${budget['unresolved_reserved_usd']:.9f}**. This conversion is an estimate; bank fees/exchange and local hardware electricity are not measured. Persistent ceilings remain USD 3, CAD 5, 300 requests. No automatic retries or provider fallback.

## Selection and retrieval

The first pre-credit full-document BM25 result remains **Recall@5 {baseline['metrics']['recall_at_5']:.6f} (51.67%) across {baseline['query_count']} queries / {baseline['corpus_size']:,} skills**; Recall@10 {baseline['metrics']['recall_at_10']:.6f}. It is a partial-protocol check, not full published-score reproduction. It uses a different representation and denominator from this small metadata selector study.

| Model | Method | Attempts / planned | Valid | Exact / valid | Candidate recall |
|---|---|---:|---:|---:|---:|
{methods}

All models share the same five queries (three single-positive, two multi-positive), N=50 gold-preserving libraries, candidate order, capped metadata, prompts and tool schema. Retrieval shortlists have at most five candidates. Listing candidate coverage is 1 by construction. Domain routing occurs only when listing/BM25/vector outputs for that query are valid; otherwise it is recorded as gated, not treated as a model answer. Thus conditional accuracy denominators can differ. CSVs give precision/recall/F1, single/multiple strata, planned coverage, tokens, latency and routing cost. Logical domain cost includes the router even if reused.

## Integration and controlled changes

Initial Qwen Flash and DeepSeek canaries returned s-prefixed native names instead of original UUIDs. Strict validation rejected them. One fixed input/schema UUID clarification, already used by the local Ollama trial, was applied uniformly to all five models before benchmark selection. Original failures remain saved; no output was repaired or sampled repeatedly for correctness. Real native metadata injection and tool isolation remain active.

GLM's pinned endpoint rejected reasoning=none with HTTP 400. Provider key cumulative usage confirmed no generation charge, and an explicit reconciliation settled that reservation at zero. GLM uses reasoning effort=low; other models request none. Output remains bounded at 512 tokens, routing at 96, temperature 0, timeout 90 seconds. This reasoning difference limits model-to-model latency/cost comparison. Provider names, model IDs, completion IDs, returned reasoning tokens, quantization tags and current price snapshots are retained. Cache-write pricing is conservatively reserved in addition to uncached prompt price for every input token.

Each successful metadata workflow has one model call and only select_skills offered. Separate activation offers bounded native read_file plus select_skills, at most four calls and two candidate reads. Actual native reads include completed ToolMessage outputs before later denials. A read is evidence of progressive disclosure, never skill execution. Downloaded instructions and links were not executed.

## Limits and saved failures

Five pilot queries are too few to establish a model or method winner. N=250/1000 live scaling remains unrun in this follow-up. The 4,192 reserved final-query texts were not consumed. Hosted endpoints/weights and cache state can change; temperature zero is not bitwise reproducibility. Local Ollama comparison is exploratory: it shares UUID clarification and query/library selection, but has different model sizes, quantization, runtime and generation controls. See the separate [local report](OLLAMA_TRIAL_REPORT.md).

{failure_text or 'No recorded failures in these configurations.'}

## Reproduce or resume

```bash
.venv/bin/python scripts/run_openrouter_models.py --stage all
.venv/bin/python scripts/report_openrouter.py
.venv/bin/python scripts/audit_openrouter.py
.venv/bin/python scripts/verify.py
.venv/bin/python pilot.py package
```

The first command requires the privately configured key and ALLOW_PAID_RUNS=1. Matching cases, including failed outputs, are reused. Do not delete ledgers or prior results. Configurations are in manifests/configs; raw traces in evidence/traces; per-model results in results/openrouter. Both notebooks default to saved/offline evidence without model calls.
"""
    (ROOT / "reports/OPENROUTER_MODELS.md").write_text(text)
    return result


if __name__ == "__main__":
    r = report()
    print(json.dumps({"models": r["models"], "budget": r["budget"]}, indent=2))
