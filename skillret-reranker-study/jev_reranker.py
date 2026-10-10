#!/usr/bin/env python3
"""Isolated, resumable BM25 -> Jev development trial; no skill execution.

The runner reads credentials ONLY from os.environ. Supply its environment from
your terminal, or use python-dotenv's launcher outside this module. No key or
request headers are written. All prior project artifacts stay unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
import numpy as np

from pilot import (ROOT, append_row, bm25_rank, digest, gold_map, load_data,
                   read_json, rows)
from reference_metrics import retrieval_metrics, selection_metrics
from agent_runner import Blocked, Budget

OUTPUT = ROOT / "results/jev"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sanitized(value):
    """Remove credential fields and any exact environment key before persistence."""
    if isinstance(value, dict):
        return {k: sanitized(v) for k, v in value.items()
                if k.lower() not in {"authorization", "api_key", "typesafe_api_key"}}
    if isinstance(value, (list, tuple)):
        return [sanitized(v) for v in value]
    if isinstance(value, str):
        for name in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY"):
            key = os.environ.get(name)
            if key:
                value = value.replace(key, "[REDACTED]")
    return value


def save_once(path, value):
    """Immutable artifact: identical replay is allowed, conflicting writes fail."""
    path = Path(path)
    text = json.dumps(sanitized(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        if path.read_text() != text:
            raise FileExistsError(f"Refusing to overwrite {path}") from None


def build_jev_questions(query, candidates, model="jev-1.13.0", prompt=None):
    """Whitelist canonical query + exact cached metadata; no labels/ranks/scores.

    The cached string already contains the name and description under the common
    cap. Keeping it combined avoids reintroducing uncapped source names/text.
    UUID order conceals BM25 display rank. Answer keys map to exact original IDs.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("empty_query")
    ids = [c["id"] for c in candidates]
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(s, str) or not s for s in ids):
        raise ValueError("invalid_candidate_ids")
    prompt = prompt or read_json("prompts/jev_relevance.json")
    state, questions = {"query": query, "candidates": {}}, {}
    for c in sorted(candidates, key=lambda c: c["id"]):
        if not isinstance(c["metadata"], str) or not c["metadata"].strip():
            raise ValueError("empty_metadata")
        sid = c["id"]
        state["candidates"][sid] = {"name_and_description": c["metadata"]}
        questions[sid] = {
            "type": "noul",
            "instructions": prompt["instructions"].format(
                candidate_path=f'`candidates[{json.dumps(sid)}].name_and_description`'),
            "criteria": dict(prompt["criteria"]),
        }
    return {"state": state, "model": model, "questions": questions}


def parse_scores(response, candidate_ids, model):
    if not isinstance(response, dict) or response.get("model") != model:
        raise ValueError("model_mismatch")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(candidate_ids):
        raise ValueError("answer_ids_mismatch")
    scores = {}
    for sid in candidate_ids:
        answer = answers[sid]
        value = answer.get("noul") if isinstance(answer, dict) else None
        if (not isinstance(answer, dict) or answer.get("type") != "noul"
                or isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("invalid_probability")
        scores[sid] = float(value)
    return scores


def rerank_with_jev(candidate_ids, scores, k=5):
    if len(set(candidate_ids)) != len(candidate_ids) or set(candidate_ids) != set(scores):
        raise ValueError("reranking_ids_mismatch")
    # Stable sorting: equal probabilities retain the original BM25 rank.
    return sorted(candidate_ids, key=lambda sid: -scores[sid])[:k]


def evaluate(ranking, top5, gold):
    """Evaluator-only multi-label diagnostics; categories may overlap."""
    gold = set(gold)
    before, covered = set(ranking[:5]) & gold, set(ranking) & gold
    result = {
        "bm25_at_5": retrieval_metrics(ranking, gold, 5),
        "bm25_at_20": retrieval_metrics(ranking, gold, 20),
        "bm25_selection": selection_metrics(ranking[:5], gold),
        "upstream_missing_ids": sorted(gold - covered),
        "upstream_retrieval_failure": covered != gold,
        "no_gold_in_candidates": not covered,
    }
    if top5 is None:
        return {**result, "jev_at_5": None, "jev_selection": None,
                "paired_recall_difference": None, "categories": ["service_failure"],
                "rescued_ids": [], "removed_ids": [], "retained_ids": [],
                "candidate_gold_missed_ids": sorted(covered)}
    after = set(top5) & gold
    categories = []
    if covered != gold:
        categories.append("upstream_retrieval_failure")
    if covered - after:
        categories.append("jev_reranking_failure")
    if after - before:
        categories.append("jev_reranking_success")
    if after & before:
        categories.append("unchanged_success")
    jev = retrieval_metrics(top5, gold, 5)
    return {**result, "jev_at_5": jev, "jev_selection": selection_metrics(top5, gold),
            "paired_recall_difference": jev["recall_at_5"] - result["bm25_at_5"]["recall_at_5"],
            "categories": categories, "rescued_ids": sorted(after - before),
            "removed_ids": sorted(before - after), "retained_ids": sorted(after & before),
            "candidate_gold_missed_ids": sorted(covered - after)}


def request_preflight(payload, cfg):
    encoded = json.dumps(payload, ensure_ascii=False).encode()
    # Byte counts are conservative screening estimates, not provider token counts.
    state_bytes = len(json.dumps(payload["state"], ensure_ascii=False).encode())
    longest = max(len(json.dumps(q, ensure_ascii=False).encode()) for q in payload["questions"].values())
    if len(encoded) + 1024 > cfg["context_tokens"] or state_bytes + longest + 1024 > cfg["state_plus_question_tokens"]:
        raise Blocked("context_preflight; no truncation permitted")
    return {"request_bytes": len(encoded), "state_plus_longest_question_bytes": state_bytes + longest}


def reserve_per_call(cfg):
    return float(Decimal(cfg["reserve_input_tokens_per_call"]) *
                 Decimal(str(cfg["input_usd_per_million"])) / Decimal(1000000) *
                 Decimal(str(cfg["budget"]["request_reserve_multiplier"])))


def prepare():
    cfg = read_json("jev.defaults.json")
    if (cfg["budget"]["max_spend_usd"] > 2 or cfg["budget"]["max_spend_cad"] > 3
            or cfg["budget"]["max_completion_requests"] > 200
            or cfg["automatic_retries"] != 0 or cfg["concurrency"] != 1):
        raise Blocked("fixed_cost_or_call_limits_exceeded")
    if (cfg["candidate_k"], cfg["final_k"], cfg["pilot_queries"]) != (20, 5, 200):
        raise Blocked("fixed_pilot_protocol_changed")
    if cfg["endpoint"] != "https://api.typesafe.ai/v1/systemone" or cfg["model"] != "jev-1.13.0":
        raise Blocked("unreviewed_endpoint_or_model")
    skills, queries, qrels = load_data()  # Existing checksum-verified loader; no downloads.
    pids = read_json("manifests/pilot_query_ids.json")
    final = read_json("manifests/final_query_ids.json")
    if len(pids) != 200 or len(set(pids)) != 200 or set(pids) & set(final) or len(final) != 4192:
        raise Blocked("pilot_reservation_mismatch")
    metadata, embedding = rows("data/metadata.jsonl"), read_json("manifests/embedding.json")
    if (digest(metadata) != embedding["metadata_hash"] or
            [m["id"] for m in metadata] != [s["id"] for s in skills]):
        raise Blocked("metadata_cache_mismatch")
    qby = {q["id"]: q["query"] for q in queries if q["id"] in set(pids)}
    gold = gold_map(qrels)
    started = time.monotonic()
    rankings = bm25_rank([m["metadata"] for m in metadata], [m["id"] for m in metadata],
                         [qby[q] for q in pids], k=20)
    retrieval_seconds = time.monotonic() - started
    previous = {r["query_id"]: r for r in rows("results/metadata_bm25_rows.jsonl")
                if r["run_id"] == embedding["spec_hash"]}
    mby = {m["id"]: m["metadata"] for m in metadata}
    cases = []
    for qid, ranking in zip(pids, rankings):
        ranking = [list(pair) for pair in ranking]
        if qid not in previous or ranking[:5] != previous[qid]["ranking"][:5]:
            raise Blocked("historical_bm25_top5_mismatch")
        if len(ranking) != 20:
            raise Blocked("short_bm25_candidates")
        candidates = [{"id": sid, "metadata": mby[sid], "bm25_score": score} for sid, score in ranking]
        payload = build_jev_questions(qby[qid], candidates, cfg["model"])
        cases.append({"query_id": qid, "query": qby[qid], "candidates": candidates,
                      "bm25_top20": ranking, "evaluator_only_gold": sorted(gold[qid]),
                      "request_preflight": request_preflight(payload, cfg)})
    git = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    spec = {"config": cfg, "dataset": read_json("manifests/dataset.json"),
            "embedding_spec_hash": embedding["spec_hash"], "metadata_hash": embedding["metadata_hash"],
            "metadata_representation": embedding["representation"],
            "metadata_implementation": "trial.vectors: first 1024 characters; tokenize; retain 126 tokens; decode; ensure <=128 including special tokens",
            "query_ids": pids, "final_ids_sha256": sha256(ROOT / "manifests/final_query_ids.json"),
            "cases_hash": digest(cases), "prompt": read_json("prompts/jev_relevance.json"),
            "code_sha256": {p: sha256(ROOT / p) for p in ["jev_reranker.py", "pilot.py", "reference_metrics.py", "agent_runner.py", "trial.py"]},
            "versions": {p: importlib.metadata.version(p) for p in ["bm25s", "numpy", "httpx", "python-dotenv"]},
            "git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "git_note": "Working tree has preexisting untracked project files; source hashes are authoritative.",
            "official_sources": read_json("evidence/jev/docs/sources.json"),
            "bm25": {"function": "pilot.bm25_rank", "method": "lucene", "k1": 1.5, "b": 0.75, "delta": 0.5, "stopwords": "en", "historical_top5_exactly_matched": 200},
            "tie_break": "descending Noul; original BM25 rank on equal probabilities",
            "request_order": "pilot manifest order; candidates and questions sorted by UUID",
            "cost_estimate": {"planned_calls": 200, "planned_judgments": 4000,
                "maximum_reserved_usd": reserve_per_call(cfg) * 200,
                "maximum_reserved_cad": reserve_per_call(cfg) * 200 * cfg["budget"]["cad_per_usd"],
                "method": "65536 input tokens (conservative full endpoint capacity) per call x published price x 1.25; output free; stages reuse prior calls",
                "cad_note": "1.50 CAD/USD planning assumption, not a verified exchange rate",
                "actual_billed_usd": None}}
    if spec["cost_estimate"]["maximum_reserved_usd"] > 2:
        raise Blocked("full_trial_estimate_exceeds_USD_2")
    run_id = digest(spec)[:20]
    folder = OUTPUT / run_id
    save_once(folder / "spec.json", spec)
    save_once(folder / "candidates.json", cases)
    if not (folder / "manifest.json").exists():
        save_once(folder / "manifest.json", {"created_at": now(), "run_id": run_id,
                  "host_platform": sys.platform, "python": sys.version.split()[0],
                  "local_bm25_seconds": retrieval_seconds, **spec})
    return folder


def check_permission():
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise Blocked('TYPESAFE_API_KEY_missing: export it privately in your terminal')
    if os.environ.get("ALLOW_PAID_RUNS") != "1":
        raise Blocked("ALLOW_PAID_RUNS_must_be_1")


def one_request(case, cfg, client, budget, case_key):
    check_permission()
    payload = build_jev_questions(case["query"], case["candidates"], cfg["model"])
    request_preflight(payload, cfg)
    ids = [sid for sid, _ in case["bm25_top20"]]
    row = {"query_id": case["query_id"], "case_key": case_key, "timestamp": now(),
           "bm25_top20": case["bm25_top20"], "request": payload,
           "evaluator_only_gold": case["evaluator_only_gold"],
           "status": "service_failure", "jev_probabilities": None, "jev_top5": None,
           "raw_response": None, "usage": None, "actual_cost_usd": None,
           "estimated_cost_from_usage_usd": None, "attempt_count": 1,
           "judgments_requested": len(ids), "model_requested": cfg["model"]}
    with budget.reserve(reserve_per_call(cfg), case_key, permission=False) as rid:
        row.update(request_id=rid, reserved_usd=reserve_per_call(cfg))
        started = time.monotonic()
        try:
            response = client.post(cfg["endpoint"], json=payload,
                headers={"Authorization": "Bearer " + os.environ["TYPESAFE_API_KEY"].strip(),
                         "Accept": "application/json"})
            row["http_status"] = response.status_code
            row["provider_request_id"] = response.headers.get("x-request-id")
            # Never retain headers. Even errors are scrubbed before writing.
            row["raw_response"] = sanitized(response.text)
            response.raise_for_status()
            body = response.json()
            row["raw_response"] = sanitized(body)
            row["model_returned"] = body.get("model")
            usage = body.get("usage")
            if isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
                                             for k in ["input_tokens", "output_tokens"]):
                row["usage"] = usage
                row["estimated_cost_from_usage_usd"] = usage["input_tokens"] * cfg["input_usd_per_million"] / 1e6
                if row["estimated_cost_from_usage_usd"] > row["reserved_usd"]:
                    raise ValueError("observed_usage_exceeds_reserve")
            else:
                raise ValueError("missing_or_invalid_usage")
            scores = parse_scores(body, ids, cfg["model"])
            row.update(status="ok", jev_probabilities=scores,
                       jev_top5=rerank_with_jev(ids, scores, cfg["final_k"]))
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
            row["error_type"] = type(exc).__name__
            # Do not format HTTP exceptions (may include URLs/headers).
            row["error_category"] = sanitized(str(exc)) if isinstance(exc, ValueError) else type(exc).__name__
        row["latency_seconds"] = time.monotonic() - started
        row["evaluation"] = evaluate(ids, row["jev_top5"], case["evaluator_only_gold"])
        # This API documents usage, not a currency-denominated bill. Retain the
        # full reserve even on success; token x price is an estimate, not billing.
        row["billing_status"] = "not_exposed; conservative_reserve_retained"
        return sanitized(row)


def summarize_jev_results(records, cfg):
    n = len(records)
    if not n:
        raise ValueError("no_results")
    valid = [r for r in records if r["status"] == "ok"]
    diffs = [r["evaluation"]["paired_recall_difference"] for r in valid]
    def mean_metric(rs, arm, metric):
        return sum(r["evaluation"][arm][metric] for r in rs) / len(rs) if rs else None
    bm5 = mean_metric(records, "bm25_at_5", "recall_at_5")
    jev5 = mean_metric(valid, "jev_at_5", "recall_at_5")
    paired_bm = mean_metric(valid, "bm25_at_5", "recall_at_5")
    ci = None
    if len(diffs) > 1:
        rng = np.random.default_rng(cfg["bootstrap_seed"])
        samples = rng.choice(np.array(diffs), size=(cfg["bootstrap_resamples"], len(diffs)), replace=True).mean(axis=1)
        ci = np.quantile(samples, [.025, .975]).tolist()
    usage = [r["usage"] for r in records if r.get("usage")]
    return {"queries_attempted": n, "successful_queries": len(valid),
        "service_failures": n - len(valid), "bm25_recall_at_5": bm5,
        "bm25_recall_at_20": mean_metric(records, "bm25_at_20", "recall_at_20"),
        "jev_recall_at_5": jev5 if len(valid) == n else None,
        "jev_recall_at_5_valid_only": jev5, "paired_bm25_recall_at_5_valid_only": paired_bm,
        "jev_recall_at_5_service_failure_lower_bound": sum(r["evaluation"]["jev_at_5"]["recall_at_5"] for r in valid) / n,
        "absolute_recall_change": jev5 - bm5 if len(valid) == n else None,
        "relative_recall_change": (jev5 - bm5) / bm5 if len(valid) == n and bm5 else None,
        "paired_mean_difference_valid_only": float(np.mean(diffs)) if diffs else None,
        "paired_bootstrap_95_ci_valid_only": ci,
        "bootstrap": {"unit": "paired query", "method": "percentile", "resamples": cfg["bootstrap_resamples"], "seed": cfg["bootstrap_seed"]},
        "improved": sum(d > 0 for d in diffs), "worsened": sum(d < 0 for d in diffs),
        "unchanged": sum(d == 0 for d in diffs),
        "full_gold_coverage_impossible_at_20": sum(r["evaluation"]["upstream_retrieval_failure"] for r in records),
        "no_gold_in_top20": sum(r["evaluation"]["no_gold_in_candidates"] for r in records),
        "queries_rescuing_gold_from_6_to_20": sum(bool(r["evaluation"]["rescued_ids"]) for r in valid),
        "rescued_gold_ids": sum(len(r["evaluation"]["rescued_ids"]) for r in valid),
        "queries_removing_bm25_gold": sum(bool(r["evaluation"]["removed_ids"]) for r in valid),
        "removed_gold_ids": sum(len(r["evaluation"]["removed_ids"]) for r in valid),
        "categories_overlapping": dict(Counter(c for r in records for c in r["evaluation"]["categories"])),
        "api_calls": sum(r["attempt_count"] for r in records),
        "judgments_requested": sum(r["judgments_requested"] for r in records),
        "judgments_valid": sum(len(r["jev_probabilities"]) for r in valid),
        "latency_seconds": {"total": sum(r.get("latency_seconds", 0) for r in records),
            "median": float(np.median([r.get("latency_seconds", 0) for r in records])),
            "p95": float(np.quantile([r.get("latency_seconds", 0) for r in records], .95))},
        "observed_input_tokens": sum(u["input_tokens"] for u in usage),
        "observed_output_tokens": sum(u["output_tokens"] for u in usage),
        "usage_missing_calls": n - len(usage), "actual_billed_usd": None,
        "estimated_cost_from_observed_usage_usd": sum(r.get("estimated_cost_from_usage_usd") or 0 for r in records),
        "conservative_reserved_usd": sum(r["reserved_usd"] for r in records),
        "all_metrics": {arm: {metric: mean_metric(valid if arm.startswith("jev") else records, arm, metric)
                              for metric in (valid or records)[0]["evaluation"].get(arm, {}) or {}}
                        for arm in ["bm25_at_5", "bm25_at_20", "jev_at_5", "bm25_selection", "jev_selection"]}}


def stage_snapshot(folder, target, cfg):
    records = rows(folder / "rows.jsonl")[:target]
    summary = summarize_jev_results(records, cfg)
    summary.update(stage_target=target, planned_queries=200, unattempted_queries=200-len(records))
    save_once(folder / f"summary-{len(records):03d}-{digest(records)[:12]}.json", summary)
    return summary


def run_jev_trial(folder, target=200, live=False):
    if target not in (5, 20, 200):
        raise ValueError("stage_must_be_5_20_or_200")
    spec = read_json(folder / "spec.json")
    cfg = spec["config"]
    if folder.name != digest(spec)[:20]:
        raise Blocked("run_spec_hash_mismatch")
    if any(sha256(ROOT / p) != expected for p, expected in spec["code_sha256"].items()):
        raise Blocked("source_changed; prepare a new version rather than mixing results")
    if spec["prompt"] != read_json("prompts/jev_relevance.json"):
        raise Blocked("prompt_changed")
    if not live:
        raise Blocked("live_flag_required")
    cases = read_json(folder / "candidates.json")
    if digest(cases) != spec["cases_hash"]:
        raise Blocked("candidate_artifact_changed")
    budget = Budget(cfg, OUTPUT / "budget.jsonl")  # Shared across all Jev run versions.
    with (folder / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prior = {r["case_key"]: r for r in rows(folder / "rows.jsonl")}
        if len(prior) != len(rows(folder / "rows.jsonl")):
            raise Blocked("duplicate_result_cases")
        reserved = {e["case_id"]: e for e in rows(budget.path) if e["event"] == "reserve"}
        with httpx.Client(timeout=cfg["timeout_seconds"], follow_redirects=False,
                          transport=httpx.HTTPTransport(retries=0), trust_env=False) as client:
            for i, case in enumerate(cases[:target]):
                if i in (5, 20):
                    prefix = rows(folder / "rows.jsonl")[:i]
                    stage_snapshot(folder, i, cfg)
                    if len(prefix) != i or any(r["status"] != "ok" for r in prefix):
                        raise Blocked(f"stage_{i}_technical_gate_failed; prior failures retained")
                    if reserve_per_call(cfg) * 200 > 2:
                        raise Blocked("full_trial_estimate_exceeds_USD_2")
                case_key = digest([folder.name, case["query_id"]])
                if case_key in prior:
                    continue
                if case_key in reserved:
                    # Never repeat an ambiguously interrupted request automatically.
                    raise Blocked("interrupted_request_with_retained_reserve; inspect ledger before any explicit reconciliation")
                row = one_request(case, cfg, client, budget, case_key)
                row["run_id"] = folder.name
                append_row(folder / "rows.jsonl", row)
                print(json.dumps({"progress": i+1, "target": target, "status": row["status"]}), flush=True)
        return stage_snapshot(folder, target, cfg)


def write_report(folder):
    import csv
    import io
    records = rows(folder / "rows.jsonl")
    manifest = read_json(folder / "manifest.json")
    summary = summarize_jev_results(records, manifest["config"])
    cfg = manifest["config"]
    final = {**manifest, "results_sha256": sha256(folder / "rows.jsonl"), "summary": summary,
             "actual_billed_usd": None, "actual_cost_note": "API exposes token usage, no billed currency amount"}
    save_once(folder / f"manifest-results-{len(records):03d}.json", final)
    save_once(folder / f"summary-{len(records):03d}.json", summary)
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=["query_id", "status", "bm25_recall_at_5", "bm25_recall_at_20", "jev_recall_at_5", "paired_difference", "rescued", "removed"])
    writer.writeheader()
    for r in records:
        e = r["evaluation"]
        writer.writerow({"query_id": r["query_id"], "status": r["status"],
            "bm25_recall_at_5": e["bm25_at_5"]["recall_at_5"], "bm25_recall_at_20": e["bm25_at_20"]["recall_at_20"],
            "jev_recall_at_5": e["jev_at_5"]["recall_at_5"] if e["jev_at_5"] else None,
            "paired_difference": e["paired_recall_difference"], "rescued": len(e["rescued_ids"]), "removed": len(e["removed_ids"])})
    write_text_once(folder / f"paired-results-{len(records):03d}.csv", table.getvalue())
    compact = {k:v for k,v in summary.items() if not isinstance(v, (dict, list))}
    table = io.StringIO(); writer = csv.DictWriter(table, fieldnames=list(compact))
    writer.writeheader(); writer.writerow(compact)
    write_text_once(folder / f"summary-{len(records):03d}.csv", table.getvalue())
    def fmt(v): return "unavailable" if v is None else f"{v:.6f}"
    delta = summary["absolute_recall_change"]
    decision = ("Worth retaining as an additional experimental condition" if delta is not None and delta > 0 else
                "No improvement established; do not promote it based on this trial")
    ci = summary["paired_bootstrap_95_ci_valid_only"]
    text = f"""# Jev reranking development trial

## Answer

{decision}. On **{len(records)} of 200 fixed pilot queries**, BM25 Recall@5 was **{fmt(summary['bm25_recall_at_5'])}**, BM25 Recall@20 was **{fmt(summary['bm25_recall_at_20'])}**, and BM25 → Jev Recall@5 was **{fmt(summary['jev_recall_at_5'])}**. Absolute change: **{fmt(delta)}** (fraction units); relative change: **{fmt(summary['relative_recall_change'])}**.

Improved / worsened / unchanged: **{summary['improved']} / {summary['worsened']} / {summary['unchanged']}**. Service failures: **{summary['service_failures']}**. Failures retain null probabilities/rankings, never invented zero scores. Valid-only Jev Recall@5 is {fmt(summary['jev_recall_at_5_valid_only'])}; treating service failures as no delivered result gives an explicitly operational lower bound of {fmt(summary['jev_recall_at_5_service_failure_lower_bound'])}.

## Protocol and provenance

- Run: `{folder.name}`; started {manifest['created_at']}; Python {manifest['python']} on `{manifest['host_platform']}`. Local computation and artifacts are on the Mac; only HTTP judgments run at TypeSafe.
- Dataset: `{manifest['dataset']['repo_id']}` revision `{manifest['dataset']['revision']}`; corpus 6,006 skills. Exactly the existing 200 pilot IDs, in saved order. The 4,192 final IDs are disjoint and unchanged; no final-query inference or evaluation.
- Existing `pilot.bm25_rank` and `reference_metrics` are reused. Recomputed BM25 top five IDs **and scores** match all 200 historical rows exactly. Relevance is binary positive qrels; per-query recall is retrieved positives / all positives, macro-averaged across queries.
- Cached `data/metadata.jsonl` is reused byte-for-byte in each skill's `name_and_description`; hash `{manifest['metadata_hash']}`. Original name + description underwent the existing 1,024-character cap, BGE tokenization/decoding of 126 tokens, and a final 128-token check including special tokens. No full bodies, uncapped source names, labels, answer counts, retrieval scores, or ranking positions enter requests.
- One request per query with 20 independent Noul questions over shared state, then highest five probabilities. Several skills may independently be relevant. Candidate display is UUID-sorted; equal probabilities retain original BM25 rank. No prompt tuning after looking at pilot outcomes.
- Pinned `jev-1.13.0`, `POST https://api.typesafe.ai/v1/systemone`, HTTP through existing httpx; no SDK installation, fallbacks, retries, context truncation, skill execution, or unrelated model benchmarks. This evaluates retrieval, not Deep Agents operation or skill execution.
- Repository has no HEAD commit; exact source hashes and installed package versions are recorded in the manifest instead of inventing a git SHA.

## Where gains and failures occur

Jev rescued previously lower-ranked gold skills in **{summary['queries_rescuing_gold_from_6_to_20']} queries**, moving **{summary['rescued_gold_ids']} gold IDs** from BM25 ranks 6–20 into the top five. It removed BM25 top-five gold skills in **{summary['queries_removing_bm25_gold']} queries**, losing **{summary['removed_gold_ids']} gold IDs**. A query can both gain and lose skills; improved/worsened refers to its net recall difference.

BM25 top 20 omitted at least one positive in **{summary['full_gold_coverage_impossible_at_20']} queries**, so complete recall was impossible there; **{summary['no_gold_in_top20']} queries** had no positives available at all. Candidate Recall@20 ({fmt(summary['bm25_recall_at_20'])}) is the reranker's recall ceiling. Diagnostics overlap for multi-positive queries: missing upstream positives, available positives outside Jev top five, rescued positives, and retained positives are saved separately. They must not be summed as exclusive categories.

## Paired comparison and supporting metrics

Each CSV row contains the paired query difference. The descriptive 95% paired percentile bootstrap interval is **{ci}**, using 10,000 paired-query resamples and seed 42; it covers {summary['successful_queries']} valid pairs. The fixed pilot is a development sample, not a population-random or untouched final test; this interval is exploratory, not a final-paper significance claim. With service failures, valid-pair intervals exclude technical failures and the all-attempt comparison remains explicitly incomplete.

Existing precision, MRR, nDCG, completeness and set metrics are in `summary-{len(records):03d}.json`. Fixed top five can include additional nongold skills; exact-set match is not the primary endpoint. No threshold or number-of-answers selection uses hidden gold counts.

## Usage, latency and cost

- API calls: **{summary['api_calls']}**; independent judgments requested / valid: **{summary['judgments_requested']} / {summary['judgments_valid']}**. Stages at 5 and 20 reuse calls within the 200 total.
- Observed provider input / output tokens: **{summary['observed_input_tokens']} / {summary['observed_output_tokens']}**; {summary['usage_missing_calls']} calls missing usage.
- Estimated cost from observed input tokens × published price: **USD {summary['estimated_cost_from_observed_usage_usd']:.8f}**, approximately **CAD {summary['estimated_cost_from_observed_usage_usd']*cfg['budget']['cad_per_usd']:.8f}** at the labelled 1.50 planning factor. **Actual billed USD is unavailable**: the API returns token usage, not a currency-denominated charge.
- Pre-run maximum reserved estimate for 200 calls: **USD {manifest['cost_estimate']['maximum_reserved_usd']:.6f}** (full 65,536-token capacity per call × USD 0.042/M input tokens × 1.25 margin). Output tokens are free under the saved pricing snapshot. USD 2 and CAD 3 ceilings, a persistent 200-call cap, sequential execution and a {cfg['timeout_seconds']}-second timeout are enforced before calls. Unknown charges retain reserves; currently reserved USD {summary['conservative_reserved_usd']:.6f}. These guards do not control unrelated account activity.
- Measured request latency: total **{summary['latency_seconds']['total']:.3f}s**, median **{summary['latency_seconds']['median']:.3f}s**, p95 **{summary['latency_seconds']['p95']:.3f}s**. This includes network/server time; it is not model-only throughput. Initial local BM25 preparation took {manifest['local_bm25_seconds']:.3f}s.

## Evidence and reproduction

Artifacts are under `results/jev/{folder.name}/`: immutable config/source/provenance manifests, candidate cache, append-only raw request/response rows, staged summaries, summary CSV, and paired-query CSV. `results/jev/budget.jsonl` persists across run versions. Offline fixture tests establish implementation behavior only; live rows establish API operation. Official source snapshots and preservation audit are in `evidence/jev/`. The two existing notebooks are unchanged. The optional vector → Jev condition was not run; the primary comparison is the only new condition.

From the repository on the Mac, with `TYPESAFE_API_KEY` exported privately:

```bash
ALLOW_PAID_RUNS=1 .venv/bin/python jev_reranker.py run --queries 200 --live
.venv/bin/python jev_reranker.py report --run-dir results/jev/{folder.name}
.venv/bin/python -m pytest -q tests/test_jev_reranker.py tests/test_reference_helpers.py
```

The first command runs the 5 → 20 → 200 technical gates and resumes exact completed cases without another charge. To launch with the locally saved key, prepend `.venv/bin/python -m dotenv -f .env run --no-override --` to the runner command (put `ALLOW_PAID_RUNS=1` before that prefix). The experiment itself only reads `os.environ`; no credential is included in artifacts. Failures are not resubmitted on resume. Preserve the ledger. Changes to source/configuration produce a new version and are subject to the same cumulative cap.

## Interpretation

This is an additional MIA5150 development/pilot condition. Public skills, synthetic queries, capped metadata, incomplete relevance labels and prior consumption of these 200 queries limit external validity. A semantically useful but unlabelled skill counts as incorrect under the existing qrels. The recommendation above concerns carrying the condition forward for a separately authorized, prespecified study; it does not authorize consuming the reserved final split.

Sources: [HTTP API](https://docs.typesafe.ai/api), [Noul](https://docs.typesafe.ai/primitives/noul), [reranking cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe), [batching](https://docs.typesafe.ai/patterns/fan-out), [current pricing and limits](https://docs.typesafe.ai/models). The cookbook's old example model was not copied; the current documented version is pinned.
"""
    report = ROOT / "reports/JEV_TRIAL.md"
    if report.exists() and report.read_text() != text:
        report = ROOT / f"reports/JEV_TRIAL-{folder.name}-{len(records):03d}.md"
    write_text_once(report, text)
    return {"report": str(report), "run_dir": str(folder), **summary}


def write_text_once(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Refusing to overwrite {path}")
    else:
        with path.open("x") as f:
            f.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "report"])
    parser.add_argument("--queries", type=int, choices=[5, 20, 200], default=200)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    try:
        folder = args.run_dir.resolve() if args.run_dir else prepare()
        if args.command == "run":
            result = run_jev_trial(folder, args.queries, args.live)
        elif args.command == "report":
            result = write_report(folder)
        else:
            result = {"prepared": str(folder), "cost_estimate": read_json(folder / "spec.json")["cost_estimate"]}
        print(json.dumps(sanitized(result), indent=2, allow_nan=False))
        return 2 if result.get("service_failures", 0) else 0
    except (Blocked, ValueError, FileExistsError) as exc:
        print(json.dumps({"status": "blocked", "reason": sanitized(str(exc))}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
