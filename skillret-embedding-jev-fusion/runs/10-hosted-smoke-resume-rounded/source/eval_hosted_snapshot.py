"""Offline evaluation of one immutable hosted-study snapshot. No network/key access.

Example from the experiment directory, after the runner writes final.json:
  ../.venv/bin/python eval_hosted_snapshot.py \
    --plan results/jev-plan-pilot-context-v1/preparation_manifest.json \
    --run-dir results/hosted-execution --snapshot SNAPSHOT_DIRECTORY \
    --baseline results/baseline_skillret_06b.json \
    --qrels ../data/qrels/test.jsonl --output-dir results/NEW-pilot-evaluation

SNAPSHOT_DIRECTORY is the actual snapshot_dir in the runner's final result.
Use a new output directory. Inputs, receipts and snapshots are never changed.
Run against original receipts; export copies can change filesystem timestamps.
Imports define transports but never construct a client or access credentials.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from evaluate import BOOTSTRAP_METRICS, deterministic_subset, evaluate, load_qrels, paired_bootstrap, rrf60
from hosted_jev_client import SpendLedger
from run_hosted_study import ledger_summary, read_plan


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def compare(labels, systems, resamples, seed):
    measured = {name: evaluate(labels, ranks) for name, ranks in systems.items()}
    return {"query_count": len(labels), "systems": measured,
            "paired_bootstrap": {name + "_minus_A": paired_bootstrap(
                measured["A"], measured[name], n_resamples=resamples, seed=seed)
                for name in ("B", "C")}}


def snapshot_metrics(plan, artifacts, baseline, labels, resamples=10000, seed=20261006):
    ids = [row["query_id"] for row in plan["rows"]]
    require(set(ids) <= set(labels), "Selected query absent from qrels")
    selected = {qid: labels[qid] for qid in ids}
    systems = {name: artifact["rankings"] for name, artifact in artifacts.items()}
    for name, artifact in artifacts.items():
        require(artifact.get("plan_sha256") == plan["plan_sha256"], "Snapshot plan hash differs")
        require(set(systems[name]) == set(ids), "Snapshot silently changed selected denominator")
    truncated_ids = []
    for row in plan["rows"]:
        qid = row["query_id"]
        pool = [candidate["skill_id"] for candidate in row["dense_top20"]]
        require(len(pool) == len(set(pool)) == 20, "Invalid fixed candidate pool")
        require(systems["A"][qid] == pool == baseline["rankings"][qid][:20], "Dense pool changed")
        b, c = systems["B"][qid], systems["C"][qid]
        require(bool(b) == bool(c), "B/C incomplete-query treatment differs")
        if b:
            require(len(b) == len(c) == 20 and set(b) == set(c) == set(pool), "Reranked pool changed")
            require(c == rrf60(pool, b), "Saved fusion is not exact fixed RRF60")
        audit = row.get("candidate_text_audit", [])
        require(len(audit) == 20 and {x["skill_id"] for x in audit} == set(pool), "Missing candidate text audit")
        require(all(type(x.get("truncated")) is bool for x in audit), "Unknown truncation status")
        if any(x["truncated"] for x in audit):
            truncated_ids.append(qid)
    result = compare(selected, systems, resamples, seed)
    require(artifacts["B"].get("actual_model_version") == artifacts["C"].get("actual_model_version"), "B/C model versions differ")
    result["actual_model_version"] = artifacts["B"].get("actual_model_version")
    result["complete_hosted_queries"] = sum(bool(systems["B"][qid]) for qid in ids)
    result["selected_query_ids"] = ids
    result["denominator_policy"] = "Every frozen selected ID; incomplete B/C rankings are empty and zero-scored."
    full_dense = evaluate(selected, {qid: baseline["rankings"][qid] for qid in ids})
    result["upstream_dense_coverage_20_50"] = full_dense["candidate_coverage"]
    result["coverage_note"] = "Coverage@50 belongs to upstream full dense rankings; B/C contain only the same 20 candidates."
    truncated = set(truncated_ids)
    result["truncation_subsets"] = {}
    for name, subset in (("any_candidate_skill_md_truncated", truncated_ids),
                         ("all_twenty_candidates_full_text", [qid for qid in ids if qid not in truncated])):
        result["truncation_subsets"][name] = (
            compare({qid: labels[qid] for qid in subset},
                    {system: {qid: ranks[qid] for qid in subset} for system, ranks in systems.items()},
                    resamples, seed) if subset else {"query_count": 0, "systems": None, "paired_bootstrap": None})
    result["subset_caveat"] = "Descriptive, nonrandom query groups defined by whether any candidate was truncated; not a causal estimate of truncation. Fixed denominators are retained within each group."
    return result


def accounting(rows):
    result = ledger_summary(SimpleNamespace(rows=lambda: rows))
    settled = [row for row in rows if row["event"] == "settled"]
    result["output_tokens_reported_for_all_settled"] = all(row.get("output_tokens") is not None for row in settled)
    result["actual_output_tokens_if_fully_reported"] = (sum(row["output_tokens"] for row in settled)
        if result["output_tokens_reported_for_all_settled"] else None)
    result["unique_settled_api_elapsed_seconds"] = sum(row.get("elapsed_seconds", 0) for row in settled)
    return result


def checkpoint_accounting(plan, run_dir, snapshot, final, final_path):
    rows = SpendLedger(run_dir).rows()  # Verifies chain, experiment identity, complete tail.
    head = final["ledger"]["ledger_head_sha256"]
    if head is None:
        rows = []
    else:
        positions = [index for index, row in enumerate(rows) if row["event_sha256"] == head]
        require(len(positions) == 1, "Snapshot ledger checkpoint is absent or ambiguous")
        rows = rows[:positions[0] + 1]
    account = accounting(rows)
    require(account["ledger_events"] == final["ledger"]["ledger_events"], "Ledger checkpoint count differs")
    request_ids = {packet["request_sha256"] for row in plan["rows"] for packet in row["batches"]}
    originals = request_ids | {row["original_transport_request_sha256"] for row in rows
                              if row["event"] == "cache_reused" and row.get("request_sha256") in request_ids}
    scope_rows = [row for row in rows if row.get("request_sha256") in originals]
    scope_account = accounting(scope_rows)
    # Scope totals may overlap other scopes; only experiment account is cumulative.
    scope_account["warning"] = "Distinct attempts serving this logical scope, including earlier reused smoke calls. Never add this to another scope's totals."
    end = max(final_path.stat().st_mtime, final["updated_at_unix"])
    invocations = []
    progress_root = run_dir / "study_progress"
    for path in progress_root.glob("*/*/final.json"):
        receipt = read(path)
        if receipt.get("plan_sha256") != plan["plan_sha256"]:
            continue
        receipt_end = max(path.stat().st_mtime, receipt["updated_at_unix"])
        if receipt_end <= end + .001:
            invocations.append({"invocation_id": receipt["invocation_id"],
                                "start": receipt["invocation_started_at_unix"], "end": receipt_end})
    require(bool(invocations), "No immutable invocation timing receipt")
    wall_sum = sum(item["end"] - item["start"] for item in invocations)
    require(wall_sum >= 0, "Invalid runner timing interval")
    scope_start = min(item["start"] for item in invocations)
    starts = [scope_start]
    for path in progress_root.glob("*/*/final.json"):
        receipt = read(path)
        if receipt["invocation_started_at_unix"] <= end:
            starts.append(receipt["invocation_started_at_unix"])
    first_attempt = account["first_attempt_unix"]
    if first_attempt is not None:
        starts.append(first_attempt)
    experiment_start = min(starts)
    new_api_elapsed = sum(row.get("elapsed_seconds", 0) for row in rows if row["event"] == "settled"
                          and any(item["start"] <= row["utc_unix"] <= item["end"] for item in invocations))
    timing = {"scope_invocations": invocations, "scope_invocation_wall_seconds": wall_sum,
              "scope_elapsed_including_gaps_seconds": end - scope_start,
              "experiment_elapsed_including_gaps_seconds": end - experiment_start,
              "new_paid_api_elapsed_in_scope_invocations_seconds": new_api_elapsed,
              "scope_non_api_wall_seconds": wall_sum - new_api_elapsed,
              "snapshot_end_unix_proxy": end,
              "cumulative_six_hour_deadline_unix": first_attempt + 21600 if first_attempt is not None else None,
              "definition": "Runner invocation start through immutable final receipt filesystem mtime (end proxy after ingestion), including local validation/checkpoint/cache/ingestion overhead. Invocation sums exclude gaps; experiment elapsed includes gaps. Parent command work before recorded runner start is not measured. Original API duration of reused responses is excluded from new API time."}
    return account, scope_account, timing


def projection(metrics, scope_account, account, timing, target_queries):
    count = metrics["query_count"]
    if metrics["complete_hosted_queries"] != count or scope_account["unresolved_attempts"] or account["unresolved_attempts"]:
        return {"status": "unavailable", "reason": "Projection requires a complete scope and no unresolved experiment attempts."}
    remaining = max(0, target_queries - count)
    per_query = timing["scope_invocation_wall_seconds"] / count
    additional_cad = Decimal(scope_account["settled_cad_with_fee_buffer"]) * Decimal(remaining) / Decimal(count)
    total_cad = Decimal(account["settled_cad_with_fee_buffer"]) + Decimal(account["held_cad_with_fee_buffer"]) + additional_cad
    return {"status": "descriptive_projection_not_execution_authorization", "target_queries": target_queries,
            "measured_queries": count, "remaining_queries": remaining,
            "observed_wall_seconds_per_query_including_local_overhead": per_query,
            "projected_additional_wall_seconds": remaining * per_query,
            "projected_experiment_wall_seconds_including_elapsed_gaps": timing["experiment_elapsed_including_gaps_seconds"] + remaining * per_query,
            "projected_additional_usage_cad_with_recorded_fee_buffer": str(additional_cad),
            "projected_total_usage_and_held_cad_with_recorded_fee_buffer": str(total_cad),
            "caveat": "Linear query-rate extrapolation from this exact measured scope; full-plan request mix and later growing-ledger overhead can differ. CAD values exclude any separately configured fixed experiment fee. No invented fee, budget gate or six-hour guarantee is asserted. Scope reuse is counted once in experiment totals."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("plan", "run-dir", "snapshot", "baseline", "qrels", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--bootstrap-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--target-queries", type=int, default=4392)
    a = p.parse_args(argv)
    require(not a.output_dir.exists(), "Use a new output directory; existing reports are immutable")
    plan = read_plan(a.plan)
    labels, baseline = load_qrels(a.qrels), read(a.baseline)
    require(len(labels) == 4392 and sum(map(len, labels.values())) == 7187, "Official qrels counts differ")
    ids = [row["query_id"] for row in plan["rows"]]
    require(ids == deterministic_subset(list(labels), len(ids)), "Frozen SHA256 query subset differs")
    final_path = a.run_dir / "study_progress" / plan["plan_sha256"] / a.snapshot.name / "final.json"
    final = read(final_path)
    require(final["plan_sha256"] == plan["plan_sha256"], "Final receipt plan differs")
    require(Path(final["snapshot_dir"]).resolve() == a.snapshot.resolve(), "Final receipt snapshot path differs")
    paths = {"A": a.snapshot / "dense_top20.json", "B": a.snapshot / "jev_rerank.json", "C": a.snapshot / "rrf60_fusion.json"}
    artifacts = {name: read(path) for name, path in paths.items()}
    measured = snapshot_metrics(plan, artifacts, baseline, labels, a.bootstrap_resamples, a.seed)
    require(measured["complete_hosted_queries"] == final["complete_queries"], "Final receipt completion differs")
    account, scope_account, timing = checkpoint_accounting(plan, a.run_dir, a.snapshot, final, final_path)
    output = {"status": "complete_scope" if measured["complete_hosted_queries"] == len(ids) else "partial_fixed_denominator",
              "scope": plan["scope"], "plan_sha256": plan["plan_sha256"], "snapshot": str(a.snapshot),
              "evaluation": measured, "experiment_unique_ledger_at_snapshot": account,
              "logical_scope_unique_attempt_usage_do_not_sum_scopes": scope_account,
              "wall_time": timing, "projection": projection(measured, scope_account, account, timing, a.target_queries),
              "source_sha256": {str(path): sha(path) for path in [a.plan, a.baseline, a.qrels, final_path, *paths.values()]},
              "ledger_checkpoint_sha256": account["ledger_head_sha256"],
              "network_calls_this_process": 0, "credentials_accessed": False}
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "evaluation.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["query_id", "system", *measured["systems"]["A"]["aggregate"]]
    with (a.output_dir / "per_query_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for name, ev in measured["systems"].items():
            for qid, row in ev["per_query"].items():
                writer.writerow({"query_id": qid, "system": name, **row})
    lines = [f"# {plan['scope'].title()} hosted snapshot", "", f"{measured['complete_hosted_queries']}/{len(ids)} queries complete; all {len(ids)} selected queries remain in metric denominators.", "",
             "| Metric | A matched dense | B Jev | C RRF60 |", "|---|---:|---:|---:|"]
    for metric in measured["systems"]["A"]["aggregate"]:
        if not metric.startswith("Coverage"):
            lines.append("| " + metric + " | " + " | ".join(f"{measured['systems'][s]['aggregate'][metric]:.6f}" for s in "ABC") + " |")
    lines += ["", "| Comparison / metric | Delta | 95% paired bootstrap CI |", "|---|---:|---|"]
    for name, bootstrap in measured["paired_bootstrap"].items():
        for metric in BOOTSTRAP_METRICS:
            value = bootstrap["metrics"][metric]
            lines.append(f"| {name} / {metric} | {value['delta']:+.6f} | [{value['ci_low']:+.6f}, {value['ci_high']:+.6f}] |")
    lines += ["", "Truncation subsets, upstream coverage@20/50, unique ledger usage, overhead-inclusive timing and descriptive projections are in [evaluation.json](evaluation.json). Scope usage overlaps other scopes; use the experiment ledger total for cumulative spend. These projections do not authorize expansion.", "",
              f"Experiment unique settlements: {account['paid_responses_settled']}; unresolved attempts: {account['unresolved_attempts']}; settled USD {account['settled_usd']}; recorded CAD with fee/buffer {account['settled_cad_with_fee_buffer']} (excluding any separate fixed fee).",
              f"Scope invocation wall time: {timing['scope_invocation_wall_seconds']:.3f}s; experiment elapsed including gaps: {timing['experiment_elapsed_including_gaps_seconds']:.3f}s."]
    (a.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "scope": plan["scope"], "selected": len(ids),
                      "completed": measured["complete_hosted_queries"], "output_dir": str(a.output_dir),
                      "network_calls": 0, "credentials_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
