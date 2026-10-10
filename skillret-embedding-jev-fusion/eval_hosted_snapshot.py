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
import math
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


def timestamp(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "Invalid timestamp: " + label)
    return value


def invocation_timing(run_dir, plan_sha256, rows, end):
    """Read immutable checkpoints; recovery windows bound missing-final invocations.

    A checkpoint proves activity through its recorded time, never a process exit.
    Recovery receipt filesystem times are deliberately not read. Final receipt
    mtimes retain the earlier report's end proxy (original files are required).
    """
    run_dir = Path(run_dir)
    recoveries = {}
    for path in sorted(run_dir.glob("recovery*/recovery.json")):
        receipt = read(path)
        if receipt.get("schema_version") != "reviewed-wsl-reboot-recovery-v1":
            continue
        key = (receipt.get("plan_sha256"), receipt.get("original_invocation_id"))
        require(key not in recoveries, "Ambiguous invocation recovery receipts")
        recoveries[key] = (receipt, path)
    invocations, all_starts = [], []
    for directory in sorted((run_dir / "study_progress").glob("*/*")):
        if not directory.is_dir():
            continue
        immutable = sorted(path for path in directory.glob("*.json")
                           if path.stem.isdigit())
        final_path = directory / "final.json"
        evidence = [(path, read(path)) for path in immutable]
        if final_path.exists():
            evidence.append((final_path, read(final_path)))
        if not evidence:
            continue
        first = evidence[0][1]
        identity = (directory.parent.name, directory.name)
        start = timestamp(first.get("invocation_started_at_unix"), "invocation start")
        if start > end:
            continue
        all_starts.append(start)
        if identity[0] != plan_sha256:
            continue
        stamps = []
        for path, receipt in evidence:
            require((receipt.get("plan_sha256"), receipt.get("invocation_id")) == identity,
                    "Immutable checkpoint invocation identity differs")
            require(receipt.get("invocation_started_at_unix") == start,
                    "Immutable checkpoint invocation start differs")
            stamp = timestamp(receipt.get("updated_at_unix"), "checkpoint update")
            require(start <= stamp, "Checkpoint predates invocation")
            if path != final_path:
                require(not stamps or stamp >= stamps[-1], "Checkpoint timestamps reversed")
                stamps.append(stamp)
        usable = [(path, receipt) for path, receipt in evidence
                  if receipt["updated_at_unix"] <= end and (path != final_path
                      or max(path.stat().st_mtime, receipt["updated_at_unix"]) <= end + .001)]
        require(bool(usable), "Invocation has no checkpoint at snapshot")
        latest_path, latest = max(usable, key=lambda item: item[1]["updated_at_unix"])
        last_observed = latest["updated_at_unix"]
        final_present = any(path == final_path for path, _ in usable)
        lower, upper = last_observed, None
        kind = "missing_final_observed_lower_bound_only"
        item = {"invocation_id": identity[1], "start": start,
                "final_receipt_present": final_present,
                "evidence_status": "final_receipt_available" if final_present else "partial_missing_final",
                "last_checkpoint_status": latest.get("status"),
                "last_checkpoint_complete_queries": latest.get("complete_queries"),
                "last_checkpoint_updated_at_unix": last_observed,
                "last_checkpoint_path": str(latest_path),
                "last_checkpoint_sha256": sha(latest_path)}
        if final_present:
            receipt = next(value for path, value in usable if path == final_path)
            proxy = max(final_path.stat().st_mtime, receipt["updated_at_unix"])
            require(proxy <= end + .001, "Final receipt end proxy exceeds snapshot")
            lower = upper = proxy
            kind = "final_receipt_filesystem_end_proxy"
            item.update(final_receipt_path=str(final_path), final_receipt_sha256=sha(final_path),
                        recorded_final_status=receipt.get("status"))
        elif identity in recoveries and timestamp(recoveries[identity][0].get("confirmed_new_wsl_boot_unix"),
                                                   "confirmed WSL boot") <= end:
            recovery, path = recoveries[identity]
            require(recovery.get("original_invocation_started_at_unix") == start,
                    "Recovery invocation start differs")
            action = timestamp(recovery.get("observed_last_action_unix"), "recovery last action")
            lower = timestamp(recovery.get("interruption_lower_bound_unix"), "interruption lower")
            upper = timestamp(recovery.get("interruption_upper_bound_unix"), "interruption upper")
            boot = timestamp(recovery.get("confirmed_new_wsl_boot_unix"), "confirmed WSL boot")
            require(last_observed <= action <= lower <= upper <= boot <= end,
                    "Recovery bounds contradict immutable checkpoint or snapshot")
            matches = [row for row in rows if row.get("event") == "reserved"
                       and row.get("utc_unix") == action
                       and row.get("plan_sha256") == plan_sha256]
            if recovery.get("original_attempt_id") is not None:
                matches = [row for row in matches
                           if row.get("attempt_id") == recovery["original_attempt_id"]]
            require(len(matches) == 1, "Recovery last action lacks unique snapshot reservation evidence")
            require(matches[0].get("request_sha256") == recovery.get("request_sha256")
                    and matches[0].get("event_sha256") == recovery.get("original_reservation_event_sha256"),
                    "Recovery reservation request or event hash differs")
            proof = recovery.get("reboot_proof", {})
            require(isinstance(proof, dict) and isinstance(proof.get("path"), str),
                    "Recovery lacks reboot proof identity")
            proof_path = (run_dir / proof["path"]).resolve()
            require(proof_path.is_relative_to(run_dir.resolve()), "Recovery proof escapes experiment directory")
            require(sha(proof_path) == proof.get("sha256"), "Recovery reboot proof hash differs")
            observed_proof = read(proof_path)
            require(isinstance(observed_proof, dict) and observed_proof.get("kind") == "wsl_reboot_observation"
                    and isinstance(observed_proof.get("boot_id"), str) and bool(observed_proof["boot_id"].strip())
                    and type(observed_proof.get("matching_live_process_count")) is int
                    and observed_proof["matching_live_process_count"] == 0,
                    "Recovery reboot/process proof is incomplete")
            observed = timestamp(observed_proof.get("observed_at_unix"), "reboot observation")
            uptime = timestamp(observed_proof.get("uptime_seconds"), "reboot uptime")
            # Allow only floating-point subtraction/serialization precision,
            # not an extra recovery-window allowance.
            boot_tolerance = 4 * max(math.ulp(observed), math.ulp(boot))
            require(uptime <= observed and action < observed - uptime
                    and math.isclose(observed - uptime, boot, rel_tol=0, abs_tol=boot_tolerance),
                    "Recovery boot bound differs from observed reboot proof")
            kind = "missing_final_reviewed_interruption_bounds"
            item.update(recovery_receipt_path=str(path), recovery_receipt_sha256=sha(path),
                        reboot_proof_path=str(proof_path), reboot_proof_sha256=proof["sha256"],
                        observed_last_action_unix=action, confirmed_new_wsl_boot_unix=boot,
                        recovery_schema_version=recovery["schema_version"])
        item.update(timing_evidence_kind=kind, end_lower_bound_unix=lower, end_upper_bound_unix=upper,
                    wall_seconds_lower_bound=lower - start,
                    wall_seconds_upper_bound=upper - start if upper is not None else None)
        invocations.append(item)
    require(bool(invocations), "No immutable invocation timing receipt")
    invocations.sort(key=lambda item: (item["start"], item["invocation_id"]))
    return invocations, all_starts


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
    invocations, starts = invocation_timing(run_dir, plan["plan_sha256"], rows, end)
    wall_lower = sum(item["wall_seconds_lower_bound"] for item in invocations)
    wall_upper = (sum(item["wall_seconds_upper_bound"] for item in invocations)
                  if all(item["wall_seconds_upper_bound"] is not None for item in invocations) else None)
    scope_start = min(item["start"] for item in invocations)
    first_attempt = account["first_attempt_unix"]
    if first_attempt is not None:
        starts.append(first_attempt)
    experiment_start = first_attempt if first_attempt is not None else min(starts)
    new_api_elapsed = sum(row.get("elapsed_seconds", 0) for row in rows if row["event"] == "settled"
                          and any(item["start"] <= row["utc_unix"] <= item["end_lower_bound_unix"]
                                  for item in invocations))
    missing_final = sum(not item["final_receipt_present"] for item in invocations)
    timing = {"scope_invocations": invocations,
              "scope_invocation_wall_seconds": wall_upper if wall_upper == wall_lower else None,
              "scope_invocation_wall_seconds_lower_bound": wall_lower,
              "scope_invocation_wall_seconds_upper_bound": wall_upper,
              "scope_invocations_missing_final": missing_final,
              "scope_elapsed_including_gaps_seconds": end - scope_start,
              "experiment_elapsed_including_gaps_seconds": end - experiment_start,
              "experiment_elapsed_start_unix": experiment_start,
              "new_paid_api_elapsed_in_scope_invocations_seconds": new_api_elapsed,
              "scope_non_api_wall_seconds": (wall_upper - new_api_elapsed
                  if wall_upper == wall_lower and not account["unresolved_attempts"] and not missing_final else None),
              "scope_non_api_wall_seconds_lower_bound": (0 if account["unresolved_attempts"] or missing_final
                  else wall_lower - new_api_elapsed),
              "scope_non_api_wall_seconds_upper_bound": wall_upper - new_api_elapsed if wall_upper is not None else None,
              "scope_wall_minus_recorded_settled_api_seconds_lower_bound": wall_lower - new_api_elapsed,
              "scope_wall_minus_recorded_settled_api_seconds_upper_bound": wall_upper - new_api_elapsed if wall_upper is not None else None,
              "snapshot_end_unix_proxy": end,
              "cumulative_six_hour_deadline_unix": first_attempt + 21600 if first_attempt is not None else None,
              "definition": "Final receipts use original filesystem mtime as an end proxy after ingestion. Missing-final invocations use immutable checkpoint activity as a lower bound and an explicit reviewed recovery window when available; a reboot timestamp is only an upper bound, never an exact exit. No success or exit is inferred for missing finals. Invocation sums exclude gaps; experiment elapsed includes gaps and the six-hour deadline remains anchored at the first ledger reservation. Parent work before runner start is unmeasured; reused API duration is excluded from new API time. Wall-minus-recorded-settled-API residuals can include unknown in-flight API time; genuine non-API lower bounds are therefore zero when missing-final or unresolved-attempt evidence remains."}
    return account, scope_account, timing


def projection(metrics, scope_account, account, timing, target_queries):
    count = metrics["query_count"]
    if not count or metrics["complete_hosted_queries"] != count:
        return {"status": "unavailable", "reason": "Projection requires a nonempty complete scope; every selected query is retained."}
    remaining = max(0, target_queries - count)
    wall_lower = timing.get("scope_invocation_wall_seconds_lower_bound", timing.get("scope_invocation_wall_seconds"))
    wall_upper = timing.get("scope_invocation_wall_seconds_upper_bound", timing.get("scope_invocation_wall_seconds"))
    require(wall_lower is not None and wall_lower >= 0 and (wall_upper is None or wall_upper >= wall_lower),
            "Invalid projection timing bounds")
    per_query_lower = wall_lower / count
    per_query_upper = wall_upper / count if wall_upper is not None else None
    additional_cad = Decimal(scope_account["settled_cad_with_fee_buffer"]) * Decimal(remaining) / Decimal(count)
    additional_usd = Decimal(scope_account["settled_usd"]) * Decimal(remaining) / Decimal(count)
    settled_cad, held_cad = Decimal(account["settled_cad_with_fee_buffer"]), Decimal(account["held_cad_with_fee_buffer"])
    settled_usd, held_usd = Decimal(account["settled_usd"]), Decimal(account["held_usd"])
    lower_cad, lower_usd = settled_cad + additional_cad, settled_usd + additional_usd
    upper_cad, upper_usd = lower_cad + held_cad, lower_usd + held_usd
    additional_wall_lower = remaining * per_query_lower
    additional_wall_upper = remaining * per_query_upper if per_query_upper is not None else None
    return {"status": "descriptive_projection_not_execution_authorization", "target_queries": target_queries,
            "measured_queries": count, "remaining_queries": remaining,
            "observed_wall_seconds_per_query_including_local_overhead": per_query_upper,
            "observed_wall_seconds_per_query_lower_bound": per_query_lower,
            "observed_wall_seconds_per_query_upper_bound": per_query_upper,
            "projected_additional_wall_seconds": additional_wall_upper,
            "projected_additional_wall_seconds_lower_bound": additional_wall_lower,
            "projected_additional_wall_seconds_upper_bound": additional_wall_upper,
            "projected_experiment_wall_seconds_including_elapsed_gaps": (
                timing["experiment_elapsed_including_gaps_seconds"] + additional_wall_upper
                if additional_wall_upper is not None else None),
            "projected_experiment_wall_seconds_lower_bound": timing["experiment_elapsed_including_gaps_seconds"] + additional_wall_lower,
            "projected_experiment_wall_seconds_upper_bound": (
                timing["experiment_elapsed_including_gaps_seconds"] + additional_wall_upper
                if additional_wall_upper is not None else None),
            "projected_additional_usage_cad_with_recorded_fee_buffer": str(additional_cad),
            "projected_additional_usage_usd": str(additional_usd),
            "projected_total_usage_and_held_cad_with_recorded_fee_buffer": str(upper_cad),
            "projected_total_usage_and_held_usd": str(upper_usd),
            "projected_total_settled_rate_cad_lower_bound": str(lower_cad),
            "projected_total_planning_commitment_cad_upper_bound": str(upper_cad),
            "projected_total_settled_rate_usd_lower_bound": str(lower_usd),
            "projected_total_planning_commitment_usd_upper_bound": str(upper_usd),
            "observed_experiment_settled_cad_with_fee_buffer": str(settled_cad),
            "observed_experiment_settled_usd": str(settled_usd),
            "held_unknown_cad_with_fee_buffer": str(held_cad), "held_unknown_usd": str(held_usd),
            "unresolved_experiment_attempts": account["unresolved_attempts"],
            "unresolved_scope_attempts": scope_account["unresolved_attempts"],
            "uncertainty_policy": "Extrapolate the complete scope's settled usage only. Add existing unresolved holds once to the experiment planning upper commitment; do not smear a historical unknown attempt into future per-query usage. Held amounts are allowances, not known charges. These cost bounds concern only existing held uncertainty under a linear settled-rate assumption, not prediction confidence intervals or guaranteed future bounds. Legacy scalar time fields carry the planning upper estimate (null if unbounded); interval fields preserve interruption uncertainty.",
            "execution_authorized": False,
            "caveat": "Linear query-rate extrapolation; full-plan request mix and growing-ledger overhead can differ. Reused smoke responses can make the observed query wall rate optimistic for fresh queries. Timing bounds include incomplete invocation work and unknown in-flight time. CAD excludes any separate fixed experiment fee. No budget gate or six-hour guarantee is asserted or reset; an unknown hold remains unresolved and this report grants no retry or expansion permission. Scope reuse is counted once in experiment totals."}


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
              f"Held unresolved allowance: USD {account['held_usd']}; CAD {account['held_cad_with_fee_buffer']} with recorded fee/buffer. This is not a known charge.",
              "Scope invocation wall time bounds: " + f"{timing['scope_invocation_wall_seconds_lower_bound']:.3f}s to "
              + (f"{timing['scope_invocation_wall_seconds_upper_bound']:.3f}s" if timing['scope_invocation_wall_seconds_upper_bound'] is not None else "unknown")
              + f"; {timing['scope_invocations_missing_final']} invocation(s) lack final receipts. Experiment elapsed including gaps: {timing['experiment_elapsed_including_gaps_seconds']:.3f}s."]
    (a.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "scope": plan["scope"], "selected": len(ids),
                      "completed": measured["complete_hosted_queries"], "output_dir": str(a.output_dir),
                      "network_calls": 0, "credentials_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
