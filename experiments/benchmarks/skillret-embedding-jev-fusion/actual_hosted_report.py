r"""Create a new English report from actual immutable smoke/pilot evaluations.

No API, key, environment-variable, model or mutable-runner operation is used.
Run after the ledger writer has stopped. Existing reports are never overwritten.
On Windows, /home/user inputs map to \\wsl.localhost\Ubuntu-26.04\home\...
using --wsl-distro; on WSL they remain normal paths. The output is a local path.

Example (the same arguments work with Windows or WSL Python):
  python actual_hosted_report.py --smoke /home/user/.../smoke-eval/evaluation.json \
    --pilot /home/user/.../pilot-eval/evaluation.json \
    --ledger /home/user/.../hosted-execution/budget_ledger.jsonl \
    --baseline /home/user/.../baseline_skillret_06b.json \
    --baseline-gate /home/user/.../baseline_gate.json \
    --qrels /home/user/code/mia5150-local/data/qrels/test.jsonl \
    --output-dir results/NEW-actual-hosted-report

Supply at least one of --smoke/--pilot. Optional --parser-diagnostic supplies
the actual first_response_precision_diagnostic.json; absent evidence is not
replaced by a claim that the repair, smoke or pilot has completed.
The named recovery-after-wsl-restart/recovery.json is read when present beside
the ledger, or supplied with --recovery-receipt. It is reporting evidence only.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re

from evaluate import evaluate, load_qrels, rrf60
from eval_hosted_snapshot import accounting
from hosted_jev_client import SpendLedger

METRICS = [f"{name}@{k}" for name in ("NDCG", "Recall", "Completeness", "MAP", "Precision") for k in (5, 10, 15)] + ["MRR@5", "MRR@10"]


def require(value, reason):
    if not value:
        raise ValueError(reason)


def local_path(value, distro):
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", distro), "Invalid WSL distribution name")
    text = str(value)
    return Path("//wsl.localhost/" + distro + text if os.name == "nt" and text.startswith("/") and not text.startswith("//") else text)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, places=6):
    return "unknown" if value is None else f"{value:.{places}f}"


def validate_scope(path, expected_scope, distro, run_dir, baseline, labels, ledger_heads):
    document = read(path)
    require(document.get("scope") == expected_scope, "Supplied evaluation scope differs")
    measured = document["evaluation"]
    ids = measured["selected_query_ids"]
    require(len(ids) == len(set(ids)) == {"smoke": 10, "pilot": 200}[expected_scope], "Frozen scope denominator differs")
    require(document["ledger_checkpoint_sha256"] in ledger_heads, "Evaluation ledger checkpoint not in verified ledger")
    snapshot = local_path(document["snapshot"], distro).resolve()
    require(snapshot.is_relative_to((run_dir / "study_snapshots").resolve()), "Snapshot escapes supplied experiment")
    filenames = {"A": "dense_top20.json", "B": "jev_rerank.json", "C": "rrf60_fusion.json"}
    artifacts = {}
    for system, filename in filenames.items():
        source = snapshot / filename
        hashes = [value for key, value in document["source_sha256"].items()
                  if local_path(key, distro).resolve() == source]
        require(hashes == [sha(source)], "Immutable snapshot hash differs")
        artifact = read(source)
        require(artifact["plan_sha256"] == document["plan_sha256"], "Snapshot plan differs")
        require(set(artifact["rankings"]) == set(ids), "Snapshot changed selected query IDs")
        artifacts[system] = artifact
    selected = {qid: labels[qid] for qid in ids}
    for qid in ids:
        a, b, c = [artifacts[system]["rankings"][qid] for system in "ABC"]
        require(a == baseline["rankings"][qid][:20] and len(set(a)) == 20, "Dense top20 pool differs")
        require(bool(b) == bool(c), "B/C incomplete-query handling differs")
        if b:
            require(len(b) == len(c) == 20 and set(b) == set(c) == set(a), "Hosted candidate pool differs")
            require(c == rrf60(a, b), "Fusion differs from fixed equal RRF60")
    for system in "ABC":
        fresh = evaluate(selected, artifacts[system]["rankings"])
        saved = measured["systems"][system]
        require(saved["query_count"] == len(ids), "Saved metric denominator differs")
        for metric in METRICS:
            require(abs(fresh["aggregate"][metric] - saved["aggregate"][metric]) < 1e-12, "Saved aggregate differs from rankings")
            require(all(abs(fresh["per_query"][q][metric] - saved["per_query"][q][metric]) < 1e-12 for q in ids), "Saved per-query value differs")
    require(measured["complete_hosted_queries"] == sum(bool(artifacts["B"]["rankings"][q]) for q in ids), "Completion count differs")
    for name, bootstrap in measured["paired_bootstrap"].items():
        require(bootstrap["query_count"] == len(ids), "Bootstrap denominator differs")
        system = name.split("_minus_")[0]
        require(system in ("B", "C"), "Unexpected bootstrap comparison")
        for metric, value in bootstrap["metrics"].items():
            expected = measured["systems"][system]["aggregate"][metric] - measured["systems"]["A"]["aggregate"][metric]
            require(abs(value["delta"] - expected) < 1e-12, "Bootstrap delta differs")
            require(all(math.isfinite(value[key]) for key in ("ci_low", "ci_high")), "Invalid confidence interval")
    return document, artifacts


def failures(scope, document, artifacts, labels, baseline, limit):
    measured = document["evaluation"]
    details = {row["query_id"]: {value["skill_id"]: value for value in row.get("scores", [])}
               for row in artifacts["B"].get("rows", [])}
    pairs, queries = [], []
    for system in ("B", "C"):
        deltas = {qid: measured["systems"][system]["per_query"][qid]["NDCG@10"] -
                  measured["systems"]["A"]["per_query"][qid]["NDCG@10"] for qid in measured["selected_query_ids"]}
        for direction in ("rise", "fall"):
            selected = [qid for qid, delta in deltas.items() if delta > 0] if direction == "rise" else [qid for qid, delta in deltas.items() if delta < 0]
            selected.sort(key=lambda qid: ((-deltas[qid] if direction == "rise" else deltas[qid]), qid))
            for qid in selected[:limit]:
                ranks = {s: {sid: rank for rank, sid in enumerate(artifacts[s]["rankings"][qid], 1)} for s in "ABC"}
                query = {"scope": scope, "comparison": system + "_minus_A", "direction": direction,
                         "query_id": qid, "delta_NDCG10": deltas[qid],
                         **{s + "_NDCG10": measured["systems"][s]["per_query"][qid]["NDCG@10"] for s in "ABC"},
                         "hosted_query_complete": bool(ranks["B"])}
                queries.append(query)
                dense50 = {sid: rank for rank, sid in enumerate(baseline["rankings"][qid], 1)}
                pool_and_misses = list(artifacts["A"]["rankings"][qid]) + sorted(set(labels[qid]) - set(ranks["A"]))
                for sid in pool_and_misses:
                    score = details.get(qid, {}).get(sid, {})
                    pairs.append({**query, "skill_id": sid, "qrel_grade": labels[qid].get(sid, 0),
                                  "dense_rank_if_top50": dense50.get(sid),
                                  "A_fixed_pool_rank": ranks["A"].get(sid), "B_native_jev_rank": ranks["B"].get(sid),
                                  "C_rrf60_rank": ranks["C"].get(sid), "native_jev_score": score.get("native_score"),
                                  "jev_confidence": score.get("confidence")})
    return queries, pairs


def metric_table(lines, measured):
    lines += ["", "| Metric | A matched dense | B Jev | C RRF60 |", "|---|---:|---:|---:|"]
    for metric in METRICS:
        lines.append("| " + metric + " | " + " | ".join(number(measured["systems"][s]["aggregate"][metric]) for s in "ABC") + " |")
    lines += ["", "| Paired comparison | Metric | Delta | 95% interval |", "|---|---|---:|---|"]
    bootstrap_notes = []
    for name, bootstrap in measured.get("paired_bootstrap", {}).items():
        for metric, value in bootstrap["metrics"].items():
            lines.append(f"| {name} | {metric} | {value['delta']:+.6f} | [{value['ci_low']:+.6f}, {value['ci_high']:+.6f}] |")
        bootstrap_notes.append(f"{name}: {bootstrap['resamples']:,} paired query resamples, seed {bootstrap['seed']}; fixed denominator {bootstrap['query_count']}.")
    lines += ["", *bootstrap_notes]


def validation_analysis(scope, document, artifacts, labels):
    """Report recorded modes; arithmetic findings never change the primary ranks."""
    pair_modes, query_modes, boolean_diagnostics, audit_rows = {}, {}, {}, []
    def numeric_fields(mapping, prefix=""):
        result = {}
        for key, value in mapping.items():
            name = prefix + str(key)
            if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", name):
                continue
            if isinstance(value, dict):
                result.update(numeric_fields(value, name + "."))
            elif isinstance(value, (int, float, bool)) and (not isinstance(value, float) or math.isfinite(value)):
                result[name] = value
            elif isinstance(value, list) and all(isinstance(item, (int, float, bool)) and math.isfinite(item) for item in value):
                result[name] = value
        return result
    scored = {row["query_id"]: row.get("scores", []) for row in artifacts["B"].get("rows", [])}
    alternative_b, alternative_c, changed_b, changed_c = {}, {}, 0, 0
    for qid in document["evaluation"]["selected_query_ids"]:
        values = scored.get(qid, [])
        modes = set()
        for score in values:
            validation = score.get("response_validation", {})
            mode = validation.get("mode") if isinstance(validation, dict) else None
            mode = mode if isinstance(mode, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", mode) else "unknown_not_recorded"
            modes.add(mode); pair_modes[mode] = pair_modes.get(mode, 0) + 1
            numeric = numeric_fields(validation) if isinstance(validation, dict) else {}
            for key, value in numeric.items():
                if isinstance(value, bool):
                    counts = boolean_diagnostics.setdefault(key, {"true": 0, "false": 0})
                    counts[str(value).lower()] += 1
            audit_rows.append({"scope": scope, "query_id": qid, "skill_id": score["skill_id"],
                               "response_validation_mode": mode, "native_score": score.get("native_score"),
                               "native_confidence": score.get("confidence"),
                               "recorded_numeric_diagnostics": json.dumps(numeric, sort_keys=True, allow_nan=False)})
        query_mode = ";".join(sorted(modes)) if modes else "no_score_records_in_snapshot"
        query_modes[query_mode] = query_modes.get(query_mode, 0) + 1
        dense = artifacts["A"]["rankings"][qid]
        if not artifacts["B"]["rankings"][qid]:
            alternative_b[qid], alternative_c[qid] = [], []
            continue
        require(len(values) == 20 and {v["skill_id"] for v in values} == set(dense), "Complete query lacks score records for sensitivity")
        positions = {sid: rank for rank, sid in enumerate(dense)}
        alternative_b[qid] = [v["skill_id"] for v in sorted(values, key=lambda v: (-v["native_score"], positions[v["skill_id"]]))]
        alternative_c[qid] = rrf60(dense, alternative_b[qid])
        changed_b += alternative_b[qid] != artifacts["B"]["rankings"][qid]
        changed_c += alternative_c[qid] != artifacts["C"]["rankings"][qid]
    selected = {qid: labels[qid] for qid in document["evaluation"]["selected_query_ids"]}
    alt = {"B": evaluate(selected, alternative_b), "C": evaluate(selected, alternative_c)}
    sensitivity = {"policy": "Native score descending; exact native-score ties use original dense rank, omitting confidence. Exploratory; primary ranking remains unchanged.",
                   "changed_B_query_orders": changed_b, "changed_C_query_orders": changed_c,
                   "query_count": len(selected), "aggregate": {name: ev["aggregate"] for name, ev in alt.items()},
                   "delta_from_primary": {name: {metric: ev["aggregate"][metric] - document["evaluation"]["systems"][name]["aggregate"][metric]
                                                  for metric in METRICS} for name, ev in alt.items()}}
    return {"pair_mode_counts": pair_modes, "query_mode_counts": query_modes,
            "recorded_boolean_diagnostic_counts": boolean_diagnostics,
            "coverage_caveat": "Counts cover score records retained in the immutable snapshot, normally complete queries; paid responses in an incomplete query may not yet appear here. No missing mode is inferred.",
            "no_confidence_tiebreak_sensitivity": sensitivity}, audit_rows


def write_csv(path, rows, empty_header):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else empty_header)
        writer.writeheader(); writer.writerows(rows)


def response_observability(rows):
    """Reservation records and preserved responses are different observations."""
    reserved = {row["attempt_id"] for row in rows if row["event"] == "reserved"}
    saved = {row["attempt_id"] for row in rows if row["event"] == "response_saved"}
    settled = {row["attempt_id"] for row in rows if row["event"] == "settled"}
    released = {row["attempt_id"] for row in rows if row["event"] == "documented_no_execution"}
    require(saved <= reserved and settled <= reserved, "Response/settlement lacks an attempt reservation")
    return {"physical_attempt_reservations": len(reserved),
            "attempts_with_preserved_http_response": len(saved),
            "attempts_with_settled_usage": len(settled),
            "attempts_without_preserved_response": len(reserved - saved - released),
            "unresolved_reservations": len(reserved - settled - released),
            "definition": "A reserved physical attempt is not proof of provider receipt or execution. A preserved response is evidence of an observed HTTP outcome. A reservation without a response can still have executed or been billed."}


def recovery_summary(path, rows, run_dir):
    if path is None or not path.exists():
        return {"status": "not_supplied"}
    receipt = read(path)
    require(receipt.get("schema_version") == "reviewed-wsl-reboot-recovery-v1", "Unsupported recovery receipt schema")
    require(receipt.get("verified") is True and receipt.get("additional_attempts") == 1, "Recovery receipt is not the reviewed single-attempt exception")
    for flag in ("retain_original_reservation", "request_bytes_and_candidate_pool_unchanged",
                 "use_first_replacement_response_regardless_of_scores", "possible_duplicate_billing_disclosed"):
        require(receipt.get(flag) is True, "Required recovery disclosure missing")
    request_hash = receipt["request_sha256"]
    require(re.fullmatch(r"[0-9a-f]{64}", request_hash), "Invalid recovery request hash")
    original, replacement = receipt["original_attempt_id"], receipt["replacement_attempt_id"]
    require(original == request_hash + "-1" and replacement == request_hash + "-2", "Recovery attempt identities differ")
    reservations = {row["attempt_id"]: row for row in rows if row["event"] == "reserved"}
    require(original in reservations, "Original recovery reservation absent from ledger")
    old = reservations[original]
    require(old["event_sha256"] == receipt["original_reservation_event_sha256"]
            and old["request_sha256"] == request_hash, "Recovery receipt does not bind the original reservation")
    require(Decimal(old["reserve_usd"]) == Decimal(receipt["original_unknown_reservation_usd"])
            and Decimal(old["reserve_cad"]) == Decimal(receipt["original_unknown_reservation_cad"]), "Original held allowance differs")
    matching = {key for key, row in reservations.items() if row.get("request_sha256") == request_hash}
    require(matching <= {original, replacement}, "More than the one reviewed replacement attempt is recorded")
    settlements = {row["attempt_id"]: row for row in rows if row["event"] == "settled"}
    released = {row["attempt_id"] for row in rows if row["event"] == "documented_no_execution"}
    saved = {row["attempt_id"] for row in rows if row["event"] == "response_saved"}
    pending = original not in settlements and original not in released
    old_final = run_dir / "study_progress" / receipt["plan_sha256"] / receipt["original_invocation_id"] / "final.json"
    return {"status": "reviewed_recovery_receipt_bound_to_ledger", "receipt_path": str(path),
            "receipt_sha256": sha(path), "request_sha256": request_hash,
            "original_attempt_id": original, "replacement_attempt_id": replacement,
            "original_invocation_id": receipt["original_invocation_id"],
            "original_final_receipt_present": old_final.exists(),
            "original_http_response_preserved_in_ledger": original in saved,
            "original_reservation_still_unresolved": pending,
            "original_unknown_held_usd": old["reserve_usd"] if pending else "0",
            "original_unknown_held_cad_with_recorded_allowances": old["reserve_cad"] if pending else "0",
            "replacement_attempt_reserved": replacement in reservations,
            "replacement_http_response_preserved": replacement in saved,
            "replacement_usage_settled": replacement in settlements,
            "one_possible_duplicate_execution_and_charge": replacement in reservations,
            "selected_response_policy": "First replacement response regardless of scores; no choice between observed outputs.",
            "interruption_lower_bound_unix": receipt["interruption_lower_bound_unix"],
            "interruption_upper_bound_unix": receipt["interruption_upper_bound_unix"],
            "authorization_definition": "Coordinator-reviewed single identical-request exception within existing CAD10/USD5 limits; not a new human approval or evidence of no original charge."}


def timing_projection_lines(document):
    timing, projected = document["wall_time"], document["projection"]
    lower = timing.get("scope_invocation_wall_seconds_lower_bound", timing.get("scope_invocation_wall_seconds"))
    upper = timing.get("scope_invocation_wall_seconds_upper_bound", timing.get("scope_invocation_wall_seconds"))
    non_api_lower = timing.get("scope_wall_minus_recorded_settled_api_seconds_lower_bound",
                               timing.get("scope_non_api_wall_seconds_lower_bound", timing.get("scope_non_api_wall_seconds")))
    non_api_upper = timing.get("scope_wall_minus_recorded_settled_api_seconds_upper_bound",
                               timing.get("scope_non_api_wall_seconds_upper_bound", timing.get("scope_non_api_wall_seconds")))
    result = ["", "### Timing and projection", "",
        f"Recorded scope invocation wall-time bounds: {number(lower, 3)} to {number(upper, 3)} s. Known API time in those invocations: {number(timing.get('new_paid_api_elapsed_in_scope_invocations_seconds'), 3)} s. Wall time minus known API time: {number(non_api_lower, 3)} to {number(non_api_upper, 3)} s; this remainder can include unobserved API time and is not necessarily pure local overhead. Experiment elapsed including gaps: {number(timing.get('experiment_elapsed_including_gaps_seconds'), 3)} s.", "", timing["definition"]]
    missing = timing.get("scope_invocations_missing_final", 0)
    if missing:
        result.append(f"\nInvocations without a final receipt: {missing}. Their last checkpoints and reviewed restart evidence supply bounds only. No final or exit receipt is invented, and an exact duration is not asserted.")
    if projected.get("status") != "descriptive_projection_not_execution_authorization":
        result.append("\nNo complete-scope wall/cost projection is available from this supplied snapshot.")
        return result
    extra_lower = projected.get("projected_additional_wall_seconds_lower_bound", projected.get("projected_additional_wall_seconds"))
    extra_upper = projected.get("projected_additional_wall_seconds_upper_bound", projected.get("projected_additional_wall_seconds"))
    total_lower = projected.get("projected_experiment_wall_seconds_lower_bound", projected.get("projected_experiment_wall_seconds_including_elapsed_gaps"))
    total_upper = projected.get("projected_experiment_wall_seconds_upper_bound", projected.get("projected_experiment_wall_seconds_including_elapsed_gaps"))
    result.append(f"\nLinear projection to {projected['target_queries']} queries: additional wall-time bounds {number(extra_lower, 1)} to {number(extra_upper, 1)} s; total experiment bounds including elapsed gaps {number(total_lower, 1)} to {number(total_upper, 1)} s. An upper planning bound is not an observed runtime or a six-hour guarantee.")
    if "projected_total_planning_commitment_cad_upper_bound" in projected:
        result.append(f"Projected settled-rate CAD valuation: {projected['projected_total_settled_rate_cad_lower_bound']}; upper planning commitment including the historical unknown held allowance once: {projected['projected_total_planning_commitment_cad_upper_bound']}. These are the evaluator's named planning bounds, not statistical cost intervals or known charges.")
        if "projected_total_planning_commitment_usd_upper_bound" in projected:
            result.append(f"Corresponding USD settled-rate projection: {projected['projected_total_settled_rate_usd_lower_bound']}; upper planning commitment: {projected['projected_total_planning_commitment_usd_upper_bound']}. Held unknown amount: USD {projected.get('held_unknown_usd', 'unknown')} / CAD {projected.get('held_unknown_cad_with_fee_buffer', 'unknown')}.")
    else:
        result.append(f"Projected recorded CAD usage/held commitment: {projected['projected_total_usage_and_held_cad_with_recorded_fee_buffer']}, excluding separately configured fixed fees.")
        scope_usd = Decimal(document["logical_scope_unique_attempt_usage_do_not_sum_scopes"]["settled_usd"])
        extra_usd = scope_usd * Decimal(projected["remaining_queries"]) / Decimal(projected["measured_queries"])
        snapshot = document["experiment_unique_ledger_at_snapshot"]
        total_usd = Decimal(snapshot["settled_usd"]) + Decimal(snapshot["held_usd"]) + extra_usd
        result.append(f"Projected additional USD usage {extra_usd}; USD usage plus held commitments at the same snapshot {total_usd}.")
    result.append("Rate extrapolation uses settled observations. Any unknown historical reservation is a planning allowance, not measured token usage, and is added once rather than multiplied across future queries. Smoke/pilot overlap is not summed. This descriptive projection does not authorize expansion or resolve billing uncertainty.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke"); parser.add_argument("--pilot")
    for name in ("ledger", "baseline", "baseline-gate", "qrels", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--parser-diagnostic")
    parser.add_argument("--recovery-receipt", help="Reviewed single-request restart recovery receipt; defaults to the named receipt beside the ledger")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args(argv)
    require(args.smoke or args.pilot, "Supply at least one actual evaluation JSON")
    require(args.examples > 0, "Example count must be positive")
    output = Path(args.output_dir)
    require(not output.exists(), "Output directory exists; use a new report version")
    resolve = lambda value: local_path(value, args.wsl_distro)
    baseline_path, gate_path, qrels_path, ledger_path = map(resolve, (args.baseline, args.baseline_gate, args.qrels, args.ledger))
    require(ledger_path.name == "budget_ledger.jsonl", "Supply the experiment ledger filename")
    baseline, gate, labels = read(baseline_path), read(gate_path), load_qrels(qrels_path)
    require(gate["status"] == "validated" and baseline["status"] == "completed", "Baseline validation is not established")
    require(len(baseline["rankings"]) == len(labels) == 4392 and gate["full_skill_count"] == 6006, "Official baseline counts differ")
    deviations = [abs(baseline["evaluation"]["aggregate"][f"NDCG@{k}"] - target) for k, target in zip((5, 10, 15), (.7902, .8112, .8187))]
    require(max(deviations) <= .005 and gate["contract_sha256"] == baseline["cache_contract_sha256"], "Baseline tolerance or cache contract differs")
    require(not any((ledger_path.parent / lock).exists() for lock in ("budget_ledger.lock", "study_runner.lock")), "Runner or ledger writer active; wait for it to stop")
    ledger_rows = SpendLedger(ledger_path.parent).rows()
    account = accounting(ledger_rows)
    observable = response_observability(ledger_rows)
    ledger_heads = {row["event_sha256"] for row in ledger_rows} | {None}
    require(not any((ledger_path.parent / lock).exists() for lock in ("budget_ledger.lock", "study_runner.lock")), "Runner or ledger writer became active; retry after it stops")
    evaluations, artifacts, source_hashes = {}, {}, {str(p): sha(p) for p in (baseline_path, gate_path, qrels_path)}
    recovery_path = resolve(args.recovery_receipt) if args.recovery_receipt else ledger_path.parent / "recovery-after-wsl-restart" / "recovery.json"
    recovery = recovery_summary(recovery_path, ledger_rows, ledger_path.parent)
    if recovery["status"] != "not_supplied":
        source_hashes[str(recovery_path)] = recovery["receipt_sha256"]
    for scope in ("smoke", "pilot"):
        supplied = getattr(args, scope)
        if supplied:
            path = resolve(supplied)
            evaluations[scope], artifacts[scope] = validate_scope(path, scope, args.wsl_distro, ledger_path.parent, baseline, labels, ledger_heads)
            source_hashes[str(path)] = sha(path)
    if "smoke" in evaluations and "pilot" in evaluations:
        require(set(evaluations["smoke"]["evaluation"]["selected_query_ids"]) <= set(evaluations["pilot"]["evaluation"]["selected_query_ids"]), "Smoke is not a pilot subset")
    lines = ["# Actual hosted study findings", "", "The full local SkillRet baseline passed its reproduction gate on all 4,392 queries and 6,006 skills. " +
             "NDCG@5/10/15: " + " / ".join(number(baseline["evaluation"]["aggregate"][f"NDCG@{k}"]) for k in (5, 10, 15)) + f"; maximum absolute official-target difference {max(deviations):.6f} (gate 0.005).", "",
             "Only the supplied immutable hosted evaluations are reported below. No missing smoke, pilot or full study result is inferred. Empty or failed selected queries keep zero scores and remain in each fixed denominator."]
    queries, pairs, validation_rows, validations = [], [], [], {}
    for scope, document in evaluations.items():
        measured = document["evaluation"]
        lines += ["", f"## {scope.title()}", "", f"{measured['complete_hosted_queries']}/{measured['query_count']} queries have complete hosted rankings. Actual model: `{measured.get('actual_model_version', 'not recorded')}`. A, B and C use the same twenty candidates."]
        metric_table(lines, measured)
        coverage = measured["upstream_dense_coverage_20_50"]
        lines += ["", "| Dense upstream coverage | Macro recall | Relevant pairs | Pair recall |", "|---|---:|---:|---:|"]
        for k in ("20", "50"):
            row = coverage[k]
            lines.append(f"| @{k} | {row['macro_recall']:.6f} | {row['relevant_pairs_found']}/{row['relevant_pairs_total']} | {row['relevant_pair_micro']:.6f} |")
        for group, subset in measured["truncation_subsets"].items():
            lines += ["", "### " + group.replace("_", " "), "", f"Queries: {subset['query_count']}."]
            if subset["query_count"]:
                metric_table(lines, subset)
        lines.append("\nThese are descriptive query groups: a query is in the truncated group if any of its twenty candidates was shortened. They are not randomized groups or a causal estimate of truncation harm.")
        validation, audit_rows = validation_analysis(scope, document, artifacts[scope], labels)
        validations[scope] = validation; validation_rows += audit_rows
        lines += ["", "### Response validation and confidence sensitivity", "", "Native provider score and confidence are preserved in the primary ranking. Structural/range/provenance validation and any arithmetic diagnostics are reported as recorded; arithmetic consistency is not silently inferred from an accepted response.", "", "| Recorded validation mode | Pair records |", "|---|---:|"]
        for mode, count in sorted(validation["pair_mode_counts"].items()):
            lines.append(f"| {mode} | {count} |")
        lines += ["", "| Per-query mode set | Queries |", "|---|---:|"]
        for mode, count in sorted(validation["query_mode_counts"].items()):
            lines.append(f"| {mode} | {count} |")
        lines.append("\n" + validation["coverage_caveat"])
        if validation["recorded_boolean_diagnostic_counts"]:
            lines += ["", "| Recorded diagnostic flag | True | False |", "|---|---:|---:|"]
            for name, counts in sorted(validation["recorded_boolean_diagnostic_counts"].items()):
                lines.append(f"| {name} | {counts['true']} | {counts['false']} |")
        sensitivity = validation["no_confidence_tiebreak_sensitivity"]
        lines += ["", sensitivity["policy"], f"Changed B query orders: {sensitivity['changed_B_query_orders']}; changed C query orders: {sensitivity['changed_C_query_orders']}. The denominator remains {sensitivity['query_count']}.", "", "| Metric | Alternative B minus primary B | Alternative C minus primary C |", "|---|---:|---:|"]
        for metric in METRICS:
            lines.append(f"| {metric} | {sensitivity['delta_from_primary']['B'][metric]:+.6f} | {sensitivity['delta_from_primary']['C'][metric]:+.6f} |")
        lines += timing_projection_lines(document)
        q, pair = failures(scope, document, artifacts[scope], labels, baseline, args.examples)
        queries += q; pairs += pair
        lines += ["", "### Largest query-level NDCG@10 changes", "", "| Comparison | Direction | Query ID | Delta | A | B | C |", "|---|---|---|---:|---:|---:|---:|"]
        for row in q:
            lines.append(f"| {row['comparison']} | {row['direction']} | `{row['query_id']}` | {row['delta_NDCG10']:+.6f} | {row['A_NDCG10']:.6f} | {row['B_NDCG10']:.6f} | {row['C_NDCG10']:.6f} |")
    lines += ["", "## Cumulative unique experiment accounting", "", f"Verified ledger head: `{account['ledger_head_sha256']}`. Physical attempt reservations: {observable['physical_attempt_reservations']}; attempts with a preserved HTTP response: {observable['attempts_with_preserved_http_response']}; settled responses: {account['paid_responses_settled']}; unresolved reservations: {account['unresolved_attempts']}; verified alias cache-reuse events: {account['alias_cache_reuse_events']}.", "", observable["definition"], "",
              f"Observed settled USD credit usage: {account['settled_usd']}. Unresolved reserved allowance: USD {account['held_usd']}, which is neither observed spend nor proof of no charge. Recorded CAD valuation of settled usage with configured fee/contingency/buffer: {account['settled_cad_with_fee_buffer']}; separate unknown held CAD allowance: {account['held_cad_with_fee_buffer']}. CAD figures exclude any separately configured fixed experiment fee and are not asserted new card charges. No hypothetical per-call fee is added.", "",
              f"Observed settled input tokens: {account['known_input_tokens_settled']}; fully reported output token total for settled responses: {account['actual_output_tokens_if_fully_reported']}. Lost/unobserved response usage is unknown and is not fabricated as zero tokens. Verified cache reuse adds no charge, but an explicitly permitted physical replacement request can duplicate an original charge. Smoke and pilot logical usage overlaps; only the unique ledger total is cumulative."]
    if recovery["status"] != "not_supplied":
        lines += ["", "## Reviewed restart-recovery protocol exception", "",
                  f"The reviewed receipt `{recovery['receipt_sha256']}` binds original attempt `{recovery['original_attempt_id']}` and the one permitted identical-request replacement `{recovery['replacement_attempt_id']}`. {recovery['authorization_definition']}", "",
                  f"Original preserved-response evidence: {recovery['original_http_response_preserved_in_ledger']}; original reservation still unresolved: {recovery['original_reservation_still_unresolved']}. Its held unknown allowance is USD {recovery['original_unknown_held_usd']} / CAD {recovery['original_unknown_held_cad_with_recorded_allowances']}. Replacement reserved: {recovery['replacement_attempt_reserved']}; replacement response preserved: {recovery['replacement_http_response_preserved']}; replacement usage settled: {recovery['replacement_usage_settled']}.", "",
                  "One possible duplicate execution and charge is explicitly disclosed if the replacement was sent. No original response is preserved, so its execution or billing outcome cannot be inferred from absent local response bytes. The exception retains the original reservation and separately accounts for the replacement; it does not erase or relabel the original attempt. Request bytes, candidates and selected query IDs remain fixed. The specified selection rule uses the first replacement response regardless of its scores; no choice between observed outputs is authorized.", "",
                  f"The interrupted original invocation `{recovery['original_invocation_id']}` has a final receipt present in the supplied filesystem: {recovery['original_final_receipt_present']}. Reviewed interruption bounds are {recovery['interruption_lower_bound_unix']} to {recovery['interruption_upper_bound_unix']} UTC epoch seconds. No final or exit receipt is reconstructed or fabricated. Primary metrics retain all frozen selected IDs; completing the metrics does not resolve the original billing uncertainty."]
    elif account["unresolved_attempts"]:
        lines.append("\nUnresolved reservations remain and no restart-recovery receipt was supplied. The report does not infer that a replacement was authorized, sent, free, or completed.")
    history = {"status": "not_supplied"}
    if args.parser_diagnostic:
        path = resolve(args.parser_diagnostic); diagnostic = read(path); source_hashes[str(path)] = sha(path)
        metadata = diagnostic["metadata"]
        response_hash = metadata["response_sha256"]
        accepted = [row for row in ledger_rows if row["event"] == "settled" and row.get("response_sha256") == response_hash]
        require(len(accepted) <= 1, "Historical response settled more than once")
        history = {"status": "historical_diagnostic_supplied", "response_sha256": response_hash,
                   "http_status": metadata["status_code"], "original_http_elapsed_seconds": metadata["elapsed_seconds"],
                   "original_error_type": diagnostic.get("parser_error", {}).get("type"),
                   "same_response_later_settled": bool(accepted)}
        lines += ["", "## Historical response-parser repair", "", f"The saved first-response diagnostic records HTTP {metadata['status_code']} with response hash `{response_hash}` and original HTTP duration {metadata['elapsed_seconds']:.6f} s. The initial score/probability consistency check rejected the serialized numeric precision. This was a local parser rejection, not a demonstrated service failure. The same preserved response is {'now uniquely settled in the ledger' if accepted else 'not recorded as settled at this ledger checkpoint'}; no repair completion is inferred without that evidence.", "",
                  "API duration excludes the investigation and repair interval. Experiment wall time includes recorded gaps; later runner timing starts are distinct from the earlier HTTP duration. Current outcomes must come from immutable reevaluated snapshots, not from changing old raw response bytes."]
    lines += ["", "## Limitations", "", "This is an exposed public benchmark with a deterministic hash-selected pilot, not a newly blinded study. Paired bootstrap intervals describe the supplied query scope; they do not establish generalization or correct multiple comparisons. Full-text and truncated groups differ in their inputs and difficulty. Reranking cannot retrieve a gold skill missing from the fixed twenty-candidate pool; @50 coverage refers only to upstream dense retrieval.", "",
              "Serialized native confidence need not reproduce a self-imposed arithmetic formula from rounded probabilities. Validation modes and numeric diagnostic flags are reported per pair/query; acceptance does not prove arithmetic agreement. Omitting or retrying queries selectively because of that disagreement could change the evaluation population. The exploratory no-confidence tie-break analysis never replaces the frozen primary results.", "",
              "Truncation followed a conservative canonical-byte admission policy, not a verified native tokenizer boundary. Wall projections include measured local overhead but can miss future growth in ledger/cache validation and a different full-plan request mix. Interrupted invocations without final receipts have timing bounds, not exact durations. Receipt mtimes are timing proxies; use original WSL artifacts, since copying may change them. A reviewed physical replacement is a disclosed protocol exception, not ordinary zero-cost cache reuse. Any original unknown reservation remains separate from observed usage pending billing evidence. No full-run completion, budget clearance, absence of duplicate charge or future runtime guarantee is inferred.", "",
              "[failure_queries.csv](failure_queries.csv) records the largest rises/falls. [failure_analysis.csv](failure_analysis.csv) gives skill IDs, gold grades, original dense ranks, native Jev ranks/scores and fusion ranks. Query text and skill text are deliberately omitted. Empty rank cells mean unavailable/not retrieved, not rank zero."]
    evidence = {"status": "actual_supplied_snapshots_reported", "scopes": evaluations, "response_validation": validations,
                "unique_experiment_ledger": account, "response_observability": observable,
                "restart_recovery": recovery, "parser_history": history, "source_sha256": source_hashes,
                "ledger_head_sha256": account["ledger_head_sha256"], "network_calls_this_process": 0, "credentials_accessed": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "report_evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "failure_queries.csv", queries, ["scope", "comparison", "direction", "query_id", "delta_NDCG10"])
    write_csv(output / "failure_analysis.csv", pairs, ["scope", "comparison", "direction", "query_id", "skill_id"])
    write_csv(output / "response_validation.csv", validation_rows, ["scope", "query_id", "skill_id", "response_validation_mode"])
    print(json.dumps({"status": "report_created", "scopes": list(evaluations), "output_dir": str(output),
                      "failure_queries": len(queries), "failure_pair_rows": len(pairs), "network_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
