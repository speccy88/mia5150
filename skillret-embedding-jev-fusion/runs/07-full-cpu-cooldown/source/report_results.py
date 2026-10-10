"""Write an evidence-backed English report; no inference, API, or credential access.

Reads only this experiment's result/protocol/cache files and runs/*/telemetry.jsonl,
guard-stop.json, and exit.json. Never reads worker logs or historical study logs.
Unrun outputs contain status markers and null rankings, never invented metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time

from evaluate import evaluate, load_qrels, paired_bootstrap, rrf60

ROOT = Path(__file__).resolve().parent
TARGETS = {"NDCG": (.7902, .8112, .8187), "Recall": (.8238, .8774, .9016),
           "Completeness": (.7083, .7894, .8265), "MAP": (.7409, .7538, .7570),
           "Precision": (.2591, .1401, .0962)}
METRICS = [f"{name}@{k}" for name in TARGETS for k in (5, 10, 15)] + ["MRR@5", "MRR@10"]
COMPARISON_METRICS = ["NDCG@5", "NDCG@10", "Recall@5", "Recall@10", "Completeness@5", "MAP@10", "MRR@5"]
UNRUN_REASON = "Hosted stage has not run in this local preparation phase; baseline validation and provider/budget gates remain required."


def read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value, only_if_missing=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if only_if_missing:
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text)
        except FileExistsError:
            return False
    else:
        temporary = path.with_name(path.name + ".report-tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return True


def save_text(path: Path, text: str):
    temporary = path.with_name(path.name + ".report-tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def csv_text(fieldnames, data):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(data)
    return stream.getvalue()


def fmt(value, places=4):
    return "—" if value is None else f"{value:.{places}f}"


def ranking_display(ids, skills):
    return "; ".join(f"{sid}: {skills.get(sid, '')}" for sid in ids)


def stable_key(query_id):
    return hashlib.sha256(query_id.encode("utf-8")).hexdigest(), query_id


def progress(experiment):
    result = []
    current_protocol = read(experiment / "protocol.json", {})
    for directory in sorted((experiment / "cache").glob("*")):
        if not directory.is_dir():
            continue
        cached_protocol = read(directory / "protocol.json", {})
        valid_current = bool(current_protocol and cached_protocol == current_protocol and cached_protocol.get("effective_rope_theta") == 1000000)
        record = {"cache_contract": directory.name,
                  "validity": "invalid_wrong_rope_theta_preserved_do_not_use" if directory.name.startswith("04df631")
                  else "current_corrected_protocol" if valid_current else "requires_current_protocol_match"}
        for kind in ("documents", "queries"):
            journal = directory / (kind + "-journal.jsonl")
            count, seconds, last_utc, incomplete_lines = 0, 0.0, None, 0
            if journal.exists():
                with journal.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            incomplete_lines += 1
                            continue
                        count += 1
                        seconds += item.get("seconds", 0)
                        last_utc = item.get("utc", last_utc)
            record[kind] = {"journaled_vectors": count, "recorded_encode_seconds": seconds,
                            "last_checkpoint_utc": last_utc, "incomplete_lines_ignored": incomplete_lines}
        result.append(record)
    return result


def resource_summary(experiment):
    result = []
    for directory in sorted((experiment / "runs").glob("*")):
        if not directory.is_dir():
            continue
        samples = []
        telemetry = directory / "telemetry.jsonl"
        if telemetry.exists():
            with telemetry.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # A concurrently appended final line can be incomplete.
        record = {"run": directory.name, "sample_count": len(samples),
                  "guard_stop": read(directory / "guard-stop.json"),
                  "exit": read(directory / "exit.json")}
        if samples:
            ages = [s["utc"] - s["host"]["utc"] for s in samples]
            swap_in = [s["swap"]["pswpin"] for s in samples]
            swap_out = [s["swap"]["pswpout"] for s in samples]
            cpu = [s["host"]["cpu_pct"] for s in samples]
            record.update({
                "first_sample_utc": samples[0]["utc"], "last_sample_utc": samples[-1]["utc"],
                "sampled_span_seconds": samples[-1]["utc"] - samples[0]["utc"],
                "min_host_available_gib": min(s["host"]["available_bytes"] for s in samples) / 1024**3,
                "min_guest_available_gib": min(s["guest_available_bytes"] for s in samples) / 1024**3,
                "max_gpu_used_mib": max(s["gpu_used_mib"] for s in samples),
                "max_gpu_temperature_c": max(s["gpu_temp_c"] for s in samples),
                "max_host_cpu_pct": max(cpu), "max_host_telemetry_age_seconds": max(ages),
                "observed_swapin_counter_change": max(swap_in) - min(swap_in),
                "observed_swapout_counter_change": max(swap_out) - min(swap_out),
                "five_consecutive_host_cpu_over80": any(all(x > 80 for x in cpu[i:i+5]) for i in range(max(0, len(cpu)-4))),
            })
        result.append(record)
    return result


def verify_stored_evaluation(stored, computed):
    if stored is None:
        return
    if stored.get("query_count") != computed["query_count"]:
        raise ValueError("Stored baseline query count differs from recomputation")
    for metric, value in computed["aggregate"].items():
        observed = stored.get("aggregate", {}).get(metric)
        if not isinstance(observed, (int, float)) or not math.isclose(value, observed, abs_tol=1e-10, rel_tol=1e-10):
            raise ValueError("Stored baseline metric differs from recomputation: " + metric)


def hosted_artifact(path, labels, baseline_rankings, kind, jev=None):
    artifact = read(path)
    if artifact is None or artifact.get("status") in ("not_run", "incomplete") or artifact.get("rankings") is None:
        return None, None
    predictions = artifact.get("rankings")
    if not isinstance(predictions, dict) or not predictions or set(predictions) - set(labels):
        raise ValueError("Invalid hosted comparison query scope")
    if artifact.get("simulated") or artifact.get("fixture"):
        raise ValueError("Synthetic results cannot enter this report")
    actual_version = artifact.get("actual_model_version")
    completed = [qid for qid, ranking in predictions.items() if ranking]
    if not completed:
        return artifact, None
    if not isinstance(actual_version, str) or not actual_version or "latest" in actual_version.lower():
        raise ValueError("Hosted results need an actual pinned model version")
    if not baseline_rankings:
        raise ValueError("Cannot compare hosted results without actual baseline rankings")
    for qid in completed:
        ranking = predictions[qid]
        pool = baseline_rankings[qid][:20]
        if len(ranking) != 20 or len(set(ranking)) != 20 or set(ranking) != set(pool):
            raise ValueError("Hosted candidate pool differs from frozen dense top20")
        if kind == "C":
            if jev is None or not jev.get("rankings", {}).get(qid):
                raise ValueError("RRF result lacks paired genuine Jev ranking")
            if ranking != rrf60(pool, jev["rankings"][qid]):
                raise ValueError("Stored RRF order is not fixed equal-weight RRF60")
    if kind == "B":
        evidence_rows = {r["query_id"]: r for r in artifact.get("rows", [])}
        for qid in completed:
            evidence = evidence_rows.get(qid, {})
            if evidence.get("status") != "complete" or len(evidence.get("scores", [])) != 20:
                raise ValueError("Completed Jev ranking lacks individual score evidence")
            if not evidence.get("batches") or any(not b.get("response_sha256") or not b.get("request_sha256") for b in evidence["batches"]):
                raise ValueError("Completed Jev ranking lacks raw-response provenance hashes")
    selected_labels = {qid: labels[qid] for qid in predictions}
    return artifact, evaluate(selected_labels, predictions)


def not_run_artifacts(results, create_baseline_placeholder, checkpoint_progress):
    for filename, system in (("jev_rerank.json", "B_jev"), ("rrf60_fusion.json", "C_rrf60")):
        save(results / filename, {"status": "not_run", "system": system, "rankings": None,
                                  "evaluation": None, "reason": UNRUN_REASON}, only_if_missing=True)
    if create_baseline_placeholder:
        save(results / "baseline_skillret_06b.json", {
            "status": "incomplete", "baseline_gate": "not_validated", "rankings": None,
            "evaluation": None, "reason": "Explicit --incomplete report snapshot; no final baseline was available.",
            "checkpoint_progress": checkpoint_progress}, only_if_missing=True)
    # Existing hosted artifacts are authoritative. Do not label their unknown usage zero.
    existing_jev = read(results / "jev_rerank.json", {})
    if existing_jev.get("status") == "not_run":
        save(results / "jev_usage.json", {
            "status": "not_run", "calls": 0, "incurred_cost": 0, "incurred_cost_currency": "CAD",
            "input_tokens": 0, "output_tokens": 0, "estimated_cost": None,
            "estimate_reason": UNRUN_REASON, "api_latency": None,
            "scope": "this preparation-only experiment; excludes other studies"}, only_if_missing=True)
    save(results / "bootstrap_ci.json", {"status": "not_run", "metrics": None,
        "reason": "No genuine paired C-minus-A comparison has been supplied."}, only_if_missing=True)


def per_query_table(labels, queries, baseline_ev, baseline_rankings, b_art, b_ev, c_art, c_ev):
    output = []
    for qid in sorted(labels):
        relevant = {sid for sid, grade in labels[qid].items() if grade > 0}
        dense = baseline_rankings.get(qid, []) if baseline_rankings else []
        row = {"query_id": qid, "query": queries.get(qid, ""), "relevant_count": len(relevant),
               "relevant_skill_ids": ";".join(sorted(relevant)),
               "dense_top5": ";".join(dense[:5]) if baseline_ev else "",
               "dense_top20": ";".join(dense[:20]) if baseline_ev else "",
               "dense_top50": ";".join(dense[:50]) if baseline_ev else "",
               "relevant_pairs_found20": len(relevant.intersection(dense[:20])) if baseline_ev else "",
               "relevant_pairs_found50": len(relevant.intersection(dense[:50])) if baseline_ev else ""}
        for name, artifact, ev in (("A", {"rankings": baseline_rankings}, baseline_ev), ("B", b_art, b_ev), ("C", c_art, c_ev)):
            values = ev["per_query"].get(qid) if ev else None
            rankings = artifact.get("rankings", {}) if artifact else {}
            row[name + "_status"] = ("complete" if rankings.get(qid) else "failed_or_missing") if values is not None else "not_run"
            for metric in METRICS + ["Coverage@20", "Coverage@50"]:
                # @50 is the upstream retrieval pool, not a 20-item reranking output.
                row[name + "_" + metric] = values[metric] if values and not (name != "A" and metric == "Coverage@50") else ""
        for metric in COMPARISON_METRICS:
            row["C_minus_A_" + metric] = (c_ev["per_query"][qid][metric] - baseline_ev["per_query"][qid][metric]) if c_ev and baseline_ev and qid in c_ev["per_query"] else ""
        output.append(row)
    return output


def failure_table(labels, queries, skills, baseline_rankings, baseline_ev, b_art, b_ev, c_art, c_ev, examples_per_category):
    categories = {}
    if baseline_ev:
        ordered = sorted(labels, key=stable_key)
        categories["dense_candidate_coverage_miss20"] = [q for q in ordered if baseline_ev["per_query"][q]["Coverage@20"] < 1]
        categories["dense_relevant_present_at_rank6_to20"] = [q for q in ordered if any(s in labels[q] and labels[q][s] > 0 for s in baseline_rankings[q][5:20])]
        categories["dense_multiskill_incomplete5"] = [q for q in ordered if sum(x > 0 for x in labels[q].values()) > 1 and baseline_ev["per_query"][q]["Completeness@5"] == 0]
    if b_ev and baseline_ev:
        delta = {q: b_ev["per_query"][q]["NDCG@10"] - baseline_ev["per_query"][q]["NDCG@10"] for q in b_ev["per_query"] if b_art["rankings"].get(q)}
        categories["jev_improves_ndcg10"] = sorted((q for q in delta if delta[q] > 0), key=lambda q: (-delta[q], stable_key(q)))
        categories["jev_hurts_ndcg10"] = sorted((q for q in delta if delta[q] < 0), key=lambda q: (delta[q], stable_key(q)))
        categories["jev_multiskill_completeness_improvement5"] = sorted((q for q in delta if sum(v > 0 for v in labels[q].values()) > 1 and b_ev["per_query"][q]["Completeness@5"] > baseline_ev["per_query"][q]["Completeness@5"]), key=stable_key)
        categories["jev_incorrectly_promoted_top5"] = sorted((q for q in delta if any(labels[q].get(s, 0) <= 0 and s not in baseline_rankings[q][:5] for s in b_art["rankings"][q][:5])), key=stable_key)
    output = []
    for category, query_ids in categories.items():
        for qid in query_ids[:examples_per_category]:
            dense = baseline_rankings[qid]
            jev = b_art.get("rankings", {}).get(qid) if b_art else None
            fusion = c_art.get("rankings", {}).get(qid) if c_art else None
            relevant = sorted(s for s, grade in labels[qid].items() if grade > 0)
            missing = sorted(set(relevant) - set(dense[:20]))
            low_hits = [f"{sid}@{rank}" for rank, sid in enumerate(dense, 1) if sid in relevant and 6 <= rank <= 20]
            interpretations = {
                "dense_candidate_coverage_miss20": "Missing relevant top20 candidates: " + ";".join(missing) + ". Reranking cannot recover these candidates.",
                "dense_relevant_present_at_rank6_to20": "Relevant low-ranked candidates already in the frozen pool: " + ";".join(low_hits) + ". This is an opportunity for reranking, not evidence of a Jev gain.",
                "dense_multiskill_incomplete5": "Dense top5 omits at least one required skill in this multi-skill query.",
                "jev_improves_ndcg10": "Observed Jev NDCG@10 gain on this query; paired aggregate intervals determine the broader conclusion.",
                "jev_hurts_ndcg10": "Observed Jev NDCG@10 loss on this query.",
                "jev_multiskill_completeness_improvement5": "Observed Jev top5 now contains all relevant skills, unlike dense top5.",
                "jev_incorrectly_promoted_top5": "Jev moves at least one qrels-nonrelevant candidate into top5 from below dense rank5.",
            }
            output.append({"category": category, "query_id": qid, "query": queries.get(qid, ""),
                "relevant_skills": ranking_display(relevant, skills), "dense_top5": ranking_display(dense[:5], skills),
                "jev_top5": ranking_display(jev[:5], skills) if jev else "",
                "rrf60_top5": ranking_display(fusion[:5], skills) if fusion else "",
                "jev_status": "complete" if jev else "not_run_or_failed", "rrf60_status": "complete" if fusion else "not_run_or_failed",
                "interpretation": interpretations[category],
                "selection": "largest observed signed NDCG delta, deterministic hash tie break" if category.startswith("jev_") and "ndcg10" in category else "ascending SHA256(query_id), first cases in category"})
    return output


def report(args):
    experiment, results = args.experiment_dir, args.experiment_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    labels = load_qrels(args.base / "data/qrels/test.jsonl")
    queries = {r["id"]: r["query"] for r in rows(args.base / "data/queries/test.jsonl")}
    skills = {r["id"]: r.get("name", "") for r in rows(args.base / "data/skills/test.jsonl")}
    checkpoints = progress(experiment)
    resources = resource_summary(experiment)
    not_run_artifacts(results, args.incomplete, checkpoints)
    baseline = read(results / "baseline_skillret_06b.json", {})
    protocol = baseline.get("protocol") or read(experiment / "protocol.json", {})
    source_hashes = {name: sha(args.base / f"data/{name}/test.jsonl") for name in ("queries","skills","qrels")}
    verified_hashes = protocol.get("test_file_sha256") == source_hashes if protocol.get("test_file_sha256") else None
    baseline_rankings = baseline.get("rankings")
    baseline_ev = None
    if isinstance(baseline_rankings, dict) and baseline_rankings:
        if set(baseline_rankings) != set(labels):
            raise ValueError("Final baseline must have rankings for every qrels query")
        if any(len(r) < 50 for r in baseline_rankings.values()):
            raise ValueError("Final baseline requires top50 for every query")
        baseline_ev = evaluate(labels, baseline_rankings)
        verify_stored_evaluation(baseline.get("evaluation"), baseline_ev)
    else:
        baseline_rankings = {}
    deviations = {f"{name}@{k}": baseline_ev["aggregate"][f"{name}@{k}"] - target
                  for name, targets in TARGETS.items() for k, target in zip((5,10,15), targets)} if baseline_ev else {}
    max_deviation = max(abs(deviations[f"NDCG@{k}"]) for k in (5,10,15)) if baseline_ev else None
    counts_match = (len(labels), len(skills), sum(sum(v > 0 for v in values.values()) for values in labels.values())) == (4392,6006,7187)
    validated = bool(baseline_ev and counts_match and max_deviation <= .005 and verified_hashes
                     and protocol.get("effective_rope_theta") == 1000000
                     and not baseline.get("cache_contract_sha256", "").startswith("04df"))
    baseline_status = "validated" if validated else "investigation_required" if baseline_ev else "incomplete_not_reproduced"
    b_art, b_ev = hosted_artifact(results / "jev_rerank.json", labels, baseline_rankings, "B")
    c_art, c_ev = hosted_artifact(results / "rrf60_fusion.json", labels, baseline_rankings, "C", b_art)
    if b_ev and c_ev and set(b_ev["per_query"]) != set(c_ev["per_query"]):
        raise ValueError("Jev and RRF comparisons must retain the same selected query denominator")
    bootstrap = None
    if c_ev:
        selected = {q: labels[q] for q in c_ev["per_query"]}
        paired_baseline = evaluate(selected, {q: baseline_rankings[q][:20] for q in selected})
        bootstrap = paired_bootstrap(paired_baseline, c_ev, n_resamples=args.bootstrap_resamples)
        bootstrap.update(status="computed_from_actual_rankings", baseline_gate=baseline_status)
        prior = read(results / "bootstrap_ci.json", {})
        if prior.get("status") == "not_run":
            save(results / "bootstrap_ci.json", bootstrap)
        elif prior != bootstrap:
            # Preserve a prior genuine calculation; save this computation separately.
            save(results / "report_bootstrap_ci.json", bootstrap)
    table = per_query_table(labels, queries, baseline_ev, baseline_rankings, b_art, b_ev, c_art, c_ev)
    save_text(results / "per_query_results.csv", csv_text(list(table[0]), table))
    failures = failure_table(labels, queries, skills, baseline_rankings, baseline_ev, b_art, b_ev, c_art, c_ev, args.examples_per_category)
    fields = ["category","query_id","query","relevant_skills","dense_top5","jev_top5","rrf60_top5","jev_status","rrf60_status","interpretation","selection"]
    save_text(results / "failure_analysis.csv", csv_text(fields, failures))
    audit = read(results / "token_audit.json", {})
    feasibility = read(results / "feasibility.json", {})
    usage = read(results / "jev_usage.json", {})
    if b_art is not None and usage.get("status") == "not_run":
        usage = {"status": "unknown_usage_with_supplied_hosted_artifact", "calls": None,
                 "incurred_cost": None, "estimated_cost": None,
                 "reason": "A prior unrun placeholder conflicted with a supplied hosted artifact. Actual usage/billing evidence is required."}
        save(results / "jev_usage.json", usage)
    comparison_evidence = {
        "baseline_gate": baseline_status,
        "baseline_full": {k: baseline_ev[k] for k in ("aggregate", "candidate_coverage", "query_count")} if baseline_ev else None,
        "jev": b_ev, "rrf60": c_ev,
        "scope_note": "Full dense baseline and selected hosted scope are separate; null means unavailable.",
    }
    save(results / "report_comparison.json", comparison_evidence)
    report_evidence = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_status": baseline_status, "dataset_counts_match_requested": counts_match,
        "baseline_evaluation_recomputed": bool(baseline_ev), "current_data_hashes_match_protocol": verified_hashes,
        "source_sha256": source_hashes, "checkpoint_progress": checkpoints, "resources": resources,
        "baseline_max_absolute_ndcg_deviation": max_deviation, "jev_evaluated": bool(b_ev), "rrf_evaluated": bool(c_ev)}
    save(results / "report_evidence.json", report_evidence)
    lines = ["# SkillRet-0.6B full-text baseline and Jev RRF60 findings", "",
        f"Report snapshot: {report_evidence['generated_utc']}. Baseline status: **{baseline_status}**.", ""]
    if baseline_ev:
        lines += [f"The full dense result contains {len(labels):,} queries, {len(skills):,} candidate skills and {sum(len(g) for g in labels.values()):,} relevant pairs. "
                  f"Maximum absolute NDCG deviation from the released target is {max_deviation:.6f}. "
                  + ("It passes the prespecified 0.005 NDCG reproduction gate." if validated else "It does not pass the full reproduction gate. Investigate before interpreting hosted comparisons."), ""]
    else:
        lines += ["The baseline has not completed. No retrieval metrics or reproduction claim are available. Journal counts below are progress evidence, not evaluated results.", ""]
    lines += [("Jev and RRF results below use supplied actual response-derived artifacts." if b_ev else "Jev has not produced a complete evaluated comparison in this phase. Its score, latency, estimated cost and improvement are unknown. No synthetic Jev result is reported."), ""]
    lines += ["## Primary comparison", "", "| System and scope | " + " | ".join(COMPARISON_METRICS) + " | Median latency (s) |", "|---|" + "---:|" * (len(COMPARISON_METRICS)+1)]
    entries = [(f"A — SkillRet-0.6B ({len(labels)} queries)", baseline_ev, baseline.get("retrieval_median_seconds"))]
    if b_ev:
        subset = {q: labels[q] for q in b_ev["per_query"]}
        matched = evaluate(subset, {q: baseline_rankings[q][:20] for q in subset})
        entries.append((f"A — matched top20 ({len(subset)} queries)", matched, None))
    api_median = usage.get("api_latency_completed_queries", {}).get("median_seconds")
    entries += [(f"B — SkillRet → Jev ({b_ev['query_count']} selected queries)" if b_ev else "B — SkillRet → Jev (not run)", b_ev, api_median if b_ev else None),
                (f"C — fixed RRF60 ({c_ev['query_count']} selected queries)" if c_ev else "C — fixed RRF60 (not run)", c_ev, api_median if c_ev else None)]
    for name, ev, latency in entries:
        lines.append("| " + name + " | " + " | ".join(fmt(ev["aggregate"].get(m)) if ev else "—" for m in COMPARISON_METRICS) + " | " + fmt(latency) + " |")
    lines += ["", "Baseline latency is query encoding plus dense dot-product search. B/C latency is API batch time when supplied, excludes retrieval, and is shared by Jev and fusion rather than a second API run. Missing or failed selected hosted queries retain empty rankings and the selected denominator. A pilot is never extrapolated into a full result.", "",
        "## All measured metrics", "", "| Metric | A full dense | A matched hosted scope | B Jev | C RRF60 |", "|---|---:|---:|---:|---:|"]
    for metric in METRICS:
        matched_value = matched["aggregate"][metric] if b_ev else None
        lines.append("| " + metric + " | " + " | ".join(fmt(value) for value in (
            baseline_ev["aggregate"][metric] if baseline_ev else None, matched_value,
            b_ev["aggregate"][metric] if b_ev else None, c_ev["aggregate"][metric] if c_ev else None)) + " |")
    lines += ["", "## Released targets and measured deviations", "", "| Metric | Released target | Actual dense | Actual minus target |", "|---|---:|---:|---:|"]
    for name, targets in TARGETS.items():
        for k, target in zip((5,10,15), targets):
            metric = f"{name}@{k}"
            actual = baseline_ev["aggregate"][metric] if baseline_ev else None
            lines.append(f"| {metric} | {target:.4f} | {fmt(actual)} | {fmt(deviations.get(metric), 6)} |")
    lines += ["", "A deviation above 0.005 triggers investigation of split, catalogue, full document construction, exact prefix, normalization, truncation, qrels and metric conventions. FP16/runtime adaptations and the dataset revision must also be considered. Metrics were recomputed from saved rankings and checked against the baseline's stored aggregates.", "",
        "## Candidate coverage and failure evidence", ""]
    if baseline_ev:
        for cutoff in ("20","50"):
            coverage = baseline_ev["candidate_coverage"][cutoff]
            lines.append(f"Dense top{cutoff}: macro candidate Recall={coverage['macro_recall']:.6f}; relevant-pair coverage={coverage['relevant_pairs_found']:,}/{coverage['relevant_pairs_total']:,} ({coverage['relevant_pair_micro']:.6f}); all-gold queries={coverage['queries_with_all_relevant']:,}/{len(labels):,}.")
    else:
        lines.append("Candidate Recall@20/@50 and relevant-pair coverage are unavailable until all dense rankings are saved.")
    lines += ["", "Valid A/B/C results use exactly the same twenty candidates. Reranking cannot recover a relevant skill missing from that pool. Dense top50 coverage describes the upstream retrieval ceiling; a 20-item reranking must not be called a top50 retrieval result.", "",
        f"`failure_analysis.csv` contains {len(failures)} deterministic examples. Baseline-only categories are coverage misses, relevant skills at ranks 6–20, and incomplete multi-skill top5 results. Actual hosted comparisons add both positive and negative cases when available. Blank Jev/RRF columns are explicit unrun/failed values, not zero-quality measurements.", "",
        "## Statistical comparison", ""]
    if bootstrap:
        for metric, item in bootstrap["metrics"].items():
            lines.append(f"- C minus A {metric}: {item['delta']:+.6f}; paired 95% CI [{item['ci_low']:+.6f}, {item['ci_high']:+.6f}].")
        lines.append("Intervals crossing zero do not establish a positive aggregate improvement. Baseline validation remains a prerequisite for interpreting these comparisons.")
    else:
        lines.append("Paired bootstrap confidence intervals are not run because there is no actual paired RRF comparison. The placeholder has null metrics.")
    lines += ["", "## Text, tokens, versions and sources", "",
        "The official document is stripped `name | description | skill_md`. The exact query prefix is `Instruct: Given a skill search query, retrieve relevant skills that match the query\\nQuery: `, with a real newline and one final space. Full text is retained in source; model input uses deterministic right-side token truncation at the configured limit. Vectors use last-token pooling and float32 L2 normalization; ranking uses dot products.", ""]
    for key in ("model","model_revision","dataset","dataset_revision","official_code_revision","max_seq_length","weight_dtype","normalization_dtype","attention","rope_theta","effective_rope_theta","batch_size","cpu_threads"):
        lines.append(f"- {key}: `{protocol.get(key, 'not recorded yet')}`")
    if protocol.get("runtime"):
        lines.append("- Runtime packages: " + "; ".join(f"{k}={v}" for k,v in protocol["runtime"].items()) + ".")
    lines += ["", f"Current source-file hashes match the run protocol: {verified_hashes if verified_hashes is not None else 'not yet recorded'}."]
    for name, value in source_hashes.items():
        lines.append(f"- {name}: `{value}`")
    if protocol.get("model_revision"):
        lines += ["", f"[Pinned model](https://huggingface.co/ThakiCloud/SKILLRET-Embedding-0.6B/tree/{protocol['model_revision']}); "
                  f"[pinned dataset](https://huggingface.co/datasets/ThakiCloud/SKILLRET/tree/{protocol.get('dataset_revision','')}); "
                  f"[official evaluation source](https://github.com/ThakiCloud/SKILLRET/blob/{protocol.get('official_code_revision','')}/skillret/eval.py). See `SOURCE_AUDIT.md` for provenance and earlier metadata-cache exclusions."]
    for kind in ("documents","queries"):
        stats = audit.get(kind)
        if stats:
            lines.append(f"{kind.title()} tokenizer counts: min={stats['min']}, median={stats['median']}, p95={stats['p95']}, max={stats['max']}, raw total={stats['total']}; over8192={stats.get('over_8192')}; over32768={stats.get('over_32768')}; truncated at this run's limit={stats['truncated_count']}.")
    lines += ["", "## Execution history, progress and resource evidence", "",
        "Recorded task-handoff history: the first feasibility attempt stopped with `No available kernel` on SM75 native grouped-query SDPA. The implementation switched to Transformers' equivalent explicit `repeat_kv` path, with memory-efficient SDPA and the quadratic math fallback disabled. This is an attention implementation adaptation, not a changed candidate catalogue or scoring objective.", "",
        "A later configuration audit found that Transformers 4.57 ignored the saved nested `rope_parameters` theta=1,000,000 and instead used its default theta=10,000. Run04 was intentionally stopped (exit -15) after 84 document vectors and before any query vectors or final result; its partial cache beginning `04df631` is invalid for ranking/reproduction and is preserved for audit only. Run03 feasibility therefore establishes resource fit only, not model correctness. The corrected loader explicitly sets theta=1,000,000 and verifies the model's actual inverse-frequency tensor before encoding. A changed loader hash, runtime, actual document/query text hashes and effective theta segregate the new cache. Only the corrected protocol can support a final reproduction claim.", ""]
    if feasibility:
        lines.append(f"Saved feasibility status={feasibility.get('status')}; tested raw token lengths={feasibility.get('raw_tokens')}; effective cap={feasibility.get('max_length')}; Torch peak reserved={feasibility.get('peak_reserved_mib')} MiB, allocated={feasibility.get('peak_allocated_mib')} MiB. These are feasibility measurements, not full-run performance. If this receipt belongs to run03, it predates the RoPE fix and proves resource fit only.")
    else:
        lines.append("The task handoff records three longest inputs passing (raw lengths 49,012 / 36,738 / 28,114), a 32,768-token cap with two truncated documents, and about 2,442 MiB Torch peak reserved. A local feasibility artifact is not available in this report snapshot; the note alone does not establish a completed baseline.")
    for item in checkpoints:
        lines.append(f"Cache `{item['cache_contract']}` ({item['validity']}): {item['documents']['journaled_vectors']:,}/6,006 document vectors and {item['queries']['journaled_vectors']:,}/4,392 query vectors journaled; recorded encoder time={item['documents']['recorded_encode_seconds'] + item['queries']['recorded_encode_seconds']:.2f}s. Cached-vector counts are not fresh inference counts for this invocation. Invalid caches must not be included in valid baseline progress.")
    for item in resources:
        lines.append(f"Run `{item['run']}`: {item['sample_count']} telemetry samples; exit={item.get('exit')}; guard stop={item.get('guard_stop')}.")
        if item["sample_count"]:
            lines.append(f"Host/guest minimum available={item['min_host_available_gib']:.3f}/{item['min_guest_available_gib']:.3f} GiB; observed GPU peak={item['max_gpu_used_mib']} MiB, max temperature={item['max_gpu_temperature_c']} C; max host CPU={item['max_host_cpu_pct']:.1f}%; five-sample CPU violation={item['five_consecutive_host_cpu_over80']}; largest host telemetry age={item['max_host_telemetry_age_seconds']:.3f}s; observed swap counter changes in/out={item['observed_swapin_counter_change']}/{item['observed_swapout_counter_change']}.")
    lines += ["", "Guard policy: at least 2 GiB free host and guest memory, no active swap, at most 7,168 MiB VRAM, temperature at most 80 C, no five consecutive host CPU samples above 80%, and host telemetry younger than eight seconds. Telemetry is sampled, not a continuous guarantee. Only this experiment's telemetry/guard receipts were read; worker and historical logs were not read by the reporter.", "",
        "## Usage, limitations and next action", ""]
    if usage.get("status") == "not_run":
        lines.append("This preparation-only experiment has zero hosted calls and zero incurred API cost. Estimated hosted cost is null because no actual smoke/pilot usage exists. The report does not inspect credential presence, model-list endpoints or provider credit.")
    else:
        lines.append("Actual supplied usage artifact: `jev_usage.json`. API median/p95 are kept separate from retrieval. Usage reported for validated caches can exclude failed or uncertain billable requests; provider billing reconciliation is still required.")
    lines.append(f"Dense retrieval median/p95={fmt(baseline.get('retrieval_median_seconds'),6)}/{fmt(baseline.get('retrieval_p95_seconds'),6)} seconds per query; document encoding={fmt(baseline.get('document_encoding_seconds'),2)} seconds (one-time cache work); model startup this run={fmt(baseline.get('startup_seconds_this_run'),2)} seconds.")
    if not b_ev:
        lines.append("No conclusion is available about Jev ordering gains, multi-skill completeness gains, redundancy with dense retrieval, category-specific harms or whether its added latency/API dependency is justified. The full study is unfinished.")
    lines.append("This is an exposed public benchmark reproduction, not a newly blinded evaluation. Keep earlier runs intact. Continue the baseline if incomplete; investigate material deviations before hosted work. Future Jev execution needs verified current version/context/prices/FX/fees/credit and the explicit all-in experiment budget; no top-up or RunPod action is implied.")
    save_text(results / "summary.md", "\n".join(lines) + "\n")
    return {"baseline_status": baseline_status, "baseline_queries_evaluated": baseline_ev["query_count"] if baseline_ev else 0,
            "jev_queries_evaluated": b_ev["query_count"] if b_ev else 0, "rrf_queries_evaluated": c_ev["query_count"] if c_ev else 0,
            "failure_examples": len(failures), "summary": str(results / "summary.md")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/home/user/code/mia5150-local"))
    parser.add_argument("--experiment-dir", type=Path, default=ROOT)
    parser.add_argument("--incomplete", action="store_true", help="Only after the worker stopped: create an absent baseline placeholder without replacing any result")
    parser.add_argument("--examples-per-category", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.examples_per_category < 1 or args.bootstrap_resamples < 1:
        parser.error("Counts must be positive")
    result = report(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
