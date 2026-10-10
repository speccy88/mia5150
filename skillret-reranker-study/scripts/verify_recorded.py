#!/usr/bin/env python3
"""Recompute the published comparison from saved evidence, without inference."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference_run"


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_query(records):
    result = {r["query_id"]: r for r in records}
    assert len(result) == len(records), "duplicate query IDs"
    return result


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert abs(actual - expected) < 1e-12, (actual, expected)


def verify_archived_hashes():
    checks = (REF / "SHA256SUMS").read_text().splitlines()
    assert len(checks) == 41
    for check in checks:
        expected, relative = check.split("  ", 1)
        path = (REF / relative).resolve()
        assert path.is_relative_to(REF.resolve())
        assert digest(path) == expected, f"historical file changed: {relative}"


def verify_paired():
    jm = read_json(REF / "jev/manifest.json")
    js = read_json(REF / "jev/spec.json")
    cm = read_json(REF / "clm/run_manifest.json")
    for name, expected in js["code_sha256"].items():
        assert digest(ROOT / name) == expected, f"source changed: {name}"
    for name, expected in cm["source_hashes"].items():
        assert digest(ROOT / name) == expected, f"source changed: {name}"
    assert jm["dataset"]["revision"] == cm["dataset_revision"]
    assert jm["metadata_hash"] == cm["metadata_hash"]

    pilot_ids = read_json(ROOT / "manifests/pilot_query_ids.json")
    jc = by_query(read_json(REF / "jev/candidates.json"))
    cc = by_query(read_json(REF / "clm/inputs.json")["cases"])
    ce = by_query(read_json(REF / "clm/evaluation_only.json"))
    jr = by_query(read_jsonl(REF / "jev/rows.jsonl"))
    cr = by_query(read_jsonl(REF / "clm/bm25_clm_raw.jsonl"))
    bm25 = by_query(read_jsonl(REF / "baselines/metadata_bm25_rows.jsonl"))
    metadata = {r["id"]: r["metadata"] for r in read_jsonl(REF / "baselines/metadata.jsonl")}
    assert len(metadata) == 6006
    assert len(pilot_ids) == 200 and len(set(pilot_ids)) == 200
    assert set(pilot_ids) == set(jc) == set(cc) == set(ce) == set(jr) == set(cr) == set(bm25)
    assert js["query_ids"] == pilot_ids == cm["pilot_ids"]

    recall = {method: [] for method in ("bm25", "jev", "clm")}
    complete = {method: 0 for method in recall}
    comparisons = []
    for qid in pilot_ids:
        jcase, ccase = jc[qid], cc[qid]
        jev, clm, baseline = jr[qid], cr[qid], ce[qid]
        assert jcase["query"] == ccase["query_text"]
        assert {c["id"]: c["metadata"] for c in jcase["candidates"]} == dict(
            zip(ccase["candidate_ids"], ccase["candidate_texts"], strict=True)
        )
        assert all(metadata[c["id"]] == c["metadata"] for c in jcase["candidates"])
        top20 = [pair[0] for pair in jcase["bm25_top20"]]
        assert top20 == baseline["ranking"]
        assert set(top20) == set(ccase["candidate_ids"]) == set(clm["candidate_ids"])
        assert [pair[0] for pair in bm25[qid]["ranking"][:5]] == top20[:5]
        assert set(jcase["evaluator_only_gold"]) == set(baseline["evaluator_only_gold"])
        assert set(jev["evaluator_only_gold"]) == set(baseline["evaluator_only_gold"])
        assert jev["status"] == clm["status"] == "ok"
        assert len(jev["jev_top5"]) == 5 and len(clm["ranking"]) == 20
        assert set(jev["jev_top5"]) <= set(top20) and set(clm["ranking"]) == set(top20)

        gold = set(baseline["evaluator_only_gold"])
        for method, predicted in (("bm25", top20[:5]), ("jev", jev["jev_top5"]),
                                  ("clm", clm["ranking"][:5])):
            score = len(gold & set(predicted)) / len(gold)
            recall[method].append(score)
            complete[method] += gold <= set(predicted)
        comparisons.append(recall["jev"][-1] - recall["clm"][-1])

    mean = {method: statistics.mean(scores) for method, scores in recall.items()}
    jsum = read_json(REF / "jev/summary-200.json")
    csum = read_json(REF / "clm/bm25_clm_summary.json")
    close(mean["bm25"], jsum["bm25_recall_at_5"])
    close(mean["jev"], jsum["jev_recall_at_5"])
    close(mean["clm"], csum["bm25_clm_recall_at_5"])
    assert complete == {"bm25": 89, "jev": 114, "clm": 51}
    close(statistics.mean(comparisons), .3275)
    assert [sum(delta > 0 for delta in comparisons), sum(delta < 0 for delta in comparisons),
            sum(delta == 0 for delta in comparisons)] == [91, 1, 108]
    assert jsum["successful_queries"] == csum["attempted"] == 200
    assert jsum["service_failures"] == csum["service_failures"] == 0
    return mean, complete, recall


def verify_llms(paired_recall):
    tables = (("OLLAMA_RESULTS.csv", "method", "qwen3.5:4b-mlx", "ollama_trial.jsonl"),
              ("OLLAMA_RESULTS.csv", "method", "qwen3.5:9b-mlx", "ollama_trial.jsonl"),
              *(("OPENROUTER_MODELS.csv", "strategy", model, f"{model}_trial.jsonl") for model in
                ("qwen38-27b", "qwen38-flash", "deepseek41-flash", "mimo-v2-6-pro", "glm53-flash")))
    llm_ids = None
    output = []
    for table, method_key, model, raw_name in tables:
        summary = next(r for r in csv.DictReader((REF / "llm" / table).open())
                       if r["model"] == model and r["method"] == "bm25" and int(r["N"]) == 50)
        raw = [r for r in read_jsonl(REF / "llm" / raw_name)
               if r.get(method_key) == "bm25" and r["N"] == 50
               and (table == "OPENROUTER_MODELS.csv" or r["model"] == model)]
        assert len(raw) == int(summary["attempted"])
        valid = [r for r in raw if r["status"] == "valid"]
        assert len(valid) == int(summary["valid"])
        assert int(summary["planned"]) == 5
        close(statistics.mean(r["selection"]["recall"] for r in valid), float(summary["recall"]))
        close(statistics.mean(r["selection"]["precision"] for r in valid), float(summary["precision"]))
        exact = sum(r["selection"]["exact_set_match"] for r in valid)
        close(exact, float(summary.get("exact", summary.get("exact_matches"))))
        if model == "glm53-flash":
            llm_ids = {r["query_id"] for r in raw}
        output.append((model, len(valid), len(raw), float(summary["recall"])))
    assert llm_ids is not None and len(llm_ids) == 5
    pilot_ids = read_json(ROOT / "manifests/pilot_query_ids.json")
    indices = [pilot_ids.index(qid) for qid in llm_ids]
    close(statistics.mean(paired_recall["jev"][i] for i in indices), .8)
    close(statistics.mean(paired_recall["clm"][i] for i in indices), .7)
    return output


def main():
    verify_archived_hashes()
    mean, complete, paired_recall = verify_paired()
    llms = verify_llms(paired_recall)
    print("Verified SHA-256 hashes for 41 published historical files")
    print("Matched 200-query evidence: identical queries, metadata, BM25 candidates, and gold labels")
    for method in ("bm25", "jev", "clm"):
        print(f"  {method:5} Recall@5 {mean[method]:.4%}; Complete@5 {complete[method]}/200")
    print("  Jev - CLM: +32.75 points; better/lower/tied: 91/1/108")
    print("Saved five-query BM25 -> LLM rows (valid/attempted, recall on valid outputs):")
    for model, valid, attempted, recall in llms:
        print(f"  {model}: {valid}/{attempted}, {recall:.2%}")
    print("No inference or paid requests were made.")


if __name__ == "__main__":
    main()
