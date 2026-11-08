"""Independent saved-ranking arithmetic and source/candidate integrity audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random

from core import STUDY, REPO, SIZES, atomic_json, digest, file_hash, full_text, jsonl


def independent(gold, ranking):
    if len(ranking) != len(set(ranking)):
        raise ValueError("Duplicate ranked ID")
    positives = {sid for sid, grade in gold.items() if grade > 0}
    out = {}
    for k in (1, 3, 5, 10, 15):
        flags = [sid in positives for sid in ranking[:k]]
        count = sum(flags)
        out[f"Hit@{k}"] = int(count > 0)
        out[f"Recall@{k}"] = count / len(positives)
        out[f"Precision@{k}"] = count / k
        out[f"Completeness@{k}"] = int(count == len(positives))
        hits = 0
        ap = 0.0
        for pos, hit in enumerate(flags, 1):
            if hit:
                hits += 1
                ap += hits / pos
        out[f"MAP@{k}"] = ap / len(positives)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(positives), k)))
        dcg = sum(int(hit) / math.log2(i + 2) for i, hit in enumerate(flags))
        out[f"NDCG@{k}"] = dcg / ideal
        out[f"MRR@{k}"] = next((1 / (i + 1) for i, hit in enumerate(flags) if hit), 0.0)
    out["Top1_ID_accuracy"] = out["Hit@1"]
    out["incorrect_top1"] = 1 - out["Hit@1"]
    out["missing_result"] = int(not ranking)
    for k in (20, 50):
        out[f"Coverage@{k}"] = len(positives.intersection(ranking[:k])) / len(positives)
    return out


def compare_evaluation(recorded, recalculated, qrels, rankings):
    """Reconcile every stored metric and both macro/micro coverage summaries."""
    assert set(recorded["per_query"]) == set(recalculated) == set(qrels)
    assert recorded["query_count"] == len(qrels)
    for qid, values in recalculated.items():
        for metric, value in recorded["per_query"][qid].items():
            assert metric in values, ("Unchecked metric", metric)
            assert abs(value - values[metric]) < 1e-12, (qid, metric)
    for metric, value in recorded["aggregate"].items():
        mean = math.fsum(r[metric] for r in recalculated.values()) / len(qrels)
        assert abs(value - mean) < 1e-12, metric
    assert recorded["incorrect_count"] == sum(r["incorrect_top1"] for r in recalculated.values())
    assert recorded["missing_count"] == sum(r["missing_result"] for r in recalculated.values())
    assert recorded["missing_query_count"] == len(set(qrels) - set(rankings))
    assert recorded["empty_rate"] == sum(not rankings.get(qid) for qid in qrels) / len(qrels)
    for k in (20, 50):
        pairs = [(set(sid for sid, grade in gold.items() if grade > 0), rankings.get(qid, [])[:k])
                 for qid, gold in qrels.items()]
        found = [len(positives.intersection(ranking)) for positives, ranking in pairs]
        total = sum(len(positives) for positives, _ in pairs)
        expected = {"relevant_pairs_found": sum(found), "relevant_pairs_total": total,
                    "queries_with_any_relevant": sum(n > 0 for n in found),
                    "queries_with_all_relevant": sum(n == len(pair[0]) for n, pair in zip(found, pairs)),
                    "macro_recall": math.fsum(n / len(pair[0]) for n, pair in zip(found, pairs)) / len(qrels),
                    "relevant_pair_micro": sum(found) / total,
                    "query_any_rate": sum(n > 0 for n in found) / len(qrels),
                    "query_all_rate": sum(n == len(pair[0]) for n, pair in zip(found, pairs)) / len(qrels),
                    "query_count": len(qrels)}
        assert set(recorded["candidate_coverage"][str(k)]) == set(expected)
        for field, value in expected.items():
            assert abs(recorded["candidate_coverage"][str(k)][field] - value) < 1e-12, (k, field)


def validate(cache):
    result = {"status": "passed_with_scope_recorded", "scaling_sizes_checked": [], "full_queries_checked": 0,
              "independent_arithmetic": True, "checks": []}
    protocol = json.loads((STUDY / "protocol.json").read_text())
    for name, sha in protocol["data_sha256"].items():
        assert file_hash(cache / "data" / (name + ".jsonl")) == sha
    skills = jsonl(cache / "data/skills.jsonl")
    docs = jsonl(STUDY / "results/documents.jsonl")
    assert len(skills) == len(docs) == 6006
    ids = {s["id"] for s in skills}
    assert len(ids) == 6006
    for skill, doc in zip(skills, docs):
        text = full_text(skill)
        assert digest(text) == doc["source_sha256"]
        assert skill["id"] == doc["skill_id"]
        end = 0
        for c in doc["chunks"]:
            assert 0 <= c["start"] <= end < c["end"] <= len(text)
            assert c["tokens"] <= protocol["chunking"]["token_budget"]
            assert digest(text[c["start"]:c["end"]]) == c["text_sha256"]
            end = c["end"]
        assert end == len(text)
    result["checks"].append("6006 complete source documents; every source character covered by valid hashed chunk spans")
    pools = json.loads((STUDY / "results/candidate_pools.json").read_text())
    historic = json.loads((REPO / "experiments/benchmarks/skill-discovery-oauth-v1-20261103/results/gpt-6.1-sol/manifest.json").read_text())["plan"]
    panel = json.loads((REPO / "experiments/benchmarks/skill-discovery-oauth-v1-20261103/data/test_queries.json").read_text())
    for n in SIZES:
        pool = pools[str(n)]["ids"]
        assert len(pool) == len(set(pool)) == n and set(pool) <= ids
        assert digest(pool) == pools[str(n)]["ordered_ids_sha256"]
        if n != 6006:
            assert all(x["candidate_ids"] == pool for x in historic if x["key"].startswith(str(n) + ":"))
        else:
            assert pool == random.Random(42).sample([s["id"] for s in skills], 6006)
        path = STUDY / f"results/scaling-n{n}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        assert len(data["rows"]) == 20
        assert {r["query_id"] for r in data["rows"]} == {f"panel-{i:02d}" for i in range(20)}
        recomputed = []
        for i, row in enumerate(data["rows"]):
            assert row["gold_id"] == panel[i]["gold_id"] and row["gold_id"] in pool
            assert set(row["ranking"]) <= set(pool)
            assert len(row["ranking"]) == min(50, n)
            assert all(math.isfinite(s) and -1.00001 <= s <= 1.00001 for s in row["scores"])
            assert all(a >= b for a, b in zip(row["scores"], row["scores"][1:]))
            metrics = independent({row["gold_id"]: 1}, row["ranking"])
            for metric, v in metrics.items():
                if metric in row:
                    assert abs(row[metric] - v) < 1e-12, (n, i, metric)
            recomputed.append(metrics)
        compare_evaluation(data["evaluation"], dict(zip((r["query_id"] for r in data["rows"]), recomputed)),
                           {r["query_id"]: {r["gold_id"]: 1} for r in data["rows"]},
                           {r["query_id"]: r["ranking"] for r in data["rows"]})
        result["scaling_sizes_checked"].append(n)
    result["checks"].append("Exact historical ordered candidate IDs; unique IDs and gold presence at all eight sizes")
    full_path = STUDY / "results/full_rankings.jsonl"
    if full_path.exists():
        rankings = jsonl(full_path)
        assert len({r["query_id"] for r in rankings}) == len(rankings)
        qrels = {}
        official_queries = jsonl(cache / "data/queries.jsonl")
        raw_qrels = jsonl(cache / "data/qrels.jsonl")
        assert len(raw_qrels) == 7187
        for row in raw_qrels:
            assert row["skill_id"] in ids and row["relevance"] == 1
            assert row["skill_id"] not in qrels.get(row["query_id"], {})
            qrels.setdefault(row["query_id"], {})[row["skill_id"]] = row["relevance"]
        assert len(official_queries) == len(qrels) == 4392
        assert {q["id"] for q in official_queries} == set(qrels)
        recalculated = {}
        for row in rankings:
            assert row["query_id"] in qrels and set(row["ranking"]) <= ids
            assert len(row["ranking"]) == 50
            assert len(row["scores"]) == 50
            assert all(math.isfinite(s) and -1.00001 <= s <= 1.00001 for s in row["scores"])
            assert all(a >= b for a, b in zip(row["scores"], row["scores"][1:]))
            recalculated[row["query_id"]] = independent(qrels[row["query_id"]], row["ranking"])
        result["full_queries_checked"] = len(rankings)
        final = STUDY / "results/full_evaluation.json"
        if final.exists():
            assert len(rankings) == 4392
            recorded = json.loads(final.read_text())
            compare_evaluation(recorded, recalculated, qrels,
                               {r["query_id"]: r["ranking"] for r in rankings})
            result["checks"].append("All 7187 unique binary qrels and all official query IDs; every stored per-query/aggregate metric and 20/50 macro/micro coverage total independently reconciled")
    summary = STUDY / "results/scaling_summary.csv"
    if summary.exists():
        for row in csv.DictReader(summary.open()):
            source = json.loads((STUDY / f"results/scaling-n{row['library_size']}.json").read_text())
            for k, v in source["evaluation"]["aggregate"].items():
                assert abs(float(row[k]) - v) < 1e-12
    result["checks"].append("Saved rankings independently rescored; JSON and CSV reconciled")
    result["completed_full_requested_scope"] = result["scaling_sizes_checked"] == list(SIZES) and result["full_queries_checked"] == 4392
    atomic_json(STUDY / "results/validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/mia-qwen")
    validate(parser.parse_args().cache)
