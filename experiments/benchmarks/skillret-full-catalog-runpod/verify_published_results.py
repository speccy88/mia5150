"""Check the committed summary against saved per-query results and test qrels.

This audits the published snapshot; it does not rerun an encoder or benchmark.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METHODS = (
    "bm25_full",
    "bge_m3_dense_full",
    "qwen3_raw_full",
    "clm_pretrained_full",
    "clm_skillret_full",
)
HEAD_SHA = "45770bd332ece2830ed624bac30c33f3ac3267eed684edf151abd0cc36fa4dd4"
PDF_SHA = "ef183a1290dfc541aef2a27cf4209f2fa11d146f76c8cfdb46fe779596246017"


def rows(path: Path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    index = json.loads((ROOT / "results/score_index.json").read_text())
    queries = set(index["query_ids"])
    skills = set(index["skill_ids"])
    assert len(queries) == len(index["query_ids"])
    assert len(skills) == len(index["skill_ids"])
    gold = defaultdict(set)
    for rel in rows(ROOT / "data/test_qrels.jsonl"):
        assert rel["relevance"] > 0
        assert rel["query_id"] in queries and rel["skill_id"] in skills
        gold[rel["query_id"]].add(rel["skill_id"])
    assert len(queries) == len(gold) == 4392 and len(skills) == 6006
    aggregate = json.loads((ROOT / "results/aggregate.json").read_text())
    assert set(aggregate) == set(METHODS)
    metrics = (
        "hit_at_1", "recall_at_1", "recall_at_5", "recall_at_10",
        "reciprocal_rank", "ndcg_at_5", "completeness_at_5",
    )
    for method in METHODS:
        seen = set()
        total = dict.fromkeys(metrics, 0.0)
        for row in rows(ROOT / "results" / f"{method}_per_query.jsonl"):
            qid = row["query_id"]
            assert qid in queries and qid not in seen
            seen.add(qid)
            ranking = row["top_10"]
            assert len(ranking) == len(set(ranking)) == 10
            assert set(ranking) <= skills
            relevant = gold[qid]
            recall_1 = int(ranking[0] in relevant) / len(relevant)
            recall_5 = len(set(ranking[:5]) & relevant) / len(relevant)
            recall_10 = len(set(ranking) & relevant) / len(relevant)
            assert math.isclose(row["hit_at_1"], int(ranking[0] in relevant), abs_tol=1e-12)
            assert math.isclose(row["recall_at_1"], recall_1, abs_tol=1e-12)
            assert math.isclose(row["recall_at_5"], recall_5, abs_tol=1e-12)
            assert math.isclose(row["recall_at_10"], recall_10, abs_tol=1e-12)
            assert math.isclose(row["completeness_at_5"], int(recall_5 == 1), abs_tol=1e-12)
            first = next((i + 1 for i, sid in enumerate(ranking) if sid in relevant), None)
            if first is not None:
                assert math.isclose(row["reciprocal_rank"], 1 / first, abs_tol=1e-12)
            else:
                assert 0 <= row["reciprocal_rank"] <= 0.1
            dcg = sum(int(sid in relevant) / math.log2(i + 2)
                      for i, sid in enumerate(ranking[:5]))
            ideal = sum(1 / math.log2(i + 2) for i in range(min(5, len(relevant))))
            assert math.isclose(row["ndcg_at_5"], dcg / ideal, abs_tol=1e-12)
            for metric in metrics:
                total[metric] += row[metric]
        assert seen == queries
        assert aggregate[method]["queries"] == len(queries)
        assert aggregate[method]["catalog_size"] == len(skills)
        for metric in metrics:
            assert math.isclose(total[metric] / len(queries), aggregate[method][metric], abs_tol=1e-12), (method, metric)
        print(f"{method}: {len(seen)} queries, Recall@5={aggregate[method]['recall_at_5']:.4f}")
    assert sha256(ROOT / "REPORT.pdf") == PDF_SHA
    head = ROOT / "checkpoints/clm_skillret/best_head.pt"
    assert sha256(head) == HEAD_SHA, "Run git lfs install --local && git lfs pull to obtain the trained head"
    print("PDF and trained head SHA-256: verified")


if __name__ == "__main__":
    main()
