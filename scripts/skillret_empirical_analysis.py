#!/usr/bin/env python3
"""Paired, query-clustered SkillRet retrieval analysis from the immutable raw archive.

Requires NumPy. No inference, API calls, training, or dataset download.
Example: python skillret_empirical_analysis.py --raw results/retrieval.jsonl.gz --out empirical_results.json
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_SHA256 = "a55bfecfa99508cebcc0d6c0238575c60bfbd919d46c2e69654f66dcd0b53a35"
METHODS = ("bm25", "vector", "hybrid")
SIZES = (10, 100, 1000, 6006)
SEEDS = (42, 43, 44)


def analyse(source: Path, bootstraps: int, random_seed: int):
    h = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != EXPECTED_SHA256:
        raise ValueError("Archive differs from the pinned, independently validated evidence")

    records = collections.defaultdict(dict)
    gold = {}
    seen = set()
    case_count = 0
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            q, seed, size = r["query_id"], int(r["seed"]), int(r["size"])
            if (q, seed, size) in seen:
                raise ValueError("Duplicate query/seed/size case")
            seen.add((q, seed, size))
            case_count += 1
            if seed not in SEEDS or size not in SIZES or set(r["methods"]) != set(METHODS):
                raise ValueError("Unexpected seed, catalog size, or method")
            for m, d in r["methods"].items():
                hits, relevant = int(d["retrieved_hits"]), int(d["gold_count"])
                if relevant <= 0 or not (0 <= hits <= relevant):
                    raise ValueError("Invalid retrieved or reference counts")
                if not np.isclose(float(d["recall5"]), hits / relevant, atol=1e-12, rtol=0):
                    raise ValueError("Raw recall mismatch")
                if len(d["ranking5"]) != len(set(d["ranking5"])) or len(d["ranking5"]) != 5:
                    raise ValueError("Invalid top-five ranking")
                if q in gold and gold[q] != relevant:
                    raise ValueError("Reference relevance count changed for the query")
                gold[q] = relevant
                records[(size, seed, m)][q] = {
                    "recall": float(d["recall5"]),
                    "mrr": float(d["mrr5"]),
                    "hits": hits,
                    "search_only_ms": 1000.0 * float(d["search_seconds"]),
                }
    query_ids = sorted(gold)
    if case_count != 52704 or len(query_ids) != 4392 or sum(gold.values()) != 7187:
        raise ValueError("Unexpected study breadth or relevance count")
    for size in SIZES:
        for seed in SEEDS:
            for method in METHODS:
                if set(records[(size, seed, method)]) != set(query_ids):
                    raise ValueError("Incomplete paired study panel")
    # Repeated full-catalog seeds do not create independent catalogs or rankings.
    for method in METHODS:
        for q in query_ids:
            if any(records[(6006, s, method)][q]["recall"] != records[(6006, 42, method)][q]["recall"]
                   or records[(6006, s, method)][q]["mrr"] != records[(6006, 42, method)][q]["mrr"]
                   for s in SEEDS):
                raise ValueError("Full-catalog retrieval differs across catalog seeds")

    # A query is the sampling unit. Average all three catalog draws within each query
    # BEFORE resampling; the 52,704 repeated conditions are not independent requests.
    rng = np.random.default_rng(random_seed)
    results = []
    comparisons = []
    for size in SIZES:
        means = {}
        for m in METHODS:
            for metric in ("recall", "mrr"):
                matrix = np.asarray([[records[(size, s, m)][q][metric] for q in query_ids]
                                     for s in SEEDS], dtype=np.float64)
                means[(m, metric)] = matrix.mean(axis=0)
                if metric == "recall":
                    results.append({
                        "size": size, "method": m,
                        "mean_recall5": float(matrix.mean()),
                        "catalog_seed_sample_sd": float(matrix.mean(axis=1).std(ddof=1)),
                        "mean_search_only_ms": float(np.mean([
                            records[(size, s, m)][q]["search_only_ms"]
                            for s in SEEDS for q in query_ids])),
                    })
        # Use the same query resamples for all method contrasts at this size.
        # Nominal 95% intervals, no multiplicity correction; exploratory except for
        # the explicitly highlighted hybrid-versus-BM25 full-catalog contrast.
        draws = rng.integers(0, len(query_ids), (bootstraps, len(query_ids)), dtype=np.int32)
        for metric in ("recall", "mrr"):
            for a, b in (("hybrid", "bm25"), ("hybrid", "vector"), ("vector", "bm25")):
                diff = means[(a, metric)] - means[(b, metric)]
                simulated = diff[draws].mean(axis=1)
                lo, hi = np.quantile(simulated, (0.025, 0.975))
                comparisons.append({
                    "size": size, "metric": metric, "contrast": f"{a} - {b}",
                    "delta_pp": float(100 * diff.mean()),
                    "ci95_pp": [float(100 * lo), float(100 * hi)],
                    "independent_query_clusters": len(query_ids),
                    "nested_catalog_seeds": len(SEEDS),
                })

    mechanism = collections.Counter()
    strata = collections.defaultdict(lambda: collections.Counter())
    for q in query_ids:
        b, v, h = (records[(6006, 42, m)][q] for m in METHODS)
        for name, competitor in (("bm25", b), ("vector", v)):
            effect = h["hits"] - competitor["hits"]
            mechanism[f"hybrid_vs_{name}_" + ("improved" if effect > 0 else "degraded" if effect < 0 else "unchanged")] += 1
        if b["hits"] == 0:
            mechanism["BM25_no_positive_in_top5"] += 1
            if h["hits"] > 0:
                mechanism["BM25_miss_recovered_by_hybrid"] += 1
            if v["hits"] > 0:
                mechanism["BM25_miss_recovered_by_vector"] += 1
        if h["hits"] > max(b["hits"], v["hits"]):
            mechanism["hybrid_outperforms_both_components"] += 1
        k = gold[q]
        strata[k]["n"] += 1
        for method, record in zip(METHODS, (b, v, h)):
            strata[k][f"{method}_sum_recall5"] += record["recall"]

    return {
        "provenance": {"source_sha256": EXPECTED_SHA256, "raw_cases": case_count,
                       "queries": len(query_ids), "positive_qrels": sum(gold.values()),
                       "catalog_seeds": list(SEEDS), "bootstrap_seed": random_seed,
                       "bootstrap_replicates": bootstraps},
        "method_size_results": results,
        "paired_query_bootstrap": comparisons,
        "full_catalog_mechanism": dict(mechanism),
        "gold_skill_strata": {str(k): {"queries": v["n"], **{
            m + "_recall5": float(v[f"{m}_sum_recall5"] / v["n"]) for m in METHODS}}
            for k, v in sorted(strata.items())},
        "limitations": [
            "Query-cluster bootstrap conditions on previously exposed synthetic queries and incomplete binary qrels.",
            "Catalog seeds are sampled distractor sets, not independent model training runs.",
            "All three full-catalog seeds are mathematically identical and are not independent replicates.",
            "Search-only milliseconds exclude query embedding, document indexing, GPU runtime, and model/API spending.",
            "The published SkillRet BM25 baseline uses a different document representation; scores are contextual only.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261121)
    args = p.parse_args()
    if args.bootstrap < 10:
        raise ValueError("At least 10 bootstrap draws are required")
    result = analyse(args.raw, args.bootstrap, args.seed)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    row = next(x for x in result["paired_query_bootstrap"] if x["size"] == 6006
               and x["metric"] == "recall" and x["contrast"] == "hybrid - bm25")
    print(json.dumps(row, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
