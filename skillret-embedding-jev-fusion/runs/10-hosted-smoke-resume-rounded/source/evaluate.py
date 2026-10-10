"""Dependency-free SkillRet metrics and fixed-pool comparison utilities.

Conventions match pytrec_eval: NDCG uses linear graded gain, relevance > 0
defines a hit, MAP@k divides by *all* relevant documents, and Precision@k
divides by k even for a short run. Completeness is 1 only when the nonempty
relevant set is completely retrieved. Every qrels query contributes to the
macro denominator, including missing/empty rankings. Values are not rounded.

Reference inspected: mia5150-local/official/skillret/eval.py::trec_eval
and mia5150-local/harness.py::evaluate. The official function rounds its
averages to five decimals; retain full precision here until presentation.
No model, network, environment-variable, or credential access occurs here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

CUTOFFS = (5, 10, 15)
COVERAGE_CUTOFFS = (20, 50)
BOOTSTRAP_METRICS = ("NDCG@10", "Recall@5", "Recall@10", "Completeness@5", "MRR@5")


def _string_keys(mapping: Mapping, name: str) -> dict:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result = {}
    for key, value in mapping.items():
        converted = str(key)
        if converted in result:
            raise ValueError(f"{name} has colliding string IDs: {converted!r}")
        result[converted] = value
    return result


def _ranking(values: Sequence, name: str) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an ordered sequence of document IDs")
    result = [str(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate document IDs")
    return result


def _labels(labels: Mapping) -> dict[str, dict[str, int]]:
    normalized = {}
    for qid, gold in _string_keys(labels, "labels").items():
        normalized[qid] = _string_keys(gold, f"labels[{qid}]")
        for sid, relevance in normalized[qid].items():
            if isinstance(relevance, bool) or not isinstance(relevance, int):
                raise ValueError(f"Relevance for {qid}/{sid} must be an integer")
    if not normalized:
        raise ValueError("Evaluation requires at least one qrels query")
    return normalized


def evaluate(labels: Mapping, rankings: Mapping) -> dict[str, Any]:
    """Evaluate qid -> {sid: integer grade} against qid -> unique ordered IDs.

    Missing ranking queries are zero-scored. Unknown ranking queries and
    duplicate document IDs are errors rather than silent denominator changes.
    Nonpositive qrels grades are nonrelevant and have zero NDCG gain.
    """
    labels = _labels(labels)
    rankings = _string_keys(rankings, "rankings")
    unknown = set(rankings) - set(labels)
    if unknown:
        raise ValueError(f"Rankings contain unknown query IDs: {sorted(unknown)[:5]}")
    rankings = {qid: _ranking(ranking, f"rankings[{qid}]") for qid, ranking in rankings.items()}
    rows = {}
    coverage = {
        str(k): {"relevant_pairs_found": 0, "relevant_pairs_total": 0,
                 "queries_with_any_relevant": 0, "queries_with_all_relevant": 0,
                 "macro_recall": 0.0}
        for k in COVERAGE_CUTOFFS
    }
    for qid, gold in labels.items():
        ranking = rankings.get(qid, [])
        relevant = {sid for sid, grade in gold.items() if grade > 0}
        n_relevant = len(relevant)
        ideal = sorted((grade for grade in gold.values() if grade > 0), reverse=True)
        values = {}
        for k in CUTOFFS:
            retrieved = ranking[:k]
            hits = len(relevant.intersection(retrieved))
            dcg = math.fsum(max(gold.get(sid, 0), 0) / math.log2(rank + 1)
                            for rank, sid in enumerate(retrieved, 1))
            idcg = math.fsum(grade / math.log2(rank + 1)
                             for rank, grade in enumerate(ideal[:k], 1))
            seen_hits = 0
            precision_at_hits = []
            for rank, sid in enumerate(retrieved, 1):
                if sid in relevant:
                    seen_hits += 1
                    precision_at_hits.append(seen_hits / rank)
            values[f"NDCG@{k}"] = dcg / idcg if idcg else 0.0
            values[f"Recall@{k}"] = hits / n_relevant if n_relevant else 0.0
            values[f"Completeness@{k}"] = float(bool(n_relevant) and hits == n_relevant)
            values[f"MAP@{k}"] = math.fsum(precision_at_hits) / n_relevant if n_relevant else 0.0
            values[f"Precision@{k}"] = hits / k
        for k in (5, 10):
            values[f"MRR@{k}"] = next((1.0 / rank for rank, sid in enumerate(ranking[:k], 1)
                                       if sid in relevant), 0.0)
        for k in COVERAGE_CUTOFFS:
            hits = len(relevant.intersection(ranking[:k]))
            recall = hits / n_relevant if n_relevant else 0.0
            values[f"Coverage@{k}"] = recall
            summary = coverage[str(k)]
            summary["relevant_pairs_found"] += hits
            summary["relevant_pairs_total"] += n_relevant
            summary["queries_with_any_relevant"] += int(hits > 0)
            summary["queries_with_all_relevant"] += int(bool(n_relevant) and hits == n_relevant)
        rows[qid] = values
    count = len(labels)
    aggregate = {metric: math.fsum(row[metric] for row in rows.values()) / count
                 for metric in next(iter(rows.values()))}
    for k in COVERAGE_CUTOFFS:
        summary = coverage[str(k)]
        summary["macro_recall"] = aggregate[f"Coverage@{k}"]
        total = summary["relevant_pairs_total"]
        summary["relevant_pair_micro"] = summary["relevant_pairs_found"] / total if total else 0.0
        summary["query_any_rate"] = summary["queries_with_any_relevant"] / count
        summary["query_all_rate"] = summary["queries_with_all_relevant"] / count
        summary["query_count"] = count
    return {"aggregate": aggregate, "per_query": rows, "query_count": count,
            "candidate_coverage": coverage,
            "empty_rate": sum(not rankings.get(qid) for qid in labels) / count,
            "missing_query_count": len(set(labels) - set(rankings)),
            "metric_conventions": {
                "ndcg_gain": "linear positive relevance",
                "binary_hit": "relevance > 0", "map_denominator": "all relevant documents",
                "precision_denominator": "cutoff k", "macro_denominator": "all qrels queries",
                "completeness": "nonempty relevant set entirely retrieved",
                "candidate_coverage": "macro recall and micro relevant-pair recall",
                "rounding": "none"}}


def deterministic_subset(query_ids: Sequence, limit: int = 200) -> list[str]:
    """Smallest SHA256(str(query_id).encode('utf-8')) hashes, then ID for ties."""
    ids = _ranking(query_ids, "query_ids")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    return sorted(ids, key=lambda qid: (hashlib.sha256(qid.encode("utf-8")).digest(), qid))[:limit]


def build_study_subsets(query_ids: Sequence) -> dict[str, list[str]]:
    study = deterministic_subset(query_ids, 200)
    return {"study_200": study, "smoke_10": study[:10]}


def _finite_scores(scores: Mapping, pool: list[str], name: str) -> dict[str, float]:
    scores = _string_keys(scores, name)
    if set(scores) != set(pool):
        raise ValueError(f"{name} must contain exactly the dense candidate pool")
    output = {}
    for sid, score in scores.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"{name}[{sid}] must be a finite numeric value")
        output[sid] = float(score)
    return output


def rank_jev(dense_ranking: Sequence, scores: Mapping, confidences: Mapping) -> list[str]:
    """Descending rubric score, descending confidence, then original dense rank."""
    dense = _ranking(dense_ranking, "dense_ranking")
    scores = _finite_scores(scores, dense, "scores")
    confidences = _finite_scores(confidences, dense, "confidences")
    positions = {sid: index for index, sid in enumerate(dense)}
    return sorted(dense, key=lambda sid: (-scores[sid], -confidences[sid], positions[sid]))


def rrf60(dense_ranking: Sequence, jev_ranking: Sequence) -> list[str]:
    """Equal-weight 1/(60+dense_rank) + 1/(60+Jev_rank), ranks starting at 1.

    Identical candidate sets are mandatory. Exact rational arithmetic ensures
    mathematically tied fusion scores use the original dense ranking.
    """
    dense = _ranking(dense_ranking, "dense_ranking")
    jev = _ranking(jev_ranking, "jev_ranking")
    if set(dense) != set(jev):
        raise ValueError("RRF60 requires identical dense and Jev candidate pools")
    dense_positions = {sid: rank for rank, sid in enumerate(dense, 1)}
    jev_positions = {sid: rank for rank, sid in enumerate(jev, 1)}
    return sorted(dense, key=lambda sid: (
        -(Fraction(1, 60 + dense_positions[sid]) + Fraction(1, 60 + jev_positions[sid])),
        dense_positions[sid]))


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def paired_bootstrap(baseline: Mapping, treatment: Mapping, *, n_resamples: int = 10000,
                     seed: int = 20261006, confidence: float = 0.95,
                     metrics: Sequence[str] = BOOTSTRAP_METRICS) -> dict[str, Any]:
    """Paired query bootstrap of treatment minus baseline (C minus A).

    Inputs are evaluate() results. Query IDs must match exactly; missing runs
    must be represented by evaluate()'s zero rows. Resample all queries with
    replacement, share each draw across metrics, and use a percentile CI.
    Standard-library Random fixes the engine and avoids a numpy dependency.
    """
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    a = _string_keys(baseline["per_query"], "baseline per_query")
    c = _string_keys(treatment["per_query"], "treatment per_query")
    if not a or set(a) != set(c):
        raise ValueError("Paired bootstrap requires the same nonempty query ID set")
    metrics = _ranking(metrics, "metrics")
    if not metrics:
        raise ValueError("At least one bootstrap metric is required")
    qids = sorted(a)
    count = len(qids)
    if baseline.get("query_count", count) != count or treatment.get("query_count", count) != count:
        raise ValueError("query_count does not match per_query rows")
    deltas = []
    summaries = {}
    for metric in metrics:
        av = [float(a[qid][metric]) for qid in qids]
        cv = [float(c[qid][metric]) for qid in qids]
        if not all(math.isfinite(value) for value in av + cv):
            raise ValueError(f"Nonfinite values for bootstrap metric {metric}")
        differences = [right - left for left, right in zip(av, cv)]
        deltas.append(differences)
        summaries[metric] = {"baseline_mean": math.fsum(av) / count,
                             "treatment_mean": math.fsum(cv) / count,
                             "delta": math.fsum(differences) / count}
    distributions = [[] for _ in metrics]
    rng = random.Random(seed)
    for _ in range(n_resamples):
        sample = [rng.randrange(count) for _ in range(count)]
        for differences, distribution in zip(deltas, distributions):
            distribution.append(math.fsum(differences[index] for index in sample) / count)
    alpha = (1 - confidence) / 2
    for metric, distribution in zip(metrics, distributions):
        distribution.sort()
        summaries[metric].update({"ci_low": _percentile(distribution, alpha),
                                  "ci_high": _percentile(distribution, 1 - alpha),
                                  "bootstrap_fraction_delta_le_zero":
                                      sum(value <= 0 for value in distribution) / n_resamples})
    return {"direction": "treatment_minus_baseline", "query_count": count,
            "resamples": n_resamples, "seed": seed, "confidence": confidence,
            "method": "paired query bootstrap; percentile CI; linear percentile interpolation",
            "rng": "python.random.Random; randrange; lexicographically sorted string query IDs",
            "query_ids_sha256": hashlib.sha256(json.dumps(qids, ensure_ascii=False,
                                                          separators=(",", ":")).encode("utf-8")).hexdigest(),
            "metrics": summaries}


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid, sid = str(row["query_id"]), str(row["skill_id"])
            if sid in result.get(qid, {}):
                raise ValueError(f"Duplicate qrel pair at line {line_number}: {qid}/{sid}")
            result.setdefault(qid, {})[sid] = row["relevance"]
    return _labels(result)


def load_rankings(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "rankings" not in payload:
        raise ValueError("Ranking JSON must contain a 'rankings' mapping")
    return _string_keys(payload["rankings"], "rankings")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", required=True, help="Official query_id/skill_id/relevance JSONL")
    parser.add_argument("--rankings", required=True, help="Baseline JSON containing a rankings mapping")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare", help="Treatment JSON; paired bootstrap reports treatment minus baseline")
    parser.add_argument("--query-ids", help="Optional JSON list explicitly defining an evaluation subset")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args(argv)
    labels = load_qrels(args.qrels)
    baseline_rankings = load_rankings(args.rankings)
    treatment_rankings = load_rankings(args.compare) if args.compare else None
    if args.query_ids:
        ids = _ranking(json.loads(Path(args.query_ids).read_text(encoding="utf-8")), "query_ids")
        unknown = set(ids) - set(labels)
        if unknown:
            raise ValueError(f"Subset contains unknown query IDs: {sorted(unknown)[:5]}")
        labels = {qid: labels[qid] for qid in ids}
        baseline_rankings = {qid: baseline_rankings[qid] for qid in ids if qid in baseline_rankings}
        if treatment_rankings is not None:
            treatment_rankings = {qid: treatment_rankings[qid] for qid in ids if qid in treatment_rankings}
    output = evaluate(labels, baseline_rankings)
    if treatment_rankings is not None:
        treatment = evaluate(labels, treatment_rankings)
        output["comparison"] = {"treatment": treatment,
                                "paired_bootstrap": paired_bootstrap(output, treatment,
                                    n_resamples=args.bootstrap_resamples, seed=args.seed)}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
