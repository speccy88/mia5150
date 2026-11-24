"""Reconstruct paired rankings and evaluation tables from frozen local evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import gzip
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np

HERE = Path(__file__).resolve().parent
CANDIDATE_COUNT = 20
TOP_COUNT = 5
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000
METRICS = (
    *(f"recall_at_{k}" for k in (1, 3, 5, 10, 20)),
    *(f"precision_at_{k}" for k in (1, 3, 5)),
    *(f"ndcg_at_{k}" for k in (5, 10)),
    *(f"mrr_at_{k}" for k in (5, 10)),
    *(f"map_at_{k}" for k in (5, 10)),
    "completeness_at_5",
)
IDENTIFIERS = ("case_id", "query_id", "catalog_size", "seed", "method")
DIAGNOSTICS = (
    "recall_improvement",
    "relative_recall_improvement",
    "improved_queries",
    "worsened_queries",
    "unchanged_queries",
    "relevant_promoted",
    "relevant_removed",
)
TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens")
LABELS = {"bm25": "BM25", "vector": "Vector", "hybrid": "Hybrid"}
METRIC_LABELS = {
    "recall": "Recall", "precision": "Precision", "ndcg": "NDCG",
    "mrr": "MRR", "map": "MAP", "completeness": "Completeness",
}
PLOT_COLUMNS = (
    "catalog_size", "method", "n_queries", "n_seeds", "recall_at_5", "recall_at_5_seed_std",
    "reranked_recall_at_5", "reranked_recall_at_5_seed_std", "recall_at_20", "recall_improvement",
    "improvement_ci_lower", "improvement_ci_upper", "ndcg_at_5", "reranked_ndcg_at_5",
    "mrr_at_5", "reranked_mrr_at_5", "latency_ms", "api_latency_ms", "api_cost_usd",
    "n_valid_cases", "n_failed_cases",
)


def load_reference(experiment: str, filename: str):
    source = HERE.parent / experiment / filename
    if not source.is_file():
        source = HERE / "references" / experiment / filename
    spec = importlib.util.spec_from_file_location(f"skillret_reference_{experiment}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference_retrieval = load_reference("skillret-reranker-study", "reference_metrics.py").retrieval_metrics
reference_fusion = load_reference("skillret-embedding-jev-fusion", "evaluate.py").evaluate


def metrics(ranking: list[str], gold: set[str]) -> dict[str, float]:
    """Use the previous SkillRet controls, with the full binary qrels denominator."""
    fusion = reference_fusion({"q": dict.fromkeys(gold, 1)}, {"q": ranking})["per_query"]["q"]
    values = {"recall_at_20": fusion["Coverage@20"]}
    for cutoff in (1, 3):
        scores = reference_retrieval(ranking, gold, cutoff)
        for metric in ("recall", "precision"):
            name = f"{metric}_at_{cutoff}"
            values[name] = scores[name]
    for metric in ("recall", "ndcg", "mrr", "map"):
        for cutoff in (5, 10):
            values[f"{metric}_at_{cutoff}"] = fusion[f"{METRIC_LABELS[metric]}@{cutoff}"]
    values["precision_at_5"] = fusion["Precision@5"]
    values["completeness_at_5"] = fusion["Completeness@5"]
    return values


def read_jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_jsonl(path: Path, rows):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode())
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...] = ()):
    temporary = path.with_name(f".{path.name}.tmp")
    fields = dict.fromkeys(columns)
    for row in rows:
        fields.update(dict.fromkeys(row))
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_attempts(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def index_outcomes(path: Path) -> dict[str, dict]:
    outcomes = {}
    for record in read_jsonl(path) if path.is_file() else ():
        case_id = record["case_id"]
        if case_id in outcomes:
            raise ValueError(f"Duplicate Decisions outcome for {case_id}")
        outcomes[case_id] = record
    return outcomes


def finite_probability(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def make_rankings(pool: dict, outcome: dict) -> dict:
    candidates = sorted(pool["candidates"], key=lambda item: item["rank"])
    baseline = [candidate["id"] for candidate in candidates]
    if len(set(baseline)) != CANDIDATE_COUNT or len(baseline) != CANDIDATE_COUNT:
        raise ValueError(f"Expected {CANDIDATE_COUNT} distinct candidates for {pool['case_id']}")
    if [item["rank"] for item in candidates] != list(range(1, CANDIDATE_COUNT + 1)):
        raise ValueError(f"Invalid original ranks for {pool['case_id']}")
    status = outcome.get("status", "unattempted")
    if status not in {"success", "failed", "unattempted"}:
        raise ValueError(f"Unexpected Decisions outcome {status!r}")
    scores = outcome.get("probabilities")
    reranked = None
    if status == "success":
        if not isinstance(scores, dict) or set(scores) != set(baseline) or not all(
            finite_probability(value) for value in scores.values()
        ):
            raise ValueError(f"Invalid successful Decisions probabilities for {pool['case_id']}")
        reranked = sorted(baseline, key=lambda skill_id: -scores[skill_id])
    elif scores is not None:
        raise ValueError(f"Unsuccessful case has probabilities for {pool['case_id']}")
    operational = baseline if status == "failed" else reranked
    return {
        **{key: pool[key] for key in (*IDENTIFIERS, "query", "catalog_ref")},
        "original_top5": baseline[:TOP_COUNT],
        "original_top20": baseline,
        "reranked_top5": reranked[:TOP_COUNT] if reranked is not None else None,
        "reranked_top20": reranked,
        "operational_top5": operational[:TOP_COUNT] if operational is not None else None,
        "operational_top20": operational,
        "status": status,
        "request_hash": outcome.get("request_hash") or pool.get("request_hash"),
        "response_reference": outcome.get("response_reference"),
        "response_model": outcome.get("response_model"),
        "request_id": outcome.get("request_id"),
        "http_status": outcome.get("http_status"),
        "cache_status": outcome.get("cache_status", "none"),
        "failure": outcome.get("failure"),
    }


def score_case(ranking: dict, gold: set[str]) -> tuple[dict, dict]:
    row = {key: ranking[key] for key in (*IDENTIFIERS, "status")}
    baseline = metrics(ranking["original_top20"], gold)
    row.update(baseline)
    row["candidate_recall_ceiling"] = baseline["recall_at_20"]
    candidate_gold = gold.intersection(ranking["original_top20"])
    row["oracle_recall_at_5"] = min(TOP_COUNT, len(candidate_gold)) / len(gold)
    comparison = {
        **{key: row[key] for key in (*IDENTIFIERS, "status")},
        "candidate_recall_ceiling": row["candidate_recall_ceiling"],
        "oracle_recall_at_5": row["oracle_recall_at_5"],
        "recall_at_5": baseline["recall_at_5"],
    }
    for ranking_key, metric_prefix in (("reranked_top20", "reranked_"), ("operational_top20", "operational_")):
        if ranking[ranking_key] is not None:
            if ranking_key == "operational_top20":
                variant = baseline if ranking["status"] == "failed" else {
                    name: row[f"reranked_{name}"] for name in METRICS
                }
            else:
                variant = metrics(ranking[ranking_key], gold)
            row.update({f"{metric_prefix}{name}": value for name, value in variant.items()})
            comparison[f"{metric_prefix}recall_at_5"] = variant["recall_at_5"]
    for ranking_key, prefix in (("reranked_top5", "valid_only_"), ("operational_top5", "operational_")):
        top5 = ranking[ranking_key]
        if top5 is None:
            continue
        before = gold.intersection(ranking["original_top5"])
        after = gold.intersection(top5)
        delta = Fraction(len(after) - len(before), len(gold))
        comparison.update({
            f"{prefix}recall_improvement": float(delta),
            f"{prefix}relative_recall_improvement": float(delta / Fraction(len(before), len(gold))) if before else None,
            f"{prefix}improved_queries": int(delta > 0),
            f"{prefix}worsened_queries": int(delta < 0),
            f"{prefix}unchanged_queries": int(delta == 0),
            f"{prefix}relevant_promoted": len(after - before),
            f"{prefix}relevant_removed": len(before - after),
            f"{prefix}promoted_ids": json.dumps(sorted(after - before), ensure_ascii=False),
            f"{prefix}removed_ids": json.dumps(sorted(before - after), ensure_ascii=False),
        })
    row.update({key: value for key, value in comparison.items() if not key.endswith("_ids")})
    return row, comparison


def mean_of(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def paired_bootstrap(rows: list[dict], field: str) -> dict:
    """Resample query clusters and preserve all seed pairs within each draw."""
    by_query = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            by_query[row["query_id"]].append(row[field])
    clusters = [by_query[query] for query in sorted(by_query)]
    n = len(clusters)
    result = {"n_bootstrap_queries": n, "improvement_ci_lower": None, "improvement_ci_upper": None}
    if not clusters:
        return result

    sums = np.array([math.fsum(cluster) for cluster in clusters], dtype=np.float64)
    counts = np.array([len(cluster) for cluster in clusters], dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    distribution = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    interval = np.quantile(distribution, (0.025, 0.975))
    result.update(improvement_ci_lower=float(interval[0]), improvement_ci_upper=float(interval[1]))
    return result


def number(value, cast=float):
    return None if value in (None, "") else cast(value)


def costs(attempts: list[dict]) -> dict:
    """Known charges and unresolved reservations are distinct, additive amounts."""
    known_charges = [float(row["estimated_cost_usd"]) for row in attempts
                     if row.get("estimated_cost_usd") not in (None, "")]
    tokens = {
        token: sum(int(row[token]) for row in attempts if row.get(token) not in (None, ""))
        for token in TOKEN_FIELDS
    }
    return {
        "api_http_attempts": sum(row.get("status") != "reserved" for row in attempts),
        "n_api_unconfirmed_reservations": sum(row.get("status") == "reserved" for row in attempts),
        "n_api_http_responses": sum(row.get("http_status") not in (None, "") for row in attempts),
        "n_api_failed_attempts": sum(row.get("status") == "failed" for row in attempts),
        "n_provider_cache_hits": sum((number(row.get("cached_input_tokens"), int) or 0) > 0 for row in attempts),
        **tokens,
        "cache_write_tokens": tokens["cache_write_input_tokens"],
        "api_cost_usd": math.fsum(known_charges),
        "api_unknown_billing_reserved_usd": math.fsum(
            float(row["reserved_cost_usd"]) for row in attempts
            if row.get("estimated_cost_usd") in (None, "")
            and row.get("reserved_cost_usd") not in (None, "")
        ),
        "n_api_unknown_billing_attempts": len(attempts) - len(known_charges),
    }


def latency_statistics(rows: list[dict], attempts: list[dict]) -> dict:
    result = {}
    for series, records, field in (
        ("retrieval", rows, "retrieval_latency_ms"),
        ("api", attempts, "api_latency_ms"),
        ("end_to_end", rows, "end_to_end_latency_ms"),
        ("local_cache", [row for row in rows if row.get("cache_status") == "local_cache_hit"], "rerank_wall_ms"),
    ):
        data = [float(row[field]) for row in records if row.get(field) not in (None, "")]
        result[f"{series}_latency_median_ms"] = float(np.median(data)) if data else None
        result[f"{series}_latency_p95_ms"] = float(np.quantile(data, 0.95)) if data else None
        result[f"n_{series}_latency_samples"] = len(data)
    return result


def summarize(identity: dict, rows: list[dict], attempts: list[dict]) -> tuple[dict, list[dict]]:
    valid = [row for row in rows if row["status"] == "success"]
    failed = [row for row in rows if row["status"] == "failed"]
    attempted = [row for row in rows if row["status"] != "unattempted"]
    complete = len(attempted) == len(rows)
    usage = costs(attempts)
    request_hashes = [row.get("request_hash") for row in rows]
    summary = {
        **identity,
        "n_queries": len({row["query_id"] for row in rows}),
        "n_seeds": len({row["seed"] for row in rows}),
        "n_cases": len(rows),
        "n_valid_cases": len(valid),
        "n_valid_queries": len({row["query_id"] for row in valid}),
        "n_valid_seeds": len({row["seed"] for row in valid}),
        "n_failed_cases": len(failed),
        "n_unattempted_cases": len(rows) - len(attempted),
        "n_attempted_cases": len(attempted),
        "complete": complete,
        "n_unique_api_requests": len(set(request_hashes)) if all(request_hashes) else None,
        "n_unique_api_requests_attempted_by_group": len({row["request_hash"] for row in attempts}),
        "n_local_cache_hits": sum(row.get("cache_status") == "local_cache_hit" for row in rows),
        **usage,
        "api_cost_per_query_usd": usage["api_cost_usd"] / len(attempted) if attempted else None,
        **latency_statistics(rows, attempts),
        "candidate_recall_ceiling": mean_of(rows, "candidate_recall_ceiling"),
        "oracle_recall_at_5": mean_of(rows, "oracle_recall_at_5"),
    }
    for metric in METRICS:
        summary[metric] = mean_of(rows, metric)
        summary[f"valid_only_original_{metric}"] = mean_of(valid, metric)
        summary[f"valid_only_reranked_{metric}"] = mean_of(valid, f"reranked_{metric}")
        summary[f"attempted_original_{metric}"] = mean_of(attempted, metric)
        summary[f"attempted_operational_{metric}"] = mean_of(attempted, f"operational_{metric}")
        summary[f"operational_reranked_{metric}"] = summary[f"attempted_operational_{metric}"] if complete else None
        summary[f"reranked_{metric}"] = summary[f"operational_reranked_{metric}"]

    intervals = []
    for population, sample, delta_field in (
        ("valid_only", valid, "valid_only_recall_improvement"),
        ("operational", attempted if complete else [], "operational_recall_improvement"),
        ("attempted", attempted, "operational_recall_improvement"),
    ):
        delta = mean_of(sample, delta_field)
        baseline = mean_of(sample, "recall_at_5")
        summary[f"{population}_recall_improvement"] = delta
        summary[f"{population}_relative_recall_improvement"] = delta / baseline if baseline else None
        for diagnostic in DIAGNOSTICS[2:]:
            prefix = "valid_only" if population == "valid_only" else "operational"
            summary[f"{population}_{diagnostic}"] = sum(row[f"{prefix}_{diagnostic}"] for row in sample) if sample else None
        interval = paired_bootstrap(sample, delta_field)
        for column in ("improvement_ci_lower", "improvement_ci_upper"):
            summary[f"{population}_{column}"] = interval[column]
        intervals.append({
            **identity, "population": population, "metric": "Recall@5",
            "n_cases": len(sample), "n_queries": interval["n_bootstrap_queries"],
            "resamples": BOOTSTRAP_RESAMPLES, "seed_rng": BOOTSTRAP_SEED,
            "recall_improvement": delta,
            "improvement_ci_lower": interval["improvement_ci_lower"],
            "improvement_ci_upper": interval["improvement_ci_upper"],
        })

    summary["recall_improvement"] = summary["operational_recall_improvement"]
    summary["relative_recall_improvement"] = summary["operational_relative_recall_improvement"]
    summary["improvement_ci_lower"] = summary["operational_improvement_ci_lower"]
    summary["improvement_ci_upper"] = summary["operational_improvement_ci_upper"]
    summary["latency_ms"] = summary["retrieval_latency_median_ms"]
    summary["api_latency_ms"] = summary["api_latency_median_ms"]
    return summary, intervals


def summarize_all(rows: list[dict], attempts: list[dict]):
    seed_groups = defaultdict(list)
    size_groups = defaultdict(list)
    for row in rows:
        seed_groups[(row["catalog_size"], row["method"], row["seed"])].append(row)
        size_groups[(row["catalog_size"], row["method"])].append(row)
    attempts_by_seed = defaultdict(list)
    attempts_by_size = defaultdict(list)
    case_index = {row["case_id"]: row for row in rows}
    for attempt in attempts:
        case = case_index.get(attempt.get("case_id"))
        if case is None:
            if attempt.get("kind") != "synthetic":
                raise ValueError(f"HTTP attempt has no logical owner: {attempt.get('attempt_id')}")
            continue
        size_key = (case["catalog_size"], case["method"])
        attempts_by_seed[(*size_key, case["seed"])].append(attempt)
        attempts_by_size[size_key].append(attempt)

    seed_results, size_results, intervals = [], [], []
    seeds_by_size = defaultdict(list)
    for key, cases in sorted(seed_groups.items()):
        size, method, seed = key
        summary, bootstrap = summarize({"catalog_size": size, "method": method, "seed": seed},
                                       cases, attempts_by_seed[key])
        seed_results.append(summary)
        seeds_by_size[(size, method)].append(summary)
        intervals.extend(bootstrap)
    for key, cases in sorted(size_groups.items()):
        size, method = key
        summary, bootstrap = summarize({"catalog_size": size, "method": method}, cases, attempts_by_size[key])
        seed_rows = seeds_by_size[key]
        fields = (
            *(f"{prefix}{metric}" for metric in METRICS
              for prefix in ("", "reranked_", "valid_only_reranked_")),
            *(f"{prefix}recall_improvement" for prefix in ("", "valid_only_", "operational_")),
        )
        for field in fields:
            values = [row[field] for row in seed_rows]
            summary[f"{field}_seed_std"] = stdev(values) if len(values) > 1 and None not in values else None
        size_results.append(summary)
        intervals.extend(bootstrap)
    return seed_results, size_results, intervals


def plot_tables(summaries: list[dict]) -> tuple[list[dict], list[dict]]:
    long = []
    for row in summaries:
        label = LABELS.get(row["method"], row["method"])
        for metric in METRICS:
            metric_name, _, cutoff = metric.partition("_at_")
            name = f"{METRIC_LABELS[metric_name]}@{cutoff}"
            for series, field, error, population in (
                (label, metric, f"{metric}_seed_std", "retrieval"),
                (f"{label} + OpenAI Decisions", f"reranked_{metric}", f"reranked_{metric}_seed_std", "operational"),
                (f"{label} + OpenAI Decisions (valid only)", f"valid_only_reranked_{metric}",
                 f"valid_only_reranked_{metric}_seed_std", "valid_only"),
            ):
                if row[field] is not None:
                    long.append({
                        "series": series, "catalog_size": row["catalog_size"], "method": row["method"],
                        "metric": name, "value": row[field], "ci_lower": None, "ci_upper": None,
                        "seed_std": row[error], "population": population,
                        **plot_population(row, population),
                    })
        for population in ("operational", "valid_only"):
            field = f"{population}_recall_improvement"
            if row[field] is not None:
                long.append({
                    "series": f"{label} + OpenAI Decisions", "catalog_size": row["catalog_size"],
                    "method": row["method"], "metric": "Recall@5 absolute improvement", "value": row[field],
                    "ci_lower": row[f"{population}_improvement_ci_lower"],
                    "ci_upper": row[f"{population}_improvement_ci_upper"],
                    "seed_std": row[f"{field}_seed_std"], "population": population,
                    **plot_population(row, population),
                })
    return summaries, long


def plot_population(row: dict, population: str) -> dict:
    if population == "valid_only":
        return {
            "complete": row["n_valid_cases"] == row["n_cases"],
            "n_queries": row["n_valid_queries"], "n_seeds": row["n_valid_seeds"],
            "n_logical_cases": row["n_valid_cases"],
        }
    return {
        "complete": population == "retrieval" or row["complete"],
        "n_queries": row["n_queries"], "n_seeds": row["n_seeds"], "n_logical_cases": row["n_cases"],
    }


def write_candidate_scores(path: Path, rows: list[dict]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("case_id", pa.string()), ("request_hash", pa.string()), ("skill_id", pa.string()),
        ("original_rank", pa.int16()), ("original_score", pa.float64()),
        ("probability", pa.float64()), ("reranked_rank", pa.int16()),
        ("operational_rank", pa.int16()), ("status", pa.string()),
    ])
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    temporary.replace(path)


def evaluate_results(results: Path, partial_retrieval_pools: Path | None = None) -> dict:
    qrels = json.loads((results / "qrels.json").read_text(encoding="utf-8"))
    outcomes = index_outcomes(results / "decision_outcomes.jsonl.gz")
    attempts = read_attempts(results / "api_usage.csv")
    pool_path = partial_retrieval_pools or results / "candidate_pools.jsonl.gz"
    rankings, per_query, comparisons, latency, scores = [], [], [], [], []
    seen_cases = set()
    for pool in read_jsonl(pool_path):
        case_id = pool["case_id"]
        if case_id in seen_cases:
            raise ValueError(f"Duplicate frozen case {case_id}")
        seen_cases.add(case_id)
        outcome = outcomes.get(case_id, {})
        ranking = make_rankings(pool, outcome)
        rankings.append(ranking)
        row, comparison = score_case(ranking, set(qrels[pool["query_id"]]))
        runtime = {
            **{key: ranking[key] for key in (*IDENTIFIERS, "status", "cache_status", "request_hash")},
            "response_reference": outcome.get("response_reference"),
            "retrieval_latency_ms": pool["retrieval_latency_ms"],
            "api_latency_ms": outcome.get("api_latency_ms") if outcome.get("cache_status") == "fresh" else None,
            "source_api_latency_ms": outcome.get("source_api_latency_ms"),
            "rerank_wall_ms": outcome.get("rerank_wall_ms"),
            "end_to_end_latency_ms": outcome.get("end_to_end_ms"),
        }
        per_query.append({**row, **runtime})
        comparisons.append(comparison)
        latency.append(runtime)
        positions = {skill_id: i for i, skill_id in enumerate(ranking["reranked_top20"] or (), start=1)}
        operational = {skill_id: i for i, skill_id in enumerate(ranking["operational_top20"] or (), start=1)}
        probabilities = outcome.get("probabilities") or {}
        for candidate in pool["candidates"]:
            skill_id = candidate["id"]
            scores.append({
                "case_id": case_id, "request_hash": ranking["request_hash"], "skill_id": skill_id,
                "original_rank": candidate["rank"], "original_score": candidate["score"],
                "probability": probabilities.get(skill_id), "reranked_rank": positions.get(skill_id),
                "operational_rank": operational.get(skill_id), "status": ranking["status"],
            })
    if outcomes.keys() - seen_cases:
        raise ValueError("Decisions outcomes contain cases outside the frozen manifest")
    if not rankings:
        raise ValueError("Frozen candidate pools are empty")
    seeds, summaries, intervals = summarize_all(per_query, attempts)
    wide, long = plot_tables(summaries)
    write_candidate_scores(results / "candidate_scores.parquet", scores)
    write_jsonl(results / "rankings.jsonl.gz", rankings)
    paired_diagnostics = tuple(f"{prefix}{field}" for prefix in ("valid_only_", "operational_")
                               for field in DIAGNOSTICS)
    write_csv(results / "per_query_metrics.csv", per_query, (
        *IDENTIFIERS, "status", *METRICS, *(f"reranked_{m}" for m in METRICS),
        *(f"operational_{m}" for m in METRICS), *paired_diagnostics,
    ))
    write_csv(results / "summary_by_seed.csv", seeds)
    write_csv(results / "summary_by_catalog_size.csv", summaries)
    write_csv(results / "plot_data.csv", wide, PLOT_COLUMNS)
    write_csv(results / "plot_data_long.csv", long)
    write_csv(results / "paired_comparisons.csv", comparisons, (
        *IDENTIFIERS, "status", "candidate_recall_ceiling", "oracle_recall_at_5", "recall_at_5",
        "reranked_recall_at_5", "operational_recall_at_5", *paired_diagnostics,
        "valid_only_promoted_ids", "valid_only_removed_ids",
        "operational_promoted_ids", "operational_removed_ids",
    ))
    write_csv(results / "bootstrap_intervals.csv", intervals)
    write_csv(results / "latency.csv", latency)
    n_valid = sum(row["n_valid_cases"] for row in summaries)
    n_failed = sum(row["n_failed_cases"] for row in summaries)
    n_unattempted = sum(row["n_unattempted_cases"] for row in summaries)
    decisions_state = "complete" if not n_unattempted else ("partial" if n_valid or n_failed else "baseline_only")
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    planned_cases = len(qrels) * math.prod(len(config[key]) for key in ("catalog_sizes", "seeds", "methods"))
    status = {
        "state": "partial_retrieval" if partial_retrieval_pools is not None else decisions_state,
        "decisions_state": decisions_state,
        "retrieval_input": pool_path.name,
        "retrieval_complete": partial_retrieval_pools is None,
        "n_planned_cases": planned_cases,
        "n_omitted_retrieval_cases": planned_cases - len(rankings) if partial_retrieval_pools is not None else 0,
        "n_cases": len(rankings), "n_valid_cases": n_valid, "n_failed_cases": n_failed,
        "n_unattempted_cases": n_unattempted, "total_usage": costs(attempts),
        "synthetic_usage": costs([row for row in attempts if row.get("kind") == "synthetic"]),
        "operational_failure_policy": "retrieval ranking fallback after a failed attempt",
        "unattempted_policy": "missing reranked and operational value; no full-group operational estimate",
    }
    temporary = results / ".evaluation_status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(results / "evaluation_status.json")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--partial-retrieval-pools", type=Path,
                        help="Explicitly evaluate an incomplete retrieval export without publishing it as the full pilot.")
    args = parser.parse_args()
    print(json.dumps(evaluate_results(args.results, args.partial_retrieval_pools), indent=2))


if __name__ == "__main__":
    main()
