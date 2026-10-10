#!/usr/bin/env python3
"""Verify the retained CPU benchmark without inference, downloads, or source imports.

Only the Python standard library is used. This reads the recorded ID-only inputs,
rankings, aggregate CSVs, timings, and an optional project SHA256SUMS manifest.
It establishes arithmetic and coverage consistency, not inference reproducibility.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys


METRICS = (
    "mrr5", "recall1", "recall5", "hit1", "hit5", "ndcg5", "complete5",
    "candidate_recall", "candidate_hit",
)
SIZES = (10, 100, 1000, 6006)
NATIVE = ("julia", "gliner", "verdict", "laya")
EMBEDDERS = ("bge-small", "e5-small", "embeddinggemma-int8")
CONDITIONS = (("bm25", "bm25"),) + tuple(
    itertools.product(EMBEDDERS, ("vector", "hybrid", "domain_vector"))
) + tuple((model, "hybrid_rerank") for model in NATIVE)
LATENCY_FIELDS = ("p50_ms", "p95_ms", "mean_ms", "serial_qps")


class ValidationError(ValueError):
    """Recorded artifacts violate the benchmark's fixed retained scope."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def number(value: str, context: str, *, nonnegative: bool = True) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{context}: expected a number") from error
    require(math.isfinite(result), f"{context}: non-finite number")
    require(not nonnegative or result >= 0, f"{context}: negative number")
    return result


def integer(value: str, context: str) -> int:
    require(isinstance(value, str) and re.fullmatch(r"\d+", value) is not None,
            f"{context}: expected a nonnegative integer")
    return int(value)


def matching_number(value: str, expected: float, context: str) -> None:
    actual = number(value, context)
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{context}: recorded value {actual} differs from recomputed {expected}")


def id_list(value: object, context: str, *, length: int | None = None) -> list[str]:
    require(isinstance(value, list), f"{context}: expected an ID list")
    require(all(isinstance(item, str) and bool(item) for item in value),
            f"{context}: expected nonempty string IDs")
    require(len(value) == len(set(value)), f"{context}: duplicate IDs")
    if length is not None:
        require(len(value) == length, f"{context}: expected {length} IDs")
    return value


def safe_path(root: Path, relative: str) -> Path:
    """Keep every read inside the explicitly selected project (including symlinks)."""
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts,
            f"unsafe relative artifact path: {relative}")
    require(not any(part == ".env" or part.startswith(".env.") for part in path.parts),
            f"refusing environment-secret artifact: {relative}")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root), f"artifact escapes project: {relative}")
    require(not any(part == ".env" or part.startswith(".env.") for part in resolved.parts),
            f"refusing environment-secret artifact: {relative}")
    require(resolved.is_file(), f"missing artifact: {relative}")
    return resolved


def load_json(root: Path, relative: str) -> object:
    try:
        return json.loads(safe_path(root, relative).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValidationError(f"{relative}: invalid JSON") from error


def load_jsonl(root: Path, relative: str) -> list[dict]:
    result = []
    with safe_path(root, relative).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValidationError(f"{relative}:{line_number}: invalid JSON") from error
            require(isinstance(row, dict), f"{relative}:{line_number}: expected an object")
            result.append(row)
    return result


def load_csv(root: Path, relative: str, required: set[str]) -> tuple[list[str], list[dict[str, str]]]:
    with safe_path(root, relative).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        require(fields is not None and len(fields) == len(set(fields)),
                f"{relative}: absent or duplicate column headers")
        require(required.issubset(fields), f"{relative}: missing required columns")
        rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values())
                for row in rows), f"{relative}: malformed CSV row")
    return fields, rows


def recompute_metrics(ranked: list[str], gold: set[str]) -> dict[str, float]:
    require(bool(gold), "cannot evaluate empty gold labels")
    require(len(ranked) == len(set(ranked)), "cannot evaluate duplicate ranked IDs")
    relevant = [int(skill in gold) for skill in ranked[:5]]
    first = next((index + 1 for index, hit in enumerate(relevant) if hit), None)
    found = sum(relevant)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(5, len(gold))))
    candidates_found = len(set(ranked) & gold)
    return {
        "mrr5": 1 / first if first else 0.0,
        "recall1": (relevant[0] if relevant else 0) / len(gold),
        "recall5": found / len(gold),
        "hit1": float(bool(relevant and relevant[0])),
        "hit5": float(bool(found)),
        "ndcg5": sum(hit / math.log2(index + 2)
                     for index, hit in enumerate(relevant)) / ideal,
        "complete5": float(found == len(gold)),
        "candidate_recall": candidates_found / len(gold),
        "candidate_hit": float(bool(candidates_found)),
    }


def percentile(values: list[float], percent: float) -> float:
    """NumPy's default linear percentile convention, implemented independently."""
    require(bool(values), "cannot take percentile of empty timings")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def verify_manifest(root: Path) -> int:
    manifest = root / "SHA256SUMS"
    if not manifest.exists():
        return 0
    entries: set[str] = set()
    lines = safe_path(root, "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        require(match is not None, f"SHA256SUMS:{index}: malformed checksum entry")
        digest, relative = match.groups()
        require(relative not in entries, f"SHA256SUMS:{index}: duplicate artifact")
        entries.add(relative)
        path = safe_path(root, relative)
        require(path != manifest.resolve(), "SHA256SUMS: cannot hash itself")
        actual = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                actual.update(block)
        require(actual.hexdigest() == digest.lower(), f"SHA256SUMS: digest mismatch: {relative}")
    require(bool(entries), "SHA256SUMS: empty manifest")
    return len(entries)


def condition_key(row: dict[str, str], context: str) -> tuple[str, str, int]:
    model, method = row["model"], row["method"]
    size = integer(row["catalog_size"], f"{context} catalog_size")
    require((model, method) in CONDITIONS and size in SIZES,
            f"{context}: unexpected model/method/catalog condition")
    return model, method, size


def verify(root: Path | str) -> dict[str, int]:
    root = Path(root).resolve()
    checksum_count = verify_manifest(root)
    base = "reference_run/"
    query_ids = id_list(load_json(root, base + "pilot32_query_ids.json"), "pilot32", length=32)
    query_set = set(query_ids)
    catalog = id_list(load_json(root, base + "full_catalog.json"), "full catalog", length=6006)
    catalog_set = set(catalog)
    require(catalog == sorted(catalog), "full catalog: IDs must be sorted")

    gold: dict[str, set[str]] = {query: set() for query in query_ids}
    qrels = load_jsonl(root, base + "qrels32.jsonl")
    for row in qrels:
        require(set(row) == {"query_id", "skill_id", "relevance"}, "qrels32: unexpected fields")
        query, skill = row["query_id"], row["skill_id"]
        require(isinstance(query, str) and query in query_set, "qrels32: unknown query ID")
        require(isinstance(skill, str) and skill in catalog_set, "qrels32: unknown skill ID")
        require(type(row["relevance"]) is int and row["relevance"] == 1,
                "qrels32: expected binary positive relevance")
        require(skill not in gold[query], "qrels32: duplicate query/skill pair")
        gold[query].add(skill)
    require(all(gold.values()), "qrels32: every matched query must have nonempty gold labels")

    pools: dict[str, list[str]] = {}
    for row in load_jsonl(root, base + "candidate_pools32.jsonl"):
        require(set(row) == {"query_id", "candidate_skill_ids"}, "candidate pools: unexpected fields")
        query = row["query_id"]
        require(isinstance(query, str) and query in query_set, "candidate pools: unknown query ID")
        require(query not in pools, "candidate pools: duplicate query ID")
        pool = id_list(row["candidate_skill_ids"], "candidate pool", length=1000)
        require(set(pool) <= catalog_set, "candidate pool: unknown skill ID")
        require(gold[query] <= set(pool[:10]), "candidate pool: sampled prefixes do not preserve all gold")
        pools[query] = pool
    require(set(pools) == query_set, "candidate pools: incomplete matched32 coverage")

    expected_grid = set(itertools.product(CONDITIONS, SIZES, query_ids))
    rankings: dict[tuple[str, str, int, str], list[str]] = {}
    metrics: dict[tuple[str, str, int], list[dict[str, float]]] = {}
    _, per_query = load_csv(root, base + "report/SkillRet_PerQuery_Matched32.csv",
                            {"model", "method", "catalog_size", "query_id", "gold_count",
                             "ranked_skill_ids", *METRICS})
    seen_grid = set()
    for index, row in enumerate(per_query, 2):
        context = f"per-query CSV row {index}"
        model, method, size = condition_key(row, context)
        query = row["query_id"]
        require(query in query_set, f"{context}: unknown query ID")
        grid_key = ((model, method), size, query)
        require(grid_key not in seen_grid, f"{context}: duplicate condition/query row")
        seen_grid.add(grid_key)
        require(integer(row["gold_count"], context + " gold_count") == len(gold[query]),
                f"{context}: gold count mismatch")
        try:
            ranked = id_list(json.loads(row["ranked_skill_ids"]), context + " ranking")
        except json.JSONDecodeError as error:
            raise ValidationError(f"{context}: invalid ranking JSON") from error
        max_depth = min(size, 20)
        require(1 <= len(ranked) <= max_depth, f"{context}: invalid ranking depth")
        require(method == "domain_vector" or len(ranked) == max_depth,
                f"{context}: incomplete ranking")
        allowed = catalog_set if size == 6006 else set(pools[query][:size])
        require(set(ranked) <= allowed, f"{context}: ranked ID outside correct catalog pool")
        calculated = recompute_metrics(ranked, gold[query])
        for metric in METRICS:
            matching_number(row[metric], calculated[metric], context + " " + metric)
        rankings[(model, method, size, query)] = ranked
        metrics.setdefault((model, method, size), []).append(calculated)
    require(seen_grid == expected_grid, "per-query CSV: incomplete exact 32 x 14 x 4 grid")
    for model in NATIVE:
        for size, query in itertools.product(SIZES, query_ids):
            require(set(rankings[(model, "hybrid_rerank", size, query)])
                    == set(rankings[("bge-small", "hybrid", size, query)]),
                    f"{model}/{size}/{query}: native shortlist differs from BGE hybrid")

    required_summary = {"model", "method", "catalog_size", "quality_queries", *METRICS,
                        "latency_queries", "latency_repeats", *LATENCY_FIELDS}
    summary_fields, summary = load_csv(root, base + "report/SkillRet_Benchmark_Matched32.csv",
                                      required_summary)
    expected_conditions = {(model, method, size) for model, method in CONDITIONS for size in SIZES}
    summaries: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in summary:
        key = condition_key(row, "summary CSV")
        require(key not in summaries, "summary CSV: duplicate condition")
        require(integer(row["quality_queries"], "quality_queries") == 32,
                "summary CSV: quality denominator must equal 32")
        for metric in METRICS:
            matching_number(row[metric], statistics.fmean(item[metric] for item in metrics[key]),
                            f"summary {key} {metric}")
        summaries[key] = row
    require(set(summaries) == expected_conditions, "summary CSV: incomplete exact 14 x 4 grid")
    full_fields, full_summary = load_csv(
        root, base + "report/SkillRet_Benchmark_Matched32_FullCatalog.csv", required_summary
    )
    require(full_fields == summary_fields, "full-catalog summary: columns differ from complete summary")
    full_by_key: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in full_summary:
        key = condition_key(row, "full-catalog summary")
        require(key[2] == 6006 and key not in full_by_key,
                "full-catalog summary: unexpected or duplicate condition")
        require(row == summaries[key], "full-catalog summary: row differs from exact summary subset")
        full_by_key[key] = row
    require(set(full_by_key) == {key for key in expected_conditions if key[2] == 6006},
            "full-catalog summary: incomplete 14-condition subset")

    observed_latency: set[tuple[str, str, int, str, int]] = set()
    timings: dict[tuple[str, str, int], list[float]] = {}
    for model in ("bm25", *EMBEDDERS, *NATIVE):
        extra = {"stage1_ms", "rerank_ms", "shortlist_matches_frozen", "fresh_shortlist_ids"}
        required = {"model", "method", "catalog_size", "query_id", "repeat", "total_ms"}
        required |= extra if model in NATIVE else {"query_encode_ms"}
        _, rows = load_csv(root, base + f"timing/{model}_latency.csv", required)
        for index, row in enumerate(rows, 2):
            context = f"{model} latency row {index}"
            key = condition_key(row, context)
            require(key[0] == model and key[2] == 6006, f"{context}: incorrect model or catalog")
            query = row["query_id"]
            repeat = integer(row["repeat"], context + " repeat")
            require(query in query_set and repeat in (0, 1), f"{context}: incorrect query or repeat")
            observation = (*key, query, repeat)
            require(observation not in observed_latency, f"{context}: duplicate latency observation")
            observed_latency.add(observation)
            total = number(row["total_ms"], context + " total_ms")
            require(total > 0, f"{context}: total latency must be positive")
            if model in NATIVE:
                stage1 = number(row["stage1_ms"], context + " stage1_ms")
                rerank = number(row["rerank_ms"], context + " rerank_ms")
                require(stage1 + rerank <= total + 1e-8, f"{context}: stage times exceed total")
                require(row["shortlist_matches_frozen"] == "True",
                        f"{context}: fresh shortlist mismatch flag")
                try:
                    fresh = id_list(json.loads(row["fresh_shortlist_ids"]), context + " fresh shortlist")
                except json.JSONDecodeError as error:
                    raise ValidationError(f"{context}: invalid fresh shortlist JSON") from error
                require(fresh == rankings[("bge-small", "hybrid", 6006, query)],
                        f"{context}: fresh shortlist differs from frozen BGE hybrid ranking")
            else:
                encode = number(row["query_encode_ms"], context + " query_encode_ms")
                require(encode <= total + 1e-8, f"{context}: query encoding exceeds total")
            timings.setdefault(key, []).append(total)
    expected_latency = {(model, method, 6006, query, repeat)
                        for (model, method), query, repeat
                        in itertools.product(CONDITIONS, query_ids, (0, 1))}
    require(observed_latency == expected_latency,
            "latency CSVs: incomplete exact 14 x 32 x 2 full-catalog coverage")
    for key, row in summaries.items():
        full = key[2] == 6006
        require(integer(row["latency_queries"], "latency_queries") == (32 if full else 0),
                f"summary {key}: wrong latency query count")
        require(integer(row["latency_repeats"], "latency_repeats") == (2 if full else 0),
                f"summary {key}: wrong latency repeat count")
        if full:
            values = timings[key]
            mean = statistics.fmean(values)
            calculated = {"p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95),
                          "mean_ms": mean, "serial_qps": 1000 / mean}
            for field, value in calculated.items():
                matching_number(row[field], value, f"summary {key} {field}")
        else:
            require(all(row[field] == "" for field in LATENCY_FIELDS),
                    f"summary {key}: sampled catalogs must have absent latency statistics")
    return {"queries": len(query_ids), "catalog_skills": len(catalog), "qrels": len(qrels),
            "per_query_rows": len(per_query), "summary_rows": len(summary),
            "full_catalog_rows": len(full_summary), "latency_observations": len(observed_latency),
            "checksum_files": checksum_count}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="CPU benchmark project directory (default: this script's directory)")
    args = parser.parse_args(argv)
    try:
        checked = verify(args.root)
    except (ValueError, OSError, csv.Error) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print("Verified recorded CPU evidence: " + ", ".join(
        f"{value} {key.replace('_', ' ')}" for key, value in checked.items()
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
