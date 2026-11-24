"""Recompute SkillRet benchmark evidence offline without importing the benchmark."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parent
SIZES = (50, 100, 250, 500, 1000, 2000, 5000)
SEEDS = (42, 43, 44)
METHODS = ("bm25", "vector", "hybrid")
REVISION = "6583d7d2ed07644d0fb8938ed8178f3a7dc42a12"
REFERENCE = Path("experiments/benchmarks/skillret-reranker-study/reference_run/jev")
PUBLIC_INPUTS = Path("experiments/scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs")
REQUIRED_CASES = 200 * len(SIZES) * len(SEEDS) * len(METHODS)
NUMERIC_TOLERANCE = 1e-11


class Audit:
    def __init__(self):
        self.failures = Counter()
        self.examples = defaultdict(list)
        self.pending = []
        self.counts = {"expected_cases": REQUIRED_CASES}

    def check(self, condition, name, location=None):
        if not condition:
            self.failures[name] += 1
            if location is not None and len(self.examples[name]) < 4:
                self.examples[name].append(str(location))
        return condition

    def require(self, path):
        if path.is_file():
            return True
        self.pending.append(path.name)
        return False

    def report(self, phase):
        status = "failed" if self.failures else "incomplete" if self.pending else "passed"
        return {
            "status": status, "phase": phase, "counts": self.counts,
            "issues": dict(self.failures), "examples": dict(self.examples),
            "pending": sorted(set(self.pending)),
        }


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def jsonl_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def case_key(row):
    return row["query_id"], int(row["catalog_size"]), int(row["seed"]), row["method"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_number(audit, recorded, expected, name, location):
    if expected is None:
        return audit.check(recorded in (None, ""), name, location)
    try:
        value = float(recorded)
    except (TypeError, ValueError):
        return audit.check(False, name, location)
    return audit.check(math.isclose(value, expected, rel_tol=NUMERIC_TOLERANCE,
                                    abs_tol=NUMERIC_TOLERANCE), name, location)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ranking_metrics(ranking, relevant):
    """Compute binary measures from hit positions with no evaluator imports."""
    total = len(relevant)
    positions = [rank for rank, sid in enumerate(ranking, 1) if sid in relevant]
    recall = {f"recall_at_{k}": sum(rank <= k for rank in positions) / total
              for k in (1, 3, 5, 10, 20)}
    precision = {f"precision_at_{k}": sum(rank <= k for rank in positions) / k
                 for k in (1, 3, 5)}
    result = recall | precision | {"completeness_at_5": float(recall["recall_at_5"] == 1.0)}
    for cutoff in (5, 10):
        hits = [rank for rank in positions if rank <= cutoff]
        dcg = math.fsum(1 / math.log2(rank + 1) for rank in hits)
        ideal = math.fsum(1 / math.log2(rank + 1) for rank in range(1, min(cutoff, total) + 1))
        result[f"ndcg_at_{cutoff}"] = dcg / ideal
        result[f"mrr_at_{cutoff}"] = 1 / hits[0] if hits else 0.0
        result[f"map_at_{cutoff}"] = math.fsum(i / rank for i, rank in enumerate(hits, 1)) / total
    return result


def compare_historical(audit, root, manifest, qrels):
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                            capture_output=True, text=True, check=False)
    repository = root.parents[2] if result.returncode else Path(result.stdout.strip())
    commit = manifest["repository_commit"]
    if not audit.check(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40,64}", commit),
                       "invalid historical reference commit"):
        return None, None
    if result.returncode:
        audit.pending.append("Git history comparison requires the repository checkout")
    else:
        history = subprocess.run([
            "git", "-C", str(repository), "diff", "--name-only", commit, "--",
            "experiments/", ":(exclude)experiments/benchmarks/skillret-openai-decisions-scaling/",
            ":(glob)**/benchmark_subset.md",
        ], capture_output=True, text=True, check=False)
        audit.check(history.returncode == 0 and not history.stdout.strip(),
                    "historical tracked files changed or Git reference unavailable")
    archived_panel = read_json(repository / REFERENCE / "manifest-results-200.json")
    audit.check(manifest["query_ids"] == archived_panel["query_ids"], "historical JEV query panel mismatch")
    historical_qrels = defaultdict(set)
    for name, expected in manifest.get("source_sha256", {}).items():
        relative_path = Path(name)
        if relative_path.name in {"qrels-test.jsonl", "queries-test.jsonl", "skills-metadata.jsonl"} and len(relative_path.parts) == 1:
            relative_path = PUBLIC_INPUTS / relative_path
        path = (repository / relative_path).resolve()
        audit.check(path.is_relative_to(repository) and path.is_file() and file_hash(path) == expected,
                    "historical source checksum mismatch", Path(name).name)
    for row in jsonl_rows(repository / PUBLIC_INPUTS / "qrels-test.jsonl"):
        if row["query_id"] in qrels and row["relevance"] > 0:
            historical_qrels[row["query_id"]].add(row["skill_id"])
    audit.check(qrels == historical_qrels, "frozen qrels differ from public positive qrels")
    documents = {
        row["id"]: row["name"].strip() + " | " + row["description"].strip()[:1024]
        for row in jsonl_rows(repository / PUBLIC_INPUTS / "skills-metadata.jsonl")
    }
    queries = {row["id"]: row["query"] for row in
               jsonl_rows(repository / PUBLIC_INPUTS / "queries-test.jsonl") if row["id"] in qrels}
    audit.check(len(documents) == 6006, "public skill corpus count mismatch")
    jev_rows = list(jsonl_rows(repository / REFERENCE / "rows.jsonl"))
    if audit.check(len(jev_rows) == 200 and {row["query_id"] for row in jev_rows} == set(qrels),
                   "historical JEV evidence panel mismatch"):
        reference_recalls = defaultdict(list)
        for row in jev_rows:
            gold = qrels[row["query_id"]]
            original = [sid for sid, score in row["bm25_top20"]]
            reference_recalls["bm25_recall_at_5"].append(len(gold.intersection(original[:5])) / len(gold))
            reference_recalls["bm25_recall_at_20"].append(len(gold.intersection(original)) / len(gold))
            reference_recalls["jev_recall_at_5"].append(len(gold.intersection(row["jev_top5"])) / len(gold))
        summary = next(csv_rows(repository / REFERENCE / "summary-200.csv"))
        for field, values in reference_recalls.items():
            compare_number(audit, summary[field], statistics.mean(values), "historical JEV summary mismatch", field)
        audit.counts["independently_scored_historical_jev_cases"] = len(jev_rows)
    return documents, queries


def catalog_memberships(audit, path, query_ids, qrels, corpus_ids, document_ids):
    expected_keys = set(product(query_ids, SIZES, SEEDS))
    memberships = {}
    ids_by_query_seed = {}
    vocab = {sid: 1 << index for index, sid in enumerate(corpus_ids or [])}
    for row in jsonl_rows(path):
        qid, size, seed = (row[name] for name in ("query_id", "catalog_size", "seed"))
        key = qid, size, seed
        if not audit.check(key in expected_keys and key not in memberships, "invalid or repeated catalog", key):
            continue
        if "document_indices" in row:
            indices = row["document_indices"]
            if not audit.check(document_ids and all(type(i) is int and 0 <= i < len(document_ids) for i in indices),
                               "invalid catalog document-index reference", key):
                continue
            ids = [document_ids[i] for i in indices]
        else:
            ids = row["skill_ids"]
        unique = set(ids)
        relevant = qrels[qid]
        reference = row["catalog_ref"]
        audit.check(len(ids) == len(unique) == size and relevant <= unique,
                    "catalog cardinality or positive qrels mismatch", key)
        audit.check(reference == canonical_hash(ids), "catalog reference hash mismatch", key)
        if corpus_ids is not None:
            query_seed = qid, seed
            if query_seed not in ids_by_query_seed:
                distractors = sorted(set(corpus_ids).difference(relevant))
                random.Random(int(canonical_hash([seed, qid, "membership"]), 16)).shuffle(distractors)
                ids_by_query_seed[query_seed] = distractors
            reference_ids = sorted(relevant) + ids_by_query_seed[query_seed][:size - len(relevant)]
            audit.check(ids == reference_ids, "catalog differs from October sampling", key)
        for sid in unique.difference(vocab):
            vocab[sid] = 1 << len(vocab)
        memberships[key] = reference, sum(vocab[sid] for sid in unique)
    audit.check(set(memberships) == expected_keys, "incomplete catalog grid")
    audit.counts["catalog_memberships"] = len(memberships)
    return memberships, vocab


def frozen_inputs(audit, root):
    result_dir = root / "results"
    full_manifest = result_dir / "experiment_manifest.json"
    partial_manifest = root / "manifests/bm25_partial_manifest.json"
    partial = not full_manifest.is_file() and partial_manifest.is_file()
    if partial:
        audit.pending.append("full frozen BM25, vector and hybrid candidate manifest")
    paths = [partial_manifest if partial else full_manifest, result_dir / "qrels.json",
             result_dir / "catalog_memberships.jsonl.gz",
             result_dir / ("bm25_partial_candidate_pools.jsonl.gz" if partial else "candidate_pools.jsonl.gz")]
    if not all([audit.require(path) for path in paths]):
        return None, None, None
    manifest_path, qrels_path, catalogs_path, pools_path = paths
    manifest = read_json(manifest_path)
    for filename, expected in manifest["sha256"].items():
        path = (root / filename).resolve()
        audit.check(path.is_relative_to(root) and path.is_file() and file_hash(path) == expected,
                    "frozen artifact checksum mismatch", Path(filename).name)
    for original in paths[1:]:
        audit.check(str(original.relative_to(root)) in manifest["sha256"],
                    "frozen artifact missing from manifest checksums", original.name)
    query_ids = manifest["query_ids"]
    audit.check(len(query_ids) == len(set(query_ids)) == 200, "wrong number of frozen query IDs")
    methods = ("bm25",) if partial else METHODS
    audit.check(tuple(manifest["sizes"]) == SIZES and tuple(manifest["seeds"]) == SEEDS
                and tuple(manifest["methods"]) == methods, "unexpected factor levels")
    revision = manifest.get("dataset_revision", manifest.get("dataset", {}).get("revision"))
    audit.check(revision == REVISION, "dataset revision mismatch")
    qrels = {qid: set(ids) for qid, ids in read_json(qrels_path).items()}
    if not audit.check(set(qrels) == set(query_ids) and all(qrels.values()), "qrels coverage mismatch"):
        return None, None, None
    documents, original_queries = compare_historical(audit, root, manifest, qrels)
    id_path = root / "manifests/catalog_document_ids.json"
    document_ids = read_json(id_path) if id_path.is_file() else None
    if document_ids is not None:
        audit.check(len(document_ids) == len(set(document_ids)) == 6006, "invalid catalog document index")
        audit.check(str(id_path.relative_to(root)) in manifest["sha256"], "catalog document index omitted from manifest")
        if documents is not None:
            audit.check(set(document_ids) == set(documents), "catalog index differs from pinned corpus")
    catalogs, vocab = catalog_memberships(audit, catalogs_path, query_ids, qrels, documents, document_ids)
    expected = set(product(query_ids, SIZES, SEEDS, methods))
    unique_cases, pools = set(), {}
    metadata_by_id = {}
    for row in jsonl_rows(pools_path):
        cid, key = row["case_id"], case_key(row)
        if not audit.check(key in expected and key not in unique_cases and cid not in pools,
                           "invalid or repeated candidate case", cid):
            continue
        unique_cases.add(key)
        pools[cid] = row
        audit.check(cid == canonical_hash([manifest["version"], REVISION, *key]), "case hash mismatch", cid)
        catalog = catalogs.get(key[:3])
        if not audit.check(catalog is not None and catalog[0] == row["catalog_ref"],
                           "case catalog reference mismatch", cid):
            continue
        if original_queries is not None:
            audit.check(row["query"] == original_queries.get(key[0]), "frozen query text mismatch", cid)
        ids = [candidate["id"] for candidate in row["candidates"]]
        audit.check(len(ids) == len(set(ids)) == 20, "candidate pool is not twenty distinct skills", cid)
        audit.check(all(vocab.get(sid, 0) & catalog[1] for sid in ids), "pool contains skill outside catalog", cid)
        audit.check(finite_number(row["retrieval_latency_ms"]) and row["retrieval_latency_ms"] >= 0,
                    "invalid retrieval timing", cid)
        for rank, candidate in enumerate(row["candidates"], 1):
            sid = candidate["id"]
            metadata = candidate["metadata"]
            audit.check(isinstance(metadata, str) and metadata_by_id.setdefault(sid, metadata) == metadata,
                        "inconsistent candidate metadata", cid)
            if documents is not None:
                audit.check(metadata == documents.get(sid), "candidate metadata differs from October rendering", cid)
            audit.check(candidate["rank"] == rank and finite_number(candidate["score"]),
                        "invalid candidate rank or score", cid)
        if all(finite_number(candidate["score"]) for candidate in row["candidates"]):
            order = sorted(row["candidates"], key=lambda candidate: (-candidate["score"], candidate["id"]))
            audit.check(order == row["candidates"], "candidate ranking disagrees with retrieval scores", cid)
    audit.check(unique_cases == expected and len(pools) == len(expected), "incomplete case grid")
    audit.check(manifest["logical_cases"] == len(expected) and manifest["candidate_count"] == 20
                and manifest["result_count"] == 5, "manifest case or ranking depth counts mismatch")
    audit.counts["logical_cases"] = len(pools)
    return pools, qrels, manifest


def expected_requests(pools, rubric):
    """Derive the label-free payload hash for every frozen pool independently."""
    by_case = {}
    names_by_hash = {}
    for cid, pool in pools.items():
        candidates = {row["id"]: {"name_and_description": row["metadata"]} for row in pool["candidates"]}
        state = json.dumps({"query": pool["query"], "candidates": candidates}, sort_keys=True,
                           ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        names = sorted(candidates)
        questions = [{"name": sid, "type": "predicate",
                      "instructions": rubric.format(candidate_id=sid)} for sid in names]
        digest = canonical_hash({"model": "gpt-6-luna", "input": state, "questions": questions})
        by_case[cid] = digest
        names_by_hash[digest] = names
    return by_case, names_by_hash


def validate_requests(audit, root, names_by_hash):
    path = root / "results/raw_requests.jsonl.gz"
    if not audit.require(path):
        return {}
    request_names = {}
    synthetic_hashes = set()
    for row in jsonl_rows(path):
        ref, payload = row["request_hash"], row["payload"]
        if not audit.check(ref not in request_names and ref == canonical_hash(payload),
                           "invalid or duplicate request hash", ref):
            continue
        if row["kind"] == "benchmark":
            if not audit.check(ref in names_by_hash, "request payload does not match a frozen pool", ref):
                continue
            request_names[ref] = names_by_hash[ref]
        elif row["kind"] == "synthetic":
            names = [question["name"] for question in payload["questions"]]
            audit.check(set(payload) == {"model", "input", "questions"} and payload["model"] == "gpt-6-luna"
                        and len(names) == len(set(names)) == 20 and names == sorted(names)
                        and all(question["type"] == "predicate" for question in payload["questions"]),
                        "invalid synthetic compatibility check", ref)
            request_names[ref] = names
            synthetic_hashes.add(ref)
        else:
            audit.check(False, "unexpected request kind", ref)
    audit.check(request_names.keys() - synthetic_hashes == names_by_hash.keys(),
                "sanitized benchmark request plan is incomplete")
    audit.check(len(synthetic_hashes) == 1, "expected one synthetic compatibility request")
    audit.counts["unique_requests"] = len(request_names) - len(synthetic_hashes)
    return request_names


def raw_predicates(body, expected_names):
    if not isinstance(body, dict) or not isinstance(body.get("model"), str):
        return None
    if not re.fullmatch(r"gpt-6-luna(?:-\d{4}-\d{2}-\d{2})?", body["model"]):
        return None
    answers = body.get("answers")
    if not isinstance(answers, list) or len(answers) != len(expected_names):
        return None
    probabilities = {}
    for answer in answers:
        if not isinstance(answer, dict) or answer.get("type") != "predicate":
            return None
        sid, probability = answer.get("name"), answer.get("probability")
        if (not isinstance(sid, str) or sid in probabilities or sid not in expected_names
                or not finite_number(probability) or not 0 <= probability <= 1):
            return None
        probabilities[sid] = probability
    return probabilities


def measured_usage(body):
    if not isinstance(body, dict):
        return None
    raw = body.get("usage")
    if not isinstance(raw, dict) or not isinstance(raw.get("input_tokens_details"), dict):
        return None
    detail = raw["input_tokens_details"]
    counts = {"input_tokens": raw.get("input_tokens"), "cached_input_tokens": detail.get("cached_tokens"),
              "cache_write_input_tokens": detail.get("cache_write_tokens"), "output_tokens": raw.get("output_tokens")}
    if any(type(v) is not int or v < 0 for v in counts.values()):
        return None
    if (raw.get("total_tokens") != counts["input_tokens"] + counts["output_tokens"]
            or counts["cached_input_tokens"] + counts["cache_write_input_tokens"] > counts["input_tokens"]
            or counts["input_tokens"] > 1_050_000):
        return None
    return counts


def priced_usage(usage, regional):
    if usage is None:
        return None
    writes, reads = usage["cache_write_input_tokens"], usage["cached_input_tokens"]
    input_tokens = usage["input_tokens"]
    nanodollars = (input_tokens - writes - reads) * 100 + reads * 10 + writes * 125
    multiplier = 2 if input_tokens > 272_000 else 1
    nanodollars *= multiplier
    return (nanodollars * 11 + 9) // 10 if regional else nanodollars


def raw_responses(audit, path, request_names, requests_by_case, regional):
    attempts, probabilities_by_attempt = {}, {}
    for row in jsonl_rows(path):
        aid, ref = row["attempt_id"], row["request_hash"]
        if not audit.check(aid not in attempts and ref in request_names, "unexpected or duplicate API attempt", aid):
            continue
        attempts[aid] = row
        if row["kind"] == "benchmark":
            audit.check(requests_by_case.get(row["case_id"]) == ref, "attempt points to wrong benchmark case", aid)
        else:
            audit.check(row["kind"] == "synthetic" and row["case_id"] == "synthetic/compatibility",
                        "invalid synthetic attempt reference", aid)
        audit.check(finite_number(row["api_latency_ms"]) and row["api_latency_ms"] >= 0,
                    "invalid fresh API latency", aid)
        body = row.get("body")
        answers = raw_predicates(body, request_names[ref])
        usage = measured_usage(body)
        http = row.get("http_status")
        succeeded = answers is not None and usage is not None and type(http) is int and 200 <= http < 300
        audit.check(row["status"] == ("success" if succeeded else "failed"), "raw API status mismatch", aid)
        audit.check(row.get("usage") == usage, "usage differs from provider response", aid)
        charge = priced_usage(usage, regional)
        guard_charge = priced_usage(usage, True)
        if charge is None and (http in {400, 401, 403, 404, 405, 413, 415, 422, 429}
                               or row.get("failure") == "proxy_error"):
            charge = 0
            guard_charge = 0
        compare_number(audit, row.get("estimated_cost_usd"), None if charge is None else charge / 1e9,
                       "response cost differs from verified pricing", aid)
        compare_number(audit, row.get("budgeted_cost_usd"), None if guard_charge is None else guard_charge / 1e9,
                       "response budget charge differs from upper-bound pricing", aid)
        if succeeded:
            audit.check(not row.get("failure"), "successful response has failure reason", aid)
            probabilities_by_attempt[aid] = answers
        else:
            audit.check(bool(row.get("failure")), "failed response has no explicit failure reason", aid)
            reported = body.get("answers", []) if isinstance(body, dict) else []
            if type(http) is int and 200 <= http < 300 and isinstance(reported, list):
                if any(isinstance(answer, dict) and answer.get("type") == "refusal" for answer in reported):
                    audit.check(row["failure"] == "predicate_refusal", "refusal lacks explicit failure reason", aid)
    audit.counts["finished_api_attempts"] = len(attempts)
    audit.counts["successful_api_attempts"] = len(probabilities_by_attempt)
    return attempts, probabilities_by_attempt


def validate_rankings(audit, root, pools, qrels, outcomes, responses, raw_scores, request_by_case):
    directory = root / "results"
    path = directory / "rankings.jsonl.gz"
    if not audit.require(path):
        return {}
    recomputed = {}
    status_counts = Counter()
    for row in jsonl_rows(path):
        cid = row["case_id"]
        if not audit.check(cid in pools and cid not in recomputed, "invalid or repeated ranking case", cid):
            continue
        pool = pools[cid]
        original = [entry["id"] for entry in pool["candidates"]]
        for key in ("query_id", "catalog_size", "seed", "method", "catalog_ref", "query"):
            audit.check(row[key] == pool[key], "ranking differs from frozen case identity", cid)
        audit.check(row["original_top20"] == original and row["original_top5"] == original[:5],
                    "original ranking differs from frozen candidates", cid)
        outcome = outcomes.get(cid, {})
        status = outcome.get("status", "unattempted")
        audit.check(status in {"success", "failed", "unattempted"} and row["status"] == status,
                    "ranking status differs from API outcome", cid)
        status_counts[status] += 1
        answer_ref = outcome.get("response_reference")
        request_ref = outcome.get("request_hash")
        scores = raw_scores.get(answer_ref) if status == "success" else None
        if outcome:
            audit.check(request_ref == request_by_case[cid], "outcome uses wrong request hash", cid)
        elif row.get("request_hash"):
            audit.check(row["request_hash"] == request_by_case[cid], "ranking uses wrong request hash", cid)
        for key in ("response_reference", "response_model", "request_id", "http_status", "failure"):
            audit.check(row.get(key) == outcome.get(key), "ranking API reference metadata mismatch", cid)
        audit.check(row.get("request_hash") == (request_ref or pool.get("request_hash")),
                    "ranking payload reference differs from recorded outcome", cid)
        cache_status = outcome.get("cache_status", "none")
        audit.check(row.get("cache_status") == cache_status, "ranking cache status mismatch", cid)
        if status == "success":
            audit.check(scores is not None and outcome.get("probabilities") == scores,
                        "successful case probabilities do not reproduce raw response", cid)
        else:
            audit.check(outcome.get("probabilities") is None, "unsuccessful case contains probabilities", cid)
        raw_response = responses.get(answer_ref, {})
        if answer_ref:
            audit.check(raw_response.get("request_hash") == request_ref and raw_response.get("status") == status,
                        "case response reference points to different payload or status", cid)
            audit.check(outcome.get("http_status") == raw_response.get("http_status")
                        and outcome.get("request_id") == raw_response.get("request_id"),
                        "case response HTTP reference mismatch", cid)
        reranked = (sorted(original, key=lambda sid: -scores[sid])
                    if scores and set(scores) == set(original) else None)
        operational = original if status == "failed" else reranked
        for variant, expected in (("reranked", reranked), ("operational", operational)):
            audit.check(row.get(f"{variant}_top20") == expected
                        and row.get(f"{variant}_top5") == (expected[:5] if expected else None),
                        "ranking violates candidate set, probability order, tie rule or fallback", cid)
        gold = qrels[pool["query_id"]]
        baseline_metrics = ranking_metrics(original, gold)
        metrics = dict(baseline_metrics)
        ceiling = metrics["recall_at_20"]
        metrics["candidate_recall_ceiling"] = ceiling
        metrics["oracle_recall_at_5"] = min(5, len(gold.intersection(original))) / len(gold)
        paired_ids = {}
        before = gold.intersection(original[:5])
        for prefix, ranking in (("reranked_", reranked), ("operational_", operational)):
            if ranking is None:
                continue
            variant_metrics = ranking_metrics(ranking, gold)
            audit.check(variant_metrics["recall_at_5"] <= ceiling and variant_metrics["recall_at_20"] == ceiling,
                        "reranking violates candidate recall ceiling", cid)
            metrics.update({prefix + name: value for name, value in variant_metrics.items()})
            name = "valid_only_" if prefix == "reranked_" else prefix
            after = gold.intersection(ranking[:5])
            hit_delta = len(after) - len(before)
            metrics[name + "recall_improvement"] = hit_delta / len(gold)
            metrics[name + "relative_recall_improvement"] = hit_delta / len(before) if before else None
            metrics[name + "improved_queries"] = int(hit_delta > 0)
            metrics[name + "worsened_queries"] = int(hit_delta < 0)
            metrics[name + "unchanged_queries"] = int(hit_delta == 0)
            metrics[name + "relevant_promoted"] = len(after - before)
            metrics[name + "relevant_removed"] = len(before - after)
            paired_ids[name + "promoted_ids"] = sorted(after - before)
            paired_ids[name + "removed_ids"] = sorted(before - after)
        fresh_latency = outcome.get("api_latency_ms") if cache_status == "fresh" else None
        if cache_status == "fresh" and raw_response:
            compare_number(audit, fresh_latency, raw_response.get("api_latency_ms"), "fresh case latency differs from API", cid)
        end, rerank_wall = outcome.get("end_to_end_ms"), outcome.get("rerank_wall_ms")
        retrieval = pool["retrieval_latency_ms"]
        if end is not None and rerank_wall is not None:
            compare_number(audit, end, retrieval + rerank_wall, "end-to-end time differs from summed stages", cid)
        timings = {"retrieval_latency_ms": retrieval, "api_latency_ms": fresh_latency,
                   "source_api_latency_ms": outcome.get("source_api_latency_ms"),
                   "rerank_wall_ms": rerank_wall, "end_to_end_latency_ms": end}
        recomputed[cid] = {"identity": case_key(pool), "status": status, "cache_status": cache_status,
                           "request_hash": row.get("request_hash"), "response_reference": answer_ref,
                           "metrics": metrics, "paired_ids": paired_ids, "timings": timings,
                           "reranked": reranked, "operational": operational, "scores": scores}
    audit.check(recomputed.keys() == pools.keys(), "missing original or reranked case ranking records")
    for name in ("success", "failed", "unattempted"):
        audit.counts[f"{name}_cases"] = status_counts[name]
    if status_counts["unattempted"]:
        audit.pending.append("Decisions outcomes for all logical cases")
    return recomputed


def csv_reference_matches(serialized, expected):
    return serialized == ("" if expected is None else str(expected))


def validate_case_exports(audit, root, cases, pools):
    directory = root / "results"
    for filename in ("per_query_metrics.csv", "paired_comparisons.csv", "latency.csv"):
        path = directory / filename
        if not audit.require(path):
            continue
        seen = set()
        for row in csv_rows(path):
            cid = row["case_id"]
            if not audit.check(cid in cases and cid not in seen, "invalid or repeated case export", f"{filename}/{cid}"):
                continue
            seen.add(cid)
            case = cases[cid]
            audit.check(case_key(row) == case["identity"] and row["status"] == case["status"],
                        "case export factors or status mismatch", f"{filename}/{cid}")
            expected = case["timings"] if filename == "latency.csv" else case["metrics"]
            columns = set(row) - {"case_id", "query_id", "catalog_size", "seed", "method", "status",
                                  "cache_status", "request_hash", "response_reference"}
            for field in columns:
                if field.endswith("_ids"):
                    value = json.loads(row[field]) if row[field] else None
                    audit.check(value == case["paired_ids"].get(field), "paired relevant skill IDs mismatch", cid)
                else:
                    number = expected.get(field, case["timings"].get(field))
                    compare_number(audit, row[field], number, f"{filename} numeric mismatch", f"{cid}/{field}")
            required = set(expected) if filename != "paired_comparisons.csv" else {
                "recall_at_5", "oracle_recall_at_5", "candidate_recall_ceiling"
            }
            audit.check(required <= columns, "case export omits expected measurements", f"{filename}/{cid}")
            for field in ("cache_status", "request_hash", "response_reference"):
                if field in row:
                    # DictReader returns text even for integer foreign keys.
                    # Require their exact CSV spelling; preserve missing as empty.
                    audit.check(csv_reference_matches(row[field], case[field]),
                                "case export request reference mismatch", f"{filename}/{cid}/{field}")
        audit.check(seen == cases.keys(), "case export coverage mismatch", filename)

    path = directory / "candidate_scores.parquet"
    if not audit.require(path):
        return
    import pyarrow.parquet as parquet

    seen = set()
    candidates = {cid: {record["id"]: record for record in pool["candidates"]} for cid, pool in pools.items()}
    table = parquet.ParquetFile(path)
    audit.check(set(table.schema_arrow.names) == {
        "case_id", "request_hash", "skill_id", "original_rank", "original_score", "probability",
        "reranked_rank", "operational_rank", "status",
    }, "unexpected Parquet candidate schema")
    for batch in table.iter_batches():
        for row in batch.to_pylist():
            cid, sid = row["case_id"], row["skill_id"]
            key = cid, sid
            if not audit.check(key not in seen and sid in candidates.get(cid, {}), "unknown or duplicate Parquet candidate", cid):
                continue
            seen.add(key)
            case = cases[cid]
            candidate = candidates[cid][sid]
            audit.check(row["original_rank"] == candidate["rank"] and row["original_score"] == candidate["score"],
                        "Parquet original candidate rank or score mismatch", cid)
            audit.check(row["status"] == case["status"] and row["request_hash"] == case["request_hash"],
                        "Parquet status or request mismatch", cid)
            audit.check(row["probability"] == (case["scores"] or {}).get(sid), "Parquet probability differs from raw answer", cid)
            for variant in ("reranked", "operational"):
                expected = case[variant].index(sid) + 1 if case[variant] is not None else None
                audit.check(row[f"{variant}_rank"] == expected, "Parquet rank differs from reconstructed ranking", cid)
    audit.check(len(seen) == 20 * len(cases), "Parquet candidate coverage mismatch")
    audit.counts["candidate_scores_checked"] = len(seen)


def mean(values):
    return math.fsum(values) / len(values) if values else None


def linear_quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap(sample, field):
    if not sample:
        return 0, None, None
    import numpy as np

    clusters = defaultdict(list)
    for case in sample:
        clusters[case["identity"][0]].append(case["metrics"][field])
    grouped = [clusters[qid] for qid in sorted(clusters)]
    values = np.array([math.fsum(pairs) for pairs in grouped])
    counts = np.array([len(pairs) for pairs in grouped])
    count = len(grouped)
    picks = np.random.default_rng(42).integers(0, count, (10_000, count))
    bootstrapped = values[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return count, linear_quantile(bootstrapped.tolist(), 0.025), linear_quantile(bootstrapped.tolist(), 0.975)


def usage_totals(attempts):
    usage_fields = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens")
    totals = {field: sum(int(attempt[field]) for attempt in attempts if attempt.get(field)) for field in usage_fields}
    costs = [float(attempt["estimated_cost_usd"]) for attempt in attempts if attempt.get("estimated_cost_usd")]
    return {
        **totals, "cache_write_tokens": totals["cache_write_input_tokens"],
        "api_http_attempts": sum(row.get("status") != "reserved" for row in attempts),
        "n_api_unconfirmed_reservations": sum(row.get("status") == "reserved" for row in attempts),
        "n_api_http_responses": sum(bool(row.get("http_status")) for row in attempts),
        "n_api_failed_attempts": sum(row.get("status") == "failed" for row in attempts),
        "n_provider_cache_hits": sum(int(row.get("cached_input_tokens") or 0) > 0 for row in attempts),
        "api_cost_usd": math.fsum(costs), "n_api_unknown_billing_attempts": len(attempts) - len(costs),
        "api_unknown_billing_reserved_usd": math.fsum(float(row["reserved_cost_usd"]) for row in attempts
                                                     if not row.get("estimated_cost_usd")),
    }


def summary_for_group(cases, attempts):
    valid = [row for row in cases if row["status"] == "success"]
    attempted = [row for row in cases if row["status"] in {"success", "failed"}]
    complete = len(attempted) == len(cases)
    usage = usage_totals(attempts)
    request_hashes = [row["request_hash"] for row in cases]
    result = {**usage, "n_queries": len({row["identity"][0] for row in cases}),
              "n_seeds": len({row["identity"][2] for row in cases}), "n_cases": len(cases), "n_valid_cases": len(valid),
              "n_valid_queries": len({row["identity"][0] for row in valid}),
              "n_valid_seeds": len({row["identity"][2] for row in valid}),
              "n_failed_cases": len(attempted) - len(valid), "n_attempted_cases": len(attempted),
              "n_unattempted_cases": len(cases) - len(attempted), "complete": complete,
              "n_unique_api_requests": len(set(request_hashes)) if all(request_hashes) else None,
              "n_unique_api_requests_attempted_by_group": len({row["request_hash"] for row in attempts}),
              "n_local_cache_hits": sum(row["cache_status"] == "local_cache_hit" for row in cases),
              "api_cost_per_query_usd": usage["api_cost_usd"] / len(attempted) if attempted else None}
    for label, values in (
        ("retrieval", [row["timings"]["retrieval_latency_ms"] for row in cases]),
        ("api", [float(row["api_latency_ms"]) for row in attempts if row.get("api_latency_ms")]),
        ("end_to_end", [row["timings"]["end_to_end_latency_ms"] for row in cases]),
        ("local_cache", [row["timings"]["rerank_wall_ms"] for row in cases if row["cache_status"] == "local_cache_hit"]),
    ):
        known = [value for value in values if value is not None]
        result[f"n_{label}_latency_samples"] = len(known)
        result[f"{label}_latency_median_ms"] = linear_quantile(known, 0.5)
        result[f"{label}_latency_p95_ms"] = linear_quantile(known, 0.95)
    result["latency_ms"] = result["retrieval_latency_median_ms"]
    result["api_latency_ms"] = result["api_latency_median_ms"]
    for output, selection, metric_prefix in (
        ("", cases, ""), ("valid_only_original_", valid, ""), ("valid_only_reranked_", valid, "reranked_"),
        ("attempted_original_", attempted, ""), ("attempted_operational_", attempted, "operational_"),
        ("operational_reranked_", attempted if complete else [], "operational_"),
    ):
        for metric in METRIC_FIELDS:
            result[output + metric] = mean([case["metrics"][metric_prefix + metric] for case in selection])
    result.update({"reranked_" + m: result["operational_reranked_" + m] for m in METRIC_FIELDS})
    for metric in ("candidate_recall_ceiling", "oracle_recall_at_5"):
        result[metric] = mean([case["metrics"][metric] for case in cases])
    intervals = {}
    for name, selection, prefix in (("valid_only", valid, "valid_only_"),
                                    ("operational", attempted if complete else [], "operational_"),
                                    ("attempted", attempted, "operational_")):
        source = prefix + "recall_improvement"
        improvement = mean([case["metrics"][source] for case in selection])
        baseline = mean([case["metrics"]["recall_at_5"] for case in selection])
        count, lower, upper = bootstrap(selection, source)
        summaries = {"recall_improvement": improvement, "relative_recall_improvement": improvement / baseline if baseline else None,
                     "improvement_ci_lower": lower, "improvement_ci_upper": upper}
        intervals[name] = {"n_queries": count, "n_cases": len(selection), "resamples": 10_000, "seed_rng": 42,
                           "improvement_ci_lower": lower, "improvement_ci_upper": upper, "recall_improvement": improvement}
        if name == "operational":
            result.update(summaries)
        for field in ("improved_queries", "worsened_queries", "unchanged_queries", "relevant_promoted", "relevant_removed"):
            summaries[field] = sum(case["metrics"][prefix + field] for case in selection) if selection else None
        result.update({f"{name}_{field}": value for field, value in summaries.items()})
    return result, intervals


METRIC_FIELDS = tuple(ranking_metrics(["x"], {"x"}))


def compare_table(audit, path, expected_rows, key_for, *, ignored_columns=()):
    if not audit.require(path):
        return
    seen = set()
    for row in csv_rows(path):
        key = key_for(row)
        if not audit.check(key in expected_rows and key not in seen, "unexpected or duplicate table row", f"{path.name}/{key}"):
            continue
        seen.add(key)
        for name, value in expected_rows[key].items():
            location = f"{path.name}/{key}/{name}"
            if not audit.check(name in row, "missing result column", location):
                continue
            if isinstance(value, (str, bool)):
                audit.check(row[name] == str(value), "textual result mismatch", location)
            else:
                compare_number(audit, row[name], value, "aggregate result mismatch", location)
        if ignored_columns:
            audit.check(set(row) <= set(expected_rows[key]) | set(ignored_columns),
                        "unexpected result columns", f"{path.name}/{key}")
    audit.check(seen == expected_rows.keys(), "result table row coverage mismatch", path.name)


def aggregate_tables(audit, root, cases, usage):
    directory = root / "results"
    groups = defaultdict(list)
    attempts = defaultdict(list)
    for case in cases.values():
        _, size, seed, method = case["identity"]
        groups[size, method, seed].append(case)
        groups[size, method, None].append(case)
    for event in usage:
        if event["kind"] != "benchmark":
            continue
        if not audit.check(event["case_id"] in cases, "API attempt has no aggregate case owner", event["attempt_id"]):
            continue
        _, size, seed, method = cases[event["case_id"]]["identity"]
        attempts[size, method, seed].append(event)
        attempts[size, method, None].append(event)
    summary_seed, summary_size, intervals = {}, {}, {}
    for (size, method, seed), group in groups.items():
        summary, paired = summary_for_group(group, attempts[size, method, seed])
        if seed is not None:
            summary_seed[size, method, seed] = summary
        else:
            summary_size[size, method] = summary
        for population, row in paired.items():
            intervals[size, method, seed, population] = row | {"metric": "Recall@5"}
    for (size, method), row in summary_size.items():
        seed_fields = [prefix + metric for metric in METRIC_FIELDS for prefix in ("", "reranked_", "valid_only_reranked_")]
        seed_fields += [prefix + "recall_improvement" for prefix in ("", "operational_", "valid_only_")]
        for field in seed_fields:
            values = [summary_seed[size, method, seed][field] for seed in SEEDS if (size, method, seed) in summary_seed]
            row[field + "_seed_std"] = statistics.stdev(values) if len(values) > 1 and None not in values else None
    size_key = lambda row: (int(row["catalog_size"]), row["method"])
    seed_key = lambda row: (*size_key(row), int(row["seed"]) if row.get("seed") else None)
    compare_table(audit, directory / "summary_by_seed.csv", summary_seed, seed_key,
                  ignored_columns=("catalog_size", "method", "seed"))
    compare_table(audit, directory / "summary_by_catalog_size.csv", summary_size, size_key,
                  ignored_columns=("catalog_size", "method"))
    compare_table(audit, directory / "plot_data.csv", summary_size, size_key,
                  ignored_columns=("catalog_size", "method"))
    compare_table(audit, directory / "bootstrap_intervals.csv", intervals,
                  lambda row: (*seed_key(row), row["population"]),
                  ignored_columns=("catalog_size", "method", "seed", "population"))
    labels = {"bm25": "BM25", "vector": "Vector", "hybrid": "Hybrid"}
    long_rows = {}
    for (size, method), source in summary_size.items():
        label = labels[method]
        populations = {
            "valid_only": {"complete": source["n_valid_cases"] == source["n_cases"], "n_queries": source["n_valid_queries"],
                           "n_seeds": source["n_valid_seeds"], "n_logical_cases": source["n_valid_cases"]},
            "retrieval": {"complete": True, "n_queries": source["n_queries"], "n_seeds": source["n_seeds"],
                          "n_logical_cases": source["n_cases"]},
            "operational": {"complete": source["complete"], "n_queries": source["n_queries"],
                            "n_seeds": source["n_seeds"], "n_logical_cases": source["n_cases"]},
        }
        for field in METRIC_FIELDS:
            kind, cutoff = field.split("_at_")
            display_metric = f"{kind.title() if kind in ('recall', 'precision', 'completeness') else kind.upper()}@{cutoff}"
            for series, prefix, population in (
                (label, "", "retrieval"),
                (f"{label} + OpenAI Decisions", "reranked_", "operational"),
                (f"{label} + OpenAI Decisions (valid only)", "valid_only_reranked_", "valid_only"),
            ):
                value = source[prefix + field]
                if value is not None:
                    long_rows[size, method, display_metric, population] = {
                        "series": series, "value": value, "seed_std": source[prefix + field + "_seed_std"],
                        "ci_lower": None, "ci_upper": None, **populations[population],
                    }
        for population in ("operational", "valid_only"):
            value = source[population + "_recall_improvement"]
            if value is not None:
                long_rows[size, method, "Recall@5 absolute improvement", population] = {
                    "series": f"{label} + OpenAI Decisions", "value": value,
                    "seed_std": source[population + "_recall_improvement_seed_std"],
                    "ci_lower": source[population + "_improvement_ci_lower"],
                    "ci_upper": source[population + "_improvement_ci_upper"], **populations[population],
                }
    compare_table(audit, directory / "plot_data_long.csv", long_rows,
                  lambda row: (*size_key(row), row["metric"], row["population"]))
    return summary_size


def same_timestamp(value):
    return isinstance(value, str) and datetime.fromisoformat(value).utcoffset() == timezone.utc.utcoffset(None)


def ledger_evidence(audit, directory, attempts, states, request_names):
    events_path = directory / "cost_ledger.jsonl"
    usage_path = directory / "api_usage.csv"
    failures_path = directory / "failures.jsonl"
    if not all([audit.require(path) for path in (events_path, usage_path, failures_path)]):
        return [], {}
    reserved, settled = {}, set()
    held, known, budgeted = Decimal(0), Decimal(0), Decimal(0)
    latest = {}
    for event in jsonl_rows(events_path):
        aid = event["attempt_id"]
        ref = event["request_hash"]
        audit.check(ref in request_names and same_timestamp(event["timestamp_utc"]), "invalid ledger payload or timestamp", aid)
        if event["event"] == "reserved":
            if audit.check(aid not in reserved, "duplicate ledger reservation", aid):
                reserved[aid] = event
                latest[ref] = aid
                compare_number(audit, event["reserved_cost_usd"], 0.28875, "request reservation amount mismatch", aid)
                held += Decimal(str(event["reserved_cost_usd"]))
                audit.check(held + budgeted <= Decimal("15"), "ledger reserves exceed hard USD budget", aid)
        elif event["event"] == "settled":
            if not audit.check(aid in reserved and aid in attempts and aid not in settled, "invalid ledger settlement", aid):
                continue
            settled.add(aid)
            raw = attempts[aid]
            original = reserved[aid]
            for key in ("case_id", "request_hash", "kind", "phase", "reserved_cost_usd"):
                audit.check(original[key] == raw[key] == event[key], "inconsistent ledger attempt identity", aid)
            for key in ("status", "http_status", "failure", "estimated_cost_usd", "budgeted_cost_usd"):
                audit.check(raw[key] == event[key], "ledger settlement differs from raw response", aid)
            audit.check(original["timestamp_utc"] == raw["started_at_utc"]
                        and event["timestamp_utc"] == raw["finished_at_utc"]
                        and event["timestamp_utc"] >= original["timestamp_utc"], "ledger event timestamps disagree", aid)
            cost = raw["estimated_cost_usd"]
            compare_number(audit, event["retained_reserve_usd"], raw["reserved_cost_usd"] if cost is None else 0.0,
                           "unknown billing reserve released or charged reserve retained", aid)
            if cost is not None:
                charge = Decimal(str(cost))
                reserve = Decimal(str(raw["reserved_cost_usd"]))
                guarded = Decimal(str(raw["budgeted_cost_usd"]))
                audit.check(0 <= charge <= guarded <= reserve, "API charge exceeds reservation", aid)
                known += charge
                budgeted += guarded
                held -= reserve
        else:
            audit.check(False, "unknown ledger event", aid)
    audit.check(set(attempts) == settled, "finished API records differ from ledger settlements")
    usage = list(csv_rows(usage_path))
    seen = set()
    for row in usage:
        aid = int(row["attempt_id"])
        if not audit.check(aid in reserved and aid not in seen, "duplicate or unreserved API usage row", aid):
            continue
        seen.add(aid)
        reservation, response = reserved[aid], attempts.get(aid, {})
        for key in ("case_id", "request_hash", "kind", "phase"):
            audit.check(row[key] == reservation[key], "usage CSV ledger identity mismatch", aid)
        audit.check(row["started_at_utc"] == reservation["timestamp_utc"]
                    and row["status"] == response.get("status", "reserved"), "usage CSV status or start mismatch", aid)
        for key in ("finished_at_utc", "request_id", "failure"):
            audit.check(row[key] == (response.get(key) or ""), "usage CSV response identity mismatch", aid)
        for key in ("estimated_cost_usd", "budgeted_cost_usd", "api_latency_ms", "rerank_wall_ms", "http_status"):
            compare_number(audit, row[key], response.get(key), "usage CSV differs from raw response", aid)
        compare_number(audit, row["reserved_cost_usd"], reservation["reserved_cost_usd"], "usage CSV reserve differs from ledger", aid)
        counts = response.get("usage") or {}
        for key in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens"):
            compare_number(audit, row[key], counts.get(key), "usage CSV tokens differ from raw response", aid)
        hit = str(counts["cached_input_tokens"] > 0) if counts else ""
        audit.check(row["provider_prompt_cache_hit"] == hit, "provider prompt-cache hit mismatch", aid)
    audit.check(seen == reserved.keys(), "incomplete API usage rows")
    unsuccessful = {aid for aid in reserved if attempts.get(aid, {}).get("status") != "success"}
    failed = set()
    for row in jsonl_rows(failures_path):
        aid = row["attempt_id"]
        if not audit.check(aid in unsuccessful and aid not in failed, "missing or duplicate failure attempt reference", aid):
            continue
        failed.add(aid)
        response, reservation = attempts.get(aid, {}), reserved[aid]
        audit.check(row["failure"] == (response.get("failure") or "dispatch_or_response_unknown")
                    and row["status"] == response.get("status", "reserved")
                    and row["http_status"] == response.get("http_status"), "failure ledger differs from attempt", aid)
        for key in ("case_id", "request_hash", "kind"):
            audit.check(row[key] == reservation[key], "failure references wrong request or case", aid)
    audit.check(failed == unsuccessful, "failure ledger coverage mismatch")
    audit.counts.update(ledger_reservations=len(reserved), ledger_settlements=len(settled),
                        estimated_api_cost_usd=float(known), budgeted_api_cost_usd=float(budgeted),
                        unresolved_cost_reservations_usd=float(held),
                        api_responses=sum(row.get("http_status") is not None for row in attempts.values()))
    if states:
        compare_number(audit, states["known_estimated_api_cost_usd"], float(known), "run state cost mismatch", "known")
        compare_number(audit, states["budgeted_api_cost_usd"], float(budgeted), "run state cost mismatch", "budgeted")
        compare_number(audit, states["held_api_reservations_usd"], float(held), "run state cost mismatch", "held")
        totals = usage_totals(usage)
        for name in states["token_totals"]:
            audit.check(states["token_totals"][name] == totals[name], "run state token totals mismatch", name)
        for name, value in {
            "ledger_reservations": len(reserved), "completed_http_attempt_records": len(attempts),
            "http_attempts_with_response": audit.counts["api_responses"], "unknown_dispatches": len(reserved) - len(attempts),
            "failed_attempt_records": sum(row["status"] == "failed" for row in attempts.values()),
            "provider_prompt_cache_hits": totals["n_provider_cache_hits"],
        }.items():
            audit.check(states.get(name) == value, "run state attempt counts mismatch", name)
    return usage, {ref: (aid, attempts.get(aid), reserved[aid]) for ref, aid in latest.items()}


def audit_api(audit, root, pools, manifest):
    directory = root / "results"
    state_path = directory / "run_state.json"
    run_state = read_json(state_path) if audit.require(state_path) else {}
    if run_state:
        audit.check(run_state["manifest_sha256"] == file_hash(directory / "experiment_manifest.json"),
                    "run state refers to different frozen manifest")
    template = read_json(root / "rubric.json")["instructions_template"]
    case_hash, names = expected_requests(pools, template)
    audit.check(manifest["unique_candidate_payloads"] == len(names), "manifest distinct payload count mismatch")
    smoke = manifest["smoke_case_ids"]
    audit.check(len(smoke) == len({case_hash.get(cid) for cid in smoke}) == 10 and all(cid in pools for cid in smoke),
                "invalid frozen smoke request panel")
    request_names = validate_requests(audit, root, names)
    response_path = directory / "raw_responses.jsonl.gz"
    attempts, scores = ({}, {})
    if audit.require(response_path):
        attempts, scores = raw_responses(audit, response_path, request_names, case_hash,
                                         run_state.get("regional_processing_price_enabled", False))
    successful_requests = [attempts[aid]["request_hash"] for aid in scores]
    audit.check(len(set(successful_requests)) == len(successful_requests), "completed request was sent twice")
    usage, latest = ledger_evidence(audit, directory, attempts, run_state, request_names)
    outcomes = {}
    path = directory / "decision_outcomes.jsonl.gz"
    if audit.require(path):
        for row in jsonl_rows(path):
            cid = row["case_id"]
            if not audit.check(cid in pools and cid not in outcomes, "duplicate or unknown outcome case", cid):
                continue
            outcomes[cid] = row
            ref = case_hash[cid]
            attempt = latest.get(ref)
            expected_status, expected_cache = "unattempted", "none"
            response = None
            if attempt:
                _, response, reservation = attempt
                expected_status = response["status"] if response else "failed"
                if response and reservation["case_id"] == cid:
                    expected_cache = "fresh"
                elif expected_status == "success":
                    expected_cache = "local_cache_hit"
            audit.check(row["status"] == expected_status and row["cache_status"] == expected_cache,
                        "case status or cache attribution differs from ledger", cid)
            audit.check(row["response_reference"] == (response or {}).get("attempt_id"),
                        "case did not reuse latest complete payload response", cid)
            audit.check(row["source_usage"] == (response or {}).get("usage")
                        and row["usage"] == ((response or {}).get("usage") if expected_cache == "fresh" else None),
                        "case token accounting double-counts or loses usage", cid)
            cost = ((response or {}).get("estimated_cost_usd") if expected_cache == "fresh"
                    else 0.0 if expected_cache == "local_cache_hit" else None)
            compare_number(audit, row["estimated_cost_usd"], cost, "case estimate double-counts or loses cost", cid)
            source_latency = (response or {}).get("api_latency_ms")
            compare_number(audit, row["source_api_latency_ms"], source_latency, "case source latency mismatch", cid)
            compare_number(audit, row["api_latency_ms"], source_latency if expected_cache == "fresh" else None,
                           "case cache latency counted as fresh inference", cid)
        audit.check(outcomes.keys() == pools.keys(), "incomplete per-case Decisions outcome export")
    if run_state:
        status_counts = Counter(row["status"] for row in outcomes.values())
        for name, value in {
            "logical_experiment_cases": len(pools), "unique_benchmark_requests": len(names),
            "unique_benchmark_requests_reserved": len({row["request_hash"] for row in usage if row["kind"] == "benchmark"}),
            "unique_benchmark_requests_completed_successfully": sum(attempts[aid]["kind"] == "benchmark" for aid in scores),
            "local_cache_hits": sum(row["cache_status"] == "local_cache_hit" for row in outcomes.values()),
            "successful_logical_cases": status_counts["success"], "failed_logical_cases": status_counts["failed"],
            "unattempted_logical_cases": status_counts["unattempted"],
        }.items():
            audit.check(run_state.get(name) == value, "run state logical and cache counts mismatch", name)
    audit.require(directory / "run_log.jsonl")
    if any(row["phase"] == "full" for row in usage):
        smoke_path = directory / "smoke_test.json"
        if audit.require(smoke_path):
            summary = read_json(smoke_path)
            audit.check(summary["gate_passed"] is True and summary["valid_case_count"] > 0
                        and [item["case_id"] for item in summary["observations"]] == smoke
                        and summary["guarded_projected_total_cost_usd"] <= summary["budget_usd"] <= 15,
                        "full run lacks validated ten-case smoke gate")
            audit.check(any(attempts[aid]["kind"] == "synthetic" for aid in scores),
                        "full run lacks successful synthetic check")
    return outcomes, attempts, scores, case_hash, usage


FIGURES = {"recall-at-5", "recall-at-20-ceiling", "reranking-improvement", "latency",
           "cost-and-token-usage", "presentation-comparison"}
OCTOBER_TABLE = Path("experiments/benchmarks/skillret-gap-closure-20261024/report/full-retrieval.csv")
PROMPT_TABLE = Path("experiments/benchmarks/openrouter-prompt-audit-20261108/results/summary.csv")
PROMPT_COMMIT = "5965356f8856f68b899ab0aa14b79de95fab506d"


def expected_figure_points(root, summaries):
    directory = root / "results"
    digest = file_hash(directory / "plot_data.csv")
    expected = {}

    def add(figure, population, method, size, field, value, *, source="plot_data.csv", source_field=None,
            bounds=(None, None), kind="", queries=None, seeds=None, SHA=digest):
        if value is not None:
            expected[figure, population, method, size, field, source_field or field] = {
                "value": value, "source": str(source), "lower": bounds[0], "upper": bounds[1],
                "uncertainty_kind": kind, "n_queries": queries, "n_seeds": seeds, "source_sha256": SHA,
            }

    for (size, method), row in summaries.items():
        labels = {"queries": row["n_queries"], "seeds": row["n_seeds"]}
        for figure in ("recall-at-5", "presentation-comparison", "recall-at-20-ceiling"):
            fields = ("recall_at_5", "reranked_recall_at_5") + (("recall_at_20",) if figure.endswith("ceiling") else ())
            for field in fields:
                value = row[field]
                sd = row.get(field + "_seed_std") if not figure.endswith("ceiling") else None
                bounds = (value - sd, value + sd) if sd is not None and value is not None else (None, None)
                add(figure, "pilot_200", method, size, field, value, bounds=bounds,
                    kind="seed_sd" if sd is not None else "", **labels)
        if row["recall_improvement"] is not None:
            add("reranking-improvement", "pilot_200", method, size, "recall_improvement", 100 * row["recall_improvement"],
                source_field="100*recall_improvement", kind="paired_ci95",
                bounds=(100 * row["improvement_ci_lower"], 100 * row["improvement_ci_upper"]), **labels)
        for kind in ("retrieval", "api", "end_to_end"):
            for statistic in ("median", "p95"):
                field = f"{kind}_latency_{statistic}_ms"
                add("latency", "pilot_200", method, size, field, row[field], **labels)
        attempted, http = row["n_attempted_cases"], row["api_http_attempts"]
        if attempted and (not http or row["n_api_unknown_billing_attempts"] != http):
            for field in ("api_cost_usd", "api_cost_per_query_usd"):
                add("cost-and-token-usage", "pilot_200", method, size, field, row[field], **labels)
        for kind, hits, denominator, count in (
            ("local", "n_local_cache_hits", "n_attempted_cases", attempted),
            ("provider", "n_provider_cache_hits", "api_http_attempts", http),
        ):
            if count:
                add("cost-and-token-usage", "pilot_200", method, size, f"{kind}_cache_hit_rate", row[hits] / count,
                    source_field=f"{hits}/({denominator})", **labels)
    for size in SIZES:
        observed = [row for (catalog_size, _), row in summaries.items() if catalog_size == size and row["input_tokens"] > 0]
        for field in ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens"):
            if observed:
                add("cost-and-token-usage", "pilot_200", "all", size, field, sum(row[field] for row in observed),
                    source_field=f"sum({field})")

    repo = root.parents[2]
    if not (repo / OCTOBER_TABLE).is_file() or not (repo / REFERENCE / "summary-200.json").is_file():
        return expected, False
    editions = defaultdict(list)
    for row in csv_rows(repo / OCTOBER_TABLE):
        editions[row["method"], int(row["size"])].append(float(row["macro_recall5"]))
    october_sha = file_hash(repo / OCTOBER_TABLE)
    for (method, size), recalls in editions.items():
        recall, sd = statistics.mean(recalls), statistics.stdev(recalls)
        add("presentation-comparison", "october_4392", method, size, "recall_at_5", recall,
            source=OCTOBER_TABLE, source_field="macro_recall5", SHA=october_sha,
            bounds=(recall - sd, recall + sd), kind="seed_sd", queries=4392, seeds=3)
    if (repo / PROMPT_TABLE).is_file():
        prompt = (repo / PROMPT_TABLE).read_bytes()
    else:
        prompt = subprocess.run(["git", "-C", str(repo), "show", f"{PROMPT_COMMIT}:{PROMPT_TABLE}"],
                                capture_output=True, check=True).stdout
    prompt_sha = hashlib.sha256(prompt).hexdigest()
    for row in csv.DictReader(prompt.decode("utf-8").splitlines()):
        if row["cohort"] == "retained_ranking" and row["library_size"] != "ALL" and int(row["n_scored"]):
            add("presentation-comparison", "openrouter_prompt_20", row["model"], int(row["library_size"]),
                "recall_at_5", float(row["recall@5"]), source=PROMPT_TABLE, source_field="recall@5",
                SHA=prompt_sha, queries=int(row["n_scored"]))
    jev_table = REFERENCE / "summary-200.json"
    jev = read_json(repo / jev_table)
    jev_sha = file_hash(repo / jev_table)
    for field in ("bm25_recall_at_5", "jev_recall_at_5"):
        add("presentation-comparison", "jev_200", "bm25", 6006, "recall_at_5", jev[field],
            source=jev_table, source_field=field, SHA=jev_sha, queries=200)
    return expected, True


def validate_figures(audit, root, summaries):
    directory = root / "results"
    points_path = directory / "figure_points.csv"
    presentation = directory / "presentation_plot_data.csv"
    export_file = directory / "figure_exports.json"
    if not all([audit.require(path) for path in (points_path, presentation, export_file, directory / "plot_data.csv")]):
        return
    expected, historical_available = expected_figure_points(root, summaries)
    points = list(csv_rows(points_path))
    exported = read_json(export_file)
    audit.check(list(csv_rows(presentation)) == [p for p in points if p["figure"] == "presentation-comparison"],
                "presentation CSV differs from rendered point export")
    audit.check(exported["figure_points_csv_sha256"] == file_hash(points_path)
                and exported["presentation_plot_data_csv_sha256"] == file_hash(presentation),
                "figure manifests omit exact exported point CSV hashes")
    point_key = lambda row: (row["figure"], row["population"], row["method"], int(row["catalog_size"]),
                             row["metric"], row["source_field"])
    checks = [row for row in points if historical_available or row["population"] == "pilot_200"]
    seen = set()
    for row in checks:
        key = point_key(row)
        if not audit.check(key in expected and key not in seen, "unexpected or duplicate figure point", key):
            continue
        seen.add(key)
        for field, value in expected[key].items():
            if isinstance(value, str):
                audit.check(row[field] == value, "figure point provenance mismatch", key)
            else:
                compare_number(audit, row[field], value, "figure point differs from its exported measurement", (*key, field))
    audit.check(seen == expected.keys(), "figure point coverage differs from available measured values")
    if not historical_available:
        audit.pending.append("historical graph source checkout")
    audit.check(set(exported["figures"]) == FIGURES, "figure manifest coverage mismatch")
    from PIL import Image

    asset_count = 0
    for slug in sorted(FIGURES):
        subset = [row for row in points if row["figure"] == slug]
        digest = canonical_hash(subset)
        item = exported["figures"].get(slug, {})
        audit.check(digest == item.get("figure_points_sha256"), "figure point hash mismatch", slug)
        stamp = f"figure_points_sha256={digest}"
        for extension in ("png", "svg", "pdf"):
            name = f"{slug}.{extension}"
            path = directory / "figures" / name
            if not audit.require(path):
                continue
            audit.check(file_hash(path) == item.get("files", {}).get(name), "figure file hash mismatch", name)
            if extension == "png":
                with Image.open(path) as image:
                    audit.check(image.format == "PNG" and min(image.info.get("dpi", (0, 0))) >= 300
                                and image.info.get("Description") == stamp,
                                "PNG data hash or 300 DPI minimum missing", name)
            else:
                audit.check(stamp.encode("ascii") in path.read_bytes(), "vector export lacks plotted source hash", name)
            asset_count += 1
    audit.counts["figure_points_checked"] = len(seen)
    audit.counts["figure_files_checked"] = asset_count


CREDENTIAL_PATTERN = re.compile(rb"\bsk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{40,}\b", re.IGNORECASE)
AUTHORIZATION_PATTERN = re.compile(rb"(?:\"authorization\"\s*:\s*\"|\bAuthorization:\s*)Bearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE)


def contains_credential(stream):
    tail = b""
    while block := stream.read(1024 * 1024):
        joined = tail + block
        if CREDENTIAL_PATTERN.search(joined) or AUTHORIZATION_PATTERN.search(joined):
            return True
        tail = joined[-4096:]
    return False


def check_secrets(audit, root, archive):
    inspected = 0
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or ".pytest_cache" in path.parts or path == archive:
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            audit.check(not contains_credential(stream), "credential pattern detected in experiment artifact", path.relative_to(root))
        inspected += 1
    if archive:
        with zipfile.ZipFile(archive) as package:
            for info in package.infolist():
                if info.is_dir():
                    continue
                with package.open(info) as stream:
                    if info.filename.endswith(".gz"):
                        with gzip.GzipFile(fileobj=stream) as expanded:
                            detected = contains_credential(expanded)
                    else:
                        detected = contains_credential(stream)
                audit.check(not detected, "credential pattern detected in archive member", Path(info.filename).name)
                inspected += 1
    audit.counts["credential_checked_files"] = inspected


def run_validation(root, inputs_only, archive):
    audit = Audit()
    active_stage = "frozen inputs"
    try:
        pools, qrels, manifest = frozen_inputs(audit, root)
        if pools and not inputs_only:
            active_stage = "API and spend evidence"
            if tuple(manifest["methods"]) == METHODS:
                outcomes, responses, scores, requests_by_case, usage = audit_api(audit, root, pools, manifest)
            else:
                outcomes, responses, scores, requests_by_case, usage = {}, {}, {}, {}, []
                audit.pending.append("live API request, response and spend evidence")
            active_stage = "rankings and per-query metrics"
            cases = validate_rankings(audit, root, pools, qrels, outcomes, responses, scores, requests_by_case)
            if cases:
                validate_case_exports(audit, root, cases, pools)
                active_stage = "aggregated metrics and bootstrap"
                summaries = aggregate_tables(audit, root, cases, usage)
                active_stage = "publication figures"
                validate_figures(audit, root, summaries)
    except (OSError, EOFError, ValueError, TypeError, KeyError, AttributeError, ImportError, subprocess.CalledProcessError) as error:
        audit.check(False, f"unreadable or invalid {active_stage}", type(error).__name__)
    try:
        check_secrets(audit, root, archive)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        audit.check(False, "artifact credential scan could not complete", type(error).__name__)
    return audit.report("inputs" if inputs_only else "results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inputs-only", action="store_true", help="Validate frozen preparation before inference.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Return zero for valid available evidence with pending work.")
    parser.add_argument("--archive", type=Path, help="Also scan the delivery ZIP and its compressed evidence for credentials.")
    parser.add_argument("--report", type=Path, help="JSON result path. Defaults to results/validation.json.")
    args = parser.parse_args()
    root = args.root.resolve()
    result = run_validation(root, args.inputs_only, args.archive.resolve() if args.archive else None)
    output = args.report or root / "results/validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2) + "\n"
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["status"] == "failed":
        return 1
    if result["status"] == "incomplete" and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
