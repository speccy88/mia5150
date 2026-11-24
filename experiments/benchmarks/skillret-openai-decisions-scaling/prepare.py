"""Freeze the exposed query panel, catalog memberships and ranked candidate pools."""

import argparse
import datetime as dt
import gzip
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from retrieve import (
    HERE, OCTOBER, Encoder, cached_embeddings, digest, historical_embeddings,
    library, model_assets, render, retrieve, sha256,
)

ROOT = HERE.parents[2]
PUBLIC_INPUTS = ROOT / "experiments/scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs"
JEV = HERE.parent / "skillret-reranker-study"
RESULTS = HERE / "results"
MANIFESTS = HERE / "manifests"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_lines(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_gzip_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as destination:
        temporary = Path(destination.name)
        try:
            with gzip.GzipFile(filename="", fileobj=destination, mode="wb", mtime=0, compresslevel=6) as compressed:
                for row in rows:
                    line = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    compressed.write((line + "\n").encode("utf-8"))
            destination.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_gzip_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def inputs(config):
    protocol = read(OCTOBER / "protocol.json")
    dataset = config["dataset"]
    if any(dataset[key] != protocol["dataset"][key] for key in ("repository", "revision", "split")):
        raise ValueError("Dataset revision or split differs from the October matched protocol")
    hashes = {name: sha256(PUBLIC_INPUTS / name) for name in protocol["source_hashes"]}
    if hashes != protocol["source_hashes"]:
        raise ValueError("Pinned October public dataset projection has changed")
    skills = json_lines(PUBLIC_INPUTS / "skills-metadata.jsonl")
    all_queries = json_lines(PUBLIC_INPUTS / "queries-test.jsonl")
    binary_qrels = json_lines(PUBLIC_INPUTS / "qrels-test.jsonl")
    documents = {skill["id"]: render(skill) for skill in skills}
    query_by_id = {query["id"]: query for query in all_queries}
    if (
        len(skills) != len(documents) or len(documents) != dataset["skill_count"]
        or len(all_queries) != len(query_by_id) or len(query_by_id) != dataset["query_count"]
        or len(binary_qrels) != dataset["positive_pair_count"]
        or digest(documents) != protocol["document_sha256"]
    ):
        raise ValueError("Pinned dataset cardinality or rendered metadata differs from October")

    labels = defaultdict(set)
    for row in binary_qrels:
        query_id, skill_id = row["query_id"], row["skill_id"]
        if row["relevance"] != 1 or query_id not in query_by_id or skill_id not in documents or skill_id in labels[query_id]:
            raise ValueError("Invalid binary qrel")
        labels[query_id].add(skill_id)
    if any(not labels[query_id] or labels[query_id] != set(query["skill_ids"]) for query_id, query in query_by_id.items()):
        raise ValueError("Pinned binary qrels differ from the official query projection")

    panel = read(JEV / "manifests/pilot_query_ids.json")
    jev_results = read(JEV / "reference_run/jev/manifest-results-200.json")
    if len(panel) != config["query_count"] or len(set(panel)) != len(panel) or panel != jev_results["query_ids"]:
        raise ValueError("Fixed JEV query panel differs from its reported run")
    queries = {query_id: query_by_id[query_id]["query"] for query_id in panel}
    qrels = {query_id: sorted(labels[query_id]) for query_id in panel}
    history = audit_jev(queries, qrels)
    provenance = {str((PUBLIC_INPUTS / name).relative_to(ROOT)): value for name, value in hashes.items()}
    return documents, queries, qrels, history, provenance


def audit_jev(queries, qrels):
    """Recompute the historical points from public raw rankings and binary qrels."""
    run = JEV / "reference_run/jev"
    paths = [JEV / "manifests/pilot_query_ids.json", JEV / "prompts/jev_relevance.json"]
    paths += [run / name for name in ("manifest-results-200.json", "candidates.json", "rows.jsonl", "summary-200.json")]
    baseline_path = JEV / "reference_run/baselines/metadata_bm25_rows.jsonl"
    paths.append(baseline_path)
    candidates = read(run / "candidates.json")
    rows = json_lines(run / "rows.jsonl")
    manifest = read(run / "manifest-results-200.json")
    summary = read(run / "summary-200.json")
    baseline = {row["query_id"]: row["ranking"] for row in json_lines(baseline_path)}
    if list(queries) != [row["query_id"] for row in candidates] or list(queries) != [row["query_id"] for row in rows]:
        raise ValueError("Saved historical results do not follow the fixed query panel")
    totals = Counter()
    expected_strata = Counter()
    for case, row in zip(candidates, rows, strict=True):
        query_id = case["query_id"]
        gold = set(qrels[query_id])
        if (
            case["query"] != queries[query_id] or set(case["evaluator_only_gold"]) != gold
            or set(row["evaluator_only_gold"]) != gold or row["status"] != "ok"
            or row["bm25_top20"] != case["bm25_top20"]
            or row["bm25_top20"][:5] != baseline[query_id][:5]
        ):
            raise ValueError(f"Historical query, labels, or candidates differ: {query_id}")
        ranked_bm25 = [skill_id for skill_id, _ in row["bm25_top20"]]
        probabilities = row["jev_probabilities"]
        if len(ranked_bm25) != 20 or len(set(ranked_bm25)) != 20 or set(ranked_bm25) != set(probabilities):
            raise ValueError(f"Historical candidate probability IDs differ: {query_id}")
        if not all(math.isfinite(score) and 0 <= score <= 1 for score in probabilities.values()):
            raise ValueError(f"Historical probability outside [0, 1]: {query_id}")
        reranked = sorted(ranked_bm25, key=lambda skill_id: -probabilities[skill_id])[:5]
        if row["jev_top5"] != reranked:
            raise ValueError(f"Historical JEV ranking cannot be reproduced: {query_id}")
        for metric, ranking in (
            ("bm25_recall_at_5", ranked_bm25[:5]), ("bm25_recall_at_20", ranked_bm25),
            ("jev_recall_at_5", reranked),
        ):
            totals[metric] += len(gold.intersection(ranking)) / len(gold)
        expected_strata[len(gold)] += 1

    points = {metric: total / len(queries) for metric, total in totals.items()}
    source_metrics = {
        "bm25_recall_at_5": summary["bm25_recall_at_5"],
        "bm25_recall_at_20": summary["bm25_recall_at_20"],
        "jev_recall_at_5": summary["jev_recall_at_5_valid_only"],
    }
    if any(not math.isclose(points[name], value, abs_tol=1e-12) for name, value in source_metrics.items()):
        raise ValueError("Historical Recall points disagree with the archived summary")
    return {
        "catalog_size": manifest["dataset"]["files"]["skills"]["rows"],
        "query_count": len(queries), "positive_count_distribution": dict(sorted(expected_strata.items())),
        "same_query_order_text_and_labels": True, "reranked_top5_reproduced": len(rows),
        "reference_bm25_top5_ids_and_scores_reproduced": len(rows),
        "recall_recomputed": points,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "upstream_run_recorded_rows_sha256": manifest["results_sha256"],
        "public_rows_sha256": sha256(run / "rows.jsonl"),
    }


def memberships_for(config, documents, queries, qrels):
    for query_id in queries:
        for seed in config["seeds"]:
            for size in config["catalog_sizes"]:
                skill_ids = library(documents, qrels[query_id], query_id, size, seed)
                yield {
                    "catalog_ref": digest(skill_ids), "query_id": query_id,
                    "catalog_size": size, "seed": seed, "skill_ids": skill_ids,
                }


def encode_memberships(memberships, document_ids):
    positions = {skill_id: i for i, skill_id in enumerate(document_ids)}
    for row in memberships:
        skill_ids = row.pop("skill_ids")
        yield row | {"document_indices": [positions[skill_id] for skill_id in skill_ids]}


def decode_memberships(memberships, document_ids):
    for row in memberships:
        indices = row.pop("document_indices")
        yield row | {"skill_ids": [document_ids[index] for index in indices]}


def embedding_inputs(config, documents, queries, work_dir, previous_arrays=None):
    settings = config["embedding"]
    assets = work_dir / "bge-small"
    if previous_arrays is None or all((assets / Path(name).name).is_file() for name in settings["assets"]):
        model_assets(settings, assets)
        encoder = Encoder(settings, assets)
        encoder.encode(["Synthetic retrieval warm-up."], query=True)
        timings = {}
        for query_id, query in queries.items():
            started = time.perf_counter()
            encoder.encode([query], query=True)
            timings[query_id] = (time.perf_counter() - started) * 1000
    else:
        timings = dict.fromkeys(queries, None)
    if previous_arrays is not None:
        docs, query_vectors, hashes = historical_embeddings(previous_arrays, documents, queries)
        evidence = {"source": "verified October snapshot", "arrays": hashes}
    else:
        docs, doc_receipt = cached_embeddings(encoder, list(documents.values()), False, work_dir, "documents")
        query_vectors, query_receipt = cached_embeddings(encoder, list(queries.values()), True, work_dir, "queries")
        evidence = {"source": "pinned asset re-encoding", "arrays": {"documents": doc_receipt, "queries": query_receipt}}
    return (docs, query_vectors, timings), evidence


def select_smoke_cases(config, unique_cases):
    chosen = []
    used = set()
    methods, sizes, seeds = config["methods"], config["catalog_sizes"], config["seeds"]
    for i in range(config["decisions"]["smoke_cases"]):
        target = (sizes[i % len(sizes)], seeds[i % len(seeds)], methods[i % len(methods)])
        eligible = [(signature, case_id) for signature, cases in unique_cases.items()
                    if signature not in used and (case_id := cases.get(target)) is not None]
        if not eligible:
            raise ValueError("Cannot find a distinct smoke payload for a frozen cell")
        signature, case_id = min(eligible, key=lambda item: digest(["smoke", 42, *item]))
        chosen.append(case_id)
        used.add(signature)
    return chosen


def verify_hashes(hashes):
    for filename, expected in hashes.items():
        if sha256(HERE / filename) != expected:
            raise ValueError(f"Frozen artifact checksum differs: {filename}")


def verify_manifest(path):
    manifest = read(path)
    verify_hashes(manifest["sha256"])
    print(f"Verified frozen manifest for {manifest['logical_cases']:,} cases: {path.name}", flush=True)


def record_hardware():
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "os": platform.system(), "release": platform.release(), "architecture": platform.machine(),
        "cpu": cpu_model, "logical_cpus_available": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "encoder_threads": 1, "blas_threads": 1,
        "python": platform.python_version(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "onnxruntime", "tokenizers", "threadpoolctl")
        },
    }


def compare_october(candidate_path, output_path):
    """Verify exact shared memberships and top-five rankings against October evidence."""
    archive = OCTOBER / "results/retrieval.jsonl.gz"
    expected_hash = read(OCTOBER / "results/retrieval-archive.json")["gzip_sha256"]
    if sha256(archive) != expected_hash:
        raise ValueError("Archived October retrieval checksum differs from its receipt")
    groups = {}
    for row in read_gzip_rows(candidate_path):
        if row["catalog_size"] in (100, 1000):
            cell = (row["query_id"], row["catalog_size"], row["seed"])
            groups.setdefault(cell, []).append(row)
    counts = defaultdict(Counter)
    for historical in read_gzip_rows(archive):
        cell = (historical["query_id"], historical["size"], historical["seed"])
        for current in groups.pop(cell, []):
            method = current["method"]
            previous = historical["methods"][method]
            top_five = [item["id"] for item in current["candidates"][:5]]
            counts[method]["cases"] += 1
            counts[method]["catalog_membership_exact"] += current["catalog_ref"] == historical["catalog_sha256"]
            counts[method]["top5_ids_exact_in_order"] += top_five == previous["ranking5"]
    if groups:
        raise ValueError("Current cases missing from the matched October retrieval archive")
    if not counts or any(row["cases"] != row["catalog_membership_exact"] for row in counts.values()):
        raise ValueError("Current overlapping catalog memberships differ from October")
    if counts["bm25"]["top5_ids_exact_in_order"] != counts["bm25"]["cases"]:
        raise ValueError("Scored BM25 does not reproduce the archived October rankings")
    result = {
        "catalog_sizes": [100, 1000], "archive_sha256": expected_hash,
        "results_by_method": counts,
        "scope": "Overlap with the archived 4,392-query retrieval results. The archive exposes only top-five IDs, so it cannot verify the full top 20.",
    }
    write_json(output_path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path.home() / ".cache/skillret-decisions", help="External encoder asset and embedding cache")
    parser.add_argument("--october-arrays", type=Path, help="Directory containing checksum-verified October arrays")
    parser.add_argument("--stage", default="full", choices=("inputs", "bm25", "full"), help="Prepare inputs, measure BM25 only, or freeze the full factorial study")
    args = parser.parse_args()
    config = read(HERE / "config.json")
    manifest_path = RESULTS / "experiment_manifest.json"
    if manifest_path.exists():
        verify_manifest(manifest_path)
        return
    started = time.perf_counter()
    documents, queries, qrels, history, source_hashes = inputs(config)
    panel_path = MANIFESTS / "query_ids.json"
    history_path = MANIFESTS / "historical_jev.json"
    input_manifest_path = MANIFESTS / "input_snapshot.json"
    document_id_path = MANIFESTS / "catalog_document_ids.json"
    qrels_path = RESULTS / "qrels.json"
    catalog_path = RESULTS / "catalog_memberships.jsonl.gz"
    document_ids = sorted(documents)
    if input_manifest_path.exists():
        verify_hashes(read(input_manifest_path)["sha256"])
    else:
        write_json(panel_path, list(queries))
        write_json(history_path, history)
        write_json(document_id_path, document_ids)
        write_json(qrels_path, qrels)
        memberships = memberships_for(config, documents, queries, qrels)
        write_gzip_rows(catalog_path, encode_memberships(memberships, document_ids))
        write_json(input_manifest_path, {
            "dataset": config["dataset"], "query_count": len(queries),
            "catalogs": len(queries) * len(config["catalog_sizes"]) * len(config["seeds"]),
            "sha256": {str(path.relative_to(HERE)): sha256(path) for path in (HERE / "config.json", panel_path, history_path, document_id_path, qrels_path, catalog_path)},
            "source_sha256": source_hashes,
        })
    if args.stage == "inputs":
        print(f"Prepared the {len(queries)}-query panel, qrels, and catalog memberships", flush=True)
        return

    embeddings = None
    if args.stage == "full":
        args.work_dir.mkdir(parents=True, exist_ok=True)
        embeddings, embedding_evidence = embedding_inputs(config, documents, queries, args.work_dir, args.october_arrays)

    counts = Counter()
    signatures = defaultdict(dict)

    def record_cases():
        memberships = decode_memberships(read_gzip_rows(catalog_path), document_ids)
        for row in retrieve(config, documents, queries, memberships, embeddings):
            counts[row["method"]] += 1
            if embeddings is not None:
                # Metadata, model and rubric are fixed in this manifest. Exact query
                # text plus sorted IDs therefore uniquely determines the API payload.
                signature = digest([row["query"], sorted(item["id"] for item in row["candidates"])])
                cell = (row["catalog_size"], row["seed"], row["method"])
                signatures[signature].setdefault(cell, row["case_id"])
            yield row

    filename = "candidate_pools.jsonl.gz" if embeddings is not None else "bm25_partial_candidate_pools.jsonl.gz"
    candidate_path = RESULTS / filename
    write_gzip_rows(candidate_path, record_cases())
    print(f"Prepared candidate pools by method: {dict(counts)}", flush=True)
    methods = config["methods"] if embeddings is not None else ["bm25"]
    cases_per_method = len(queries) * len(config["catalog_sizes"]) * len(config["seeds"])
    if counts != dict.fromkeys(methods, cases_per_method):
        raise ValueError("Frozen candidate pools do not cover the factorial design")
    comparison_name = "october_agreement.json" if embeddings is not None else "october_bm25_agreement.json"
    comparison_path = HERE / "evidence" / comparison_name
    compare_october(candidate_path, comparison_path)
    paths = [panel_path, history_path, input_manifest_path, document_id_path, qrels_path, catalog_path, candidate_path, comparison_path]
    paths += [HERE / name for name in ("config.json", "prepare.py", "retrieve.py")]
    for name in ("protocol.json", "core.py", "fast_bm25.py", "embedding-full-manifest.json"):
        path = OCTOBER / name
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)
    overview = ROOT / "benchmark_subset.md"
    if overview.is_file():
        source_hashes[overview.name] = sha256(overview)
    manifest = {
        "version": config["version"], "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dataset": config["dataset"], "query_ids": list(queries), "sizes": config["catalog_sizes"],
        "seeds": config["seeds"], "methods": methods, "logical_cases": sum(counts.values()),
        "candidate_count": config["candidate_count"], "result_count": config["result_count"],
        "metadata_representation": "strip name and description, join with ` | `, cap description at 1024 characters",
        "catalog_sampling": "October core.library: all positives, then prefix of SHA-256 seeded shuffled sorted distractors",
        "catalog_encoding": "Ordered document_indices into manifests/catalog_document_ids.json; catalog_ref hashes the decoded IDs",
        "timing_scope": "search per catalog and seed, plus one separately measured batch-one query forward shared across dense cases for each query when available",
        "hardware_and_packages": record_hardware(), "source_sha256": source_hashes,
        "preparation_elapsed_seconds": time.perf_counter() - started,
    }
    if embeddings is None:
        expected_cases = cases_per_method * len(config["methods"])
        manifest.update(
            stage="bm25_partial", expected_logical_cases=expected_cases,
            expected_logical_judgments=expected_cases * config["candidate_count"],
        )
        output_path = MANIFESTS / "bm25_partial_manifest.json"
    else:
        manifest.update(
            stage="full", logical_judgments=sum(counts.values()) * config["candidate_count"],
            unique_candidate_payloads=len(signatures),
            smoke_case_ids=select_smoke_cases(config, signatures), embedding=config["embedding"] | embedding_evidence,
        )
        paths.append(HERE / "rubric.json")
        recovery_receipt = HERE / "evidence/bge_asset_recovery.json"
        if recovery_receipt.is_file():
            paths.append(recovery_receipt)
        output_path = manifest_path
    manifest["sha256"] = {str(path.relative_to(HERE)): sha256(path) for path in paths}
    write_json(output_path, manifest)
    verify_manifest(output_path)


if __name__ == "__main__":
    main()
