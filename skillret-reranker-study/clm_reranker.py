"""Bounded CLM ranking; evaluation stays local, outside model requests.

Reuses the pilot's metadata, BM25, reservations and multi-positive metrics.
No skill bodies or instructions are executed. No CUDA/CLM imports on the Mac.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from reference_metrics import retrieval_metrics, selection_metrics

QUESTION = "Which candidate skill is most relevant to satisfying this request?"
MODEL = "clm-latest"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_new(path, data):
    """Exclusive creation prevents overwriting prior evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def append_jsonl(path, row):
    import os
    with Path(path).open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def new_run(root):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    p = Path(root) / "results" / ("clm_runpod_" + stamp)
    p.mkdir(parents=True, exist_ok=False)
    return p


def build_clm_candidates(query_text, candidate_ids, metadata, case_id):
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("empty canonical query")
    ids = list(candidate_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be nonempty and unique")
    # Display order depends on IDs only, never BM25 score, labels or relevance.
    ids.sort(key=lambda x: hashlib.sha256(f"42\0{case_id}\0{x}".encode()).digest())
    texts = [metadata[x] for x in ids]
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("empty candidate text")
    return {"query_id": case_id, "query_text": query_text,
            "candidate_ids": ids, "candidate_texts": texts}


def rank_payload(case):
    # Explicit field allowlist. Query labels and IDs cannot enter the request.
    return {"model": MODEL, "context": case["query_text"], "question": QUESTION,
            "answers": list(case["candidate_texts"]), "temperature": 1.0}


def parse_rank(response, case):
    if response.get("model") != MODEL:
        raise ValueError("unexpected CLM model")
    ranked = response.get("ranked")
    ids, texts = case["candidate_ids"], case["candidate_texts"]
    if not isinstance(ranked, list) or len(ranked) != len(ids):
        raise ValueError("incomplete ranked response")
    available = defaultdict(deque)
    for index, (skill_id, text) in enumerate(zip(ids, texts)):
        available[text].append((index, skill_id))
    mapped = []
    for expected_rank, row in enumerate(ranked, 1):
        if row.get("rank") != expected_rank:
            raise ValueError("invalid rank sequence")
        text, prob = row.get("candidate"), row.get("prob")
        if not isinstance(text, str) or not available[text]:
            raise ValueError("unknown or repeated candidate")
        if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not math.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError("invalid probability")
        index, skill_id = available[text].popleft()
        mapped.append((skill_id, float(prob), index))
    if abs(sum(x[1] for x in mapped) - 1) > 1e-4:
        raise ValueError("probabilities do not sum to one")
    if any(a[1] < b[1] for a, b in zip(mapped, mapped[1:])):
        raise ValueError("response is not sorted by probability")
    # Identical text maps in input order. Probability ties use that same order.
    mapped.sort(key=lambda x: (-x[1], x[2]))
    return [x[0] for x in mapped]


def rank_with_clm(client, base_url, case, timeout=120):
    started = time.perf_counter()
    record = {"query_id": case["query_id"], "request": rank_payload(case),
              "candidate_ids": case["candidate_ids"], "status": "SERVICE_FAILURE"}
    try:
        response = client.post(base_url.rstrip("/") + "/v1/rank",
                               json=record["request"], timeout=timeout)
        record["http_status"] = response.status_code
        record["response_text"] = response.text
        record["server_latency_ms"] = response.headers.get("X-CLM-Latency-Ms")
        response.raise_for_status()
        record["response"] = response.json()
        record["ranking"] = parse_rank(record["response"], case)
        record["status"] = "ok"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["latency_seconds"] = time.perf_counter() - started
    return record


def evaluate_row(baseline, inference):
    gold = set(baseline["evaluator_only_gold"])
    b = baseline["ranking"]
    c = inference.get("ranking", []) if inference["status"] == "ok" else []
    bm = retrieval_metrics(b, gold, 5)
    cm = retrieval_metrics(c, gold, 5)
    labels = []
    missing = sorted(gold - set(b[:20]))
    gained, lost = sorted((set(c[:5]) - set(b[:5])) & gold), sorted((set(b[:5]) - set(c[:5])) & gold)
    if missing:
        labels.append("UPSTREAM_RETRIEVAL_FAILURE")
    if inference["status"] != "ok":
        labels.append("SERVICE_FAILURE")
    else:
        if gained:
            labels.append("CLM_RERANKING_SUCCESS")
        if lost:
            labels.append("CLM_RERANKING_FAILURE")
        if not gained and not lost:
            labels.append("UNCHANGED_SUCCESS" if gold & set(c[:5]) else "UNCHANGED_FAILURE")
    return {"query_id": baseline["query_id"], "status": inference["status"],
            "bm25": bm, "candidate": retrieval_metrics(b, gold, 20), "clm": cm,
            "selection": selection_metrics(c[:5], gold),
            "delta_recall_at_5": cm["recall_at_5"] - bm["recall_at_5"],
            "missing_gold_in_top20": missing, "gained_gold": gained, "lost_gold": lost,
            "failure_labels": labels, "latency_seconds": inference.get("latency_seconds")}


def percentile(values, q):
    v = sorted(values)
    if not v:
        return None
    i = (len(v) - 1) * q
    lo = int(i)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (i - lo)


def latency_summary(values):
    return {"mean_seconds": statistics.mean(values) if values else None,
            "median_seconds": statistics.median(values) if values else None,
            "p95_seconds": percentile(values, .95)}


def summarize_clm_results(baselines, raw, bootstrap=5000):
    by = {r["query_id"]: r for r in raw}
    if len(by) != len(raw):
        raise ValueError("duplicate inference query IDs")
    if set(by) - {b['query_id'] for b in baselines}:
        raise ValueError("inference contains unknown query IDs")
    evaluated = [evaluate_row(b, by[b["query_id"]]) for b in baselines if b["query_id"] in by]
    if not evaluated:
        return {"planned": len(baselines), "attempted": 0, "status": "NOT_RUN"}, []
    deltas = [r["delta_recall_at_5"] for r in evaluated]
    rng = random.Random(42)
    samples = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(bootstrap))
    mean = lambda group, key: statistics.mean(r[group][key] for r in evaluated)
    base, clm = mean("bm25", "recall_at_5"), mean("clm", "recall_at_5")
    return {"planned": len(baselines), "attempted": len(evaluated),
            "unrun": len(baselines) - len(evaluated),
            "status": "COMPLETE" if len(evaluated) == len(baselines) else "PARTIAL",
            "bm25_recall_at_5": base, "bm25_recall_at_20": mean("candidate", "recall_at_20"),
            "bm25_clm_recall_at_5": clm, "absolute_delta": clm - base,
            "relative_delta": (clm - base) / base if base else None,
            "paired_bootstrap_95_ci": [percentile(samples, .025), percentile(samples, .975)],
            "improved": sum(d > 0 for d in deltas), "worsened": sum(d < 0 for d in deltas),
            "unchanged": sum(d == 0 for d in deltas),
            "queries_with_any_gold_missing_top20": sum(bool(r["missing_gold_in_top20"]) for r in evaluated),
            "queries_with_all_gold_missing_top20": sum(r["candidate"]["recall_at_20"] == 0 for r in evaluated),
            "service_failures": sum(r["status"] != "ok" for r in evaluated),
            "failure_policy": "attempted failures score zero; unrun queries excluded and counted",
            "clm_complete_at_5": mean("clm", "completeness_at_5"),
            "clm_precision_at_5": mean("clm", "precision_at_5"), "clm_f1": mean("selection", "f1"),
            **latency_summary([r["latency_seconds"] for r in evaluated if r["status"] == "ok"])}, evaluated


def prepare_trial(root, run):
    """Only pilot query text is retained; remote input contains no qrels."""
    from pilot import bm25_rank, digest, gold_map
    root, run = Path(root), Path(run)
    manifest = json.loads((root / "manifests/dataset.json").read_text())
    emb = json.loads((root / "manifests/embedding.json").read_text())
    pids = json.loads((root / "manifests/pilot_query_ids.json").read_text())
    final_ids = set(json.loads((root / "manifests/final_query_ids.json").read_text()))
    if len(pids) != 200 or len(set(pids)) != 200 or set(pids) & final_ids:
        raise ValueError("pilot/final reservation mismatch")
    for info in manifest["files"].values():
        if sha256(root / info["path"]) != info["sha256"]:
            raise ValueError("pinned dataset checksum mismatch")
    metadata_rows = read_jsonl(root / "data/metadata.jsonl")
    if digest(metadata_rows) != emb["metadata_hash"] or emb["dataset_revision"] != manifest["revision"]:
        raise ValueError("metadata provenance mismatch")
    metadata = {s["id"]: s["metadata"] for s in metadata_rows}
    wanted, queries, qrels = set(pids), {}, []
    with (root / "data/queries.jsonl").open() as f:
        for line in f:
            q = json.loads(line)
            if q["id"] in wanted:
                queries[q["id"]] = q["query"]
    with (root / "data/qrels.jsonl").open() as f:
        for line in f:
            q = json.loads(line)
            if q["query_id"] in wanted:
                qrels.append(q)
    gold = gold_map(qrels)
    start = time.perf_counter()
    ranks = bm25_rank(list(metadata.values()), list(metadata), [queries[q] for q in pids], k=20)
    wall = time.perf_counter() - start
    saved = {r["query_id"]: r for r in read_jsonl(root / "results/metadata_bm25_rows.jsonl")
             if r["run_id"] == emb["spec_hash"]}
    baselines, requests, changed = [], [], []
    for qid, ranked in zip(pids, ranks):
        ids = [p[0] for p in ranked]
        if ids[:5] != [p[0] for p in saved[qid]["ranking"][:5]]:
            changed.append(qid)
        baselines.append({"query_id": qid, "ranking": ids, "scores": [p[1] for p in ranked],
                          "evaluator_only_gold": sorted(gold[qid]),
                          "metrics": {**retrieval_metrics(ids, gold[qid], 5), **retrieval_metrics(ids, gold[qid], 20)}})
        requests.append(build_clm_candidates(queries[qid], ids, metadata, qid))
    if changed:
        write_new(run / "bm25-prefix-mismatch.json", changed)
        raise ValueError("BM25 top20 changed saved top5; inspect tie/prefix evidence")
    write_new(run / "inputs.json", {"cases": requests, "cache_library": [metadata[k] for k in sorted(metadata)[:1000]]})
    write_new(run / "evaluation_only.json", baselines)
    mean = lambda k: statistics.mean(x["metrics"][k] for x in baselines)
    baseline_summary = {"queries": 200, "corpus_size": len(metadata),
                        "bm25_recall_at_5": mean("recall_at_5"),
                        "bm25_recall_at_20": mean("recall_at_20"),
                        "top5_matches_saved_all_queries": True, "wall_seconds": wall}
    write_new(run / "bm25_baseline.json", baseline_summary)
    spec = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "PREPARED",
            "dataset_revision": manifest["revision"], "pilot_ids": pids,
            "metadata_hash": emb["metadata_hash"], "representation": emb["representation"],
            "inputs_sha256": sha256(run / "inputs.json"),
            "evaluation_sha256": sha256(run / "evaluation_only.json"),
            "question": QUESTION, "temperature": 1.0, "top_k": 5, "candidate_k": 20,
            "action_cache": "512MiB CPU; fresh process before main trial",
            "encoder": "Qwen/Qwen3-8B", "dtype": "bfloat16", "pooling": "LAST",
            "max_model_len": 2048, "truncate_prompt_tokens": "disabled; reject oversize",
            "quantization": None, "cpu_offload_gb": 0, "seed": 42,
            "source_hashes": {p: sha256(root / p) for p in ["pilot.py", "reference_metrics.py", "clm_reranker.py"]},
            "optional_direct": "NOT_RUN", "optional_vector": "NOT_RUN"}
    write_new(run / "run_manifest.json", spec)
    return baseline_summary


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
