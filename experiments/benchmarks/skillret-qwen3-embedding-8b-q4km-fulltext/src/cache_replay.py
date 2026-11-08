"""Measure verified cache-only execution and replay every official saved ranking.

This command makes no model/API request. It requires a completed primary run.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from core import (STUDY, SIZES, PREFIX, atomic_json, digest, file_hash, full_text,
                  jsonl, latency_summary, rank_skills)


def replay(cache):
    results = STUDY / "results"
    assert (results / "full_evaluation.json").exists(), "Complete the primary benchmark first"
    model = json.loads((results / "model.json").read_text())
    identity = {k: model[k] for k in ("protocol_sha256", "model_digest", "runtime", "normalization", "endpoint")}
    assert identity["protocol_sha256"] == file_hash(STUDY / "protocol.json")
    root = cache / "vectors" / digest(identity)

    def read(text, kind):
        key = digest({"text_sha256": digest(text), "identity": identity, "kind": kind})
        path = root / (key + ".npy")
        receipt = json.loads((root / (key + ".json")).read_text())
        assert receipt["text_sha256"] == digest(text)
        assert receipt["vector_sha256"] == file_hash(path)
        v = np.load(path, allow_pickle=False)
        assert v.shape == (4096,) and v.dtype == np.float32 and np.isfinite(v).all()
        assert abs(float(np.linalg.norm(v)) - 1) < 2e-5
        return v

    source_start = time.perf_counter()
    skills = jsonl(cache / "data/skills.jsonl")
    documents = jsonl(results / "documents.jsonl")
    pools = json.loads((results / "candidate_pools.json").read_text())
    id_index = {s["id"]: i for i, s in enumerate(skills)}
    source_seconds = time.perf_counter() - source_start
    # Reconstruct the exact original index row order, retaining every original ID.
    seen, order = set(), []
    for size in SIZES:
        for sid in pools[str(size)]["ids"]:
            if sid not in seen:
                seen.add(sid)
                order.append(id_index[sid])
    start = time.perf_counter()
    vectors, owners = [], []
    for i in order:
        doc = documents[i]
        text = full_text(skills[i])
        assert digest(text) == doc["source_sha256"]
        for c in doc["chunks"]:
            part = text[c["start"]:c["end"]]
            assert digest(part) == c["text_sha256"]
            vectors.append(read(part, "document"))
            owners.append(i)
    matrix = np.stack(vectors)
    owners = np.array(owners, dtype=np.int64)
    tie_keys = [d["source_sha256"] for d in documents]
    index_seconds = time.perf_counter() - start
    saved = {r["query_id"]: r for r in jsonl(results / "full_rankings.jsonl")}
    queries = jsonl(cache / "data/queries.jsonl")
    assert len(saved) == len(queries) == 4392
    timings = []
    for q in queries:
        start = time.perf_counter()
        v = read(PREFIX + q["query"], "query")
        read_seconds = time.perf_counter() - start
        search_start = time.perf_counter()
        indices, scores = rank_skills(matrix, owners, v, range(6006), tie_keys)
        search_seconds = time.perf_counter() - search_start
        ranked = [skills[i]["id"] for i in indices]
        assert ranked == saved[q["id"]]["ranking"], q["id"]
        assert np.allclose(scores, saved[q["id"]]["scores"], rtol=0, atol=1e-6)
        timings.append({"query_id": q["id"], "cache_read_s": read_seconds,
                        "search_s": search_seconds, "total_s": time.perf_counter() - start})
        if len(timings) % 500 == 0:
            print("cache replay", len(timings), flush=True)
    summary = {"status": "completed", "model_api_requests": 0, "skills": 6006,
               "chunks_loaded": len(vectors), "queries": 4392, "all_rankings_identical": True,
               "maximum_score_tolerance": 1e-6, "source_loading_s": source_seconds,
               "index_cache_validation_and_assembly_s": index_seconds,
               "index_cached_skills_per_second": 6006 / index_seconds,
               "query_cache_read": latency_summary([r["cache_read_s"] for r in timings]),
               "query_search": latency_summary([r["search_s"] for r in timings]),
               "query_total_cached": latency_summary([r["total_s"] for r in timings]),
               "model_digest": identity["model_digest"], "protocol_sha256": identity["protocol_sha256"]}
    atomic_json(results / "cache_replay.json", summary)
    with (results / "cache_replay_per_query.jsonl").open("w") as f:
        for row in timings:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/mia-qwen")
    replay(parser.parse_args().cache)
