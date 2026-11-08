"""Export lossless public vectors; replay all retrieval without model or network."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from core import (STUDY, SIZES, PREFIX, atomic_json, digest, evaluate, file_hash,
                  full_text, jsonl, rank_skills)
from validate import compare_evaluation, independent

VECTORS = STUDY / "vectors"
ARCHIVE = VECTORS / "index-v1.npz"


def checked_vector(root, text_hash, identity, kind):
    key = digest({"text_sha256": text_hash, "identity": identity, "kind": kind})
    path = root / (key + ".npy")
    meta = json.loads((root / (key + ".json")).read_text())
    assert meta["text_sha256"] == text_hash and meta["vector_sha256"] == file_hash(path)
    vector = np.load(path, allow_pickle=False)
    assert vector.dtype == np.float32 and vector.shape == (4096,)
    assert np.isfinite(vector).all() and abs(float(np.linalg.norm(vector)) - 1) < 2e-5
    return vector


def write_archive(path, arrays):
    """Fixed ZIP metadata; numeric/string NPY members only, never pickle."""
    temporary = path.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, array in sorted(arrays.items()):
            assert array.dtype.kind in "fiuU" and not array.dtype.hasobject
            data = io.BytesIO()
            np.save(data, array, allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data.getvalue(), compresslevel=6)
    temporary.replace(path)


def export(cache):
    results = STUDY / "results"
    assert json.loads((results / "progress.json").read_text())["stage"] == "completed"
    model = json.loads((results / "model.json").read_text())
    identity = {k: model[k] for k in ("protocol_sha256", "model_digest", "runtime", "normalization", "endpoint")}
    assert identity["protocol_sha256"] == file_hash(STUDY / "protocol.json")
    root = cache / "vectors" / digest(identity)
    skills = jsonl(cache / "data/skills.jsonl")
    documents = jsonl(results / "documents.jsonl")
    queries = jsonl(cache / "data/queries.jsonl")
    qrels = jsonl(cache / "data/qrels.jsonl")
    panel = json.loads((STUDY.parents[2] / "experiments/benchmarks/skill-discovery-oauth-v1-20261103/data/test_queries.json").read_text())
    pools = json.loads((results / "candidate_pools.json").read_text())
    skill_index = {s["id"]: i for i, s in enumerate(skills)}
    query_index = {q["id"]: i for i, q in enumerate(queries)}
    seen, order = set(), []
    for size in SIZES:
        for sid in pools[str(size)]["ids"]:
            if sid not in seen:
                seen.add(sid)
                order.append(skill_index[sid])
    vectors, owners, hashes = [], [], []
    for i in order:
        doc = documents[i]
        assert digest(full_text(skills[i])) == doc["source_sha256"]
        for chunk in doc["chunks"]:
            vectors.append(checked_vector(root, chunk["text_sha256"], identity, "document"))
            owners.append(i)
            hashes.append(chunk["text_sha256"])
    arrays = {
        "document_vectors": np.stack(vectors),
        "chunk_skill_indices": np.asarray(owners, dtype=np.int32),
        "chunk_text_sha256": np.asarray(hashes),
        "skill_ids": np.asarray([s["id"] for s in skills]),
        "skill_source_sha256": np.asarray([d["source_sha256"] for d in documents]),
        "official_query_vectors": np.stack([checked_vector(root, digest(PREFIX + q["query"]), identity, "query") for q in queries]),
        "official_query_ids": np.asarray([q["id"] for q in queries]),
        "official_query_sha256": np.asarray([digest(q["query"]) for q in queries]),
        "panel_query_vectors": np.stack([checked_vector(root, digest(PREFIX + q["query"]), identity, "query") for q in panel]),
        "panel_query_ids": np.asarray([f"panel-{i:02d}" for i in range(20)]),
        "panel_gold_ids": np.asarray([q["gold_id"] for q in panel]),
        "qrel_query_indices": np.asarray([query_index[q["query_id"]] for q in qrels], dtype=np.int32),
        "qrel_skill_indices": np.asarray([skill_index[q["skill_id"]] for q in qrels], dtype=np.int32),
        "qrel_relevance": np.asarray([q["relevance"] for q in qrels], dtype=np.uint8),
    }
    VECTORS.mkdir(exist_ok=True)
    write_archive(ARCHIVE, arrays)
    atomic_json(VECTORS / "manifest.json", {
        "format": "numpy NPZ, lossless ZIP; no pickle; float32 unit vectors",
        "file": ARCHIVE.name, "sha256": file_hash(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
        "model_digest": identity["model_digest"], "protocol_sha256": identity["protocol_sha256"],
        "skills": 6006, "chunks": 6913, "official_queries": 4392, "scaling_queries": 20, "qrels": 7187,
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256_contiguous_bytes": digest(np.ascontiguousarray(value).tobytes())}
                   for key, value in arrays.items()},
        "raw_source_text_included": False, "model_weights_included": False,
        "qrels_source_sha256": json.loads((STUDY / "protocol.json").read_text())["data_sha256"]["qrels"],
    })
    print("exported", ARCHIVE.name, ARCHIVE.stat().st_size, "bytes", flush=True)


def load_archive(path, manifest):
    assert file_hash(path) == manifest["sha256"], "Missing LFS payload or corrupt vector archive"
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(manifest["arrays"])
        arrays = {key: archive[key] for key in archive.files}
    for key, value in arrays.items():
        expected = manifest["arrays"][key]
        assert list(value.shape) == expected["shape"] and str(value.dtype) == expected["dtype"]
        assert digest(np.ascontiguousarray(value).tobytes()) == expected["sha256_contiguous_bytes"]
    for key in ("document_vectors", "official_query_vectors", "panel_query_vectors"):
        values = arrays[key]
        assert values.dtype == np.float32 and values.shape[1] == 4096 and np.isfinite(values).all()
        assert np.max(np.abs(np.linalg.norm(values, axis=1) - 1)) < 2e-5
    return arrays


def replay():
    started = time.perf_counter()
    manifest = json.loads((VECTORS / "manifest.json").read_text())
    assert manifest["protocol_sha256"] == file_hash(STUDY / "protocol.json")
    arrays = load_archive(ARCHIVE, manifest)
    loading = time.perf_counter() - started
    ids = arrays["skill_ids"].tolist()
    qids = arrays["official_query_ids"].tolist()
    assert len(ids) == len(set(ids)) == 6006 and len(qids) == len(set(qids)) == 4392
    assert arrays["document_vectors"].shape == (6913, 4096)
    index = {sid: i for i, sid in enumerate(ids)}
    owners = arrays["chunk_skill_indices"]
    assert set(owners.tolist()) == set(range(6006))
    pools = json.loads((STUDY / "results/candidate_pools.json").read_text())
    ties = arrays["skill_source_sha256"]
    scaling_count = 0
    for size in SIZES:
        saved = json.loads((STUDY / f"results/scaling-n{size}.json").read_text())
        # Match the original accumulating index and its exact row order.
        active = set()
        for n in SIZES:
            active.update(index[sid] for sid in pools[str(n)]["ids"])
            if n == size:
                break
        mask = np.isin(owners, list(active))
        matrix, selected_owners = arrays["document_vectors"][mask], owners[mask]
        rankings, labels = {}, {}
        for qi, qid in enumerate(arrays["panel_query_ids"].tolist()):
            selected, scores = rank_skills(matrix, selected_owners, arrays["panel_query_vectors"][qi],
                                          [index[sid] for sid in pools[str(size)]["ids"]], ties)
            ranked = [ids[i] for i in selected]
            assert ranked == saved["rows"][qi]["ranking"]
            assert np.allclose(scores, saved["rows"][qi]["scores"], rtol=0, atol=1e-6)
            rankings[qid] = ranked
            labels[qid] = {str(arrays["panel_gold_ids"][qi]): 1}
            scaling_count += 1
        rows = {qid: independent(gold, rankings[qid]) for qid, gold in labels.items()}
        compare_evaluation(saved["evaluation"], rows, labels, rankings)
    saved = {r["query_id"]: r for r in jsonl(STUDY / "results/full_rankings.jsonl")}
    labels = {qid: {} for qid in qids}
    assert len(arrays["qrel_relevance"]) == 7187 and np.all(arrays["qrel_relevance"] == 1)
    for qi, si, relevance in zip(arrays["qrel_query_indices"], arrays["qrel_skill_indices"], arrays["qrel_relevance"]):
        assert ids[si] not in labels[qids[qi]]
        labels[qids[qi]][ids[si]] = int(relevance)
    rankings = {}
    for qi, qid in enumerate(qids):
        selected, scores = rank_skills(arrays["document_vectors"], owners, arrays["official_query_vectors"][qi], range(6006), ties)
        rankings[qid] = [ids[i] for i in selected]
        assert rankings[qid] == saved[qid]["ranking"]
        assert np.allclose(scores, saved[qid]["scores"], rtol=0, atol=1e-6)
        if (qi + 1) % 500 == 0:
            print("published replay", qi + 1, flush=True)
    evaluation = evaluate(labels, rankings)
    original = json.loads((STUDY / "results/full_evaluation.json").read_text())
    assert evaluation == original
    compare_evaluation(original, {qid: independent(gold, rankings[qid]) for qid, gold in labels.items()}, labels, rankings)
    receipt = {"status": "completed", "skills": 6006, "chunks": 6913, "official_queries": 4392,
               "scaling_cases": scaling_count, "model_api_requests": 0, "network_requests": 0,
               "original_cache_files_used": 0, "all_rankings_identical": True,
               "all_metrics_identical": True, "maximum_score_tolerance": 1e-6,
               "archive_loading_validation_s": loading, "total_wall_s": time.perf_counter() - started,
               "archive_sha256": manifest["sha256"]}
    atomic_json(STUDY / "results/published_vector_replay.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("export", "replay"))
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/mia-qwen")
    args = parser.parse_args()
    export(args.cache) if args.action == "export" else replay()
