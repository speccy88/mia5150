"""Fetch pinned public inputs outside Git; freeze audits before inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import urllib.request

import numpy as np
from tokenizers import Tokenizer

from core import STUDY, REPO, SIZES, PREFIX, atomic_json, chunks, digest, file_hash, full_text, jsonl

DATA_REV = "6583d7d2ed07644d0fb8938ed8178f3a7dc42a12"
TOKENIZER_REV = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
HASHES = {"skills": "1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81",
          "queries": "4d07a35f40e597f8468c6dab480c95a1f9b40ecbde80274081ba2e7c643d0f6d",
          "qrels": "edbfae0b8109de460d54e27e6235b1b47316f34956744db0571ad1b4ddb860af"}


def prepare(cache):
    cache = Path(cache)
    for name, expected in HASHES.items():
        target = cache / "data" / f"{name}.jsonl"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".download")
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/{DATA_REV}/data/{name}/test.jsonl", tmp)
            if file_hash(tmp) != expected:
                raise ValueError(f"Downloaded {name} hash mismatch")
            tmp.replace(target)
        if file_hash(target) != expected:
            raise ValueError(f"Pinned {name} hash mismatch")
    tok_path = cache / "tokenizer/tokenizer.json"
    if not tok_path.exists():
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://huggingface.co/Qwen/Qwen3-Embedding-8B/resolve/{TOKENIZER_REV}/tokenizer.json", tok_path)
    tokenizer = Tokenizer.from_file(str(tok_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    skills = jsonl(cache / "data/skills.jsonl")
    queries = jsonl(cache / "data/queries.jsonl")
    labels = jsonl(cache / "data/qrels.jsonl")
    assert len(skills) == len({s["id"] for s in skills}) == 6006
    assert len(queries) == len({q["id"] for q in queries}) == 4392
    assert len(labels) == 7187
    ids = {s["id"] for s in skills}
    qrels = {}
    for row in labels:
        qid, sid = str(row["query_id"]), str(row["skill_id"])
        assert sid in ids and sid not in qrels.setdefault(qid, {})
        qrels[qid][sid] = row["relevance"]
    assert len(qrels) == 4392 and all(set(q["skill_ids"]) == set(qrels[str(q["id"])]) for q in queries)
    panel_path = REPO / "experiments/benchmarks/skill-discovery-oauth-v1-20261103/data/test_queries.json"
    panel = json.loads(panel_path.read_text())
    assert len(panel) == 20
    historical_path = REPO / "experiments/benchmarks/skill-discovery-oauth-v1-20261103/results/gpt-6.1-sol/manifest.json"
    historical = json.loads(historical_path.read_text())["plan"]
    pools = {}
    for n in SIZES:
        if n != 6006:
            previous = [r["candidate_ids"] for r in historical if r["key"].startswith(str(n) + ":")]
            assert len(previous) == 20 and all(p == previous[0] for p in previous)
            pool = previous[0]
        else:
            pool = [s["id"] for s in random.Random(42).sample(skills, n)]
        assert len(pool) == len(set(pool)) == n and set(pool) <= ids
        assert all(q["gold_id"] in pool for q in panel)
        pools[str(n)] = {"ids": pool, "ordered_ids_sha256": digest(pool),
                         "source": "new seed42 sample, full catalog" if n == 6006 else "exact historical manifest IDs and order",
                         "gold_positions": [pool.index(q["gold_id"]) + 1 for q in panel], "absent_gold": 0}
    atomic_json(STUDY / "results/candidate_pools.json", pools)
    records = []
    for i, s in enumerate(skills):
        text = full_text(s)
        c = chunks(text, tokenizer)
        n = len(tokenizer.encode(text, add_special_tokens=False).ids)
        records.append({"skill_id": s["id"], "source_index": i, "source_sha256": digest(text),
                        "body_sha256": digest(s["skill_md"]), "source_characters": len(text),
                        "source_utf8_bytes": len(text.encode()), "original_tokens": n,
                        "covered_characters": len(text), "chunks": c})
        if (i + 1) % 500 == 0:
            print("audited", i + 1, flush=True)
    out = STUDY / "results/documents.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    lengths = np.array([r["original_tokens"] for r in records])
    audit = {"skills": 6006, "full_queries": 4392, "qrel_pairs": 7187, "panel_queries": 20,
             "original_tokens_total": int(lengths.sum()), "original_tokens_mean": float(lengths.mean()),
             "original_tokens_median": float(np.median(lengths)), "original_tokens_max": int(lengths.max()),
             "skills_requiring_chunking": sum(len(r["chunks"]) > 1 for r in records),
             "chunks": sum(len(r["chunks"]) for r in records),
             "chunk_tokens_including_overlap": sum(c["tokens"] for r in records for c in r["chunks"]),
             "source_characters": sum(r["source_characters"] for r in records),
             "source_coverage_percent": 100.0, "missing_documents": 0,
             "qrel_set_size_counts": dict(Counter(len(v) for v in qrels.values())),
             "duplicate_names": len(skills) - len({s["name"] for s in skills}),
             "tokenizer_sha256": file_hash(tok_path), "documents_manifest_sha256": file_hash(out),
             "data_sha256": HASHES, "panel_sha256": file_hash(panel_path),
             "historical_manifest_sha256": file_hash(historical_path)}
    atomic_json(STUDY / "results/input_audit.json", audit)
    protocol = {"version": 1, "model": "qwen3-embedding:8b-q4_K_M", "quantization": "Q4_K_M",
                "model_manifest_sha256": "64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b",
                "model_blob_sha256": "3fcd3febec8b3fd64435204db75bf0dd73b91e8d0661e0331acfe7e7c3120b85",
                "dimensions": 4096, "api": "/api/embed", "truncate": False, "batch_size": 1,
                "options": {"num_ctx": 4096, "num_batch": 512, "num_thread": 2, "seed": 42},
                "query_prefix": PREFIX, "document_format": "name + ' | ' + description + ' | ' + complete skill_md; no stripping",
                "chunking": {"version": 1, "token_budget": 4094, "overlap_tokens": 128,
                             "markdown_boundary_lookback_tokens": 256, "source_spans": "Unicode character offsets; exact source slices"},
                "aggregation": "maximum chunk cosine per skill, deduplicate before top50",
                "normalization": "float32 L2", "ties": "ascending SHA256 of exact full text; no pool-position/ID/gold features",
                "ranking_depth": 50, "sizes": list(SIZES), "new_sample_seed": 42,
                "dataset": "ThakiCloud/SKILLRET", "dataset_revision": DATA_REV,
                "data_sha256": HASHES, "tokenizer_revision": TOKENIZER_REV,
                "tokenizer_sha256": file_hash(tok_path), "documents_manifest_sha256": file_hash(out),
                "source_git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "bootstrap": {"seed": 42, "resamples": 10000, "unit": "paired question cluster across sizes"},
                "relevance": {"scaling": "single gold skill ID; all 20 queries retained", "official": "all 7187 binary qrels; denominator 4392"},
                "optional_metadata_ablation": "unattempted; primary full-text priority"}
    target = STUDY / "protocol.json"
    if target.exists():
        frozen = json.loads(target.read_text())
        # A published/reproduction checkout has a new Git HEAD. Keep the
        # original source receipt while requiring identical scientific inputs.
        protocol["source_git_revision"] = frozen["source_git_revision"]
        if frozen != protocol:
            raise ValueError("Frozen protocol mismatch; preserve prior run")
    atomic_json(target, protocol)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/mia-qwen")
    prepare(parser.parse_args().cache)
