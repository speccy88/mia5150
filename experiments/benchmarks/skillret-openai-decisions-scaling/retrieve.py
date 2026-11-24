"""October matched retrieval with exact first-stage scores preserved."""

import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

os.environ.update(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    TOKENIZERS_PARALLELISM="false",
)

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OCTOBER = HERE.parent / "skillret-gap-closure-20261024"
sys.path.insert(0, str(OCTOBER))
try:
    from core import digest, library, rank, render, rrf
    from fast_bm25 import SparseBM25
finally:
    sys.path.pop(0)


class ScoredBM25(SparseBM25):
    """Use October's sparse postings and expose its per-catalog scores."""

    def scored_search(self, query, ids, depth):
        global_index = np.fromiter((self.pos[skill_id] for skill_id in ids), dtype=np.int64, count=len(ids))
        lengths = self.lengths[global_index]
        denominator = 1.5 * (0.25 + 0.75 * lengths / (lengths.mean() or 1))
        scores = np.zeros(len(ids))
        local_index = np.full(len(self.ids), -1, dtype=np.int64)
        local_index[global_index] = np.arange(len(ids))
        for term, query_frequency in Counter(re.findall(r"\w+", query.casefold())).items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            positions = local_index[posting[0]]
            present = positions >= 0
            positions = positions[present]
            frequencies = posting[1][present]
            document_frequency = len(positions)
            idf = math.log1p((len(ids) - document_frequency + 0.5) / (document_frequency + 0.5))
            scores[positions] += (
                query_frequency * idf * frequencies * 2.5 / (frequencies + denominator[positions])
            )
        by_id = dict(zip(ids, map(float, scores), strict=True))
        return rank(by_id, ids, depth), by_id


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def model_assets(spec, directory):
    """Download only pinned public assets and verify bytes before use."""
    directory.mkdir(parents=True, exist_ok=True)
    for relative_path, expected_hash in spec["assets"].items():
        path = directory / Path(relative_path).name
        if not path.exists():
            request = urllib.request.Request(
                f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{relative_path}"
            )
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                temporary = Path(output.name)
                try:
                    with urllib.request.urlopen(request, timeout=120) as source:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
                    output.flush()
                    if sha256(temporary) != expected_hash:
                        raise ValueError(f"Pinned embedding asset checksum mismatch: {path.name}")
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        if sha256(path) != expected_hash:
            raise ValueError(f"Pinned embedding asset checksum mismatch: {path.name}")


class Encoder:
    """Use the frozen October ONNX model, tokenizer and FP32 CPU settings."""

    def __init__(self, spec, assets):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.spec = spec
        self.tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=spec["max_tokens"])
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(str(assets / "model.onnx"), options, providers=["CPUExecutionProvider"])
        self.inputs = {input_spec.name for input_spec in self.session.get_inputs()}

    def encode(self, texts, *, query):
        if query:
            texts = [self.spec["query_prefix"] + text for text in texts]
        encoded = self.tokenizer.encode_batch(texts)
        fields = {"input_ids": "ids", "attention_mask": "attention_mask", "token_type_ids": "type_ids"}
        inputs = {
            key: np.asarray([getattr(item, field) for item in encoded], dtype=np.int64)
            for key, field in fields.items() if key in self.inputs
        }
        output, = self.session.run(["last_hidden_state"], inputs)
        cls = output[:, 0, :]
        normalized = cls / np.linalg.norm(cls, axis=1, keepdims=True)
        if normalized.shape != (len(texts), 384) or normalized.dtype != np.float32 or not np.isfinite(normalized).all():
            raise ValueError("Pinned embedding model returned invalid normalized vectors")
        return normalized


def cached_embeddings(encoder, texts, query, directory, name):
    """Resume encoding from the last complete batch of 128 inputs."""
    identity = digest([encoder.spec, query, texts])
    path = directory / f"{name}-{identity}.npy"
    progress = directory / f"{name}-{identity}.progress"
    completed = int(progress.read_text()) if progress.is_file() and path.is_file() else 0
    if completed < 0 or completed > len(texts):
        raise ValueError("Embedding progress outside input bounds")
    vectors = np.lib.format.open_memmap(
        path, mode="r+" if completed else "w+", dtype="float32", shape=(len(texts), 384),
    )
    if vectors.shape != (len(texts), 384) or vectors.dtype != np.float32:
        raise ValueError("Cached embedding shape or dtype mismatch")
    batch_size = encoder.spec["batch_size"]
    started = time.perf_counter()
    for start in range(completed, len(texts), 128):
        stop = min(start + 128, len(texts))
        for offset in range(start, stop, batch_size):
            end = min(offset + batch_size, stop)
            vectors[offset:end] = encoder.encode(texts[offset:end], query=query)
        vectors.flush()
        temporary = progress.with_suffix(".tmp")
        temporary.write_text(str(stop))
        temporary.replace(progress)
        if stop == len(texts) or stop % 1024 == 0:
            print(f"Encoded {name}: {stop:,}/{len(texts):,}", flush=True)
    return vectors, {"sha256": sha256(path), "elapsed_seconds": time.perf_counter() - started, "reused_rows": completed}


def historical_embeddings(directory, document_ids, query_ids):
    """Verify both saved October arrays before using their exact vectors."""
    manifest = json.loads((OCTOBER / "embedding-full-manifest.json").read_text(encoding="utf-8"))
    vectors = {}
    for name in ("docs.npy", "queries-full.npy"):
        path = directory / name
        if sha256(path) != manifest["arrays"][name]:
            raise ValueError(f"October embedding array checksum mismatch: {name}")
        vectors[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    document_positions = {skill_id: i for i, skill_id in enumerate(manifest["document_ids"])}
    query_positions = {query_id: i for i, query_id in enumerate(manifest["query_ids"])}
    docs = vectors["docs.npy"][[document_positions[skill_id] for skill_id in document_ids]]
    queries = vectors["queries-full.npy"][[query_positions[query_id] for query_id in query_ids]]
    return docs, queries, manifest["arrays"]


def retrieve(config, documents, queries, memberships, embeddings=None):
    """Yield ranked candidate rows; dense timings include a batch-one query forward if measured."""
    bm25 = ScoredBM25(documents)
    positions = {skill_id: i for i, skill_id in enumerate(documents)}
    dense_positions = {query_id: i for i, query_id in enumerate(queries)}
    depth = config["fusion"]["source_depth"]
    constant = config["fusion"]["rrf_constant"]
    with threadpool_limits(limits=1, user_api="blas"):
        for membership in memberships:
            query_id = membership["query_id"]
            query = queries[query_id]
            skill_ids = membership["skill_ids"]
            started = time.perf_counter()
            bm_rank, bm_scores = bm25.scored_search(query, skill_ids, depth)
            bm_ms = (time.perf_counter() - started) * 1000
            methods = [("bm25", bm_rank, bm_scores, bm_ms, 0.0)]
            if embeddings is not None:
                doc_vectors, query_vectors, query_timings = embeddings
                started = time.perf_counter()
                selected = [positions[skill_id] for skill_id in skill_ids]
                similarities = doc_vectors[selected] @ query_vectors[dense_positions[query_id]]
                vector_scores = dict(zip(skill_ids, map(float, similarities), strict=True))
                vector_rank = rank(vector_scores, skill_ids, depth)
                vector_ms = (time.perf_counter() - started) * 1000

                started = time.perf_counter()
                hybrid_rank = rrf(bm_rank, vector_rank, k=constant)
                hybrid_scores = {}
                for ranking in (bm_rank, vector_rank):
                    for position, skill_id in enumerate(ranking, 1):
                        hybrid_scores[skill_id] = hybrid_scores.get(skill_id, 0) + 1 / (constant + position)
                fusion_ms = (time.perf_counter() - started) * 1000
                forward_ms = query_timings[query_id]
                methods.extend([
                    ("vector", vector_rank, vector_scores, vector_ms, forward_ms),
                    ("hybrid", hybrid_rank, hybrid_scores, bm_ms + vector_ms + fusion_ms, forward_ms),
                ])
            for method, ordered, scores, search_ms, forward_ms in methods:
                candidate_count = config["candidate_count"]
                if len(ordered) < candidate_count:
                    raise ValueError("Catalog produces too few candidates")
                case_id = digest([
                    config["version"], config["dataset"]["revision"], query_id,
                    membership["catalog_size"], membership["seed"], method,
                ])
                yield {
                    "case_id": case_id, "query_id": query_id, "query": query,
                    "catalog_ref": membership["catalog_ref"], "catalog_size": membership["catalog_size"],
                    "seed": membership["seed"], "method": method,
                    "retrieval_latency_ms": search_ms + (forward_ms or 0.0),
                    "retrieval_search_ms": search_ms, "query_embedding_ms": forward_ms,
                    "candidates": [
                        {"id": skill_id, "score": scores[skill_id], "rank": position, "metadata": documents[skill_id]}
                        for position, skill_id in enumerate(ordered[:candidate_count], 1)
                    ],
                }
