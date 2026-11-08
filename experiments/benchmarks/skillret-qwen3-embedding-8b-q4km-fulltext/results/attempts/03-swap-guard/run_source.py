"""Resumable local Ollama execution; no hosted inference or raw-text exports."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import threading
import time

import numpy as np
import psutil
import requests
from tokenizers import Tokenizer

from core import (STUDY, REPO, PREFIX, SIZES, atomic_json, bootstrap, digest, evaluate,
                  file_hash, full_text, jsonl, latency_summary, rank_skills, unit)

BASE = "http://127.0.0.1:11434"
RESULTS = STUDY / "results"


def utc():
    return datetime.now(timezone.utc).isoformat()


def public_runtime(value):
    """Remove home-directory identifiers from vendor/runtime metadata exports."""
    if isinstance(value, dict):
        return {k: public_runtime(v) for k, v in value.items()}
    if isinstance(value, list):
        return [public_runtime(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"/(?:home|Users)/[^/\s]+", "<user-home>", value)
    return value


def api(route, payload=None, timeout=900):
    r = requests.get(BASE + route, timeout=timeout) if payload is None else requests.post(BASE + route, json=payload, timeout=timeout)
    if not r.ok:
        # Do not export server messages that could echo source input.
        raise RuntimeError(f"Ollama {route} HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


class Monitor:
    def __init__(self, stage):
        self.stage = stage
        self.protocol_sha256 = file_hash(STUDY / "protocol.json")
        self.done = threading.Event()
        self.problem = None
        self.swap_start = psutil.swap_memory()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def sample(self):
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], text=True)
        gpu = [float(x.strip()) for x in raw.strip().split(",")]
        row = {"utc": utc(), "stage": self.stage, "protocol_sha256": self.protocol_sha256, "gpu_memory_mib": gpu[0],
               "gpu_utilization_percent": gpu[1], "gpu_temperature_c": gpu[2], "gpu_power_w": gpu[3],
               "system_used_bytes": mem.used, "system_available_bytes": mem.available,
               "swap_used_bytes": swap.used, "swap_in_bytes": swap.sin, "swap_out_bytes": swap.sout}
        if mem.available < 768 * 1024**2:
            self.problem = "Available system RAM below 768 MiB"
        if swap.sin > self.swap_start.sin or swap.sout > self.swap_start.sout:
            self.problem = "Swap activity increased"
        if gpu[0] > 7680 or gpu[2] > 80:
            self.problem = "GPU memory or temperature guard"
        return row

    def loop(self):
        with (RESULTS / "telemetry.jsonl").open("a") as f:
            while not self.done.is_set():
                try:
                    f.write(json.dumps(self.sample()) + "\n")
                    f.flush()
                except Exception as e:
                    self.problem = "Telemetry unavailable: " + type(e).__name__
                self.done.wait(1)

    def check(self):
        if self.problem:
            raise RuntimeError(self.problem)

    def close(self):
        self.done.set()
        self.thread.join()


class Encoder:
    def __init__(self, cache, protocol, monitor):
        self.cache = cache
        self.protocol = protocol
        self.monitor = monitor
        tags = api("/api/tags")["models"]
        entry = next(x for x in tags if x["name"] == protocol["model"])
        assert entry["digest"].removeprefix("sha256:") == protocol["model_manifest_sha256"]
        show = api("/api/show", {"model": protocol["model"]})
        assert show["details"]["quantization_level"] == "Q4_K_M"
        assert show["model_info"]["qwen3.embedding_length"] == 4096
        assert "embedding" in show["capabilities"]
        self.identity = {"protocol_sha256": file_hash(STUDY / "protocol.json"),
                         "model_digest": entry["digest"], "runtime": api("/api/version"),
                         "normalization": "float32 L2", "endpoint": "/api/embed"}
        previous = RESULTS / "model.json"
        if previous.exists():
            old = json.loads(previous.read_text())
            if any(old[k] != v for k, v in self.identity.items()):
                raise ValueError("Model/runtime/protocol changed; preserve the frozen run")
        self.root = cache / "vectors" / digest(self.identity)
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_json(self.root / "identity.json", self.identity)
        atomic_json(RESULTS / "model.json", public_runtime({**self.identity, "details": show["details"],
                    "model_info": show["model_info"], "model_size_bytes": entry["size"], "runtime_metadata_paths_redacted": True}))
        self.tokenizer = Tokenizer.from_file(str(cache / "tokenizer/tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def native_token_count(self, text):
        """Audit full input with this model's local GGUF tokenizer on anomalies.

        Ollama parses native special-token spellings, which the HF audit
        tokenizer can split. This endpoint tokenizes only; inference stays on
        /api/embed. No endpoint outside loopback is ever used.
        """
        matches = []
        blob = self.protocol["model_blob_sha256"]
        for process in psutil.process_iter(["cmdline"]):
            args = process.info["cmdline"] or []
            if (args and args[0].endswith("/llama-server") and "--model" in args
                    and args[args.index("--model") + 1].endswith(blob) and "--port" in args):
                matches.append(args[args.index("--port") + 1])
        if len(matches) != 1:
            raise ValueError("Cannot identify the model's local native tokenizer")
        response = requests.post("http://127.0.0.1:" + matches[0] + "/tokenize",
                                 json={"content": text, "add_special": True, "parse_special": True}, timeout=30)
        response.raise_for_status()
        return len(response.json()["tokens"])

    def encode(self, text, kind, use_cache=True):
        key = digest({"text_sha256": digest(text), "identity": self.identity, "kind": kind})
        path = self.root / (key + ".npy")
        meta = self.root / (key + ".json")
        if use_cache and meta.exists() and path.exists():
            t = time.perf_counter()
            receipt = json.loads(meta.read_text())
            if file_hash(path) != receipt["vector_sha256"] or receipt["text_sha256"] != digest(text):
                raise ValueError("Corrupt cached embedding")
            v = np.load(path, allow_pickle=False)
            if v.dtype != np.float32 or not np.isfinite(v).all() or abs(float(np.linalg.norm(v)) - 1) > 2e-5:
                raise ValueError("Invalid cached vector")
            assert v.shape == (4096,)
            return v, {**receipt, "cache_hit": True, "cache_read_s": time.perf_counter() - t}
        self.monitor.check()
        t = time.perf_counter()
        response = api("/api/embed", {"model": self.protocol["model"], "input": text,
                       "truncate": False, "dimensions": 4096, "keep_alive": "30m",
                       "options": self.protocol["options"]})
        elapsed = time.perf_counter() - t
        self.monitor.check()
        assert len(response["embeddings"]) == 1
        v = unit(response["embeddings"][0])
        original_tokens = len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        actual = response["prompt_eval_count"]
        # Ollama may add EOS framing. Fewer evaluated tokens means a parity error.
        native_verified = None
        if actual != original_tokens + 1:
            native_verified = self.native_token_count(text)
            if native_verified != actual:
                raise ValueError(f"Native tokenizer/runtime token mismatch: native={native_verified}, runtime={actual}")
        receipt = {"key": key, "text_sha256": digest(text), "kind": kind,
                   "latency_s": elapsed, "original_tokens": original_tokens, "prompt_eval_count": actual,
                   "runtime_total_s": response["total_duration"] / 1e9,
                   "runtime_load_s": response["load_duration"] / 1e9, "utc": utc(),
                   "normalization_dtype": "float32", "cache_hit": False,
                   "native_tokenizer_verified_count": native_verified}
        if use_cache:
            with path.with_suffix(".tmp").open("wb") as f:
                np.save(f, v, allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            path.with_suffix(".tmp").replace(path)
            receipt["vector_sha256"] = file_hash(path)
            atomic_json(meta, receipt)
        with (RESULTS / "embedding_calls.jsonl").open("a") as f:
            f.write(json.dumps(receipt) + "\n")
        return v, receipt


def hardware():
    return {"os": platform.freedesktop_os_release()["PRETTY_NAME"], "python": platform.python_version(),
            "cpu": subprocess.check_output(["lscpu"], text=True),
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"], text=True).strip(),
            "system_ram_bytes": psutil.virtual_memory().total,
            "swap_bytes": psutil.swap_memory().total,
            "cuda_toolkit_compiler": "not installed; Ollama bundles CUDA inference libraries",
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "requests", "tokenizers", "psutil", "matplotlib")},
            "ollama_environment": {"OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_NO_CLOUD": "1"},
            "gpu_workload_preflight": "No compute processes; 9 MiB total device use at initial inspection",
            "memory_type": "DDR3 reported by user; DMI inspection requires unavailable root access"}


def feasibility(encoder, skills, documents):
    probes = [min(documents, key=lambda r: r["original_tokens"]),
              min(documents, key=lambda r: abs(r["original_tokens"] - 1500)),
              max(documents, key=lambda r: max(c["tokens"] for c in r["chunks"]))]
    rows = []
    for doc in probes:
        c = max(doc["chunks"], key=lambda r: r["tokens"])
        text = full_text(skills[doc["source_index"]])[c["start"]:c["end"]]
        v, receipt = encoder.encode(text, "feasibility", use_cache=False)
        rows.append({"skill_id": doc["skill_id"], **receipt, "norm": float(np.linalg.norm(v)),
                     "resident_models": public_runtime(api("/api/ps"))})
        atomic_json(RESULTS / "feasibility_progress.json", {"status": "partial", "probes": rows})
        print("feasibility", receipt["original_tokens"], receipt["latency_s"], flush=True)
    v2, rep = encoder.encode(text, "feasibility_repeat", use_cache=False)
    maximum = float(np.abs(v - v2).max())
    if maximum > 1e-4:
        raise ValueError("Repeat embedding determinism check failed")
    initial_cold = RESULTS / "attempts/01-admission-boundary/embedding_calls.jsonl"
    cold = jsonl(initial_cold)[0] if initial_cold.exists() else rows[0]
    atomic_json(RESULTS / "feasibility.json", {"status": "passed", "probes": rows,
                "repeat_max_abs_difference": maximum, "repeat": rep,
                "cold_start_load_s": cold["runtime_load_s"],
                "cold_start_first_embedding_s": cold["latency_s"],
                "cold_start_source": "first genuine unloaded-model request; retained admission-boundary attempt" if initial_cold.exists() else "this attempt",
                "model_reconfiguration_load_s_this_attempt": rows[0]["runtime_load_s"],
                "context": 4096, "batch_size": 1, "context_increased": False})


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(cache, stage):
    started = time.monotonic()
    monitor = Monitor(stage)
    try:
        protocol = json.loads((STUDY / "protocol.json").read_text())
        sources = {str(p.relative_to(STUDY)): file_hash(p) for p in sorted((STUDY / "src").glob("*.py"))}
        atomic_json(RESULTS / f"source-{digest(sources)[:16]}.json", sources)
        atomic_json(RESULTS / "hardware.json", hardware())
        skills = jsonl(cache / "data/skills.jsonl")
        documents = jsonl(RESULTS / "documents.jsonl")
        assert len(skills) == len(documents) == 6006
        assert file_hash(RESULTS / "documents.jsonl") == protocol["documents_manifest_sha256"]
        encoder = Encoder(cache, protocol, monitor)
        if stage == "feasibility" or not (RESULTS / "feasibility.json").exists():
            feasibility(encoder, skills, documents)
        if stage == "feasibility":
            return
        id_to_index = {s["id"]: i for i, s in enumerate(skills)}
        tie_keys = np.asarray([d["source_sha256"] for d in documents])
        assert len(set(tie_keys.tolist())) == 6006
        pools = json.loads((RESULTS / "candidate_pools.json").read_text())
        panel = json.loads((REPO / "experiments/benchmarks/skill-discovery-oauth-v1-20261103/data/test_queries.json").read_text())
        query_vectors = []
        query_receipts = []
        for q in panel:
            v, r = encoder.encode(PREFIX + q["query"], "query")
            query_vectors.append(v)
            query_receipts.append(r)
        vectors, chunk_owners, completed = [], [], set()
        indexing, scaling_rows, summaries = [], [], []
        for size in SIZES:
            pool = pools[str(size)]["ids"]
            additional = [sid for sid in pool if sid not in completed]
            index_start = time.monotonic()
            new_calls = cached_calls = new_tokens = 0
            for sid in additional:
                doc = documents[id_to_index[sid]]
                text = full_text(skills[doc["source_index"]])
                assert digest(text) == doc["source_sha256"]
                for c in doc["chunks"]:
                    part = text[c["start"]:c["end"]]
                    assert digest(part) == c["text_sha256"]
                    v, receipt = encoder.encode(part, "document")
                    vectors.append(v)
                    chunk_owners.append(id_to_index[sid])
                    if receipt["cache_hit"]:
                        cached_calls += 1
                    else:
                        new_calls += 1
                        new_tokens += receipt["prompt_eval_count"]
                completed.add(sid)
                if len(completed) % 10 == 0:
                    atomic_json(RESULTS / "progress.json", {"stage": "indexing", "utc": utc(), "skills_completed": len(completed),
                                "target_skills": 6006, "chunks_loaded": len(vectors), "current_size": size})
                    print("indexed", len(completed), "/6006; chunks", len(vectors), flush=True)
            index_seconds = time.monotonic() - index_start
            indexing.append({"library_size": size, "additional_skills": len(additional), "cumulative_skills": len(completed),
                             "stage_wall_s": index_seconds, "uncached_calls": new_calls, "cached_calls": cached_calls,
                             "uncached_prompt_tokens": new_tokens, "new_skills_per_second": len(additional) / index_seconds,
                             "uncached_tokens_per_second_wall": new_tokens / index_seconds})
            with (RESULTS / "indexing_history.jsonl").open("a") as history:
                history.write(json.dumps({"utc": utc(), **indexing[-1]}) + "\n")
            matrix = np.stack(vectors)
            owners = np.asarray(chunk_owners, dtype=np.int64)
            eligible = [id_to_index[sid] for sid in pool]
            previous_scaling = RESULTS / f"scaling-n{size}.json"
            if previous_scaling.exists():
                previous = json.loads(previous_scaling.read_text())
                assert previous["identity"] == encoder.identity
                result = previous["evaluation"]
                scaling_rows.extend(previous["rows"])
                summaries.append({"model": protocol["model"], "library_size": size, "queries": 20, **result["aggregate"],
                                  "incorrect_count": result["incorrect_count"], "missing_count": result["missing_count"]})
                write_csv(RESULTS / "scaling_summary.csv", summaries)
                write_csv(RESULTS / "indexing_stages.csv", indexing)
                print("RESUMED SCALING", size, flush=True)
                continue
            labels, rankings, rows = {}, {}, []
            for qi, q in enumerate(panel):
                total_start = time.perf_counter()
                fresh_query, fresh_receipt = encoder.encode(PREFIX + q["query"], "query_latency_probe", use_cache=False)
                t = time.perf_counter()
                selected, scores = rank_skills(matrix, owners, fresh_query, eligible, tie_keys)
                latency = time.perf_counter() - t
                total_latency = time.perf_counter() - total_start
                # Cache-only search uses the precomputed query; measure separately.
                cached_start = time.perf_counter()
                cached_selected, _ = rank_skills(matrix, owners, query_vectors[qi], eligible, tie_keys)
                cached_latency = time.perf_counter() - cached_start
                assert selected.tolist() == cached_selected.tolist()
                ranked = [skills[i]["id"] for i in selected]
                assert len(ranked) == len(set(ranked)) and set(ranked) <= set(pool)
                qid = f"panel-{qi:02d}"
                labels[qid] = {q["gold_id"]: 1}
                rankings[qid] = ranked
                r = fresh_receipt
                rows.append({"model": protocol["model"], "library_size": size, "query_id": qid,
                             "gold_id": q["gold_id"], "ranking": ranked, "scores": scores.tolist(),
                             "query_sha256": digest(q["query"]), "query_embedding_uncached_s": r["latency_s"],
                             "query_vector_reused_across_sizes": False, "similarity_search_s": latency,
                             "total_retrieval_s": total_latency, "cached_query_search_s": cached_latency,
                             "serving_component_sum_s": r["latency_s"] + latency})
            result = evaluate(labels, rankings)
            for row in rows:
                row.update(result["per_query"][row["query_id"]])
            atomic_json(RESULTS / f"scaling-n{size}.json", {"library_size": size, "status": "completed", "identity": encoder.identity, "evaluation": result, "rows": rows})
            scaling_rows.extend(rows)
            summaries.append({"model": protocol["model"], "library_size": size, "queries": 20, **result["aggregate"],
                              "incorrect_count": result["incorrect_count"], "missing_count": result["missing_count"]})
            write_csv(RESULTS / "scaling_summary.csv", summaries)
            write_csv(RESULTS / "indexing_stages.csv", indexing)
            print("SCALING", size, result["aggregate"]["Top1_ID_accuracy"], result["aggregate"]["NDCG@10"], flush=True)
        atomic_json(RESULTS / "content_coverage.json", {"skills_processed": len(completed),
                    "chunks_processed": len(vectors), "source_coverage_percent": 100.0,
                    "failed_or_missing_documents": 0,
                    "source_characters": sum(r["source_characters"] for r in documents),
                    "original_tokens": sum(r["original_tokens"] for r in documents),
                    "tokens_including_overlap": sum(c["tokens"] for r in documents for c in r["chunks"]),
                    "protocol_sha256": encoder.identity["protocol_sha256"], "model_digest": encoder.identity["model_digest"]})
        confidence = {metric: bootstrap([[next(r[metric] for r in scaling_rows if r["query_id"] == f"panel-{qi:02d}" and r["library_size"] == n)
                                         for n in SIZES] for qi in range(20)])
                      for metric in ("Top1_ID_accuracy", "Recall@10", "NDCG@10", "MRR@10")}
        atomic_json(RESULTS / "scaling_confidence.json", confidence)
        with (RESULTS / "scaling_rankings.jsonl").open("w") as f:
            for row in scaling_rows:
                f.write(json.dumps(row) + "\n")
        write_csv(RESULTS / "scaling_per_query.csv", [{k: v for k, v in r.items() if k not in ("ranking", "scores")} for r in scaling_rows])
        atomic_json(RESULTS / "progress.json", {"stage": "scaling_completed", "skills_completed": 6006, "scaling_cases": 160, "utc": utc()})
        if stage == "scaling":
            return
        queries = jsonl(cache / "data/queries.jsonl")
        qrels = {}
        for r in jsonl(cache / "data/qrels.jsonl"):
            qrels.setdefault(r["query_id"], {})[r["skill_id"]] = r["relevance"]
        rankings, query_rows = {}, []
        saved_path = RESULTS / "full_rankings.jsonl"
        saved = {r["query_id"]: r for r in jsonl(saved_path)} if saved_path.exists() else {}
        assert set(saved) <= {q["id"] for q in queries}
        with saved_path.open("a") as output:
            for q in queries:
                qid = q["id"]
                if qid in saved:
                    row = saved[qid]
                    assert row["query_sha256"] == digest(q["query"])
                else:
                    total_start = time.perf_counter()
                    v, receipt = encoder.encode(PREFIX + q["query"], "query")
                    t = time.perf_counter()
                    selected, scores = rank_skills(matrix, owners, v, range(6006), tie_keys)
                    latency = time.perf_counter() - t
                    row = {"query_id": qid, "query_sha256": digest(q["query"]),
                           "ranking": [skills[i]["id"] for i in selected], "scores": scores.tolist(),
                           "query_embedding_uncached_s": receipt["latency_s"], "similarity_search_s": latency,
                           "total_retrieval_s_this_execution": time.perf_counter() - total_start,
                           "serving_component_sum_s": receipt["latency_s"] + latency, "cache_hit": receipt["cache_hit"]}
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                assert len(row["ranking"]) == len(set(row["ranking"])) == 50
                assert set(row["ranking"]) <= set(id_to_index)
                rankings[qid] = row["ranking"]
                query_rows.append(row)
                if len(rankings) % 50 == 0:
                    print("full queries", len(rankings), "/4392", flush=True)
                    atomic_json(RESULTS / "progress.json", {"stage": "full_queries", "queries_completed": len(rankings), "utc": utc()})
        assert len(rankings) == 4392
        result = evaluate(qrels, rankings)
        atomic_json(RESULTS / "full_evaluation.json", result)
        atomic_json(RESULTS / "full_confidence.json", {k: bootstrap([r[k] for r in result["per_query"].values()]) for k in ("NDCG@10", "Recall@10", "MRR@10")})
        write_csv(RESULTS / "full_per_query.csv", [{"model": protocol["model"], "library_size": 6006, "query_id": r["query_id"],
                  **{k: v for k, v in r.items() if k not in ("query_id", "ranking", "scores")}, **result["per_query"][r["query_id"]]} for r in query_rows])
        atomic_json(RESULTS / "progress.json", {"stage": "completed", "utc": utc(), "skills_completed": 6006, "scaling_cases": 160, "full_queries": 4392,
                    "wall_seconds_this_invocation": time.monotonic() - started})
    except Exception as e:
        # Class/message only; stack traces and absolute local paths stay outside Git.
        atomic_json(RESULTS / "failure.json", {"utc": utc(), "stage": stage, "exception": type(e).__name__,
                    "message": public_runtime(str(e)), "wall_seconds": time.monotonic() - started})
        raise
    finally:
        monitor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/mia-qwen")
    parser.add_argument("--stage", choices=("feasibility", "scaling", "full"), default="full")
    args = parser.parse_args()
    main(args.cache, args.stage)
