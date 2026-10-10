#!/usr/bin/env python3
"""Run only on Linux/CUDA, inside an isolated pod venv; no Runpod credentials."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clm_reranker import (QUESTION, append_jsonl, latency_summary, rank_with_clm,
                          sha256, write_csv, write_new)


def stop(process):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_ready(client, url, process, deadline):
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}; inspect server log")
        try:
            response = client.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("server readiness deadline exceeded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", type=float, required=True)
    args = ap.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("Official BF16 reference requires Linux/CUDA; Mac remains preparation/evaluation only")
    import httpx
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("NVIDIA GPU with BF16 support required")
    base = Path.cwd()
    out = base / "out"
    out.mkdir(exist_ok=False)
    versions = json.loads((base / "versions.json").read_text())
    data = json.loads((base / "inputs.json").read_text())
    os.environ["HF_HOME"] = str(base / "model-cache")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    gpu = subprocess.check_output(["nvidia-smi"], text=True)
    (out / "gpu-info.txt").write_text(gpu)
    env = {"platform": platform.platform(), "python": sys.version, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
           "versions": {p: importlib.metadata.version(p) for p in ["torch", "vllm", "contrastive-lm", "transformers"]}}
    (out / "environment.txt").write_text(json.dumps(env, indent=2) + "\n")
    (out / "requirements.remote.freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    encoder = clm = None
    finish = threading.Event()

    def telemetry():
        with (out / "gpu_telemetry.csv").open("x") as f:
            f.write("timestamp,name,memory.used_MiB,memory.total_MiB,utilization.gpu_percent\n")
            while not finish.is_set():
                r = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                f.write(r.stdout)
                f.flush()
                finish.wait(2)

    thread = threading.Thread(target=telemetry, daemon=True)
    thread.start()
    requests_used = 0
    def reserve():
        nonlocal requests_used
        if requests_used >= 300 or time.time() + 125 >= args.deadline:
            raise TimeoutError("request cap or experiment deadline reached")
        requests_used += 1
        append_jsonl(out / "request_reservations.jsonl", {"attempt": requests_used, "utc_epoch": time.time(), "max_seconds": 120})

    with httpx.Client(trust_env=False) as client:
        try:
            ck = versions["checkpoint"]
            checkpoint = hf_hub_download(ck["repo"], "CLM_v0.1-8B.pt", revision=ck["revision"])
            expected = next(f["lfs"]["sha256"] for f in ck["files"] if f["rfilename"] == "CLM_v0.1-8B.pt")
            if sha256(checkpoint) != expected:
                raise ValueError("checkpoint SHA-256 mismatch")
            write_new(out / "clm_versions.json", {**versions, "checkpoint_sha256": expected,
                       "checkpoint_bytes": Path(checkpoint).stat().st_size, "checkpoint_filename": Path(checkpoint).name,
                       "head_device": "cpu"})
            enc = versions["encoder"]
            tokenizer = AutoTokenizer.from_pretrained(enc["repo"], revision=enc["revision"], trust_remote_code=False)
            cmd = [str(Path(sys.executable).parent / "vllm"), "serve", enc["repo"],
                   "--revision", enc["revision"], "--tokenizer-revision", enc["revision"],
                   "--served-model-name", "qwen3-8b", "--runner", "pooling", "--enforce-eager",
                   "--pooler-config", '{"pooling_type":"LAST"}', "--enable-prefix-caching",
                   "--max-model-len", "2048", "--gpu-memory-utilization", "0.80", "--max-num-seqs", "32",
                   "--dtype", "bfloat16", "--cpu-offload-gb", "0", "--host", "127.0.0.1", "--port", "8090"]
            write_new(out / "vllm-command.json", cmd)
            with (out / "vllm.log").open("x") as log:
                encoder = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            models = wait_ready(client, "http://127.0.0.1:8090/v1/models", encoder, min(args.deadline - 180, time.time() + 1200))
            write_new(out / "encoder_models.json", models)
            vectors, latencies = [], []
            for text in ["A customer was charged twice.", "A customer was charged twice.", "A telescope observes distant stars."]:
                reserve()
                t = time.perf_counter()
                response = client.post("http://127.0.0.1:8090/v1/embeddings", json={"model": "qwen3-8b", "input": [text]}, timeout=120)
                response.raise_for_status()
                vectors.append(np.asarray(response.json()["data"][0]["embedding"], dtype=float))
                latencies.append(time.perf_counter() - t)
            same = float(np.max(np.abs(vectors[0] - vectors[1])))
            different = float(np.linalg.norm(vectors[0] - vectors[2]))
            smoke = {"dimension": len(vectors[0]), "finite": bool(all(np.isfinite(v).all() for v in vectors)),
                     "same_text_max_abs_difference": same, "different_text_l2": different,
                     "first_seconds": latencies[0], "warm_seconds": latencies[1]}
            write_new(out / "encoder_smoke.json", smoke)
            if not smoke["finite"] or smoke["dimension"] != 4096 or same > 1e-4 or different < 1e-4:
                raise ValueError("encoder numerical smoke failed")
            runtime_log = (out / "vllm.log").read_text()
            gpu_used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip().split("\n")[0])
            verification = {"bf16_in_runtime_log": "bfloat16" in runtime_log,
                            "gpu_used_MiB": gpu_used, "configuration": {"runner": "pooling", "pooling": "LAST",
                            "dtype": "bfloat16", "quantization": None, "cpu_offload_gb": 0},
                            "residency_evidence": "pinned full weights; offload disabled; CUDA startup logs and nvidia-smi"}
            write_new(out / "runtime_verification.json", verification)
            if not verification["bf16_in_runtime_log"] or gpu_used < 14000:
                raise ValueError("BF16 GPU runtime verification failed")

            def start_clm(label):
                nonlocal clm
                stop(clm)
                command = [str(Path(sys.executable).parent / "clm-serve"), "--host", "127.0.0.1", "--port", "8700",
                           "--emb-url", "http://127.0.0.1:8090/v1/embeddings", "--emb-model", "qwen3-8b",
                           "--ckpt", checkpoint, "--device", "cpu", "--action-cache", "512MiB",
                           "--max-tokens", "0", "--no-download", "--no-ui"]
                with (out / "clm.log").open("a") as log:
                    log.write(f"\nSTART {label}\n"); log.flush()
                    clm = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                health = wait_ready(client, "http://127.0.0.1:8700/health", clm, min(time.time() + 120, args.deadline - 130))
                if not health.get("embedder") or health.get("mock") or "clm-latest" not in health.get("models", []):
                    raise ValueError("CLM health validation failed")
                return health

            write_new(out / "clm_health.json", start_clm("smoke"))
            write_new(out / "clm_models.json", client.get("http://127.0.0.1:8700/v1/models").json())
            smoke_case = {"query_id": "smoke-billing", "query_text": "A customer reports being charged twice for the same invoice.",
                          "candidate_ids": ["billing", "technical", "shipping", "sales"],
                          "candidate_texts": ["billing support", "technical support", "shipping support", "sales support"]}
            reserve()
            sm = rank_with_clm(client, "http://127.0.0.1:8700", smoke_case)
            sm["expected_top_id"] = "billing"
            sm["semantic_smoke_passed"] = sm.get("ranking", [None])[0] == "billing"
            sm["semantic_smoke_policy"] = "record incorrect prediction; only protocol/service failure gates the benchmark"
            write_new(out / "clm_rank_smoke.json", sm)
            if sm["status"] != "ok":
                raise ValueError("ranking service/protocol smoke failed")
            reserve()
            typed = client.post("http://127.0.0.1:8700/v1/systemone", json={"state": smoke_case["query_text"],
                "model": "clm-latest", "questions": {
                "noul": {"type": "noul", "instructions": "Is this a billing problem?"},
                "choice": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Charges and refunds", "shipping": "Package deliveries"}},
                "score": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Low urgency", "Moderate urgency", "High urgency"]}}}, timeout=120)
            write_new(out / "clm_typed_smoke.json", {"status": typed.status_code, "body": typed.text})
            typed.raise_for_status()
            start_clm("main-fresh-action-cache")
            consecutive_errors = 0
            for case in data["cases"]:
                state = case["query_text"].strip() + "\n\n" + QUESTION
                lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in [state] + case["candidate_texts"]]
                if max(lengths) > 2048:
                    row = {"query_id": case["query_id"], "status": "CONTEXT_EXCLUSION", "max_tokens": max(lengths), "limit": 2048,
                           "error": "No truncation; query/candidate exceeds encoder context", "latency_seconds": None}
                else:
                    reserve()
                    row = rank_with_clm(client, "http://127.0.0.1:8700", case)
                    consecutive_errors = consecutive_errors + 1 if row["status"] != "ok" else 0
                append_jsonl(out / "bm25_clm_raw.jsonl", row)
                print(json.dumps({"query_id": case["query_id"], "status": row["status"]}), flush=True)
                if consecutive_errors >= 3:
                    raise RuntimeError("three consecutive service failures; no automatic retries")
            write_new(out / "main_cache_final.json", client.get("http://127.0.0.1:8700/health").json())
            cache_rows = []
            for n in [10, 100, 1000]:
                if time.time() + 240 >= args.deadline:
                    break
                start_clm(f"cache-{n}-fresh")
                before = client.get("http://127.0.0.1:8700/health").json()
                times = []
                for i in range(21):
                    reserve()
                    case = {"query_id": f"cache-{n}-{i}", "query_text": data["cases"][i]["query_text"],
                            "candidate_ids": [str(j) for j in range(n)], "candidate_texts": data["cache_library"][:n]}
                    row = rank_with_clm(client, "http://127.0.0.1:8700", case)
                    row.update({"candidate_count": n, "phase": "cold_action" if i == 0 else "warm_action"})
                    append_jsonl(out / "cache_raw.jsonl", row)
                    if row["status"] != "ok":
                        raise RuntimeError("cache benchmark request failed; preserved without retry")
                    times.append(row["latency_seconds"])
                after = client.get("http://127.0.0.1:8700/health").json()
                write_new(out / f"cache_health_{n}.json", {"before": before, "after": after})
                cache_rows.append({"candidates": n, "cold_action_seconds": times[0], "warm_queries": len(times) - 1,
                                   **latency_summary(times[1:]), "input_tokens": "not exposed by /v1/rank",
                                   "caveat": "CLM caches reset; vLLM prefix cache retained; warm states are different queries"})
                # Persist each completed size even if a later size runs out of time.
                write_csv(out / f"cache_benchmark_{n}.csv", [cache_rows[-1]])
            write_csv(out / "cache_benchmark.csv", cache_rows)
            write_new(out / "completion.json", {"status": "complete", "requests_used": requests_used,
                       "optional_direct": "NOT_RUN", "optional_vector": "NOT_RUN"})
            return 0
        except Exception:
            (out / "failure.txt").write_text(traceback.format_exc())
            return 2
        finally:
            stop(clm)
            stop(encoder)
            finish.set()
            thread.join(timeout=12)


if __name__ == "__main__":
    raise SystemExit(main())
