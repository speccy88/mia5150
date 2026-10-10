"""Run frozen BGE and Qwen encoder preprocessing on the RunPod GPU."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from data_stage import read_jsonl, write_json

BGE_REV = "5617a9f61b028005a4858fdac845db406aefb181"
QWEN_REV = "b968826d9c46dd6066d109eabc6255188de91218"


def rows(root: Path, split: str, kind: str):
    return read_jsonl(root / "data" / f"{split}_{kind}.jsonl")


def put_vectors(root: Path, model: str, split: str, kind: str, records, vectors: np.ndarray, recipe: dict):
    ids = [r["id"] for r in records]
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1)
    assert arr.shape[0] == len(ids) and len(set(ids)) == len(ids)
    assert np.isfinite(arr).all() and np.allclose(norms, 1, atol=2e-3)
    d = root / "vectors" / model
    d.mkdir(parents=True, exist_ok=True)
    name = f"{split}_{kind}"
    np.save(d / f"{name}.npy", arr)
    write_json(d / f"{name}.json", {"ids": ids, "shape": arr.shape, "dtype": str(arr.dtype),
               "norm_min": float(norms.min()), "norm_max": float(norms.max()),
               "text_sha256": hashlib.sha256("\n".join(r["text"] for r in records).encode()).hexdigest(),
               "recipe": recipe})


def bge(root: Path):
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer("BAAI/bge-m3", revision=BGE_REV, device="cuda", trust_remote_code=False)
    model.max_seq_length = 8192
    recipe = {"model": "BAAI/bge-m3", "revision": BGE_REV, "mode": "dense only",
              "pooling": "SentenceTransformer published model", "max_seq_length": 8192,
              "normalized": True, "inference_device": "cuda", "stored_dtype": "float32"}
    for kind in ("skills", "queries"):
        records = rows(root, "test", kind)
        lengths = [len(model.tokenizer.encode(r["text"], add_special_tokens=True)) for r in records]
        assert max(lengths) <= 8192, "BGE text would be truncated"
        t = time.perf_counter()
        unique = list(dict.fromkeys(r["text"] for r in records))
        arr_u = model.encode(unique, batch_size=32, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=True)
        lookup = {s: i for i, s in enumerate(unique)}
        arr = arr_u[[lookup[r["text"]] for r in records]]
        put_vectors(root, "bge_m3", "test", kind, records, arr, recipe)
        write_json(root / "evidence" / f"bge_{kind}_timing.json",
                   {"records": len(records), "unique_texts": len(unique),
                    "tokens": sum(lengths), "encode_seconds": time.perf_counter() - t,
                    "max_tokens": max(lengths), "gpu_peak_bytes": torch.cuda.max_memory_allocated()})
    skill_bank = np.load(root / "vectors/bge_m3/test_skills.npy", mmap_mode="r")
    observations = []
    online_vectors = []
    for r in rows(root, "test", "queries")[:20]:
        t = time.perf_counter()
        q = model.encode([r["text"]], normalize_embeddings=True, convert_to_numpy=True)[0]
        online_vectors.append(q)
        encode = time.perf_counter() - t
        t2 = time.perf_counter(); _ = np.argmax(np.asarray(skill_bank) @ q)
        observations.append({"query_id":r["id"], "encode_seconds":encode,
                             "score_top1_seconds":time.perf_counter()-t2,
                             "text_to_top1_seconds":time.perf_counter()-t})
    write_json(root / "results/bge_m3_online_latency.json", observations)
    put_vectors(root, "bge_m3", "online", "queries", rows(root, "test", "queries")[:20],
                np.stack(online_vectors), recipe)
    del model
    torch.cuda.empty_cache()


def start_vllm(root: Path, deadline: float):
    import requests
    import sys
    cmd = [str(Path(sys.executable).parent / "vllm"), "serve", "Qwen/Qwen3-8B",
           "--revision", QWEN_REV, "--tokenizer-revision", QWEN_REV,
           "--served-model-name", "qwen3-8b", "--runner", "pooling",
           "--override-pooler-config", '{"pooling_type":"LAST"}', "--dtype", "bfloat16",
           "--enforce-eager", "--max-model-len", "2048", "--max-num-seqs", "32",
           "--gpu-memory-utilization", "0.80", "--cpu-offload-gb", "0",
           "--host", "127.0.0.1", "--port", "8090"]
    write_json(root / "evidence/vllm_command.json", cmd)
    log = (root / "logs/vllm.log").open("w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    until = min(deadline - 300, time.time() + 1200)
    while time.time() < until:
        if p.poll() is not None:
            raise RuntimeError(f"vLLM exited {p.returncode}; see logs/vllm.log")
        try:
            if requests.get("http://127.0.0.1:8090/v1/models", timeout=5).status_code == 200:
                return p, log
        except requests.RequestException:
            pass
        time.sleep(3)
    p.terminate()
    raise TimeoutError("vLLM did not become ready before startup deadline")


def qwen(root: Path, deadline: float):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from clm.embedder import Embedder
    from clm.engine import Engine
    from clm.heads import HeadPair

    ck = hf_hub_download("Contrastive-LM/CLM-v0.1-8B", "CLM_v0.1-8B.pt",
                         revision="e939398d4556fcd9400c76fa8c5a513202f42b0a")
    dst = root / "checkpoints/clm_pretrained/CLM_v0.1-8B.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        import shutil
        shutil.copy2(ck, dst)
    original_hash = hashlib.sha256(dst.read_bytes()).hexdigest()
    assert original_hash == "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5", "Pinned CLM checkpoint hash mismatch"
    write_json(root / "checkpoints/clm_pretrained/config.json", {"model": "Contrastive-LM/CLM-v0.1-8B",
               "revision": "e939398d4556fcd9400c76fa8c5a513202f42b0a", "sha256": original_hash})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision=QWEN_REV, trust_remote_code=False)
    all_sets = {(split, kind): rows(root, split, kind) for split in ("train", "test")
                for kind in ("skills", "queries")}
    token_stats = {}
    for (split, kind), recs in all_sets.items():
        lengths = [len(tok.encode(r["text"], add_special_tokens=True)) for r in recs]
        token_stats[f"{split}_{kind}"] = {"count": len(recs), "tokens": sum(lengths),
                                          "max": max(lengths), "over_2048": sum(x > 2048 for x in lengths)}
    write_json(root / "data/qwen_token_audit.json", token_stats)
    assert all(v["over_2048"] == 0 for v in token_stats.values()), "predeclared shared shortening required"

    p, log = start_vllm(root, deadline)
    try:
        embedder = Embedder(url="http://127.0.0.1:8090/v1/embeddings", model="qwen3-8b",
                            max_tokens=2048, batch=32, cache_size=200000)
        heads = HeadPair("clm-latest", str(dst), "cpu").ensure()
        assert heads.cfg and heads.n_params > 0
        write_json(root / "checkpoints/clm_pretrained/architecture.json",
                   {"cfg": heads.cfg, "parameters": heads.n_params, "projection_dim": heads.proj_dim,
                    "input_dim": 4096, "scale": heads.scale})
        # Compare our offline path to the pinned upstream Engine's actual ranking path.
        examples = [(f"Synthetic request {i}: extract table data {i}",
                     [f"table extractor {i}", f"image classifier {i}", f"file backup {i}"])
                    for i in range(10)]
        eng = Engine(embedder=embedder, checkpoint=str(dst), device="cpu", action_cache=0)
        parity = []
        parity_q, parity_s = [], []
        parity_qrows, parity_srows = [], []
        for query, candidates in examples:
            qv, _ = embedder.embed([query]); sv, _ = embedder.embed(candidates)
            parity_q.append(qv[0]); parity_qrows.append({"id":str(len(parity_qrows)),"text":query})
            for candidate, vector in zip(candidates, sv):
                parity_srows.append({"id":str(len(parity_srows)),"text":candidate})
                parity_s.append(vector)
            zq, zs = heads.project(qv, sv)
            offline = np.argsort(-(zq @ zs.T)[0], kind="stable").tolist()
            online = eng.rank(query, candidates, instructions=None)
            expected = [candidates.index(x["candidate"]) for x in online]
            assert offline == expected
            parity.append({"query": query, "rank_indices": offline,
                           "pre_head_norm": float(np.linalg.norm(qv)),
                           "projected_norm": float(np.linalg.norm(zq))})
        write_json(root / "evidence/clm_parity.json", {"examples": parity,
                   "checkpoint_sha256": original_hash, "official_engine": True})
        parity_recipe = {"model":"Qwen/Qwen3-8B", "revision":QWEN_REV,
                         "pooling":"LAST", "dtype":"bfloat16", "purpose":"CLM parity"}
        put_vectors(root, "qwen_prehead", "parity", "queries", parity_qrows,
                    np.stack(parity_q), parity_recipe)
        put_vectors(root, "qwen_prehead", "parity", "skills", parity_srows,
                    np.stack(parity_s), parity_recipe)

        # Representative pilot: sample equally from query length quartiles.
        train = all_sets[("train", "queries")]
        ordered = sorted(train, key=lambda x: len(tok.encode(x["text"], add_special_tokens=True)))
        rng = random.Random(42)
        pilot = []
        for j in range(4):
            band = ordered[j * len(ordered) // 4:(j + 1) * len(ordered) // 4]
            pilot.extend(rng.sample(band, 64))
        t0 = time.perf_counter()
        pilot_vec, pilot_tokens = embedder.embed([r["text"] for r in pilot])
        put_vectors(root, "qwen_prehead", "pilot", "queries", pilot, pilot_vec,
                    {"model":"Qwen/Qwen3-8B", "revision":QWEN_REV, "pooling":"LAST", "dtype":"bfloat16"})
        pilot_elapsed = time.perf_counter() - t0
        pilot_rate = pilot_tokens / max(pilot_elapsed, 1e-6)
        assert pilot_rate > 0 and np.isfinite(pilot_vec).all()
        remaining = max(0, deadline - time.time() - 3600)  # train, report, sync reserve
        fixed_tokens = sum(token_stats[f"{s}_{k}"]["tokens"] for s,k in
                           [("train","skills"),("test","skills"),("test","queries")])
        per_query = token_stats["train_queries"]["tokens"] / len(train)
        split = json.loads((root / "data/train_validation_split.json").read_text())
        val_ids = set(split["validation_query_ids"])
        train_ids = set(split["training_query_ids"])
        group_recs = {}
        for r in train:
            if r["id"] in train_ids: group_recs.setdefault(r["group"], []).append(r["id"])
        from collections import defaultdict
        train_gold = defaultdict(set)
        for rel in rows(root, "train", "qrels"):
            train_gold[rel["query_id"]].add(rel["skill_id"])
        strata = {"single":[], "multiple":[]}
        for group, ids in group_recs.items():
            bucket = "multiple" if any(len(train_gold[q]) > 1 for q in ids) else "single"
            strata[bucket].append(group)
        rng = random.Random(42)
        for bucket in strata.values():
            bucket.sort(); rng.shuffle(bucket)
        # Deterministic proportional interleaving preserves the gold-count mix.
        groups = []
        totals = {label: len(bucket) for label, bucket in strata.items()}
        used = {label: 0 for label in strata}
        while any(strata.values()):
            label = min((label for label in strata if strata[label]),
                        key=lambda label: (used[label] / totals[label], label))
            groups.append(strata[label].pop()); used[label] += 1
        choice = None
        for n in [len(groups), 20000, 10000, 5000]:
            n = min(n, len(groups))
            selected_ids = {q for g in groups[:n] for q in group_recs[g]}
            needed = fixed_tokens + per_query * (len(val_ids) + len(selected_ids))
            if 1.5 * needed / pilot_rate <= 0.65 * remaining:
                choice = {"groups": n, "selected_ids": selected_ids, "estimated_seconds": 1.5 * needed / pilot_rate}
                break
        if choice is None:
            raise RuntimeError("measured Qwen throughput cannot fit even 5000 groups plus full test")
        selected = choice["selected_ids"] | val_ids
        subset = [r for r in train if r["id"] in selected]
        write_json(root / "data/training_cohort.json", {"seed": 42, "selected_training_groups": choice["groups"],
                   "selection": "seeded proportional interleaving of single/multiple-gold query groups",
                   "selected_training_group_ids": sorted(groups[:choice["groups"]]),
                   "excluded_training_group_ids": sorted(groups[choice["groups"]:]),
                   "selected_training_query_ids": sorted(choice["selected_ids"]),
                   "validation_query_ids": sorted(val_ids), "excluded_training_query_ids": sorted(train_ids-choice["selected_ids"]),
                   "pilot_seconds": pilot_elapsed, "pilot_tokens": pilot_tokens,
                   "pilot_tokens_per_second": pilot_rate, "estimated_seconds_with_1_5_margin": choice["estimated_seconds"]})
        recipe = {"model": "Qwen/Qwen3-8B", "revision": QWEN_REV, "dtype": "bfloat16",
                  "pooling": "LAST", "pre_head_normalization": "L2 float32",
                  "max_tokens": 2048, "instruction": None, "stored_dtype": "float32"}
        for split_name, kind, recs in [("train", "skills", all_sets[("train","skills")]),
                                      ("train", "queries", subset),
                                      ("test", "skills", all_sets[("test","skills")]),
                                      ("test", "queries", all_sets[("test","queries")])]:
            if time.time() > deadline - 2400:
                raise TimeoutError("budget deadline reserve reached before Qwen corpus complete")
            d = root / "vectors/qwen_prehead"
            if (d / f"{split_name}_{kind}.npy").exists():
                continue
            unique = list(dict.fromkeys(r["text"] for r in recs))
            outputs = []
            for offset in range(0, len(unique), 512):
                block = unique[offset:offset+512]
                shard = d / "shards" / f"{split_name}_{kind}_{offset:07d}.npy"
                shard.parent.mkdir(parents=True, exist_ok=True)
                if shard.exists():
                    x = np.load(shard)
                    assert x.shape == (len(block), 4096)
                else:
                    x, _ = embedder.embed(block)
                    np.save(shard, x)
                outputs.append(x)
                write_json(root / "evidence/qwen_progress.json",
                           {"split": split_name, "kind": kind, "unique_done": min(offset+512,len(unique)),
                            "unique_total": len(unique), "at_epoch": time.time()})
            arr_u = np.concatenate(outputs)
            ix = {s:i for i,s in enumerate(unique)}
            put_vectors(root, "qwen_prehead", split_name, kind, recs,
                        arr_u[[ix[r["text"]] for r in recs]], recipe)
        skill_bank = np.load(root / "vectors/qwen_prehead/test_skills.npy", mmap_mode="r")
        qrows = all_sets[("test", "queries")]
        observations = []
        online_vectors = []
        fresh = Embedder(url="http://127.0.0.1:8090/v1/embeddings", model="qwen3-8b",
                         max_tokens=2048, batch=1, cache_size=0)
        for r in qrows[:20]:
            t = time.perf_counter(); q, _ = fresh.embed([r["text"]])
            online_vectors.append(q[0])
            encode = time.perf_counter() - t
            t2 = time.perf_counter(); _ = np.argmax(np.asarray(skill_bank) @ q[0])
            observations.append({"query_id":r["id"], "encode_seconds":encode,
                                 "score_top1_seconds":time.perf_counter()-t2,
                                 "text_to_top1_seconds":time.perf_counter()-t})
        write_json(root / "results/qwen3_online_latency.json", observations)
        # These online samples are generated vectors too, even when text repeats the corpus.
        put_vectors(root, "qwen_prehead", "online", "queries", qrows[:20],
                    np.stack(online_vectors), recipe)
    finally:
        p.terminate()
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired: p.kill(); p.wait()
        log.close()
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == original_hash
