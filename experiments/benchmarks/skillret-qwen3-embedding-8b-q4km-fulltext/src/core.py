"""Full-source chunking, strict float32 cosine retrieval and TREC metrics."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
SIZES = (50, 100, 250, 500, 1000, 2000, 5000, 6006)
PREFIX = "Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: "


def digest(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    elif not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def full_text(skill):
    for field in ("name", "description", "skill_md"):
        if not isinstance(skill.get(field), str):
            raise ValueError(f"Missing/non-string {field}")
    if not skill["skill_md"].strip():
        raise ValueError("Missing full body")
    # Deliberately retain *all* source whitespace, including leading/trailing body bytes.
    return skill["name"] + " | " + skill["description"] + " | " + skill["skill_md"]


def chunks(text, tokenizer, budget=4094, overlap=128):
    """Character-exact spans; token offsets never decode/rewrite original text.

    Prefer a Markdown section boundary within the last 256 tokens of a window.
    Retokenize each selected substring because BPE boundaries can change. Two
    admission slots are reserved by Ollama (actual GGUF framing adds one EOS).
    """
    if not 0 <= overlap < budget // 2:
        raise ValueError("Invalid overlap")
    enc = tokenizer.encode(text, add_special_tokens=False)
    if len(enc.ids) <= budget:
        return [{"start": 0, "end": len(text), "tokens": len(enc.ids), "text_sha256": digest(text)}]
    sections = [m.start() for m in re.finditer(r"(?m)^#{1,6}\s", text)]
    result = []
    start = 0
    while start < len(text):
        remaining = tokenizer.encode(text[start:], add_special_tokens=False)
        if len(remaining.ids) <= budget:
            end = len(text)
        else:
            # Offset of next token avoids losing characters between token offsets.
            end = start + remaining.offsets[budget][0]
            low = start + remaining.offsets[max(budget // 2, budget - 256)][0]
            preferred = [s for s in sections if low <= s <= end]
            if preferred:
                end = preferred[-1]
            while len(tokenizer.encode(text[start:end], add_special_tokens=False).ids) > budget:
                end -= 1
        if end <= start:
            raise ValueError("Chunking failed to advance")
        tokenized = tokenizer.encode(text[start:end], add_special_tokens=False)
        result.append({"start": start, "end": end, "tokens": len(tokenized.ids),
                       "text_sha256": digest(text[start:end])})
        if end == len(text):
            break
        next_start = (end if overlap == 0 or len(tokenized.ids) < 2 else
                      start + tokenized.offsets[min(len(tokenized.ids) - 1, max(1, len(tokenized.ids) - overlap))][0])
        if not start < next_start <= end:
            raise ValueError("Invalid chunk advance")
        start = next_start
    assert result[0]["start"] == 0 and result[-1]["end"] == len(text)
    assert all(b["start"] <= a["end"] for a, b in zip(result, result[1:]))
    return result


def unit(vector, dimensions=4096):
    a = np.asarray(vector, dtype=np.float32)
    if a.shape != (dimensions,) or not np.isfinite(a).all():
        raise ValueError("Invalid embedding shape or values")
    norm = np.linalg.norm(a)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Zero/nonfinite embedding norm")
    a = a / np.float32(norm)
    if abs(float(np.linalg.norm(a)) - 1) > 2e-5:
        raise ValueError("Normalization failed")
    return a


def rank_skills(vectors, chunk_skill_indices, query, eligible, tie_keys, top_k=50):
    """Max chunk cosine; exact ties use full-text hashes, never pool positions."""
    eligible = np.asarray(eligible, dtype=np.int64)
    if len(set(eligible.tolist())) != len(eligible):
        raise ValueError("Duplicate eligible IDs")
    scores = np.full(int(max(chunk_skill_indices)) + 1, -np.inf, dtype=np.float32)
    np.maximum.at(scores, chunk_skill_indices, vectors @ query)
    keys = np.asarray(tie_keys)[eligible]
    if len(set(keys.tolist())) != len(keys):
        raise ValueError("Tie keys must be unique source-content hashes")
    selected = np.lexsort((keys, -scores[eligible]))[:top_k]
    idx = eligible[selected]
    return idx, scores[idx]


def trec_module():
    path = STUDY.parent / "skillret-embedding-jev-fusion/evaluate.py"
    spec = importlib.util.spec_from_file_location("historical_trec_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extra_metrics(gold, ranking):
    relevant = {sid for sid, value in gold.items() if value > 0}
    if len(ranking) != len(set(ranking)):
        raise ValueError("Duplicate ranked skill")
    values = {"Top1_ID_accuracy": float(bool(ranking) and ranking[0] in relevant),
              "incorrect_top1": int(not ranking or ranking[0] not in relevant),
              "missing_result": int(not ranking)}
    for k in (1, 3, 5, 10):
        hits = len(set(ranking[:k]) & relevant)
        values[f"Hit@{k}"] = float(hits > 0)
        values[f"Recall@{k}"] = hits / len(relevant) if relevant else 0.0
        values[f"Precision@{k}"] = hits / k
    return values


def evaluate(gold, rankings):
    result = trec_module().evaluate(gold, rankings)
    for qid in gold:
        result["per_query"][qid].update(extra_metrics(gold[qid], rankings.get(qid, [])))
    result["aggregate"] = {k: math.fsum(row[k] for row in result["per_query"].values()) / len(gold)
                           for k in next(iter(result["per_query"].values()))}
    result["incorrect_count"] = sum(r["incorrect_top1"] for r in result["per_query"].values())
    result["missing_count"] = sum(r["missing_result"] for r in result["per_query"].values())
    return result


def bootstrap(matrix, seed=42, count=10000):
    """Rows are independent question clusters, columns repeated sizes/configs."""
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    rng = np.random.default_rng(seed)
    means = np.empty((count, a.shape[1]), dtype=np.float64)
    for start in range(0, count, 50):
        end = min(count, start + 50)
        indices = rng.integers(0, len(a), size=(end - start, len(a)))
        means[start:end] = a[indices].mean(axis=1)
    return {"mean": a.mean(axis=0).tolist(),
            "per_column_95ci": np.quantile(means, [.025, .975], axis=0).T.tolist(),
            "across_columns_mean": float(a.mean()),
            "across_columns_95ci": np.quantile(means.mean(axis=1), [.025, .975]).tolist(),
            "question_clusters": len(a), "resamples": count, "seed": seed}


def latency_summary(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return None
    return {"count": len(a), "mean_s": float(a.mean()), "median_s": float(np.median(a)),
            "p50_s": float(np.quantile(a, .5)), "p95_s": float(np.quantile(a, .95)),
            "maximum_s": float(a.max())}
