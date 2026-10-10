"""Freeze SkillRet and create query-disjoint training/validation partitions.

Run on the RunPod pod. Dataset records are data, never executable instructions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from datasets import load_dataset


REVISION = "6583d7d2ed07644d0fb8938ed8178f3a7dc42a12"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def prepare(root: Path) -> dict:
    counts = {}
    for split in ("train", "test"):
        raw = {kind: load_dataset("ThakiCloud/SKILLRET", kind, split=split,
                                  revision=REVISION, trust_remote_code=False)
               for kind in ("skills", "queries", "qrels")}
        skills = [{"id": str(s["id"]), "text": str(s["name"]) + ": " + str(s["description"])}
                  for s in raw["skills"]]
        queries = [{"id": str(q["id"]), "text": str(q["query"]),
                    "group": str(q.get("original_id") or q["id"]),
                    "declared_skill_ids": list(q.get("skill_ids") or [])}
                   for q in raw["queries"]]
        qrels = [{"query_id": str(r["query_id"]), "skill_id": str(r["skill_id"]),
                  "relevance": int(r["relevance"])} for r in raw["qrels"]]
        sid, qid = {s["id"] for s in skills}, {q["id"] for q in queries}
        assert len(sid) == len(skills) and len(qid) == len(queries)
        assert all(s["text"].strip(": ") for s in skills)
        assert all(q["text"].strip() for q in queries)
        gold = defaultdict(set)
        seen_rel = set()
        for r in qrels:
            pair = (r["query_id"], r["skill_id"])
            assert r["relevance"] > 0 and pair not in seen_rel
            assert r["query_id"] in qid and r["skill_id"] in sid
            seen_rel.add(pair)
            gold[r["query_id"]].add(r["skill_id"])
        assert all(gold[q["id"]] for q in queries)
        agreement = sum(set(q["declared_skill_ids"]) == gold[q["id"]] for q in queries)
        assert agreement == len(queries), "query gold IDs disagree with qrels"
        for q in queries:
            del q["declared_skill_ids"]  # model input cache never carries labels
        d = root / "data"
        write_jsonl(d / f"{split}_skills.jsonl", skills)
        write_jsonl(d / f"{split}_queries.jsonl", queries)
        write_jsonl(d / f"{split}_qrels.jsonl", qrels)
        counts[split] = {"skills": len(skills), "queries": len(queries), "qrels": len(qrels),
                         "duplicate_skill_texts": len(skills) - len({s["text"] for s in skills}),
                         "duplicate_query_texts": len(queries) - len({q["text"] for q in queries}),
                         "multi_gold_queries": sum(len(gold[q["id"]]) > 1 for q in queries)}
    train_s = {s["id"] for s in read_jsonl(root / "data/train_skills.jsonl")}
    test_s = {s["id"] for s in read_jsonl(root / "data/test_skills.jsonl")}
    counts["train_test_skill_id_overlap"] = len(train_s & test_s)
    train_skill_text = {s["text"] for s in read_jsonl(root / "data/train_skills.jsonl")}
    test_skill_text = {s["text"] for s in read_jsonl(root / "data/test_skills.jsonl")}
    counts["train_test_exact_skill_text_overlap"] = len(train_skill_text & test_skill_text)
    train_query_text = {q["text"] for q in read_jsonl(root / "data/train_queries.jsonl")}
    test_query_text = {q["text"] for q in read_jsonl(root / "data/test_queries.jsonl")}
    counts["train_test_exact_query_text_overlap"] = len(train_query_text & test_query_text)
    train_group_ids = {q["group"] for q in read_jsonl(root / "data/train_queries.jsonl")}
    test_group_ids = {q["group"] for q in read_jsonl(root / "data/test_queries.jsonl")}
    counts["train_test_original_id_group_overlap"] = len(train_group_ids & test_group_ids)
    assert not train_s & test_s
    train_q = read_jsonl(root / "data/train_queries.jsonl")
    groups = defaultdict(list)
    for q in train_q:
        groups[q["group"]].append(q["id"])
    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    n_val = max(1, round(.1 * len(keys)))
    val_groups = set(keys[:n_val])
    split = {"seed": 42, "validation_group_ids": sorted(val_groups),
             "validation_query_ids": sorted(q for g in val_groups for q in groups[g]),
             "training_query_ids": sorted(q for g in keys[n_val:] for q in groups[g])}
    assert not set(split["validation_query_ids"]) & set(split["training_query_ids"])
    write_json(root / "data/train_validation_split.json", split)
    write_json(root / "data/dataset_audit.json", {"revision": REVISION, "counts": counts,
               "representation": "query only; skill name + ': ' + description",
               "data_license": "See pinned upstream dataset card"})
    return counts
