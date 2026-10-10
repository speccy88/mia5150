"""Small, standard-library-only controls for the LOCAL pilot protocol.

These are not a claim to reproduce SkillRet's official evaluator. Compare the
pinned official implementation before making that claim. No LLM or network here.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable, Sequence


def _ids(values: Iterable[str], name: str) -> list[str]:
    result = list(values)
    if any(not isinstance(x, str) or not x for x in result):
        raise ValueError(f"{name} must contain nonempty string IDs")
    return result


def retrieval_metrics(ranking: Sequence[str], gold: Iterable[str], k: int) -> dict[str, float]:
    """Binary relevance; Precision@k denominator is k, even if fewer returned.

    A duplicate ranking is a retriever error, not an opportunity to earn twice
    for the same skill. Completeness is 1 iff the entire nonempty gold set occurs
    in the first k unique ranked positions. Values are fractions, not percents.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    ranked = _ids(ranking, "ranking")
    if len(set(ranked)) != len(ranked):
        raise ValueError("duplicate IDs in ranking")
    expected = set(_ids(gold, "gold"))
    if not expected:
        raise ValueError("this pilot requires a nonempty gold set")
    top = ranked[:k]
    hits = [i for i, skill_id in enumerate(top, start=1) if skill_id in expected]
    recall = len(hits) / len(expected)
    dcg = sum(1 / math.log2(i + 1) for i in hits)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(expected)) + 1))
    return {
        f"recall_at_{k}": recall,
        f"precision_at_{k}": len(hits) / k,
        f"mrr_at_{k}": 1 / hits[0] if hits else 0.0,
        f"ndcg_at_{k}": dcg / idcg,
        f"completeness_at_{k}": float(expected.issubset(top)),
    }


def selection_metrics(prediction: Sequence[str], gold: Iterable[str]) -> dict[str, float | int]:
    """Set-based selection scores, plus a duplicate diagnostic.

    Do not remove unknown IDs before calling: they are false positives.
    Apply format/tool/candidate validity outside this function. Duplicates can
    yield set-exact match but must separately invalidate a strict output record.
    """
    returned = _ids(prediction, "prediction")
    predicted = set(returned)
    expected = set(_ids(gold, "gold"))
    if not expected:
        raise ValueError("this pilot requires a nonempty gold set")
    hits = len(predicted & expected)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_match": float(predicted == expected),
        "duplicate_predictions": len(returned) - len(predicted),
    }


def make_nested_catalogs(
    corpus_ids: Sequence[str],
    gold: Iterable[str],
    sizes: Sequence[int],
    query_id: str,
    seed: int = 42,
) -> dict[int, tuple[str, ...]]:
    """Evaluator-only, per-query, gold-preserving distractor-scaling design.

    Gold is used ONLY to build this conditional experiment, never as input to
    a retriever/LLM. Candidate display order is independent of gold membership.
    Input corpus ordering cannot change membership. Every larger catalogue
    contains each smaller catalogue. This is NOT a shared global corpus.
    """
    ids = _ids(corpus_ids, "corpus_ids")
    if len(set(ids)) != len(ids):
        raise ValueError("corpus IDs must be unique")
    expected = set(_ids(gold, "gold"))
    if not expected or not expected.issubset(ids):
        raise ValueError("gold must be nonempty and fully present in the corpus")
    if not query_id:
        raise ValueError("query_id must be nonempty")
    if not sizes or any(isinstance(n, bool) or not isinstance(n, int) for n in sizes):
        raise ValueError("sizes must be a nonempty sequence of integers")
    if min(sizes) < len(expected) or max(sizes) > len(ids):
        raise ValueError("every size must fit all gold IDs and not exceed the corpus")
    seed_material = f"{seed}\0{query_id}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(seed_material).digest(), "big")
    distractors = sorted(set(ids) - expected)
    random.Random(stable_seed).shuffle(distractors)

    def order_key(skill_id: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(seed_material + b"\0display\0" + skill_id.encode("utf-8")).digest()
        return digest, skill_id

    return {
        n: tuple(sorted(expected | set(distractors[: n - len(expected)]), key=order_key))
        for n in sorted(set(sizes))
    }
