"""Small tested controls, NOT the complete benchmark or a trained model.

Dependencies: numpy; torch only for multi_positive_loss.
The production pipeline must independently validate IDs, data splits and model parity.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from decimal import Decimal

import numpy as np


def stable_rank(skill_ids: Sequence[str], scores: Sequence[float]) -> list[str]:
    """Descending score; stable lexical skill-ID tie-break. Reject bad inputs."""
    ids = list(skill_ids)
    values = np.asarray(scores, dtype=np.float64)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Skill IDs must be nonempty and unique.")
    if not all(isinstance(s, str) and s for s in ids):
        raise ValueError("Each skill ID must be a nonempty string.")
    if values.ndim != 1 or len(values) != len(ids) or not np.isfinite(values).all():
        raise ValueError("Scores must be finite and align one-to-one with skill IDs.")
    order = np.lexsort((np.asarray(ids, dtype=str), -values))
    return [ids[int(i)] for i in order]


def retrieval_metrics(ranking: Sequence[str], gold_ids: Sequence[str], k: int = 5) -> dict[str, float]:
    """One query's binary-relevance metrics; RR uses the supplied FULL ranking."""
    ranking, gold = list(ranking), set(gold_ids)
    if k <= 0 or not gold:
        raise ValueError("k must be positive; gold must be nonempty.")
    if len(set(ranking)) != len(ranking):
        raise ValueError("Duplicate predictions are not permitted.")
    relevant = [int(s in gold) for s in ranking]
    dcg = sum(x / math.log2(i + 2) for i, x in enumerate(relevant[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {
        "hit_at_1": float(bool(relevant) and relevant[0] == 1),
        "recall_at_1": sum(relevant[:1]) / len(gold),
        f"recall_at_{k}": sum(relevant[:k]) / len(gold),
        "reciprocal_rank": next((1 / (i + 1) for i, x in enumerate(relevant) if x), 0.0),
        f"ndcg_at_{k}": dcg / ideal,
        f"completeness_at_{k}": float(gold.issubset(ranking[:k])),
    }


def positive_mask(
    query_ids: Sequence[str], skill_ids: Sequence[str], qrels: Mapping[str, set[str]]
) -> np.ndarray:
    """All known positives, including skills shared across different queries."""
    if not query_ids or not skill_ids:
        raise ValueError("Batch must have queries and candidate skills.")
    if len(set(query_ids)) != len(query_ids) or len(set(skill_ids)) != len(skill_ids):
        raise ValueError("Batch queries and skills must each be unique.")
    mask = np.asarray([[s in qrels[q] for s in skill_ids] for q in query_ids], dtype=bool)
    if not mask.any(axis=1).all():
        raise ValueError("Every query must have at least one batch positive.")
    return mask


def multi_positive_loss(logits, positives):
    """Uniform-positive bidirectional CE. Zero-positive columns skip reverse CE.

    `logits` already includes the trained scale. Gradients flow into logits/heads.
    This is a documented many-to-many adaptation, not the unmodified CLM loss.
    """
    import torch
    import torch.nn.functional as F

    if logits.ndim != 2 or min(logits.shape) == 0 or positives.shape != logits.shape:
        raise ValueError("Expected nonempty [queries, skills] logits and matching mask.")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("Logits must be finite floating-point values.")
    if positives.dtype != torch.bool:
        raise ValueError("Positives must be a boolean relevance mask.")
    p = positives.to(device=logits.device, dtype=torch.float32)
    row_count, col_count = p.sum(1), p.sum(0)
    if not (row_count > 0).all():
        raise ValueError("Every query requires at least one positive.")
    x = logits.float()
    forward = -((p / row_count[:, None]) * F.log_softmax(x, dim=1)).sum(1).mean()
    valid = col_count > 0
    target = p[:, valid] / col_count[valid][None, :]
    reverse = -(target * F.log_softmax(x[:, valid], dim=0)).sum(0).mean()
    return (forward + reverse) / 2


def nested_catalog(
    all_skill_ids: Sequence[str], gold_ids: Sequence[str], size: int, seed: int, query_id: str
) -> list[str]:
    """Gold-present, seeded nested SETS; randomized order, no gold-first ordering."""
    ids, gold = sorted(all_skill_ids), set(gold_ids)
    if len(ids) != len(set(ids)) or not gold or not gold.issubset(ids):
        raise ValueError("Catalog IDs must be unique and include the nonempty gold set.")
    if not len(gold) <= size <= len(ids):
        raise ValueError("Requested size must fit all positives and not exceed catalog.")
    def rng(label: str) -> random.Random:
        key = f"{seed}|{query_id}|{label}".encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:16], "big"))
    wrong = [s for s in ids if s not in gold]
    rng("distractors").shuffle(wrong)
    chosen = sorted(gold) + wrong[:size - len(gold)]
    rng(f"order:{size}").shuffle(chosen)
    return chosen


def authorized_job_budget(starting_balance: float, cap: float = 6.50, reserve: float = 1.00) -> float:
    """Pure arithmetic control. This DOES NOT monitor billing or shut down a pod."""
    numbers = [Decimal(str(x)) for x in (starting_balance, cap, reserve)]
    if any(not n.is_finite() or n < 0 for n in numbers):
        raise ValueError("Balance, cap and reserve must be finite and nonnegative.")
    balance, ceiling, cushion = numbers
    return float(max(Decimal(0), min(ceiling, balance - cushion)))
