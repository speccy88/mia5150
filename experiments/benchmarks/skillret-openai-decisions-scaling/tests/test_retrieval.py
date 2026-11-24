"""Offline regression checks for catalog sampling and October ranking equivalence."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from prepare import decode_memberships, encode_memberships, memberships_for
    from retrieve import ScoredBM25, SparseBM25, digest, retrieve, rrf
finally:
    sys.path.pop(0)


def configuration():
    return {
        "version": "test", "dataset": {"revision": "test"},
        "candidate_count": 2, "seeds": [42, 43], "catalog_sizes": [3, 6],
        "fusion": {"source_depth": 100, "rrf_constant": 60},
    }


def test_gold_is_retained_and_distractors_are_nested():
    ids = sorted("abcdef")
    qrels = {"qid": ["a", "d"]}
    membership = list(memberships_for(configuration(), ids, qrels, qrels))
    assert len(membership) == 4
    for short, long in zip(membership[::2], membership[1::2], strict=True):
        assert short["seed"] == long["seed"]
        assert short["skill_ids"] == long["skill_ids"][:3]
        assert short["skill_ids"][:2] == ["a", "d"]
        assert long["catalog_ref"] == digest(long["skill_ids"])
    encoded = list(encode_memberships((dict(row) for row in membership), ids))
    assert list(decode_memberships(encoded, ids)) == membership


@pytest.mark.parametrize("query", ["irrelevant token", "mars mars rover", "mars orbit", "mars orbit mars orbit rover"])
def test_scored_bm25_matches_historical_rankings(query):
    docs = {"z": "Mars rover", "b": "Venus", "c": "Mars orbiter orbit", "d": "Mars orbit", "e": "Earth"}
    catalog = ["d", "b", "z", "e", "c"]
    order, scores = ScoredBM25(docs).scored_search(query, catalog, 4)
    assert order == SparseBM25(docs).search(query, catalog, 4)
    assert set(scores) == set(catalog)
    assert all(np.isfinite(score) for score in scores.values())
    assert order == sorted(catalog, key=lambda skill: (-scores[skill], skill))[:4]


def test_hybrid_uses_full_first_stage_depth_and_reports_rank_scores():
    docs = {"a": "Mars rover", "b": "Mars orbit", "c": "Jupiter"}
    queries = {"q": "Mars rover"}
    skill_ids = list(docs)
    memberships = [{"query_id": "q", "catalog_ref": digest(skill_ids), "catalog_size": 3, "seed": 42, "skill_ids": skill_ids}]
    document_vectors = np.array([[0.0, 1], [0.5, 0], [1, 0]], dtype="float32")
    query_vectors = np.array([[1, 0]], dtype="float32")
    bm, vector, hybrid = retrieve(configuration(), docs, queries, memberships, (document_vectors, query_vectors, {"q": 2.0}))
    assert {row["method"] for row in (bm, vector, hybrid)} == {"bm25", "vector", "hybrid"}
    assert [item["id"] for item in bm["candidates"]] == ["a", "b"]
    assert [item["id"] for item in vector["candidates"]] == ["c", "b"]
    assert [item["id"] for item in hybrid["candidates"]] == rrf(["a", "b", "c"], ["c", "b", "a"])[:2]
    assert [item["rank"] for item in hybrid["candidates"]] == [1, 2]
    assert hybrid["candidates"][0]["score"] == pytest.approx(1 / 61 + 1 / 63)
    assert hybrid["retrieval_latency_ms"] == pytest.approx(hybrid["retrieval_search_ms"] + 2.0)
