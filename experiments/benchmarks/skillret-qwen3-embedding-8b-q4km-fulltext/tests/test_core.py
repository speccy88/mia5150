import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import bootstrap, chunks, evaluate, full_text, rank_skills, unit


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        class Encoded:
            ids = list(range(len(text)))
            offsets = [(i, i + 1) for i in range(len(text))]
        return Encoded()


def test_complete_unicode_source_coverage_and_overlap():
    text = "  name | desc |\n# Heading\n" + ("héllo 中文 🦆 e\u0301\n" * 80) + "  "
    parts = chunks(text, CharacterTokenizer(), budget=128, overlap=12)
    covered = set()
    for c in parts:
        covered.update(range(c["start"], c["end"]))
        assert len(text[c["start"]:c["end"]]) <= 128
    assert covered == set(range(len(text)))
    assert all(b["start"] < a["end"] for a, b in zip(parts, parts[1:]))
    assert parts == chunks(text, CharacterTokenizer(), budget=128, overlap=12)


def test_whitespace_and_missing_body():
    assert full_text({"name": " n ", "description": " d ", "skill_md": "\n body  "}) == " n  |  d  | \n body  "
    with pytest.raises(ValueError):
        full_text({"name": "n", "description": "d", "skill_md": " "})


def test_direct_boundary_and_invalid_overlap():
    tok = CharacterTokenizer()
    assert len(chunks("x" * 128, tok, budget=128, overlap=12)) == 1
    assert len(chunks("x" * 129, tok, budget=128, overlap=12)) == 2
    with pytest.raises(ValueError):
        chunks("x", tok, budget=128, overlap=64)


def test_float32_cosine_and_invalid_vectors():
    a = unit([3, 4, 0], dimensions=3)
    b = unit([0, 1, 0], dimensions=3)
    assert a.dtype == np.float32
    assert float(a @ b) == pytest.approx(.8)
    for invalid in ([0, 0, 0], [float("nan"), 1, 0], [1, 2]):
        with pytest.raises(ValueError):
            unit(invalid, dimensions=3)


def test_chunk_max_deduplicates_and_excludes_out_of_pool():
    vectors = np.array([[1, 0], [.8, .6], [1, 0], [0, 1]], dtype=np.float32)
    owners = np.array([0, 0, 1, 2])
    idx, scores = rank_skills(vectors, owners, np.array([1, 0], dtype=np.float32), [1, 0], ["b", "a", "c"])
    assert idx.tolist() == [1, 0]  # content hashes decide ties, not UUID/gold/position.
    assert scores.tolist() == [1, 1]
    idx2, _ = rank_skills(vectors, owners, np.array([1, 0], dtype=np.float32), [0, 1], ["b", "a", "c"])
    assert idx.tolist() == idx2.tolist()
    idx, _ = rank_skills(vectors, owners, np.array([1, 0], dtype=np.float32), [2], ["b", "a", "c"])
    assert idx.tolist() == [2]


def test_official_multiple_positives_and_missing_query_denominator():
    result = evaluate({"q1": {"a": 1, "b": 1}, "q2": {"c": 1}}, {"q1": ["x", "a", "b"]})
    row = result["per_query"]["q1"]
    assert row["Recall@1"] == 0
    assert row["Recall@3"] == 1
    assert row["Precision@5"] == .4
    assert row["MAP@5"] == pytest.approx((.5 + 2 / 3) / 2)
    assert row["MRR@10"] == .5
    assert row["NDCG@5"] == pytest.approx((1 / math.log2(3) + .5) / (1 + 1 / math.log2(3)))
    assert result["aggregate"]["Recall@3"] == .5
    assert result["missing_count"] == 1
    with pytest.raises(ValueError):
        evaluate({"q": {"a": 1}}, {"q": ["a", "a"]})


def test_single_gold_id_is_not_name_equivalence():
    r = evaluate({"q": {"id-a": 1}}, {"q": ["id-b", "id-a"]})["per_query"]["q"]
    assert r["Top1_ID_accuracy"] == 0
    assert r["Hit@3"] == r["Recall@3"] == 1
    assert r["MAP@10"] == r["MRR@10"] == .5


def test_cluster_bootstrap_preserves_repeated_question_dependence():
    values = [0] * 10 + [1] * 10
    a = bootstrap(values, count=1000)
    b = bootstrap(np.repeat(np.array(values)[:, None], 8, axis=1), count=1000)
    assert a["question_clusters"] == b["question_clusters"] == 20
    assert a["across_columns_95ci"] == b["across_columns_95ci"]
    assert b == bootstrap(np.repeat(np.array(values)[:, None], 8, axis=1), count=1000)
