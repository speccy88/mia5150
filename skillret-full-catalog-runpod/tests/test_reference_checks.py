import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from starter.reference_checks import (
    authorized_job_budget, multi_positive_loss, nested_catalog,
    positive_mask, retrieval_metrics, stable_rank,
)


def test_known_answer_ranking():
    m = retrieval_metrics(["X", "A", "Y", "C"], ["A", "C"])
    assert m["hit_at_1"] == 0
    assert m["recall_at_1"] == 0
    assert m["recall_at_5"] == 1
    assert m["reciprocal_rank"] == 0.5
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert m["ndcg_at_5"] == pytest.approx(expected)
    assert m["completeness_at_5"] == 1


def test_hit_is_not_multigold_recall():
    m = retrieval_metrics(["A", "X", "Y"], ["A", "B", "C"])
    assert m["hit_at_1"] == 1
    assert m["recall_at_1"] == pytest.approx(1 / 3)
    assert m["completeness_at_5"] == 0


def test_all_correct():
    m = retrieval_metrics(["A", "C", "X"], ["A", "C"])
    assert m["ndcg_at_5"] == 1
    assert m["recall_at_5"] == 1
    assert m["reciprocal_rank"] == 1


def test_wrong_ranking_is_not_a_success():
    assert all(v == 0 for v in retrieval_metrics(["X", "Y"], ["A"]).values())


def test_empty_result_not_dropped():
    assert all(v == 0 for v in retrieval_metrics([], ["A"]).values())


def test_rr_is_full_not_at_five():
    m = retrieval_metrics(["W", "X", "Y", "Z", "B", "A"], ["A"])
    assert m["recall_at_5"] == 0
    assert m["reciprocal_rank"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("ranking,gold,k", [(["A", "A"], ["A"], 5), (["A"], [], 5), (["A"], ["A"], 0)])
def test_invalid_metric_inputs_rejected(ranking, gold, k):
    with pytest.raises(ValueError):
        retrieval_metrics(ranking, gold, k)


def test_stable_ties_and_order():
    assert stable_rank(["Z", "B", "A"], [1, 0, 0]) == ["Z", "A", "B"]
    assert stable_rank(["B", "A", "Z"], [0, 0, 1]) == ["Z", "A", "B"]


@pytest.mark.parametrize("ids,scores", [(["A", "A"], [1, 2]), (["A"], [np.nan]), (["A"], [np.inf]), (["A"], [1, 2])])
def test_invalid_score_inputs_rejected(ids, scores):
    with pytest.raises(ValueError):
        stable_rank(ids, scores)


def test_shared_skill_is_positive_for_both_queries():
    m = positive_mask(["q1", "q2"], ["A", "B", "X"], {"q1": {"A", "B"}, "q2": {"B"}})
    np.testing.assert_array_equal(m, [[True, True, False], [False, True, False]])


def test_missing_batch_positive_rejected():
    with pytest.raises(ValueError):
        positive_mask(["q1"], ["X"], {"q1": {"A"}})


def test_loss_reduces_to_symmetric_infonce():
    x = torch.tensor([[2., -1.], [0.5, 3.]], requires_grad=True)
    labels = torch.arange(2)
    expected = (F.cross_entropy(x, labels) + F.cross_entropy(x.T, labels)) / 2
    actual = multi_positive_loss(x, torch.eye(2, dtype=torch.bool))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_multigold_loss_with_negative_only_column():
    x = torch.tensor([[2., 2., -2.], [-1., 3., 0.]], requires_grad=True)
    p = torch.tensor([[True, True, False], [False, True, False]])
    a = F.log_softmax(x, dim=1)
    forward = -((a[0, 0] + a[0, 1]) / 2 + a[1, 1]) / 2
    b = F.log_softmax(x[:, :2], dim=0)
    reverse = -(b[0, 0] + (b[0, 1] + b[1, 1]) / 2) / 2
    loss = multi_positive_loss(x, p)
    torch.testing.assert_close(loss, (forward + reverse) / 2)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad[:, 2] > 0).all()  # negative candidate remains in forward denominator


def test_loss_batch_permutation_invariant():
    x = torch.tensor([[2., 1., 0.], [0., 2., 1.]])
    p = torch.tensor([[True, True, False], [False, True, False]])
    rows, cols = torch.tensor([1, 0]), torch.tensor([2, 0, 1])
    torch.testing.assert_close(multi_positive_loss(x, p), multi_positive_loss(x[rows][:, cols], p[rows][:, cols]))


def test_invalid_loss_mask_rejected():
    with pytest.raises(ValueError):
        multi_positive_loss(torch.zeros(2, 2), torch.tensor([[False, False], [True, False]]))


def test_nested_catalog_is_reproducible_and_gold_complete():
    ids = [f"s{i:03d}" for i in range(100)]
    gold = ["s002", "s051"]
    a = nested_catalog(ids, gold, 10, 42, "q7")
    b = nested_catalog(ids, gold, 30, 42, "q7")
    assert a == nested_catalog(ids[::-1], gold, 10, 42, "q7")
    assert set(gold).issubset(a) and set(a).issubset(b)
    assert len(a) == len(set(a)) == 10
    assert set(nested_catalog(ids, gold, 100, 42, "q7")) == set(ids)
    assert a != nested_catalog(ids, gold, 10, 43, "q7")


def test_gold_not_always_prepended():
    ids = [f"s{i:03d}" for i in range(100)]
    gold = ["s002", "s051"]
    positions = [nested_catalog(ids, gold, 20, 42, f"q{i}")[:2] for i in range(10)]
    assert not all(set(p) == set(gold) for p in positions)


@pytest.mark.parametrize("size", [1, 101])
def test_invalid_catalog_size_rejected(size):
    with pytest.raises(ValueError):
        nested_catalog([str(i) for i in range(100)], ["1", "2"], size, 42, "q")


@pytest.mark.parametrize("balance,expected", [(7.58, 6.5), (6, 5), (0.5, 0), (100, 6.5)])
def test_budget_never_spends_reserve(balance, expected):
    assert authorized_job_budget(balance) == expected


def test_budget_rejects_bad_number():
    with pytest.raises(ValueError):
        authorized_job_budget(float("nan"))
