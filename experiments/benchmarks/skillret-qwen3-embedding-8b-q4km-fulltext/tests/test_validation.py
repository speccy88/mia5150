"""Independent audit must catch correct-looking scores with wrong denominators."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import evaluate
from validate import compare_evaluation, independent


def evidence():
    labels = {"two-positives": {"a": 1, "b": 1}, "missing": {"c": 1}}
    rankings = {"two-positives": ["x", "a", "b"]}
    independent_rows = {qid: independent(gold, rankings.get(qid, [])) for qid, gold in labels.items()}
    return labels, rankings, independent_rows, evaluate(labels, rankings)


def test_independent_audit_covers_missing_queries_and_macro_micro_denominators():
    labels, rankings, rows, recorded = evidence()
    assert rows["two-positives"]["MAP@5"] == pytest.approx((.5 + 2 / 3) / 2)
    assert rows["missing"]["Coverage@50"] == 0
    assert recorded["candidate_coverage"]["50"]["macro_recall"] == .5
    assert recorded["candidate_coverage"]["50"]["relevant_pair_micro"] == pytest.approx(2 / 3)
    compare_evaluation(recorded, rows, labels, rankings)


def test_independent_audit_rejects_scalar_and_coverage_corruption():
    labels, rankings, rows, recorded = evidence()
    for section, key in (("aggregate", "Recall@10"), ("per_query", "MAP@10"),
                         ("coverage", "relevant_pairs_total")):
        changed = deepcopy(recorded)
        if section == "per_query":
            changed[section]["two-positives"][key] += .01
        elif section == "coverage":
            changed["candidate_coverage"]["50"][key] += 1
        else:
            changed[section][key] += .01
        with pytest.raises(AssertionError):
            compare_evaluation(changed, rows, labels, rankings)
