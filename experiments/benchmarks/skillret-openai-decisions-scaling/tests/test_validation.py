"""Check independent offline arithmetic and evidence rejection."""

import gzip
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate import Audit, contains_credential, priced_usage, raw_predicates, raw_responses, ranking_metrics


def test_ranking_metrics_use_all_positive_qrels():
    ranking = ["distractor", "a", "b", "unused", "others"]
    result = ranking_metrics(ranking, {"a", "b", "outside_top20"})
    assert result["map_at_5"] == pytest.approx((1 / 2 + 2 / 3) / 3)
    assert result["recall_at_20"] == pytest.approx(2 / 3)
    assert result["precision_at_5"] == pytest.approx(2 / 5)
    assert result["mrr_at_5"] == 1 / 2
    assert result["completeness_at_5"] == 0


@pytest.mark.parametrize("change", ["duplicate", "unlisted", "out_of_range", "boolean", "nonfinite", "refusal"])
def test_invalid_raw_predicates_do_not_become_probabilities(change):
    first = {"type": "predicate", "name": "a", "probability": 0.1}
    second = {"type": "predicate", "name": "b", "probability": 0.2}
    body = {"model": "gpt-6-luna", "answers": [first, second]}
    assert raw_predicates(body, ["a", "b"]) == {"a": 0.1, "b": 0.2}
    second.update({
        "duplicate": {"name": "a"}, "unlisted": {"name": "c"},
        "out_of_range": {"probability": 1.01}, "boolean": {"probability": True},
        "nonfinite": {"probability": float("nan")}, "refusal": {"type": "refusal"},
    }[change])
    assert raw_predicates(body, ["a", "b"]) is None


def test_raw_attempt_uses_provider_usage_and_both_cost_bases(tmp_path):
    usage = {"input_tokens": 1000, "cached_input_tokens": 100, "cache_write_input_tokens": 200, "output_tokens": 30}
    assert priced_usage(usage, False) == 96000
    assert priced_usage(usage, True) == 105600
    body = {"model": "gpt-6-luna", "answers": [{"type": "predicate", "name": "a", "probability": 0.8}],
            "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 200},
                      "output_tokens": 30, "total_tokens": 1030}}
    row = {"attempt_id": 1, "request_hash": "fixed", "case_id": "test", "kind": "benchmark", "api_latency_ms": 12,
           "body": body, "http_status": 200, "status": "success", "usage": usage,
           "estimated_cost_usd": 0.000096, "budgeted_cost_usd": 0.0001056, "failure": None}
    path = tmp_path / "raw_responses.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    audit = Audit()
    _, probabilities = raw_responses(audit, path, {"fixed": ["a"]}, {"test": "fixed"}, False)
    assert probabilities == {1: {"a": 0.8}}
    assert audit.report("results")["status"] == "passed"


def test_scan_flags_credentials_crossing_chunk_boundary():
    safe = b'Example header: Authorization: Bearer ${OPENAI_API_KEY}'
    assert not contains_credential(io.BytesIO(safe))
    synthetic_secret = b"sk-" + b"proj-" + b"z" * 85
    assert contains_credential(io.BytesIO(b" " * (1024 * 1024 - 3) + synthetic_secret))
