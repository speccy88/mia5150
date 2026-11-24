"""Offline paired-evaluation checks for success, fallback and pending cases."""

import csv
import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest


source = Path(__file__).resolve().parents[1] / "evaluate.py"
spec = importlib.util.spec_from_file_location("skillret_scaling_evaluation", source)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def pool(case_id, query_id, seed=42):
    return {
        "case_id": case_id, "query_id": query_id, "query": f"Query {query_id}",
        "catalog_size": 50, "seed": seed, "method": "bm25", "catalog_ref": f"catalog-{case_id}",
        "retrieval_latency_ms": 2.5,
        "candidates": [{"id": f"s{i:02}", "score": 20 - i, "rank": i}
                       for i in range(1, 21)],
    }


def outcome(case_id):
    probabilities = {f"s{i:02}": 0.2 for i in range(1, 21)}
    probabilities.update(s06=0.8, s13=0.8)
    return {
        "case_id": case_id, "request_hash": "a" * 64, "status": "success",
        "probabilities": probabilities, "cache_status": "fresh", "response_reference": 1,
        "http_status": 200, "api_latency_ms": 10.0, "end_to_end_ms": 12.5,
    }


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_metrics_reuse_full_qrels_denominator():
    scores = evaluation.metrics(["a", "x", "b"], {"a", "b", "c"})
    assert scores["recall_at_3"] == pytest.approx(2 / 3)
    assert scores["precision_at_5"] == pytest.approx(2 / 5)
    assert scores["map_at_5"] == pytest.approx((1 + 2 / 3) / 3)
    assert scores["recall_at_20"] == pytest.approx(2 / 3)
    assert scores["completeness_at_5"] == 0


def test_tie_breaker_and_failed_outcome():
    first = evaluation.make_rankings(pool("a", "q"), outcome("a"))
    assert first["reranked_top5"] == ["s06", "s13", "s01", "s02", "s03"]
    assert set(first["reranked_top20"]) == set(first["original_top20"])
    failed = evaluation.make_rankings(pool("a", "q"), {"status": "failed"})
    assert failed["reranked_top20"] is None
    assert failed["operational_top20"] == failed["original_top20"]
    unattempted = evaluation.make_rankings(pool("a", "q"), {})
    assert unattempted["reranked_top20"] is None
    assert unattempted["operational_top20"] is None


def test_invalid_probabilities_fail_evaluation():
    record = outcome("a")
    record["probabilities"]["s03"] = float("nan")
    with pytest.raises(ValueError, match="Invalid successful"):
        evaluation.make_rankings(pool("a", "q"), record)
    record["probabilities"]["s03"] = True
    with pytest.raises(ValueError, match="Invalid successful"):
        evaluation.make_rankings(pool("a", "q"), record)


def test_baseline_only_partial_and_complete_outputs(tmp_path):
    qrels = {"a": ["s06", "s13"], "b": ["s06"]}
    pools = [pool("a", "a"), pool("b", "b")]
    (tmp_path / "qrels.json").write_text(json.dumps(qrels), encoding="utf-8")
    evaluation.write_jsonl(tmp_path / "candidate_pools.jsonl.gz", pools)

    baseline = evaluation.evaluate_results(tmp_path)
    assert baseline["state"] == "baseline_only"
    assert baseline["n_unattempted_cases"] == 2
    baseline_plot = csv_rows(tmp_path / "plot_data.csv")[0]
    assert float(baseline_plot["recall_at_5"]) == 0
    assert baseline_plot["reranked_recall_at_5"] == ""
    assert pq.read_table(tmp_path / "candidate_scores.parquet")["probability"].null_count == 40

    evaluation.write_jsonl(tmp_path / "decision_outcomes.jsonl.gz", [outcome("a")])
    evaluation.write_csv(tmp_path / "api_usage.csv", [{
        "attempt_id": 1, "request_hash": "a" * 64, "case_id": "a", "kind": "benchmark", "status": "success",
        "api_latency_ms": "10.0", "input_tokens": "105", "cached_input_tokens": "100",
        "cache_write_input_tokens": "0", "output_tokens": "0", "estimated_cost_usd": "0.001",
    }])
    partial = evaluation.evaluate_results(tmp_path)
    assert partial["state"] == "partial"
    assert partial["n_valid_cases"] == partial["n_unattempted_cases"] == 1
    partial_plot = csv_rows(tmp_path / "plot_data.csv")[0]
    assert partial_plot["operational_reranked_recall_at_5"] == ""
    assert float(partial_plot["valid_only_reranked_recall_at_5"]) == 1
    assert float(partial_plot["api_latency_median_ms"]) == 10
    assert float(partial_plot["api_cost_usd"]) == pytest.approx(0.001)
    assert int(partial_plot["n_provider_cache_hits"]) == 1
    assert pq.read_table(tmp_path / "candidate_scores.parquet")["probability"].null_count == 20

    evaluation.write_jsonl(tmp_path / "decision_outcomes.jsonl.gz", [
        outcome("a"), {"case_id": "b", "status": "failed", "failure": "service_failure"},
    ])
    complete = evaluation.evaluate_results(tmp_path)
    assert complete["state"] == "complete"
    assert complete["n_valid_cases"] == complete["n_failed_cases"] == 1
    measured = csv_rows(tmp_path / "plot_data.csv")[0]
    assert float(measured["reranked_recall_at_5"]) == pytest.approx(0.5)
    assert float(measured["recall_improvement"]) == pytest.approx(0.5)
    assert float(measured["recall_at_20"]) == float(measured["reranked_recall_at_20"]) == 1
    assert float(measured["valid_only_reranked_recall_at_5"]) == 1
    assert int(measured["operational_improved_queries"]) == 1
    assert int(measured["operational_unchanged_queries"]) == 1


def test_bootstrap_keeps_query_seed_pairs_together():
    rows = [{"query_id": q, "delta": d} for q, d in [("one", 1), ("one", -1), ("two", 0)]]
    interval = evaluation.paired_bootstrap(rows, "delta")
    assert interval["n_bootstrap_queries"] == 2
    assert interval["improvement_ci_lower"] == interval["improvement_ci_upper"] == 0
