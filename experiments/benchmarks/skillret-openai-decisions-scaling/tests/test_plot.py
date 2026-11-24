"""Check figure inputs against missing measurements and published coverage."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("skillret_scaling_plot", EXPERIMENT / "plot.py")
plot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plot
spec.loader.exec_module(plot)


def test_unattempted_api_does_not_produce_zero_cost_or_recall(tmp_path):
    source = tmp_path / "plot_data.csv"
    row = {
        "catalog_size": 50, "method": "bm25", "n_queries": 200, "n_seeds": 3,
        "recall_at_5": 0.85, "recall_at_5_seed_std": 0.03, "recall_at_20": 0.93,
        "n_attempted_cases": 0, "api_http_attempts": 0, "n_local_cache_hits": 0,
        "api_cost_usd": 0, "api_cost_per_query_usd": "", "input_tokens": 0,
        "cached_input_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0,
        "reranked_recall_at_5": "", "recall_improvement": "",
    }
    with source.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    points = plot.pilot_points(source)
    assert {point.figure for point in points} == {
        "recall-at-5", "recall-at-20-ceiling", "presentation-comparison",
    }
    point = next(point for point in points if point.figure == "recall-at-5")
    assert (point.lower, point.upper) == pytest.approx((0.82, 0.88))
    exported = tmp_path / "points.csv"
    plot.write_points(exported, points)
    encoded = json.dumps(
        plot.read_csv(exported), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    assert plot.points_hash(points) == hashlib.sha256(encoded).hexdigest()


def test_historical_points_retain_original_coverage_and_catalog_sizes():
    points = plot.published_points(EXPERIMENT.parent.parent.parent, EXPERIMENT / "results")
    prompt = [point for point in points if point.population == "openrouter_prompt_20"]
    assert len(prompt) == 61
    assert sum(point.n_queries for point in prompt) == 1184
    assert len([point for point in prompt if point.catalog_size == 5000]) == 7
    assert min(point.n_queries for point in prompt) == 5
    historical = [point for point in points if point.population == "jev_200"]
    assert {point.catalog_size for point in historical} == {6006}
    assert [point.value for point in historical] == pytest.approx([0.6058333333333333, 0.7091666666666667])
