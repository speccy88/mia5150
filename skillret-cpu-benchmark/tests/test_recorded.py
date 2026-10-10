"""Offline consistency and corruption tests for the retained CPU evidence.

Fixtures copy only ID/ranking/CSV artifacts. They never import inference sources,
download models, or change the retained reference run.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpu_verify_recorded", PROJECT / "verify_recorded.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


@pytest.fixture
def recorded(tmp_path):
    root = tmp_path / "benchmark"
    source = PROJECT / "reference_run"
    paths = [source / name for name in (
        "pilot32_query_ids.json", "qrels32.jsonl", "full_catalog.json", "candidate_pools32.jsonl"
    )]
    paths += list((source / "report").glob("*.csv"))
    paths += list((source / "timing").glob("*.csv"))
    for path in paths:
        target = root / "reference_run" / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return root


def read_csv(root, relative):
    path = root / "reference_run" / relative
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(root, relative, fields, rows):
    path = root / "reference_run" / relative
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(root, name):
    with (root / "reference_run" / name).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(root, name, rows):
    (root / "reference_run" / name).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_complete_retained_evidence_passes(recorded):
    assert verifier.verify(recorded) == {
        "queries": 32, "catalog_skills": 6006, "qrels": 53,
        "per_query_rows": 1792, "summary_rows": 56, "full_catalog_rows": 14,
        "latency_observations": 896, "checksum_files": 0,
    }


def test_metric_denominators_and_cutoff():
    actual = verifier.recompute_metrics(["x", "a", "b"], {"a", "b"})
    assert actual == {
        "mrr5": 0.5, "recall1": 0, "recall5": 1, "hit1": 0, "hit5": 1,
        "ndcg5": (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)),
        "complete5": 1, "candidate_recall": 1, "candidate_hit": 1,
    }
    ranked = [str(index) for index in range(7)]
    actual = verifier.recompute_metrics(ranked, set(ranked))
    assert actual["recall1"] == 1 / 7
    assert actual["recall5"] == 5 / 7
    assert actual["complete5"] == 0
    assert actual["candidate_recall"] == 1
    assert verifier.recompute_metrics(["x1", "x2", "x3", "x4", "x5", "a"], {"a"})["mrr5"] == 0


@pytest.mark.parametrize("ranked,gold,reason", [
    (["a"], set(), "empty gold"), (["a", "a"], {"a"}, "duplicate ranked IDs"),
])
def test_invalid_metric_inputs_rejected(ranked, gold, reason):
    with pytest.raises(verifier.ValidationError, match=reason):
        verifier.recompute_metrics(ranked, gold)


@pytest.mark.parametrize("metric", verifier.METRICS)
def test_each_per_query_metric_is_recomputed(recorded, metric):
    relative = "report/SkillRet_PerQuery_Matched32.csv"
    fields, rows = read_csv(recorded, relative)
    rows[0][metric] = str(float(rows[0][metric]) + 0.02)
    write_csv(recorded, relative, fields, rows)
    with pytest.raises(verifier.ValidationError, match=metric + ": recorded value"):
        verifier.verify(recorded)


def test_duplicate_quality_observation_rejected(recorded):
    relative = "report/SkillRet_PerQuery_Matched32.csv"
    fields, rows = read_csv(recorded, relative)
    rows[-1] = dict(rows[0])
    write_csv(recorded, relative, fields, rows)
    with pytest.raises(verifier.ValidationError, match="duplicate condition/query"):
        verifier.verify(recorded)


def test_unknown_query_id_rejected(recorded):
    relative = "report/SkillRet_PerQuery_Matched32.csv"
    fields, rows = read_csv(recorded, relative)
    rows[0]["query_id"] = "q-not-in-matched32"
    write_csv(recorded, relative, fields, rows)
    with pytest.raises(verifier.ValidationError, match="unknown query ID"):
        verifier.verify(recorded)


def test_ranked_id_must_belong_to_correct_sampled_pool(recorded):
    relative = "report/SkillRet_PerQuery_Matched32.csv"
    fields, rows = read_csv(recorded, relative)
    pools = {row["query_id"]: row["candidate_skill_ids"]
             for row in read_jsonl(recorded, "candidate_pools32.jsonl")}
    ranked = json.loads(rows[0]["ranked_skill_ids"])
    ranked[-1] = pools[rows[0]["query_id"]][99]
    rows[0]["ranked_skill_ids"] = json.dumps(ranked)
    write_csv(recorded, relative, fields, rows)
    with pytest.raises(verifier.ValidationError, match="outside correct catalog pool"):
        verifier.verify(recorded)


def test_native_ranking_must_reorder_same_shortlist(recorded):
    relative = "report/SkillRet_PerQuery_Matched32.csv"
    fields, rows = read_csv(recorded, relative)
    row = next(row for row in rows if row["model"] == "julia" and row["catalog_size"] == "100")
    pools = {item["query_id"]: item["candidate_skill_ids"]
             for item in read_jsonl(recorded, "candidate_pools32.jsonl")}
    gold = {item["skill_id"] for item in read_jsonl(recorded, "qrels32.jsonl")
            if item["query_id"] == row["query_id"]}
    ranked = json.loads(row["ranked_skill_ids"])
    replacement = next(skill for skill in pools[row["query_id"]][:100]
                       if skill not in ranked and skill not in gold)
    position = next(index for index in range(5, len(ranked)) if ranked[index] not in gold)
    ranked[position] = replacement
    row["ranked_skill_ids"] = json.dumps(ranked)
    write_csv(recorded, relative, fields, rows)
    with pytest.raises(verifier.ValidationError, match="native shortlist differs"):
        verifier.verify(recorded)


def test_duplicate_candidate_pool_id_rejected(recorded):
    rows = read_jsonl(recorded, "candidate_pools32.jsonl")
    rows[0]["candidate_skill_ids"][-1] = rows[0]["candidate_skill_ids"][0]
    write_jsonl(recorded, "candidate_pools32.jsonl", rows)
    with pytest.raises(verifier.ValidationError, match="duplicate IDs"):
        verifier.verify(recorded)


def test_empty_gold_query_rejected(recorded):
    rows = read_jsonl(recorded, "qrels32.jsonl")
    missing = rows[0]["query_id"]
    write_jsonl(recorded, "qrels32.jsonl", [row for row in rows if row["query_id"] != missing])
    with pytest.raises(verifier.ValidationError, match="nonempty gold"):
        verifier.verify(recorded)


@pytest.mark.parametrize("corruption,reason", [
    ("duplicate", "duplicate query/skill"), ("wrong_skill", "unknown skill ID"),
])
def test_invalid_qrels_rejected(recorded, corruption, reason):
    rows = read_jsonl(recorded, "qrels32.jsonl")
    if corruption == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0]["skill_id"] = "unknown-skill"
    write_jsonl(recorded, "qrels32.jsonl", rows)
    with pytest.raises(verifier.ValidationError, match=reason):
        verifier.verify(recorded)


def test_aggregate_metric_mismatch_rejected(recorded):
    relative = "report/SkillRet_Benchmark_Matched32.csv"
    fields, rows = read_csv(recorded, relative)
    rows[0]["recall5"] = str(float(rows[0]["recall5"]) - 0.01)
    write_csv(recorded, relative, fields, rows)
    with pytest.raises(verifier.ValidationError, match="summary .* recall5"):
        verifier.verify(recorded)


def test_full_catalog_table_must_be_exact_subset(recorded):
    relative = "report/SkillRet_Benchmark_Matched32_FullCatalog.csv"
    fields, rows = read_csv(recorded, relative)
    rows[0]["latency_context"] = "changed context"
    write_csv(recorded, relative, fields, rows)
    with pytest.raises(verifier.ValidationError, match="exact summary subset"):
        verifier.verify(recorded)


@pytest.mark.parametrize("field", verifier.LATENCY_FIELDS)
def test_each_latency_statistic_is_recomputed(recorded, field):
    for relative in ("report/SkillRet_Benchmark_Matched32.csv",
                     "report/SkillRet_Benchmark_Matched32_FullCatalog.csv"):
        fields, rows = read_csv(recorded, relative)
        row = next(row for row in rows if row["catalog_size"] == "6006" and row["model"] == "bm25")
        row[field] = str(float(row[field]) + 1)
        write_csv(recorded, relative, fields, rows)
    with pytest.raises(verifier.ValidationError, match=field + ": recorded value"):
        verifier.verify(recorded)


def test_missing_latency_repeat_rejected(recorded):
    relative = "timing/bm25_latency.csv"
    fields, rows = read_csv(recorded, relative)
    write_csv(recorded, relative, fields, rows[:-1])
    with pytest.raises(verifier.ValidationError, match="14 x 32 x 2"):
        verifier.verify(recorded)


def test_duplicate_latency_observation_rejected(recorded):
    relative = "timing/bm25_latency.csv"
    fields, rows = read_csv(recorded, relative)
    rows[-1] = dict(rows[0])
    write_csv(recorded, relative, fields, rows)
    with pytest.raises(verifier.ValidationError, match="duplicate latency observation"):
        verifier.verify(recorded)


def test_timing_native_shortlist_mismatch_rejected(recorded):
    relative = "timing/julia_latency.csv"
    fields, rows = read_csv(recorded, relative)
    rows[0]["fresh_shortlist_ids"] = json.dumps(list(reversed(json.loads(rows[0]["fresh_shortlist_ids"]))))
    write_csv(recorded, relative, fields, rows)
    with pytest.raises(verifier.ValidationError, match="fresh shortlist differs"):
        verifier.verify(recorded)


def test_linear_percentile_convention():
    assert verifier.percentile([40, 10, 30, 20], 50) == 25
    assert verifier.percentile([40, 10, 30, 20], 95) == pytest.approx(38.5)


def test_manifest_validation_and_changed_file_detection(recorded):
    relative = "reference_run/pilot32_query_ids.json"
    path = recorded / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (recorded / "SHA256SUMS").write_text(f"{digest}  {relative}\n", encoding="utf-8")
    assert verifier.verify(recorded)["checksum_files"] == 1
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(verifier.ValidationError, match="digest mismatch"):
        verifier.verify(recorded)


def test_manifest_cannot_read_outside_project(recorded):
    (recorded / "SHA256SUMS").write_text("0" * 64 + "  ../private-file\n", encoding="utf-8")
    with pytest.raises(verifier.ValidationError, match="unsafe relative artifact path"):
        verifier.verify(recorded)


def test_manifest_cannot_read_environment_secret(recorded):
    (recorded / "SHA256SUMS").write_text("0" * 64 + "  .env\n", encoding="utf-8")
    with pytest.raises(verifier.ValidationError, match="refusing environment-secret"):
        verifier.verify(recorded)


def test_cli_json_corruption_returns_nonzero(recorded, capsys):
    (recorded / "reference_run" / "pilot32_query_ids.json").write_text("{invalid", encoding="utf-8")
    assert verifier.main(["--root", str(recorded)]) == 1
    assert "Verification failed: reference_run/pilot32_query_ids.json: invalid JSON" in capsys.readouterr().err


def test_cli_failure_returns_nonzero_without_traceback(tmp_path, capsys):
    assert verifier.main(["--root", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert "Verification failed: missing artifact" in output.err
    assert "Traceback" not in output.err
