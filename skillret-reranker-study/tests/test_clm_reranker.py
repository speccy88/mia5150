import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from clm_reranker import (build_clm_candidates, evaluate_row, new_run, parse_rank,
                          rank_payload, rank_with_clm, read_jsonl, append_jsonl,
                          summarize_clm_results, write_new)


def case():
    return {"query_id": "q", "query_text": "Fix the payment issue", "candidate_ids": ["a", "b", "c"],
            "candidate_texts": ["billing", "network", "shipping"]}


def response():
    return {"model": "clm-latest", "ranked": [
        {"rank": 1, "candidate": "network", "prob": .6},
        {"rank": 2, "candidate": "billing", "prob": .3},
        {"rank": 3, "candidate": "shipping", "prob": .1}]}


def test_request_has_no_evaluation_fields():
    c = case() | {"skill_ids": ["SECRET_GOLD"], "gold": ["SECRET_GOLD"], "k": 2, "skill_names": ["HINT"]}
    payload = rank_payload(c)
    assert set(payload) == {"model", "context", "question", "answers", "temperature"}
    assert "SECRET_GOLD" not in json.dumps(payload) and "HINT" not in json.dumps(payload)
    assert c["query_id"] not in payload
    assert payload["answers"] == c["candidate_texts"]


def test_candidate_order_does_not_depend_on_retrieval_rank():
    md = {"a": "billing", "b": "network", "c": "shipping"}
    assert build_clm_candidates("query", ["a", "b", "c"], md, "q") == build_clm_candidates("query", ["c", "a", "b"], md, "q")


def test_correct_external_mapping():
    assert parse_rank(response(), case()) == ["b", "a", "c"]


def test_duplicate_text_and_ties_are_stable():
    c = case(); c["candidate_texts"] = ["billing", "billing", "shipping"]
    r = {"model": "clm-latest", "ranked": [{"rank": i + 1, "candidate": t, "prob": 1 / 3}
                                             for i, t in enumerate(["shipping", "billing", "billing"])]}
    assert parse_rank(r, c) == ["a", "b", "c"]


@pytest.mark.parametrize("bad", ["model", "missing", "unknown", "duplicate", "nan", "sum", "rank", "unsorted"])
def test_malformed_response_rejected(bad):
    r = response()
    if bad == "model": r["model"] = "clm-raw"
    if bad == "missing": r["ranked"].pop()
    if bad == "unknown": r["ranked"][0]["candidate"] = "unknown"
    if bad == "duplicate": r["ranked"][1]["candidate"] = "network"
    if bad == "nan": r["ranked"][0]["prob"] = float("nan")
    if bad == "sum": r["ranked"][0]["prob"] = .1
    if bad == "rank": r["ranked"][0]["rank"] = 4
    if bad == "unsorted":
        r["ranked"][0]["prob"], r["ranked"][1]["prob"] = .3, .6
    with pytest.raises(ValueError): parse_rank(r, case())


def test_http_failure_preserved_once():
    calls = []
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(502, text="encoder offline")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        row = rank_with_clm(client, "http://localhost:8700", case())
    assert row["status"] == "SERVICE_FAILURE" and row["http_status"] == 502
    assert row["response_text"] == "encoder offline" and len(calls) == 1
    assert "ranking" not in row


def test_successful_response_and_latency():
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response()))) as client:
        row = rank_with_clm(client, "http://localhost:8700", case())
    assert row["status"] == "ok" and row["ranking"] == ["b", "a", "c"]
    assert row["latency_seconds"] >= 0


def test_multi_positive_mixed_gains_losses_and_upstream():
    b = {"query_id": "q", "ranking": ["a", "b", "x", "y", "z", "c"], "evaluator_only_gold": ["a", "b", "c", "d"]}
    i = {"status": "ok", "ranking": ["a", "c", "x", "y", "z", "b"], "latency_seconds": 1}
    r = evaluate_row(b, i)
    assert r["bm25"]["recall_at_5"] == .5 and r["clm"]["recall_at_5"] == .5
    assert r["candidate"]["recall_at_20"] == .75
    assert r["clm"]["precision_at_5"] == .4
    assert r["selection"]["f1"] == pytest.approx(4 / 9)
    assert set(r["failure_labels"]) == {"UPSTREAM_RETRIEVAL_FAILURE", "CLM_RERANKING_SUCCESS", "CLM_RERANKING_FAILURE"}
    assert r["missing_gold_in_top20"] == ["d"]


def test_paired_summary_keeps_failed_and_unrun_denominators():
    bs = [{"query_id": q, "ranking": ["x", "y", "z", "w", "v", "gold"], "evaluator_only_gold": ["gold"]} for q in ["q1", "q2", "q3"]]
    raw = [{"query_id": "q1", "status": "ok", "ranking": ["gold", "x", "y", "z", "w", "v"], "latency_seconds": 1},
           {"query_id": "q2", "status": "SERVICE_FAILURE", "latency_seconds": 2}]
    summary, _ = summarize_clm_results(bs, raw, bootstrap=100)
    assert summary["attempted"] == 2 and summary["unrun"] == 1
    assert summary["bm25_clm_recall_at_5"] == .5 and summary["service_failures"] == 1
    assert summary["median_seconds"] == 1 and summary["improved"] == 1


def test_serialization_and_exclusive_results(tmp_path):
    p = tmp_path / "rows.jsonl"
    append_jsonl(p, {"text": "a\u2028b", "status": "SERVICE_FAILURE"})
    assert read_jsonl(p)[0]["text"] == "a\u2028b"
    out = tmp_path / "saved.json"
    write_new(out, {"prior": True})
    with pytest.raises(FileExistsError): write_new(out, {"prior": False})
    assert json.loads(out.read_text()) == {"prior": True}
    assert new_run(tmp_path) != new_run(tmp_path)


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("clm_run", Path(__file__).resolve().parents[1] / "scripts/run_clm_trial.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_budget_persists_usd_and_cad_ceiling(runner):
    p = runner.budget_plan(.49, 0)
    assert p["reserved_usd"] < 3 and p["reserved_cad"] == p["reserved_usd"] * 1.5
    with pytest.raises(RuntimeError): runner.budget_plan(.49, 5)
    with pytest.raises(RuntimeError): runner.budget_plan(10, 4.9)


def test_cleanup_refuses_unattributed_pod(runner, tmp_path):
    with pytest.raises(ValueError):
        runner.terminate_created(None, {"pod_id": "someone-elses-pod"}, tmp_path / "cleanup.json")


def test_cleanup_deletes_then_verifies_404(runner, tmp_path, monkeypatch):
    calls = []
    def handle(request):
        calls.append(request.method)
        if calls == ["GET"]:
            return httpx.Response(200, json={"name": "mine", "status": "RUNNING"})
        if request.method == "DELETE": return httpx.Response(204)
        return httpx.Response(404)
    monkeypatch.setattr(runner.time, "sleep", lambda x: None)
    with httpx.Client(base_url="https://api.runpod.io/v2", transport=httpx.MockTransport(handle)) as client:
        r = runner.terminate_created(client, {"created_by_this_run": True, "pod_id": "abc", "pod_name": "mine"}, tmp_path / "cleanup.json")
    assert r["terminated"] and calls == ["GET", "DELETE", "GET"]


def test_remote_bundle_excludes_qrels_and_secrets(runner, tmp_path):
    import tarfile
    write_new(tmp_path / "inputs.json", {"cases": [case()]})
    write_new(tmp_path / "evaluation_only.json", {"gold": "PRIVATE_EVALUATOR_LABEL"})
    (tmp_path / ".env").write_text("SECRET=DO_NOT_COPY")
    bundle = runner.remote_bundle(tmp_path, 9999999999)
    with tarfile.open(bundle) as tf:
        assert set(tf.getnames()) == {"clm_reranker.py", "reference_metrics.py", "clm_remote_worker.py", "inputs.json", "versions.json", "setup.sh"}
        content = b"\n".join(tf.extractfile(x).read() for x in tf.getmembers())
        assert b"PRIVATE_EVALUATOR_LABEL" not in content and b"DO_NOT_COPY" not in content
