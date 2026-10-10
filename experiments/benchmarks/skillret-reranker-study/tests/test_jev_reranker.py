"""Offline controls: HTTP fixtures are never reported as live Jev evidence."""
from copy import deepcopy
import json
import math

import httpx
import pytest

import jev_reranker as j
from agent_runner import Blocked, Budget
from pilot import read_json, rows, digest


@pytest.fixture
def cfg():
    return read_json("jev.defaults.json")


@pytest.fixture
def candidates():
    return [{"id": f"00000000-0000-0000-0000-{i:012d}",
             "metadata": f"skill {i} : retained capped description", "bm25_score": 30-i,
             "skill_md": "BODY_SENTINEL", "gold": "GOLD_SENTINEL", "major": "DOMAIN_SENTINEL"}
            for i in range(20)]


def case(candidates):
    return {"query_id": "pilot", "query": "CANONICAL_QUERY",
            "candidates": candidates, "bm25_top20": [[c["id"], c["bm25_score"]] for c in candidates],
            "evaluator_only_gold": [candidates[0]["id"], candidates[8]["id"]]}


def reply(ids, probabilities=None):
    probabilities = probabilities or [.2] * len(ids)
    return {"model": "jev-1.13.0", "answers": {sid: {"type": "noul", "noul": p}
            for sid, p in zip(ids, probabilities)}, "usage": {"input_tokens": 1000, "output_tokens": 200}}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ["MAX_SPEND_USD", "MAX_SPEND_CAD", "CAD_PER_USD"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "TEST_ONLY_typesafe_secret_value")
    monkeypatch.setenv("ALLOW_PAID_RUNS", "1")


def test_candidate_whitelist_and_exact_metadata(candidates):
    payload = j.build_jev_questions("CANONICAL_QUERY", list(reversed(candidates)))
    text = json.dumps(payload)
    assert payload["state"]["query"] == "CANONICAL_QUERY"
    assert set(payload["state"]) == {"query", "candidates"}
    for bad in ["BODY_SENTINEL", "GOLD_SENTINEL", "DOMAIN_SENTINEL", "bm25_score", "evaluator_only_gold", "skill_names", '"k"']:
        assert bad not in text
    assert list(payload["questions"]) == sorted(c["id"] for c in candidates)
    for c in candidates:
        assert payload["state"]["candidates"][c["id"]] == {"name_and_description": c["metadata"]}
        assert c["id"] in payload["questions"][c["id"]]["instructions"]
        assert "Several skills" in payload["questions"][c["id"]]["criteria"]["true"]
    assert all(q["type"] == "noul" for q in payload["questions"].values())


def test_scores_sort_and_equal_ties_keep_bm25_order(candidates):
    ids = [c["id"] for c in candidates]
    probabilities = [.1]*20
    probabilities[12], probabilities[8], probabilities[0] = .99, .8, .8
    scores = j.parse_scores(reply(ids, probabilities), ids, "jev-1.13.0")
    assert j.rerank_with_jev(ids, scores) == [ids[12], ids[0], ids[8], ids[1], ids[2]]
    assert set(scores) == set(ids)


@pytest.mark.parametrize("bad", [-.1, 1.1, True, None, "0.5", float("nan"), float("inf")])
def test_invalid_probability_fails(bad):
    with pytest.raises(ValueError, match="invalid_probability"):
        j.parse_scores(reply(["exact-id"], [bad]), ["exact-id"], "jev-1.13.0")


def test_exact_ids_no_autocorrection_and_pinned_model():
    for r in [reply(["ID-TYPO"]), {**reply(["id"]), "answers": {}},
              {**reply(["id"]), "model": "jev-preview"}]:
        with pytest.raises(ValueError):
            j.parse_scores(r, ["id"], "jev-1.13.0")


def test_multi_positive_metrics_and_overlapping_outcomes():
    ranking = [str(i) for i in range(20)]
    e = j.evaluate(ranking, ["1", "6", "7", "8", "9"], {"0", "1", "6", "21"})
    assert e["bm25_at_5"]["recall_at_5"] == .5
    assert e["bm25_at_20"]["recall_at_20"] == .75
    assert e["jev_at_5"]["recall_at_5"] == .5
    assert e["paired_recall_difference"] == 0
    assert e["rescued_ids"] == ["6"] and e["removed_ids"] == ["0"]
    assert e["retained_ids"] == ["1"] and e["upstream_missing_ids"] == ["21"]
    assert set(e["categories"]) == {"upstream_retrieval_failure", "jev_reranking_failure", "jev_reranking_success", "unchanged_success"}
    assert e["jev_at_5"]["completeness_at_5"] == 0
    assert e["jev_selection"]["exact_set_match"] == 0


def test_effective_http_payload_and_usage_cost_not_billed(tmp_path, candidates, cfg):
    captured = []
    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        assert request.url == cfg["endpoint"]
        assert "TEST_ONLY_typesafe_secret_value" not in request.content.decode()
        return httpx.Response(200, json=reply(list(body["questions"])))
    ledger = Budget(cfg, tmp_path / "budget.jsonl")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        r = j.one_request(case(candidates), cfg, client, ledger, "case1")
    assert len(captured) == 1 and r["status"] == "ok"
    assert r["jev_top5"] == [c["id"] for c in candidates[:5]]
    assert r["actual_cost_usd"] is None
    assert r["estimated_cost_from_usage_usd"] == pytest.approx(.000042)
    assert ledger.totals()["unresolved_reserved_usd"] == j.reserve_per_call(cfg)
    j.save_once(tmp_path / "result.json", r)
    assert "TEST_ONLY_typesafe_secret_value" not in (tmp_path / "result.json").read_text()
    assert "authorization" not in json.dumps(r).lower()


@pytest.mark.parametrize("mode", ["429", "timeout", "bad_json", "missing_answer", "missing_usage"])
def test_api_failures_are_explicit_and_never_retried(tmp_path, candidates, cfg, mode):
    attempts = []
    def handler(request):
        attempts.append(1)
        if mode == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if mode == "429":
            return httpx.Response(429, json={"error": "rate limited"})
        if mode == "bad_json":
            return httpx.Response(200, text="not-json")
        result = reply([c["id"] for c in candidates])
        if mode == "missing_answer":
            result["answers"].pop(candidates[-1]["id"])
        else:
            result.pop("usage")
        return httpx.Response(200, json=result)
    ledger = Budget(cfg, tmp_path / "budget.jsonl")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        r = j.one_request(case(candidates), cfg, client, ledger, "case1")
    assert len(attempts) == 1 and r["status"] == "service_failure"
    assert r["jev_probabilities"] is None and r["jev_top5"] is None
    assert r["evaluation"]["paired_recall_difference"] is None
    assert ledger.totals()["requests"] == 1
    assert ledger.totals()["unresolved_reserved_usd"] > 0


def test_missing_env_permission_blocks_before_network(tmp_path, candidates, cfg, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    def handler(_):
        pytest.fail("network reached without environment key")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(Blocked, match="TYPESAFE_API_KEY_missing"):
            j.one_request(case(candidates), cfg, client, Budget(cfg, tmp_path / "ledger.jsonl"), "case")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-key")
    monkeypatch.setenv("ALLOW_PAID_RUNS", "0")
    with pytest.raises(Blocked, match="ALLOW_PAID_RUNS"):
        j.check_permission()


def test_summary_paired_denominators_failures_and_bootstrap(cfg):
    rs = []
    for i, (gold, top5) in enumerate([({"8"}, ["8", "1", "2", "3", "4"]),
                                    ({"0", "8"}, ["0", "8", "2", "3", "4"]),
                                    ({"0"}, None)]):
        rs.append({"status": "ok" if top5 else "service_failure",
            "evaluation": j.evaluate([str(i) for i in range(20)], top5, gold),
            "jev_probabilities": {str(i): .5 for i in range(20)} if top5 else None,
            "attempt_count": 1, "judgments_requested": 20, "usage": None,
            "reserved_usd": .01, "latency_seconds": .2})
    s = j.summarize_jev_results(rs, cfg)
    assert s["bm25_recall_at_5"] == .5
    assert s["jev_recall_at_5"] is None and s["absolute_recall_change"] is None
    assert s["jev_recall_at_5_valid_only"] == 1
    assert s["paired_bm25_recall_at_5_valid_only"] == .25
    assert s["paired_mean_difference_valid_only"] == .75
    assert s["paired_bootstrap_95_ci_valid_only"] == [.5, 1.0]
    assert (s["improved"], s["worsened"], s["unchanged"], s["service_failures"]) == (2, 0, 0, 1)
    assert s["jev_recall_at_5_service_failure_lower_bound"] == pytest.approx(2/3)
    assert s["actual_billed_usd"] is None


def test_immutable_serialization_and_secret_scrubbing(tmp_path):
    p = tmp_path / "result.json"
    value = {"label": "unicode\u2028line", "api_key": "should disappear", "raw": "TEST_ONLY_typesafe_secret_value"}
    j.save_once(p, value)
    original = p.read_bytes()
    j.save_once(p, value)
    assert p.read_bytes() == original and "[REDACTED]" in p.read_text()
    with pytest.raises(FileExistsError):
        j.save_once(p, {"new": 1})
    assert p.read_bytes() == original
    q = tmp_path / "summary.csv"
    j.write_text_once(q, "old\n")
    with pytest.raises(FileExistsError):
        j.write_text_once(q, "new\n")


def test_full_run_reserve_and_context_rejection(cfg, candidates):
    assert j.reserve_per_call(cfg) * 200 == pytest.approx(.688128)
    with pytest.raises(Blocked, match="context_preflight"):
        j.request_preflight(j.build_jev_questions("x"*65000, candidates), cfg)


def test_budget_persists_and_enforces_call_and_currency_caps(tmp_path, cfg):
    cfg = deepcopy(cfg)
    cfg["budget"].update(max_completion_requests=1, max_spend_usd=.01, max_spend_cad=.01)
    path = tmp_path / "ledger.jsonl"
    budget = Budget(cfg, path)
    with pytest.raises(Blocked, match="budget_limit"):
        with budget.reserve(.007, "case", permission=False):
            pytest.fail("CAD guard did not block")
    with budget.reserve(.001, "first", permission=False):
        pass
    again = Budget(cfg, path)
    assert again.totals()["unresolved_reserved_usd"] == .001
    with pytest.raises(Blocked, match="request_cap"):
        with again.reserve(.001, "second", permission=False):
            pytest.fail("request cap did not block")


def test_resume_reuses_results_and_preserves_failures(tmp_path, cfg, candidates, monkeypatch):
    cases = [{**case(candidates), "query_id": f"q{i}"} for i in range(20)]
    spec = {"config": cfg, "cases_hash": digest(cases), "code_sha256": {},
            "prompt": read_json("prompts/jev_relevance.json")}
    folder = tmp_path / digest(spec)[:20]
    j.save_once(folder / "spec.json", spec)
    j.save_once(folder / "candidates.json", cases)
    monkeypatch.setattr(j, "OUTPUT", tmp_path)
    calls = []
    def fake(c, config, client, budget, case_key):
        calls.append(c["query_id"])
        good = c["query_id"] != "q2"
        ids = [x[0] for x in c["bm25_top20"]]
        return {"query_id": c["query_id"], "case_key": case_key,
                "status": "ok" if good else "service_failure", "attempt_count": 1,
                "judgments_requested": 20, "jev_probabilities": dict.fromkeys(ids, .5) if good else None,
                "reserved_usd": .01, "evaluation": j.evaluate(ids, ids[:5] if good else None, c["evaluator_only_gold"])}
    monkeypatch.setattr(j, "one_request", fake)
    first = j.run_jev_trial(folder, 5, True)
    before = (folder / "rows.jsonl").read_bytes()
    again = j.run_jev_trial(folder, 5, True)
    assert first == again and len(calls) == 5
    assert (folder / "rows.jsonl").read_bytes() == before
    with pytest.raises(Blocked, match="stage_5_technical_gate_failed"):
        j.run_jev_trial(folder, 20, True)
    assert len(calls) == 5 and len(rows(folder / "rows.jsonl")) == 5
