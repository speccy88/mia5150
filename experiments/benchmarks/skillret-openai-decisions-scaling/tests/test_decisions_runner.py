import asyncio
import gzip
import json

import httpx
import pytest

from decisions_reranker import (
    ENDPOINT,
    MAX_REQUEST_NANO_USD,
    MODEL,
    InvalidDecision,
    Usage,
    build_payload,
    canonical_json,
    parse_probabilities,
    request_digest,
    reranked_candidate_ids,
    sanitize,
)
from run_benchmark import (
    ROOT,
    Case,
    ExecutionLimits,
    Executor,
    Ledger,
    Plan,
    RateLimiter,
    Request,
    RunBlocked,
    execute_live,
    export,
    synthetic_request,
)

RUBRIC = json.loads((ROOT / "rubric.json").read_text(encoding="utf-8"))
LIMITS = ExecutionLimits(15_000_000_000, 3, 60_000, 50, 1, 2, False)


def candidates():
    return [
        {"id": f"skill_{index:02d}", "metadata": str(index), "rank": index + 1, "score": 2 / (index + 1)}
        for index in range(20)
    ]


def request(text):
    payload = build_payload(text, candidates(), RUBRIC)
    encoded = canonical_json(payload)
    return Request(request_digest(encoded), encoded, [question["name"] for question in payload["questions"]], "benchmark")


def response_body(candidate_ids, probability=0.7):
    return {
        "model": MODEL,
        "answers": [
            {"name": candidate_id, "type": "predicate", "probability": probability}
            for candidate_id in candidate_ids
        ],
        "usage": {
            "input_tokens": 1_000,
            "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 300},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 1_020,
        },
    }


def test_canonical_request_excludes_retrieval_fields_and_is_rank_invariant():
    original = candidates()
    reverse = [
        dict(item, rank=rank, score=0, method="bm25", relevant=True)
        for rank, item in enumerate(reversed(original), 1)
    ]
    left = build_payload("Compute résumé statistics.", original, RUBRIC)
    right = build_payload("Compute résumé statistics.", reverse, RUBRIC)
    assert canonical_json(left) == canonical_json(right)
    assert "résumé".encode() in canonical_json(left)
    assert set(left) == {"input", "model", "questions"}
    state = json.loads(left["input"])
    assert set(state) == {"query", "candidates"}
    assert {"name_and_description": "0"} == state["candidates"]["skill_00"]
    assert [question["name"] for question in left["questions"]] == [item["id"] for item in original]
    changed = [dict(original[0], metadata="updated metadata"), *original[1:]]
    assert request_digest(canonical_json(build_payload(state["query"], changed, RUBRIC))) != request_digest(canonical_json(left))


@pytest.mark.parametrize(
    ("modification", "failure"),
    [
        (lambda answers: answers.pop(), "incomplete_answers"),
        (lambda answers: answers[1].update(name=answers[0]["name"]), "unexpected_or_duplicate_answer_name"),
        (lambda answers: answers[0].update(probability=True), "invalid_probability"),
        (lambda answers: answers[0].update(probability=float("nan")), "invalid_probability"),
        (lambda answers: answers[0].update(probability=float("inf")), "invalid_probability"),
        (lambda answers: answers[0].update(probability=-0.1), "invalid_probability"),
        (lambda answers: answers[0].update(type="refusal"), "predicate_refusal"),
    ],
)
def test_incomplete_invalid_and_refused_responses(modification, failure):
    prepared = request("test query")
    body = response_body(prepared.candidate_ids)
    modification(body["answers"])
    with pytest.raises(InvalidDecision, match=failure):
        parse_probabilities(body, prepared.candidate_ids)


def test_probabilities_preserve_original_rank_for_ties():
    original = candidates()
    ordered = list(reversed(original))
    prepared = request("test query")
    probabilities = parse_probabilities(response_body(prepared.candidate_ids), prepared.candidate_ids)
    probabilities[original[10]["id"]] = 0.99
    assert reranked_candidate_ids(ordered, probabilities)[:5] == ["skill_10", "skill_00", "skill_01", "skill_02", "skill_03"]


def test_usage_cost_partition_and_premiums():
    measured = Usage.from_response(response_body(request("query").candidate_ids))
    assert measured.cost_nano_usd(False) == 89_500
    assert measured.cost_nano_usd(True) == 98_450
    assert Usage(272_001, 0, 0, 0).cost_nano_usd(False) == 54_400_200
    assert Usage(1_050_000, 0, 1_050_000, 10).cost_nano_usd(True) == MAX_REQUEST_NANO_USD
    with pytest.raises(InvalidDecision, match="invalid_usage"):
        Usage(1_000, 500, 501, 0)


def test_sensitive_headers_and_masked_keys_removed():
    value = {
        "Authorization": "sample",
        "nested": ["Bearer mock-sample-auth-token-more", "API key sk-abc***123"],
        "body": "custom-api-token",
    }
    assert sanitize(value, "custom-api-token") == {
        "nested": ["[REDACTED]", "API key [REDACTED]"],
        "body": "[REDACTED]",
    }


def test_persistent_budget_refuses_unknown_reservations_and_resends(tmp_path):
    first, second = request("first query"), request("second query")
    ledger_path = tmp_path / "ledger.sqlite"
    with Ledger(ledger_path, "a" * 64, False) as ledger:
        assert ledger.reserve(first, "first", "full", MAX_REQUEST_NANO_USD, 5)
        assert ledger.reserve(second, "second", "full", MAX_REQUEST_NANO_USD, 5, wait_for_active=True) is None
    with Ledger(ledger_path, "a" * 64, False) as ledger:
        assert ledger.totals()["held_nano_usd"] == MAX_REQUEST_NANO_USD
        with pytest.raises(RunBlocked, match="reservation would exceed"):
            ledger.reserve(second, "second", "full", MAX_REQUEST_NANO_USD, 5)

        async def check():
            def unexpected(_):
                pytest.fail("An unresolved dispatch must not be repeated.")

            async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
                row = await Executor(ledger, client, LIMITS, "test").score(first, "first", "full")
            assert row["status"] == "reserved"

        asyncio.run(check())


def test_resume_retries_prior_explicit_rejection_once_and_reuses_success(tmp_path):
    prepared = request("test query")
    statuses = iter([401, 200])
    requests = []

    def handler(sent):
        requests.append(sent)
        status = next(statuses)
        content = response_body(prepared.candidate_ids) if status == 200 else {
            "error": "Incorrect API key test-credential",
        }
        return httpx.Response(status, json=content, headers={"x-request-id": "mock-request"})

    async def invoke(ledger):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = Executor(ledger, client, LIMITS, "test-credential")
            return await executor.score(prepared, "case", "full")

    with Ledger(tmp_path / "ledger.sqlite", "a" * 64, False) as ledger:
        first = asyncio.run(invoke(ledger))
        assert first["failure"] == "http_401"
        assert first["known_cost_nano_usd"] == 0
        assert "test-credential" not in first["response_body"]
        second = asyncio.run(invoke(ledger))
        third = asyncio.run(invoke(ledger))
        assert second["status"] == third["status"] == "success"
        assert second["attempt_id"] == third["attempt_id"]
        assert ledger.totals()["reservations"] == 2

    assert len(requests) == 2
    for sent in requests:
        assert str(sent.url) == ENDPOINT
        assert sent.content == prepared.encoded


def test_ledger_preserves_ambiguous_billing_when_transport_fails(tmp_path):
    prepared = request("test query")
    attempt_count = 0

    def handler(sent):
        nonlocal attempt_count
        attempt_count += 1
        raise httpx.ReadTimeout("Mock connection lost.", request=sent)

    async def invoke(ledger):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await Executor(ledger, client, LIMITS, "test").score(prepared, "case", "full")

    with Ledger(tmp_path / "ledger.sqlite", "a" * 64, False) as ledger:
        first = asyncio.run(invoke(ledger))
        assert first["status"] == "failed"
        assert first["failure"] == "transport_error"
        assert first["known_cost_nano_usd"] is None
        assert ledger.totals()["held_nano_usd"] == MAX_REQUEST_NANO_USD
        assert asyncio.run(invoke(ledger))["attempt_id"] == first["attempt_id"]
    assert attempt_count == 1


def test_end_to_end_smoke_then_unique_full_requests_and_cached_logical_cases(tmp_path, monkeypatch):
    test_requests = [request(f"query {index}") for index in range(11)]
    cases = {
        f"case{index}": Case(f"case{index}", "qid", prepared.request_hash, candidates(), 2.0)
        for index, prepared in enumerate(test_requests)
    }
    cases["duplicate"] = Case("duplicate", "qid", test_requests[0].request_hash, candidates(), 2.0)
    plan = Plan(
        cases, {item.request_hash: item for item in test_requests},
        {item.request_hash: f"case{index}" for index, item in enumerate(test_requests)},
        [f"case{index}" for index in range(10)], {"qid": ["skill_04"]}, "b" * 64,
    )
    synthetic = synthetic_request(ROOT / "rubric.json")
    calls = []

    def handler(sent):
        assert str(sent.url) == ENDPOINT
        payload = json.loads(sent.content)
        assert set(json.loads(payload["input"])) == {"query", "candidates"}
        calls.append(request_digest(sent.content))
        body = response_body([entry["name"] for entry in payload["questions"]])
        return httpx.Response(200, json=body)

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        "run_benchmark.httpx.AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs),
    )

    with Ledger(tmp_path / "ledger.sqlite", plan.manifest_sha256, False) as ledger:
        state = asyncio.run(execute_live(plan, synthetic, ledger, tmp_path, LIMITS, "test", False))
        assert state == "complete"
        assert calls == [synthetic.request_hash, *(item.request_hash for item in test_requests)]
        final = export(tmp_path, plan, synthetic, ledger, LIMITS, state, None)
        assert final["logical_experiment_cases"] == 12
        assert final["local_cache_hits"] == 1
        assert final["completed_http_attempt_records"] == 12
        assert final["token_totals"]["input_tokens"] == 12_000
        assert json.loads((tmp_path / "smoke_test.json").read_text())["gate_passed"]

        with gzip.open(tmp_path / "decision_outcomes.jsonl.gz", "rt") as stream:
            outcomes = {row["case_id"]: row for line in stream if (row := json.loads(line))}
        assert outcomes["case0"]["cache_status"] == "fresh"
        assert outcomes["duplicate"]["cache_status"] == "local_cache_hit"
        assert outcomes["duplicate"]["api_latency_ms"] is None
        assert outcomes["duplicate"]["probabilities"] == outcomes["case0"]["probabilities"]
        assert outcomes["duplicate"]["response_reference"] == outcomes["case0"]["response_reference"]
        assert asyncio.run(execute_live(plan, synthetic, ledger, tmp_path, LIMITS, "test", False)) == "complete"
        assert len(calls) == 12


def test_rate_limit_retries_once_and_respects_provider_delay(tmp_path, monkeypatch):
    prepared = request("rate limit query")
    statuses = iter([429, 200])
    delays = []
    monkeypatch.setattr(RateLimiter, "pause", lambda self, delay: delays.append(delay))

    def handler(sent):
        status = next(statuses)
        body = response_body(prepared.candidate_ids) if status == 200 else {
            "error": {"code": "rate_limit_exceeded"},
        }
        return httpx.Response(status, json=body, headers={"Retry-After": "7"})

    async def invoke(ledger):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await Executor(ledger, client, LIMITS, "test").score(prepared, "case", "full")

    with Ledger(tmp_path / "ledger.sqlite", "a" * 64, False) as ledger:
        assert asyncio.run(invoke(ledger))["status"] == "success"
        assert [row["http_status"] for row in ledger.attempts()] == [429, 200]
    assert delays == [7, 7]


def test_project_quota_rejection_stops_without_retry(tmp_path):
    prepared = request("project quota query")
    calls = []

    def handler(sent):
        calls.append(sent)
        return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})

    async def invoke(ledger):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = Executor(ledger, client, LIMITS, "test")
            row = await executor.score(prepared, "case", "full")
            assert row["failure"] == "http_429_insufficient_quota"
            assert executor.run_stop is not None

    with Ledger(tmp_path / "ledger.sqlite", "a" * 64, False) as ledger:
        asyncio.run(invoke(ledger))
        assert len(calls) == 1
        assert ledger.totals()["held_nano_usd"] == 0
