"""Meaningful controls; HTTP responses here are fixtures, never live results."""

import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
import json
import os
from pathlib import Path
import time
from typing import Dict, Tuple

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
import agent_runner as ar
import pilot


def selected(ids):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "select_skills", "args": {"skill_ids": ids}, "id": "select"}
        ],
    )


def test_jsonl_unicode_separator(tmp_path):
    p = tmp_path / "unicode.jsonl"
    p.write_text(json.dumps({"text": "a\u2028b"}, ensure_ascii=False) + "\n")
    assert pilot.rows(p) == [{"text": "a\u2028b"}]


def test_real_native_metadata_no_body_no_extra_files():
    fs = ar.fixtures()
    model = ar.GuardedModel(responses=[selected([fs[0]["id"]])])
    result = ar.run_agent("CANONICAL_QUERY_ONLY", fs[:2], model)
    prompt = json.dumps(model.calls[0]["messages"])
    assert all("s-" + s["id"] in prompt for s in fs[:2])
    assert fs[2]["id"] not in prompt and "BODY_ONLY_SENTINEL" not in prompt
    assert {t["function"]["name"] for t in model.calls[0]["tools"]} == {"select_skills"}
    assert len(model.calls) == 1
    assert set(result["state_file_paths"]) == {
        f"/skills/s-{s['id']}/SKILL.md" for s in fs[:2]
    }
    assert [
        m["data"]["content"] for m in model.calls[0]["messages"] if m["type"] == "human"
    ] == ["CANONICAL_QUERY_ONLY"]
    assert all(
        x not in prompt for x in ['skill_ids": ["gold', "evaluator_only_gold", "qrels"]
    )


@pytest.mark.parametrize("ids,status", [(["unknown"], "unknown_ids"), ([], "valid")])
def test_selection_preserves_invalid_and_empty(ids, status):
    result = ar.run_agent(
        "query", ar.fixtures()[:1], ar.GuardedModel(responses=[selected(ids)])
    )
    assert result["status"] == status and result["selected"] == ids


def test_duplicate_selection_invalid():
    sid = ar.fixtures()[0]["id"]
    result = ar.run_agent(
        "query", ar.fixtures()[:1], ar.GuardedModel(responses=[selected([sid, sid])])
    )
    assert result["status"] == "duplicate_ids"


def test_unoffered_tool_never_executes():
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "execute",
                "args": {"command": "touch /tmp/forbidden"},
                "id": "bad",
            }
        ],
    )
    result = ar.run_agent(
        "query", ar.fixtures()[:1], ar.GuardedModel(responses=[response])
    )
    assert result["status"] == "malformed_output" and len(result["calls"]) == 1


def test_activation_read_is_real_native_tool():
    fs = ar.fixtures()[:1]
    sid = fs[0]["id"]
    read = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_file",
                "args": {"file_path": f"/skills/s-{sid}/SKILL.md", "limit": 100},
                "id": "read",
            }
        ],
    )
    r = ar.run_agent(
        "query", fs, ar.GuardedModel(responses=[read, selected([sid])]), True
    )
    assert (
        r["status"] == "valid"
        and "BODY_ONLY_SENTINEL_0" in r["reads"][0]["output"]["content"]
    )
    assert "BODY_ONLY_SENTINEL_0" not in json.dumps(r["calls"][0]["messages"])
    assert "BODY_ONLY_SENTINEL_0" in json.dumps(r["calls"][1]["messages"])


@pytest.mark.parametrize(
    "args,error",
    [
        ({"file_path": "/etc/passwd"}, "read_outside_candidates"),
        (
            {
                "file_path": "/skills/s-00000000-0000-0000-0000-000000000001/SKILL.md",
                "limit": 1000,
            },
            "read_window_limit",
        ),
    ],
)
def test_read_denial(args, error):
    read = AIMessage(
        content="", tool_calls=[{"name": "read_file", "args": args, "id": "bad-read"}]
    )
    with pytest.raises(ar.Blocked, match=error):
        ar.run_agent(
            "query", ar.fixtures()[:1], ar.GuardedModel(responses=[read]), True
        )


def test_budget_survives_restart_and_ambiguous_failure(tmp_path, monkeypatch):
    for k in ["MAX_SPEND_USD", "MAX_SPEND_CAD", "CAD_PER_USD"]:
        monkeypatch.delenv(k, raising=False)
    cfg = pilot.config()
    cfg["budget"].update(max_spend_usd=1, max_spend_cad=1.5, max_completion_requests=2)
    path = tmp_path / "ledger.jsonl"
    a = ar.Budget(cfg, path)
    with pytest.raises(RuntimeError):
        with a.reserve(0.6, "one", permission=False):
            raise RuntimeError("ambiguous network failure")
    b = ar.Budget(cfg, path)
    assert b.totals()["unresolved_reserved_usd"] == 0.6
    with pytest.raises(ar.Blocked, match="budget_limit"):
        with b.reserve(0.5, "two", permission=False):
            pass
    with b.reserve(0.1, "two", permission=False) as rid:
        b.settle(rid, 0.01)
    assert b.totals()["accounted_usd"] == 0.61
    with pytest.raises(ar.Blocked, match="request_cap"):
        with b.reserve(0.01, "three", permission=False):
            pass


def test_cad_guard_independent(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_PER_USD", "2")
    cfg = pilot.config()
    cfg["budget"].update(max_spend_usd=3, max_spend_cad=1)
    with pytest.raises(ar.Blocked, match="budget_limit"):
        with ar.Budget(cfg, tmp_path / "budget.jsonl").reserve(
            0.51, "case", permission=False
        ):
            pass


def test_missing_permission_does_not_call_adapter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("ALLOW_PAID_RUNS", "0")
    called = []
    monkeypatch.setattr(ar, "make_adapter", lambda *a, **k: called.append(True))
    with pytest.raises(ar.Blocked, match="credentials_or_permission_missing"):
        ar.GuardedModel().bind_tools([ar.select_skills]).invoke("query")
    assert not called


def pin():
    return {
        "model": "qwen/qwen3-8b",
        "provider_tag": "alibaba",
        "provider_name": "Alibaba",
        "pricing": {"prompt": "0.000000117", "completion": "0.000000455"},
        "reasoning": {"effort": "none"},
        "max_prompt_tokens": 98304,
        "context_length": 131072,
        "max_completion_tokens": 8192,
    }


def test_actual_adapter_http_serialization_no_retry(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-credential")
    attempts = []

    def respond(request):
        attempts.append(
            {
                "body": json.loads(request.content),
                "timeout": request.extensions.get("timeout"),
            }
        )
        return httpx.Response(
            503, json={"error": {"message": "fixture unavailable", "code": 503}}
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    adapter = ar.make_adapter(pin(), pilot.config(), client)
    with pytest.raises(Exception):
        adapter._generate(
            [HumanMessage(content="query")],
            tools=[ar.convert_to_openai_tool(ar.select_skills)],
            tool_choice="auto",
            max_tokens=512,
            retries=None,
            timeout_ms=90000,
        )
    assert len(attempts) == 1
    body = attempts[0]["body"]
    assert [x["function"]["name"] for x in body["tools"]] == ["select_skills"]
    assert body["tool_choice"] == "auto" and body["temperature"] == 0
    assert (
        body["provider"]["only"] == ["alibaba"]
        and body["provider"]["allow_fallbacks"] is False
    )
    assert body["reasoning"]["effort"] == "none" and body["plugins"] == []
    assert body["model"] == "qwen/qwen3-8b" and body["max_tokens"] == 512
    assert attempts[0]["timeout"]["read"] == 90.0


def test_actual_adapter_usage_and_completion_id(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-credential")

    def respond(request):
        return httpx.Response(
            200,
            json={
                "id": "fixture-generation",
                "model": "qwen/qwen3-8b",
                "provider": "Alibaba",
                "created": 123,
                "object": "chat.completion",
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call",
                                    "type": "function",
                                    "function": {
                                        "name": "select_skills",
                                        "arguments": '{"skill_ids":[]}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost": 0.00001,
                },
            },
        )

    adapter = ar.make_adapter(
        pin(), pilot.config(), httpx.Client(transport=httpx.MockTransport(respond))
    )
    result = adapter._generate(
        [HumanMessage(content="query")],
        tools=[ar.convert_to_openai_tool(ar.select_skills)],
        tool_choice="auto",
        max_tokens=512,
        retries=None,
    )
    assert result.llm_output["id"] == "fixture-generation"
    message = result.generations[0].message
    assert (
        message.usage_metadata["input_tokens"] == 10
        and message.response_metadata["cost"] == 0.00001
    )
    assert message.response_metadata["provider"] == "Alibaba"


def test_official_metric_control_parity():
    import pytrec_eval

    source = ast.parse(
        (pilot.ROOT / "evidence/official_skillret_eval.py.txt").read_text()
    )
    fn = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "trec_eval"
    )
    ns = {"Dict": Dict, "Tuple": Tuple, "pytrec_eval": pytrec_eval}
    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]),
            "reviewed-official-evaluator",
            "exec",
        ),
        ns,
    )
    truth = {"q1": {"a": 1, "b": 1}, "q2": {"z": 1}, "q3": {"w": 1}}
    rankings = {"q1": ["x", "b", "a"], "q2": ["z"], "q3": []}
    result = ns["trec_eval"](
        truth,
        {q: {s: 10 - i for i, s in enumerate(r)} for q, r in rankings.items()},
        (5, 10),
    )
    for k in [5, 10]:
        ms = [pilot.retrieval_metrics(r, truth[q], k) for q, r in rankings.items()]
        for off, own in [
            ("Recall", "recall"),
            ("Precision", "precision"),
            ("NDCG", "ndcg"),
            ("Completeness", "completeness"),
        ]:
            assert result[f"{off}@{k}"] == round(
                sum(m[f"{own}_at_{k}"] for m in ms) / 3, 5
            )


def test_saved_real_baseline_denominator_and_independent_recall(reference_json):
    # Historical evidence is never placed in a fresh user's runtime results/.
    summary = reference_json("results/baseline_summary.jsonl", jsonl=True)[-1]
    raw = [
        r
        for r in reference_json("results/baseline_rows.jsonl", jsonl=True)
        if r["run_id"] == summary["run_id"]
    ]
    assert len(raw) == summary["query_count"] == 200
    actual = (
        sum(
            len({s for s, score in r["ranking"][:5]} & set(r["evaluator_only_gold"]))
            / len(r["evaluator_only_gold"])
            for r in raw
        )
        / 200
    )
    assert actual == pytest.approx(summary["metrics"]["recall_at_5"])
    assert summary["corpus_size"] == 6006


@pytest.mark.parametrize("missing_cost", [False, True])
def test_guarded_model_debits_or_retains_reserve(tmp_path, monkeypatch, missing_cost):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-credential")
    for k in ["MAX_SPEND_USD", "MAX_SPEND_CAD", "CAD_PER_USD"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(ar, "require_permission", lambda: None)
    budget = ar.Budget(pilot.config(), tmp_path / "ledger.jsonl")
    monkeypatch.setattr(ar, "Budget", lambda cfg=None: budget)
    monkeypatch.setattr(
        ar, "append_row", lambda path, row: pilot.append_row(tmp_path / Path(path), row)
    )
    factory = ar.make_adapter
    attempts = []

    def respond(request):
        attempts.append(1)
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if not missing_cost:
            usage["cost"] = 0.00001
        return httpx.Response(
            200,
            json={
                "id": "fixture-generation",
                "model": "qwen/qwen3-8b",
                "provider": "Alibaba",
                "created": 123,
                "object": "chat.completion",
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call",
                                    "type": "function",
                                    "function": {
                                        "name": "select_skills",
                                        "arguments": '{"skill_ids":[]}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": usage,
            },
        )

    monkeypatch.setattr(
        ar,
        "make_adapter",
        lambda p, c: factory(
            p, c, httpx.Client(transport=httpx.MockTransport(respond))
        ),
    )
    model = ar.GuardedModel(pin=pin())
    if missing_cost:
        with pytest.raises(ar.Blocked, match="missing_cost"):
            model.bind_tools([ar.select_skills]).invoke("query")
        assert (
            budget.totals()["unresolved_reserved_usd"] > 0
            and budget.totals()["actual_usd"] == 0
        )
    else:
        model.bind_tools([ar.select_skills]).invoke("query")
        assert (
            budget.totals()["actual_usd"] == 0.00001
            and budget.totals()["unresolved_reserved_usd"] == 0
        )
    assert attempts == [1] and budget.totals()["requests"] == 1
    assert model.calls[0]["completion_id"] == "fixture-generation"
    assert model.calls[0]["actual_provider"] == "Alibaba"
    with pytest.raises(ar.Blocked, match="per_workflow_call_cap"):
        model.bind_tools([ar.select_skills]).invoke("second forbidden call")
    assert attempts == [1]


def test_context_estimate_counts_tokens_not_mapping_keys():
    count = ar.token_estimate([HumanMessage(content="hello")], [])
    assert count > 2
    assert (
        ar.context_estimate(
            {"input_tokens_estimated": 1000, "input_token_upper_bound": 10000}
        )
        == 2124
    )
