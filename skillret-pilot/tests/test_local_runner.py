"""Local adapter transport fixtures; real-model outcomes live in results/ollama."""

import httpx
import json
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from agent_runner import Blocked, select_skills, convert_to_openai_tool
from local_runner import adapter, LocalModel, SETTINGS, local_contract, UUID_CONTRACT


def test_blocked_trace_recovery_counts_results_not_requests():
    from local_runner import native_read_events

    request = {
        "type": "ai",
        "data": {
            "tool_calls": [
                {
                    "id": "read-1",
                    "name": "read_file",
                    "args": {"file_path": "/skills/candidate/SKILL.md", "limit": 100},
                }
            ]
        },
    }
    response = {
        "type": "tool",
        "data": {
            "name": "read_file",
            "tool_call_id": "read-1",
            "content": "native body",
            "status": "success",
        },
    }
    row = {
        "calls": [
            {"messages": [request]},
            {"messages": [request, response]},
            {"messages": [request, response]},
        ],
        "reads": [],
        "status": "blocked",
    }
    events = native_read_events(row)
    assert len(events) == 1 and events[0]["output"]["content"] == "native body"
    assert native_read_events({"calls": [{"messages": [request]}], "reads": []}) == []


def test_local_contract_preserves_query_and_native_metadata():
    messages = [
        SystemMessage(content="native metadata fixture"),
        HumanMessage(content="canonical query"),
    ]
    tools = [convert_to_openai_tool(select_skills)]
    revised, schemas = local_contract(messages, tools)
    assert revised[0].content == "native metadata fixture\n\n" + UUID_CONTRACT
    assert revised[1].content == "canonical query"
    assert messages[0].content == "native metadata fixture"
    assert (
        "pattern"
        in schemas[0]["function"]["parameters"]["properties"]["skill_ids"]["items"]
    )
    assert (
        "pattern"
        not in tools[0]["function"]["parameters"]["properties"]["skill_ids"]["items"]
    )
    block = SystemMessage(content=[{"type": "text", "text": "native metadata"}])
    revised, _ = local_contract([block, messages[1]], tools)
    assert revised[0].content == [
        *block.content,
        {"type": "text", "text": UUID_CONTRACT},
    ]
    assert len(block.content) == 1


def test_ollama_exact_wire_and_usage():
    sent = {}
    raw = {}
    count = []

    def respond(request):
        count.append(1)
        return httpx.Response(
            200,
            json={
                "model": "fixture-local",
                "created_at": "2026-09-26T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "select_skills",
                                "arguments": {"skill_ids": []},
                            }
                        }
                    ],
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 20,
                "eval_count": 10,
                "total_duration": 1000000,
            },
        )

    client = adapter(
        {"model": "fixture-local"}, sent, raw, httpx.MockTransport(respond)
    )
    client.num_predict = 96
    result = client._generate(
        [HumanMessage(content="canonical query")],
        tools=[convert_to_openai_tool(select_skills)],
        stream=False,
    )
    assert count == [1]
    assert sent["model"] == "fixture-local" and sent["think"] is False
    assert sent["options"]["num_ctx"] == 32768 and sent["options"]["num_predict"] == 96
    assert sent["options"]["temperature"] == 0 and sent["options"]["seed"] == 42
    assert sent["stream"] is False and [
        t["function"]["name"] for t in sent["tools"]
    ] == ["select_skills"]
    assert result.generations[0].message.usage_metadata["input_tokens"] == 20
    assert result.generations[0].message.tool_calls[0]["args"] == {"skill_ids": []}
    client._client._client.close()


def test_local_allowlist_and_context_prevent_network():
    model = LocalModel(pin={"model": "never-called", "digest": "fixture"})
    with pytest.raises(Blocked, match="effective_tool_allowlist_mismatch"):
        model._generate([HumanMessage(content="query")], tools=[])
    with pytest.raises(Blocked, match="local_context_limit_no_truncation"):
        model.bind_tools([select_skills]).invoke("x" * SETTINGS["num_ctx"])
    assert not model.calls


def test_ollama_no_transport_retry():
    sent = {}
    raw = {}
    count = []

    def respond(request):
        count.append(1)
        return httpx.Response(503, json={"error": "fixture unavailable"})

    client = adapter(
        {"model": "fixture-local"}, sent, raw, httpx.MockTransport(respond)
    )
    with pytest.raises(Exception):
        client._generate([HumanMessage(content="query")], stream=False)
    assert count == [1]
    client._client._client.close()
