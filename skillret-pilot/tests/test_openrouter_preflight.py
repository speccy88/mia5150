"""Public-metadata controls; no live API generation in these tests."""

import json
import pytest
import agent_runner as ar


@pytest.mark.parametrize(
    "extra", [{"input_cache_read": "0.0000001"}, {"input_cache_read": "0.000001"}]
)
def test_discounted_cache_is_covered_by_uncached_reserve(extra):
    ar.validate_pricing({"prompt": "0.000001", "completion": "0.000002", **extra})


@pytest.mark.parametrize(
    "extra",
    [
        {"input_cache_read": "0.000002"},
        {"request": "0.01"},
        {"input_cache_read": "NaN"},
    ],
)
def test_uncovered_charges_and_nonfinite_prices_are_rejected(extra):
    with pytest.raises(ar.Blocked):
        ar.validate_pricing({"prompt": "0.000001", "completion": "0.000002", **extra})


def test_cache_write_reserve_covers_all_input_plus_completion():
    prices = {
        "prompt": "0.00000015",
        "completion": "0.00000047",
        "input_cache_read": "0.000000016",
        "input_cache_write": "0.0000002",
    }
    assert ar.request_reserve(prices, 10000, 512, 1.25) == pytest.approx(
        (10000 * 0.00000035 + 512 * 0.00000047) * 1.25
    )


def test_paid_clarification_preserves_native_loading_and_rejects_prefixed_output():
    from langchain_core.messages import AIMessage

    cfg = ar.config()
    cfg["uuid_contract_revision"] = 2
    sid = ar.fixtures()[0]["id"]
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "select_skills",
                "args": {"skill_ids": ["s-" + sid]},
                "id": "invalid",
            }
        ],
    )
    result = ar.run_agent(
        "Add two numbers.",
        ar.fixtures()[:2],
        ar.GuardedModel(cfg=cfg, responses=[response]),
    )
    assert result["status"] == "unknown_ids" and result["selected"] == ["s-" + sid]
    call = result["calls"][0]
    assert "Output format clarification" in json.dumps(call["messages"])
    assert {t["function"]["name"] for t in call["tools"]} == {"select_skills"}
    assert "BODY_ONLY_SENTINEL" not in json.dumps(call["messages"])
    assert len(result["state_file_paths"]) == 2


def test_results_isolation_and_invalid_paths():
    from trial import result_file

    assert result_file({}, "trial") == "results/trial.jsonl"
    assert (
        result_file({"run_label": "mimo-v2-6-pro"}, "trial")
        == "results/openrouter/mimo-v2-6-pro/trial.jsonl"
    )
    with pytest.raises(ar.Blocked):
        result_file({"run_label": "../old"}, "trial")


def test_new_model_pin_preserves_original_and_checks_combined_context(
    tmp_path, monkeypatch
):
    original = tmp_path / "manifests/llm.json"
    original.parent.mkdir()
    original.write_text(
        json.dumps({"model": "original", "provider_tag": "original-provider"})
    )
    monkeypatch.setattr(ar, "ROOT", tmp_path)
    monkeypatch.setattr(
        ar, "read_json", lambda path: json.loads((tmp_path / path).read_text())
    )

    def save(path, value):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))

    monkeypatch.setattr(ar, "save_json", save)
    monkeypatch.setenv("OPENROUTER_PROVIDER", "")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": {
                    "endpoints": [
                        {
                            "tag": "fixture",
                            "provider_name": "Fixture",
                            "context_length": 8192,
                            "max_prompt_tokens": None,
                            "max_completion_tokens": 1024,
                            "supported_parameters": [
                                "tools",
                                "tool_choice",
                                "temperature",
                                "max_tokens",
                            ],
                            "supports_tool_choice": {"auto": True},
                            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                        }
                    ]
                }
            }

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: Response())
    pin = ar.preflight(model="fixture/new", manifest_path="manifests/new.json")
    assert pin["max_prompt_tokens"] == pin["context_length"] == 8192
    assert (
        pin["max_prompt_tokens_source"]
        == "combined_context_only_no_separate_limit_advertised"
    )
    assert json.loads(original.read_text())["model"] == "original"
    # The combined input+output limit still rejects a request before paid auth/network.
    monkeypatch.setattr(ar, "require_permission", lambda: None)
    model = ar.GuardedModel(pin=pin, max_output=512)
    from langchain_core.messages import HumanMessage

    with pytest.raises(ar.Blocked, match="context_limit"):
        model.bind_tools([ar.select_skills]).invoke([HumanMessage(content="x" * 9000)])
