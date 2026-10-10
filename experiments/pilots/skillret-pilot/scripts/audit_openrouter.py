"""Read-only validation of saved real OpenRouter evidence and report arithmetic."""

from pathlib import Path
import json
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import rows, read_json, save_json, selection_metrics
from pilot import ROOT
from agent_runner import Budget
from local_runner import native_read_events
from scripts.run_openrouter_models import MODELS
from scripts.report_openrouter import current_records, report

checks = {
    "selectors": 0,
    "routers": 0,
    "activation_cases": 0,
    "native_reads": 0,
    "verified_paid_responses": 0,
}
queries = {
    q["id"]: q["query"]
    for q in rows("data/queries.jsonl")
    if q["id"] in read_json("manifests/live_query_ids.json")[:5]
}


def inspect(call, allowed, query=None):
    wire = call["wire_request"]
    pin = call["pin"]
    assert {t["function"]["name"] for t in wire["tools"]} == allowed
    assert (
        wire["model"] == pin["model"]
        and wire["temperature"] == 0
        and wire["stream"] is False
    )
    assert wire["provider"]["allow_fallbacks"] is False
    assert (
        wire["provider"]["only"] == wire["provider"]["order"] == [pin["provider_tag"]]
    )
    assert wire["reasoning"] == pin["reasoning"]
    assert not wire["plugins"]
    assert call["input_token_upper_bound"] + wire["max_tokens"] <= pin["context_length"]
    assert "evaluator_only_gold" not in json.dumps(
        wire
    ) and "skill_names" not in json.dumps(wire)
    if query is not None:
        assert [m["content"] for m in wire["messages"] if m["role"] == "user"] == [
            query
        ]
    if call.get("actual_usd") is not None:
        assert call["raw_response"]["provider"] == pin["provider_name"]
        assert call["raw_response"]["model"] == pin["model"]
        assert call["actual_usd"] <= call["reserved_usd"]
        assert call["raw_response"]["usage"]["cost"] == call["actual_usd"]


for label in MODELS:
    cfg, pin, run_id, trial, activation, smoke, routes = current_records(label)
    catalog_path = f"manifests/catalogs-{run_id}.json"
    if (ROOT / catalog_path).exists():
        for catalog in read_json(catalog_path):
            local = read_json(
                "manifests/ollama/catalogs/" + catalog["query_id"] + ".json"
            )
            assert catalog["ordered_ids"] == local["ordered_ids"]
    for r in trial:
        decision = r.get("decision")
        calls = decision["calls"] if decision else r.get("failed_calls", [])
        if decision:
            assert len(calls) == 1
            expected = {f"/skills/s-{s}/SKILL.md" for s in r["candidate_ids"]}
            assert set(decision["state_file_paths"]) == expected
        for c in calls:
            if not c.get("wire_request"):
                continue
            inspect(c, {"select_skills"}, queries[r["query_id"]])
            if decision:
                prompt = json.dumps(c["wire_request"]["messages"][0]["content"])
                assert (
                    set(re.findall(r"/skills/s-[0-9a-f-]{36}/SKILL\.md", prompt))
                    == expected
                )
                assert "BODY_ONLY_SENTINEL_" not in prompt
                assert "Output format clarification" in prompt
        if r["status"] == "valid":
            assert r["selection"] == selection_metrics(
                r["returned_ids"], set(r["evaluator_only_gold"])
            )
        checks["selectors"] += bool(calls)
    for r in routes:
        for c in r["calls"]:
            inspect(c, {"route_domain"}, queries[r["query_id"]])
            assert "/skills/" not in json.dumps(c["wire_request"])
            checks["routers"] += 1
    for r in activation:
        reads = native_read_events(r)
        assert len(r["calls"]) <= 4 and len(reads) <= 2
        for c in r["calls"]:
            if c.get("wire_request"):
                inspect(c, {"read_file", "select_skills"}, queries.get(r["query_id"]))
        prompt = (
            json.dumps(r["calls"][0]["wire_request"]["messages"][0]["content"])
            if r["calls"]
            else ""
        )
        expected = set(re.findall(r"/skills/s-[0-9a-f-]{36}/SKILL\.md", prompt))
        for read in reads:
            assert read["args"]["file_path"] in expected
            assert (
                read["args"].get("offset", 0) == 0
                and 1 <= read["args"].get("limit", 100) <= 100
            )
        if r["status"] == "valid":
            assert any(
                x["args"]["file_path"]
                in {f"/skills/s-{s}/SKILL.md" for s in r["selected"]}
                for x in reads
            )
        checks["activation_cases"] += 1
        checks["native_reads"] += len(reads)

responses = {r["request_id"]: r for r in rows("evidence/traces/responses.jsonl")}
events = rows("results/budget.jsonl")
settled = {e["request_id"]: e["actual_usd"] for e in events if e["event"] == "settle"}
reserved = {
    e["request_id"]: e["reserved_usd"] for e in events if e["event"] == "reserve"
}
for rid, c in responses.items():
    assert rid in reserved and c["actual_usd"] == settled[rid]
    assert c["actual_usd"] <= reserved[rid]
    checks["verified_paid_responses"] += 1
assert abs(sum(settled.values()) - Budget().totals()["actual_usd"]) < 1e-9
summary = report()
assert (
    sum(m["actual_usd_all_configs"] for m in summary["models"])
    == sum(c["actual_usd"] for c in responses.values())
    or abs(
        sum(m["actual_usd_all_configs"] for m in summary["models"])
        - sum(c["actual_usd"] for c in responses.values())
    )
    < 1e-9
)
result = {
    "status": "passed",
    "time": time.time(),
    "checks": checks,
    "budget": Budget().totals(),
    "scope": "Real saved wire requests, canonical queries, tool/provider isolation, candidate paths, native reads, metric arithmetic and cost reconciliation. Does not prove execution success.",
}
save_json("evidence/openrouter-audit-" + str(time.time_ns()) + ".json", result)
print(json.dumps(result, indent=2))
