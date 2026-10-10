"""Validate saved scaling evidence, candidate isolation, reuse and arithmetic."""

from pathlib import Path
import json
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import rows, read_json, save_json, digest
from agent_runner import Budget
from scripts.report_scaling import report

spec = read_json("manifests/openrouter_scaling.json")
records = [
    r
    for r in rows("results/openrouter/scaling/trial.jsonl")
    if r["run_id"] == spec["run_id"]
]
assert len({r["case_key"] for r in records}) == len(records)
catalog = {
    (r["query_id"], r["N"]): r["ordered_ids"]
    for r in read_json("manifests/openrouter_scaling_catalogs.json")
}
queries = {
    q["id"]: q["query"]
    for q in rows("data/queries.jsonl")
    if q["id"] in spec["query_ids"]
}
source = {
    r["case_key"]: r
    for label in spec["models"]
    for r in rows(f"results/openrouter/{label}/trial.jsonl")
}
routes = {
    r["case_key"]: r
    for label in spec["models"]
    for r in rows(f"results/openrouter/{label}/routes.jsonl")
}
routes.update(
    {r["case_key"]: r for r in rows("results/openrouter/scaling/routes.jsonl")}
)
for qid in spec["query_ids"]:
    available = [n for n in spec["sizes"] if (qid, n) in catalog]
    for small, large in zip(available, available[1:]):
        assert set(catalog[qid, small]) < set(catalog[qid, large])


def inspect(call, allowed, query):
    wire = call["wire_request"]
    pin = call["pin"]
    assert {t["function"]["name"] for t in wire["tools"]} == allowed
    assert wire["model"] == pin["model"] and wire["temperature"] == 0
    assert wire["provider"]["allow_fallbacks"] is False
    assert (
        wire["provider"]["only"] == wire["provider"]["order"] == [pin["provider_tag"]]
    )
    assert wire["reasoning"] == pin["reasoning"] and not wire["plugins"]
    assert [m["content"] for m in wire["messages"] if m["role"] == "user"] == [query]
    assert not any(
        k in json.dumps(wire)
        for k in ["evaluator_only_gold", "skill_names", "original_query"]
    )
    if call.get("actual_usd") is not None:
        assert call["actual_usd"] == call["raw_response"]["usage"]["cost"]
        assert call["actual_usd"] <= call["reserved_usd"]
        assert call["raw_response"]["provider"] == pin["provider_name"]
        assert call["raw_response"]["model"] == pin["model"]


counts = {
    "cases": len(records),
    "reused": 0,
    "selector_wire_calls": 0,
    "router_wire_calls": 0,
}
route_seen = set()
for r in records:
    assert r["case_key"] == digest(
        [spec["run_id"], r["model"], r["query_id"], r["N"], r["strategy"]]
    )
    library = catalog[r["query_id"], r["N"]]
    gold = set(r["evaluator_only_gold"])
    candidates = set(r["candidate_ids"])
    assert (
        len(library) == r["N"] and gold <= set(library) and candidates <= set(library)
    )
    assert r["candidate_ids"] == [s for s in library if s in candidates]
    if "candidate_recall" in r:
        assert r["candidate_recall"] == len(candidates & gold) / len(gold)
        assert r["candidate_completeness"] == float(gold <= candidates)
    if r.get("source_reused"):
        old = source[r["source_case_key"]]
        assert all(
            r[k] == old[k]
            for k in [
                "query_id",
                "N",
                "strategy",
                "status",
                "candidate_ids",
                "returned_ids",
            ]
        )
        counts["reused"] += 1
    decision = r.get("decision")
    calls = decision["calls"] if decision else r.get("failed_calls", [])
    if decision:
        assert len(calls) == 1
        expected = {f"/skills/s-{s}/SKILL.md" for s in candidates}
        assert set(decision["state_file_paths"]) == expected
    for c in calls:
        if not c.get("wire_request"):
            continue
        inspect(c, {"select_skills"}, queries[r["query_id"]])
        counts["selector_wire_calls"] += 1
        if decision:
            paths = set(
                re.findall(
                    r"/skills/s-[0-9a-f-]{36}/SKILL\.md",
                    json.dumps(c["wire_request"]["messages"][0]["content"]),
                )
            )
            assert paths == expected
    if r["status"] == "valid":
        chosen = set(r["returned_ids"])
        assert chosen <= candidates
        assert r["selection"]["exact_set_match"] == float(chosen == gold)
        assert r["selection"]["recall"] == len(chosen & gold) / len(gold)
        assert r["selection"]["precision"] == (
            len(chosen & gold) / len(chosen) if chosen else 0
        )
    if r.get("route_case_key"):
        route = routes[r["route_case_key"]]
        if route["case_key"] not in route_seen:
            for c in route["calls"]:
                inspect(c, {"route_domain"}, queries[r["query_id"]])
                counts["router_wire_calls"] += 1
            route_seen.add(route["case_key"])
        if "logical_usage" in r:
            physical = calls + route["calls"]
            assert (
                abs(
                    r["logical_usage"]["actual_usd"]
                    - sum(c["actual_usd"] for c in physical)
                )
                < 1e-12
            )
            assert r["logical_usage"]["input_tokens"] == sum(
                c["usage"]["input_tokens"] for c in physical
            )

summary = report()
events = rows("results/budget.jsonl")
settled = {e["request_id"]: e["actual_usd"] for e in events if e["event"] == "settle"}
assert abs(sum(settled.values()) - summary["budget"]["actual_usd"]) < 1e-9
assert (
    abs(
        sum(r["new_actual_usd"] for r in summary["costs"])
        - summary["incremental_actual_usd"]
    )
    < 1e-9
)
assert sum(r["valid"] for r in summary["summary"]) == summary["valid"]
assert sum(r["attempted"] for r in summary["summary"]) == summary["attempted"]
result = {
    "status": "passed",
    "counts": counts,
    "budget": Budget().totals(),
    "time": time.time(),
    "caveat": "Small conditional scaling study, not task execution or a general model ranking.",
}
save_json("evidence/openrouter-scaling-audit-" + str(time.time_ns()) + ".json", result)
print(json.dumps(result, indent=2))
