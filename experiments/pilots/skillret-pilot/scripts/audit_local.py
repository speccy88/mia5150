"""Audit saved real Ollama wire traces; makes no generation requests."""

from pathlib import Path
import json
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import rows, read_json, save_json
from local_runner import native_read_events

qids = read_json("manifests/live_query_ids.json")[:5]
queries = {q["id"]: q["query"] for q in rows("data/queries.jsonl") if q["id"] in qids}
records = rows("results/ollama/trial.jsonl")
routes = rows("results/ollama/routes.jsonl")
activations = rows("results/ollama/activation.jsonl")
checks = {
    "selection_decisions": 0,
    "router_calls": 0,
    "activation_cases": 0,
    "native_reads": 0,
}


def inspect_call(call, allowed, query=None):
    wire = call["wire_request"]
    assert {t["function"]["name"] for t in wire["tools"]} == allowed
    assert wire["think"] is False and wire["stream"] is False
    assert wire["options"]["num_ctx"] == 32768 and wire["options"]["temperature"] == 0
    assert wire["options"]["seed"] == 42
    assert call["raw_response"]["model"] == call["model"]
    assert not call["raw_response"]["message"].get("thinking")
    humans = [m["content"] for m in wire["messages"] if m["role"] == "user"]
    if query is not None:
        assert humans == [query]
    # Evaluation objects never enter the wire payload.
    assert not any(
        k in json.dumps(wire)
        for k in ["evaluator_only_gold", "skill_names", "original_query"]
    )
    assert call["input_token_upper_bound"] + wire["options"]["num_predict"] <= 32768


for row in records:
    decision = row.get("decision")
    if not decision or not decision["calls"]:
        continue
    assert len(decision["calls"]) == 1
    expected = {f"/skills/s-{s}/SKILL.md" for s in row["candidate_ids"]}
    assert set(decision["state_file_paths"]) == expected
    for call in decision["calls"]:
        inspect_call(call, {"select_skills"}, queries[row["query_id"]])
        paths = set(
            re.findall(
                r"/skills/s-[0-9a-f-]{36}/SKILL\.md",
                call["wire_request"]["messages"][0]["content"],
            )
        )
        assert paths == expected
        assert "BODY_ONLY_SENTINEL_" not in json.dumps(call["wire_request"])
    if row["status"] == "valid":
        predicted = set(row["returned_ids"])
        gold = set(row["evaluator_only_gold"])
        assert row["selection"]["exact_set_match"] == float(predicted == gold)
        assert (
            abs(row["selection"]["recall"] - len(predicted & gold) / len(gold)) < 1e-12
        )
        assert (
            abs(
                row["selection"]["precision"]
                - (len(predicted & gold) / len(predicted) if predicted else 0)
            )
            < 1e-12
        )
    checks["selection_decisions"] += 1

for row in routes:
    for call in row["calls"]:
        inspect_call(call, {"route_domain"})
        wire = call["wire_request"]
        assert [m["content"] for m in wire["messages"] if m["role"] == "user"][
            0
        ] in queries.values()
        assert "/skills/" not in json.dumps(wire)
        checks["router_calls"] += 1

for row in activations:
    reads = native_read_events(row)
    assert len(row["calls"]) <= 4 and len(reads) <= 2
    for call in row["calls"]:
        inspect_call(call, {"select_skills", "read_file"})
    paths = set(row.get("state_file_paths", []))
    if not paths:
        candidate = next(
            r["candidate_ids"]
            for r in records
            if r["run_id"] == row["run_id"]
            and r["query_id"] == row["query_id"]
            and r["method"] == "vector"
        )
        paths = {f"/skills/s-{sid}/SKILL.md" for sid in candidate}
    for read in reads:
        assert read["args"]["file_path"] in paths
        content = read["output"]["content"]
        notice = "\n[Explicit 12000-character read truncation]"
        assert read["args"].get("limit", 100) <= 100
        if read.get("truncated_chars") or content.endswith(notice):
            assert content[12000:] == notice
        else:
            assert len(content) <= 12000
    if row["status"] == "valid":
        selected = {f"/skills/s-{s}/SKILL.md" for s in row["selected"]}
        assert any(r["args"]["file_path"] in selected for r in row["reads"])
    checks["activation_cases"] += 1
    checks["native_reads"] += len(reads)

ledger = rows("results/ollama/requests.jsonl")
assert len(ledger) <= 100 and len({r["request_id"] for r in ledger}) == len(ledger)
result = {
    "status": "passed",
    "time": time.time(),
    "checks": checks,
    "local_attempts": len(ledger),
    "scope": "Saved real wire messages, tools, canonical query, candidate-only paths and native disclosure. Not a semantic-accuracy test.",
}
save_json("evidence/ollama/audit-" + str(time.time_ns()) + ".json", result)
print(json.dumps(result, indent=2))
