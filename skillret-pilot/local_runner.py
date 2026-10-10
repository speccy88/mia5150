"""Small, user-authorized Ollama follow-up; separate from the OpenRouter protocol.

Reuses the real Deep Agents graph and candidate-only StateBackend. No remote
endpoint, paid API, model download, skill execution or fallback is permitted.
"""

from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import csv
import fcntl
import importlib.metadata as md
import json
from pathlib import Path
import time
import uuid

import httpx
import numpy as np
from langchain_core.messages import messages_to_dict, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import Field
from agent_runner import GuardedModel, Blocked, run_agent, fixtures, sanitize
from pilot import (
    ROOT,
    save_json,
    read_json,
    rows,
    append_row,
    digest,
    make_nested_catalogs,
    bm25_rank,
    retrieval_metrics,
    selection_metrics,
)
from trial import experiment_data, route, usage_totals

BASE_URL = "http://127.0.0.1:11434"
SETTINGS = {
    "num_ctx": 32768,
    "num_predict": 512,
    "temperature": 0.0,
    "seed": 42,
    "reasoning": False,
    "timeout_seconds": 90,
    "keep_alive": "5m",
    "max_requests": 100,
    "query_count": 5,
    "library_size": 50,
    "shortlist": 5,
    "retry_count": 0,
}
UUID_CONTRACT = """Output format clarification: every select_skills.skill_ids item must
be the 36-character original UUID, containing only hexadecimal digits and four
hyphens. The native skill name starts with s-; omit those first two characters
when forming a selection ID. A name starting with s- or a file path is invalid
in skill_ids. For read_file, retain the complete displayed path, including s-.
This formatting rule does not change which skills to select."""


def local_contract(messages, tools):
    """One documented, fixed contract clarification; never repair model output."""
    messages = list(messages)
    tools = deepcopy(tools)
    if any(t["function"]["name"] == "select_skills" for t in tools):
        if messages and messages[0].type == "system":
            content = messages[0].content
            content = (
                content + "\n\n" + UUID_CONTRACT
                if isinstance(content, str)
                else [*content, {"type": "text", "text": UUID_CONTRACT}]
            )
            messages[0] = messages[0].model_copy(update={"content": content})
        else:
            messages.insert(0, SystemMessage(content=UUID_CONTRACT))
        for tool in tools:
            if tool["function"]["name"] == "select_skills":
                field = tool["function"]["parameters"]["properties"]["skill_ids"]
                field["description"] = (
                    "Original 36-character UUIDs only; omit the native name prefix s-. Never return paths."
                )
                field["items"]["pattern"] = (
                    "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
                )
    return messages, tools


def inventory():
    with httpx.Client(base_url=BASE_URL, timeout=10, trust_env=False) as client:
        version = client.get("/api/version")
        version.raise_for_status()
        tags = client.get("/api/tags")
        tags.raise_for_status()
        observed = {"server_version": version.json(), "models": []}
        for item in tags.json()["models"]:
            info = client.post("/api/show", json={"model": item["name"]})
            info.raise_for_status()
            info = info.json()
            if (
                item.get("remote_host")
                or info.get("remote_host")
                or "cloud" in item["name"]
            ):
                continue
            observed["models"].append(
                {
                    "model": item["name"],
                    "digest": item["digest"],
                    "size_bytes": item["size"],
                    "details": info.get("details"),
                    "model_info": info.get("model_info"),
                    "capabilities": info.get("capabilities", []),
                    "thinking": info.get("thinking"),
                    "parameters": info.get("parameters"),
                    "server_version": version.json()["version"],
                }
            )
    save_json(f"evidence/ollama/inventory-{time.time_ns()}.json", observed)
    return observed


def pins(selected=None):
    info = inventory()
    available = [
        x for x in info["models"] if selected is None or x["model"] == selected
    ]
    if not available:
        raise Blocked("requested_local_model_not_installed")
    out = []
    for model in sorted(available, key=lambda m: m["size_bytes"]):
        if "tools" not in model["capabilities"]:
            raise Blocked("model_does_not_advertise_tools")
        if False not in (model.get("thinking") or {}).get("values", []):
            raise Blocked("cannot_disable_thinking")
        limits = [
            v for k, v in model["model_info"].items() if k.endswith(".context_length")
        ]
        if not limits or min(limits) < SETTINGS["num_ctx"]:
            raise Blocked("insufficient_advertised_context")
        spec = {
            "engine": "local_ollama",
            "base_url": BASE_URL,
            **model,
            "settings": SETTINGS,
            "dataset_revision": read_json("manifests/dataset.json")["revision"],
            "metadata_hash": read_json("manifests/embedding.json")["metadata_hash"],
            "versions": {
                p: md.version(p)
                for p in ["deepagents", "langchain-core", "langchain-ollama", "ollama"]
            },
            "protocol_code": {
                p: digest((ROOT / p).read_text().split("\ndef report():", 1)[0])
                for p in ["agent_runner.py", "reference_metrics.py", "local_runner.py"]
            },
            "local_contract_revision": 2,
            "local_contract": UUID_CONTRACT,
        }
        run_id = digest(spec)
        save_json(f"manifests/ollama/{run_id}.json", spec)
        out.append((run_id, spec))
    return out


def adapter(pin, request_capture, response_capture, transport=None):
    def on_request(request):
        if str(request.url) != BASE_URL + "/api/chat":
            raise Blocked("nonlocal_or_unexpected_endpoint")
        request_capture.update(sanitize(json.loads(request.content)))

    def on_response(response):
        response.read()
        response_capture.update(sanitize(response.json()))

    kwargs = {
        "timeout": SETTINGS["timeout_seconds"],
        "trust_env": False,
        "follow_redirects": False,
        "event_hooks": {"request": [on_request], "response": [on_response]},
    }
    if transport is not None:
        kwargs["transport"] = transport
    return ChatOllama(
        model=pin["model"],
        base_url=BASE_URL,
        reasoning=False,
        temperature=0,
        num_ctx=SETTINGS["num_ctx"],
        num_predict=SETTINGS["num_predict"],
        seed=42,
        keep_alive=SETTINGS["keep_alive"],
        sync_client_kwargs=kwargs,
        async_client_kwargs={
            "trust_env": False,
            "timeout": SETTINGS["timeout_seconds"],
        },
    )


class LocalModel(GuardedModel):
    """Local counterpart to the guarded OpenRouter boundary; same graph/hooks."""

    run_id: str = ""
    ledger_path: Path = Field(
        default_factory=lambda: ROOT / "results/ollama/requests.jsonl"
    )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if len(self.calls) >= self.max_calls:
            raise Blocked("per_workflow_call_cap")
        tools = kwargs.get("tools", [])
        if {x["function"]["name"] for x in tools} != self.allowed:
            raise Blocked("effective_tool_allowlist_mismatch")
        messages, tools = local_contract(messages, tools)
        payload = {"messages": messages_to_dict(messages), "tools": tools}
        bound = len(json.dumps(payload, ensure_ascii=False).encode()) + 2048
        if bound + self.max_output > SETTINGS["num_ctx"]:
            raise Blocked("local_context_limit_no_truncation")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        # The process lock spans the physical request; no parallel local generations.
        with self.ledger_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if len(rows(self.ledger_path)) >= SETTINGS["max_requests"]:
                raise Blocked("local_request_cap")
            request_id = str(uuid.uuid4())
            entry = {
                "request_id": request_id,
                "run_id": self.run_id,
                "case_key": self.case_id,
                "model": self.pin["model"],
                "model_digest": self.pin["digest"],
                "time": time.time(),
                "api_charge_usd": 0,
            }
            append_row(self.ledger_path, entry)
            call = {
                **entry,
                **payload,
                "mode": "real_local_inference",
                "input_token_upper_bound": bound,
                "actual_usd": 0,
                "cost_scope": "local API charge only; electricity/hardware not measured",
            }
            self.calls.append(call)
            append_row("evidence/ollama/requests.jsonl", sanitize(call))
            sent = {}
            raw = {}
            client = adapter(self.pin, sent, raw)
            started = time.monotonic()
            try:
                client.num_predict = self.max_output
                result = client._generate(
                    messages, stop=stop, tools=tools, stream=False
                )
                answer = result.generations[0].message
                call.update(
                    response=answer.model_dump(mode="json"),
                    usage=answer.usage_metadata,
                    actual_model=raw.get("model"),
                    completion_id=request_id,
                    done_reason=raw.get("done_reason"),
                )
                if raw.get("model") != self.pin["model"]:
                    raise Blocked("local_model_changed")
                if not raw.get("done") or raw.get("done_reason") == "length":
                    raise Blocked("local_output_incomplete")
                if not answer.usage_metadata:
                    raise Blocked("local_usage_missing")
                if raw.get("message", {}).get("thinking"):
                    raise Blocked("thinking_was_not_disabled")
                if (
                    raw.get("prompt_eval_count", 0) + self.max_output
                    >= SETTINGS["num_ctx"]
                ):
                    raise Blocked("context_capacity_observed")
                return result
            except Exception as e:
                call.update(error_type=type(e).__name__, error=sanitize(str(e)))
                raise Blocked(type(e).__name__ + ": " + sanitize(str(e))) from None
            finally:
                call.update(
                    wire_request=sent,
                    raw_response=raw,
                    latency_seconds=time.monotonic() - started,
                )
                append_row("evidence/ollama/responses.jsonl", sanitize(call))
                client._client._client.close()


def decision(run_id, pin, key, query, candidates, activation=False):
    model = LocalModel(run_id=run_id, pin=pin, case_id=key)
    start = time.monotonic()
    try:
        result = run_agent(query, candidates, model, activation)
    except Exception as e:
        result = {
            "status": "blocked",
            "selected": [],
            "calls": sanitize(model.calls),
            "reads": [],
            "error_category": type(e).__name__ + ": " + sanitize(str(e)),
        }
    result.update(workflow_wall_seconds=time.monotonic() - start)
    return result


def canary(run_id, pin):
    key = digest([run_id, "canary"])
    previous = [
        r for r in rows("results/ollama/canaries.jsonl") if r["case_key"] == key
    ]
    if previous:
        return previous[-1]
    result = decision(run_id, pin, key, "Add two numbers.", fixtures()[:2])
    row = {"case_key": key, "run_id": run_id, "model": pin["model"], **result}
    row["correct"] = row["status"] == "valid" and row["selected"] == [
        fixtures()[0]["id"]
    ]
    append_row("results/ollama/canaries.jsonl", row)
    return row


def smoke(selected=None):
    output = []
    for run_id, pin in pins(selected):
        result = canary(run_id, pin)
        output.append(
            {
                "model": pin["model"],
                "run_id": run_id,
                "status": result["status"],
                "correct": result["correct"],
                "seconds": result["workflow_wall_seconds"],
                "usage": usage_totals(result["calls"]),
            }
        )
    return output


def run(selected=None):
    by, qby, gold, metadata, v, indices, qindices = experiment_data()
    mby = {m["id"]: m for m in metadata}
    qids = read_json("manifests/live_query_ids.json")[:5]
    output = []
    for run_id, pin in pins(selected):
        check = canary(run_id, pin)
        if not check["correct"]:
            output.append(
                {"model": pin["model"], "status": "canary_failed", "run_id": run_id}
            )
            continue
        previous = {r["case_key"]: r for r in rows("results/ollama/trial.jsonl")}
        cached = {r["case_key"]: r for r in rows("results/ollama/routes.jsonl")}
        current = []
        stop_model = False
        for qid in qids:
            library = list(make_nested_catalogs(list(by), gold[qid], [50], qid)[50])
            save_json(
                f"manifests/ollama/catalogs/{qid}.json",
                {"query_id": qid, "N": 50, "ordered_ids": library},
            )
            for method in ["listing", "bm25", "vector", "domain_vector"]:
                key = digest([run_id, qid, 50, method])
                start = time.monotonic()
                routed = None
                if key in previous:
                    current.append(previous[key])
                    continue
                row = {
                    "case_key": key,
                    "run_id": run_id,
                    "model": pin["model"],
                    "query_id": qid,
                    "N": 50,
                    "method": method,
                    "evaluator_only_gold": sorted(gold[qid]),
                    "candidate_ids": [],
                    "returned_ids": [],
                    "attempt_count": 0,
                    "status": "blocked",
                }
                try:
                    if method == "listing":
                        ranked = library
                    elif method == "bm25":
                        ranked = [
                            s
                            for s, score in bm25_rank(
                                [mby[s]["metadata"] for s in library],
                                library,
                                [qby[qid]["query"]],
                                5,
                            )[0]
                        ]
                    else:
                        pool = library
                        if method == "domain_vector":
                            prior = [
                                r
                                for r in current
                                if r["query_id"] == qid
                                and r["method"] in ["listing", "bm25", "vector"]
                            ]
                            if len(prior) != 3 or any(
                                r["status"] != "valid" for r in prior
                            ):
                                raise Blocked("selector_prerequisite_failed")
                            rkey = digest([run_id, qid, "route"])
                            if rkey in cached:
                                routed = cached[rkey]
                            else:
                                router = LocalModel(
                                    run_id=run_id,
                                    pin=pin,
                                    case_id=rkey,
                                    max_output=96,
                                    allowed={"route_domain"},
                                )
                                rstart = time.monotonic()
                                try:
                                    major = route(qby[qid]["query"], router)
                                except Blocked as e:
                                    append_row(
                                        "results/ollama/routes.jsonl",
                                        {
                                            "case_key": rkey,
                                            "run_id": run_id,
                                            "status": "failed",
                                            "calls": sanitize(router.calls),
                                            "error": str(e),
                                        },
                                    )
                                    raise
                                routed = {
                                    "case_key": rkey,
                                    "run_id": run_id,
                                    "major": major,
                                    "status": "valid",
                                    "calls": router.calls,
                                    "seconds": time.monotonic() - rstart,
                                }
                                append_row("results/ollama/routes.jsonl", routed)
                                cached[rkey] = routed
                            if routed["status"] != "valid":
                                raise Blocked("previous_router_failure")
                            pool = [
                                s for s in library if mby[s]["major"] == routed["major"]
                            ]
                            row.update(
                                predicted_major=routed["major"],
                                routing_retention=len(set(pool) & gold[qid])
                                / len(gold[qid]),
                            )
                            if not pool:
                                row.update(
                                    status="empty_domain",
                                    candidate_recall=0,
                                    candidate_completeness=0,
                                    selection=selection_metrics([], gold[qid]),
                                )
                                raise Blocked("empty_domain_no_fallback")
                        scores = (
                            v["docs"][[indices[s] for s in pool]]
                            @ v["queries"][qindices[qid]]
                        )
                        ranked = [
                            pool[i] for i in np.argsort(-scores, kind="stable")[:5]
                        ]
                    display = [s for s in library if s in set(ranked)]
                    row.update(
                        candidate_ids=display,
                        retrieval_ranking=ranked,
                        candidate_recall=len(set(display) & gold[qid]) / len(gold[qid]),
                        candidate_completeness=float(gold[qid] <= set(display)),
                        retrieval_seconds=time.monotonic() - start,
                    )
                    result = decision(
                        run_id, pin, key, qby[qid]["query"], [by[s] for s in display]
                    )
                    row.update(
                        status=result["status"],
                        returned_ids=result["selected"],
                        decision=result,
                        attempt_count=len(result["calls"]),
                        selection=selection_metrics(result["selected"], gold[qid]),
                        logical_usage=usage_totals(
                            result["calls"] + (routed["calls"] if routed else [])
                        ),
                        route_case_key=routed["case_key"] if routed else None,
                        logical_seconds=time.monotonic() - start,
                    )
                    if result["status"] == "blocked":
                        stop_model = True
                except Blocked as e:
                    row["error_category"] = str(e)
                append_row("results/ollama/trial.jsonl", sanitize(row))
                current.append(row)
                print(
                    json.dumps(
                        {
                            "model": pin["model"],
                            "query": qid,
                            "method": method,
                            "status": row["status"],
                            "exact": row.get("selection", {}).get("exact_set_match"),
                        }
                    ),
                    flush=True,
                )
                if stop_model:
                    break
            if stop_model:
                break
        # Disclosure is a separate task, never included in selection accuracy.
        if not stop_model:
            cases = [
                ("fixture-" + str(i), text, fixtures())
                for i, text in enumerate(
                    [
                        "Add two numbers.",
                        "Translate a French paragraph.",
                        "Draw a square.",
                    ]
                )
            ]
            for qid in qids[:2]:
                candidate = next(
                    (
                        r["candidate_ids"]
                        for r in current
                        if r["query_id"] == qid and r["method"] == "vector"
                    ),
                    [],
                )
                cases.append((qid, qby[qid]["query"], [by[s] for s in candidate]))
            for label, query, candidates in cases:
                key = digest([run_id, label, "activation"])
                if any(
                    r["case_key"] == key
                    for r in rows("results/ollama/activation.jsonl")
                ):
                    continue
                result = decision(run_id, pin, key, query, candidates, True)
                append_row(
                    "results/ollama/activation.jsonl",
                    {
                        "case_key": key,
                        "run_id": run_id,
                        "model": pin["model"],
                        "query_id": label,
                        **result,
                    },
                )
                print(
                    json.dumps(
                        {
                            "model": pin["model"],
                            "activation": label,
                            "status": result["status"],
                            "reads": len(result["reads"]),
                        }
                    ),
                    flush=True,
                )
                if result["status"] == "blocked":
                    break
        output.append(
            {
                "model": pin["model"],
                "run_id": run_id,
                "cases": len(current),
                "valid": sum(r["status"] == "valid" for r in current),
            }
        )
    return output


def report():
    all_rows = rows("results/ollama/trial.jsonl")
    canaries = rows("results/ollama/canaries.jsonl")
    latest = {r["model"]: r["run_id"] for r in canaries}
    records = list(
        {
            r["case_key"]: r for r in all_rows if r["run_id"] == latest[r["model"]]
        }.values()
    )
    table = []
    for model, run_id in latest.items():
        for method in ["listing", "bm25", "vector", "domain_vector"]:
            group = [
                r for r in records if r["model"] == model and r["method"] == method
            ]
            valid = [r for r in group if r["status"] == "valid"]
            average = lambda values: sum(values) / len(values) if values else None
            table.append(
                {
                    "model": model,
                    "method": method,
                    "N": 50,
                    "planned": 5,
                    "recorded": len(group),
                    "valid": len(valid),
                    "attempted": sum(r.get("attempt_count", 0) > 0 for r in group),
                    "exact_matches": sum(
                        r["selection"]["exact_set_match"] for r in valid
                    ),
                    "conditional_exact_match": average(
                        [r["selection"]["exact_set_match"] for r in valid]
                    ),
                    "precision": average([r["selection"]["precision"] for r in valid]),
                    "recall": average([r["selection"]["recall"] for r in valid]),
                    "f1": average([r["selection"]["f1"] for r in valid]),
                    "retrieval_cases": sum("candidate_recall" in r for r in group),
                    "candidate_recall": average(
                        [
                            r["candidate_recall"]
                            for r in group
                            if "candidate_recall" in r
                        ]
                    ),
                    "mean_workflow_seconds": average(
                        [r["logical_seconds"] for r in group if "logical_seconds" in r]
                    ),
                    "mean_logical_tokens": average(
                        [
                            r["logical_usage"]["input_tokens"]
                            + r["logical_usage"]["output_tokens"]
                            for r in group
                            if "logical_usage" in r
                        ]
                    ),
                }
            )
    path = ROOT / "reports/OLLAMA_RESULTS.csv"
    if table:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    activation = [
        r
        for r in rows("results/ollama/activation.jsonl")
        if r["run_id"] == latest.get(r["model"])
    ]
    read_events = [
        {"case_key": r["case_key"], "model": r["model"], "query_id": r["query_id"], **e}
        for r in activation
        for e in native_read_events(r)
    ]
    save_json(
        "evidence/ollama/derived_native_reads.json",
        {
            "source": "deduplicated native ToolMessage results; requests without a result are excluded",
            "events": read_events,
        },
    )
    calls = rows("evidence/ollama/responses.jsonl")
    strata = []
    for model in latest:
        for method in ["listing", "bm25", "vector", "domain_vector"]:
            for label in ["single", "multiple"]:
                group = [
                    r
                    for r in records
                    if r["model"] == model
                    and r["method"] == method
                    and (len(r["evaluator_only_gold"]) == 1) == (label == "single")
                ]
                valid = [r for r in group if r["status"] == "valid"]
                strata.append(
                    {
                        "model": model,
                        "method": method,
                        "gold_group": label,
                        "recorded": len(group),
                        "valid": len(valid),
                        "exact_matches": sum(
                            r["selection"]["exact_set_match"] for r in valid
                        ),
                    }
                )
    with (ROOT / "reports/OLLAMA_SINGLE_MULTI.csv").open("w", newline="") as f:
        if strata:
            writer = csv.DictWriter(f, fieldnames=list(strata[0]))
            writer.writeheader()
            writer.writerows(strata)
    summary = {
        "runs": latest,
        "selection": table,
        "canaries": [
            {
                "model": r["model"],
                "run_id": r["run_id"],
                "status": r["status"],
                "correct": r["correct"],
                "selected": r["selected"],
            }
            for r in canaries
        ],
        "failures": [
            {
                "model": r["model"],
                "method": r["method"],
                "query_id": r["query_id"],
                "status": r["status"],
                "error": r.get("error_category"),
            }
            for r in records
            if r["status"] != "valid"
        ],
        "requests_total_including_preserved_attempts": len(
            rows("results/ollama/requests.jsonl")
        ),
        "http_requests": sum(bool(c.get("wire_request")) for c in calls),
        "completed_requests": sum("usage" in c for c in calls),
        "api_charge_usd": 0,
        "power_and_hardware_cost": "not measured",
        "physical_tokens": {
            **usage_totals(calls),
            "cached_tokens": None,
            "reasoning_tokens": None,
        },
        "router": [
            {
                "model": m,
                "routes": len(
                    [r for r in records if r["model"] == m and "predicted_major" in r]
                ),
                "mean_gold_retention": average(
                    [
                        r["routing_retention"]
                        for r in records
                        if r["model"] == m and "routing_retention" in r
                    ]
                ),
            }
            for m in latest
        ],
        "activation": [
            {
                "model": m,
                "cases": len([r for r in activation if r["model"] == m]),
                "planned": 5,
                "valid": sum(
                    r["model"] == m and r["status"] == "valid" for r in activation
                ),
                "fixture_valid": sum(
                    r["model"] == m
                    and r["status"] == "valid"
                    and r["query_id"].startswith("fixture-")
                    for r in activation
                ),
                "real_valid": sum(
                    r["model"] == m
                    and r["status"] == "valid"
                    and not r["query_id"].startswith("fixture-")
                    for r in activation
                ),
                "native_reads": sum(e["model"] == m for e in read_events),
                "real_native_reads": sum(
                    e["model"] == m and not e["query_id"].startswith("fixture-")
                    for e in read_events
                ),
                "failures": [
                    {
                        "query_id": r["query_id"],
                        "status": r["status"],
                        "error": r.get("error_category"),
                    }
                    for r in activation
                    if r["model"] == m and r["status"] != "valid"
                ],
            }
            for m in latest
        ],
    }
    save_json("reports/ollama_summary.json", summary)
    fmt = lambda value: f"{value:.3f}" if value is not None else "—"
    table_text = "\n".join(
        f"| {r['model']} | {r['method']} | {r['valid']}/{r['planned']} | {str(int(r['exact_matches']))+'/'+str(r['valid']) if r['valid'] else '— (0 valid)'} | {fmt(r['candidate_recall'])} ({r['retrieval_cases']}) | {fmt(r['mean_workflow_seconds'])} |"
        for r in table
    )
    failure_counts = defaultdict(int)
    for r in summary["failures"]:
        failure_counts[(r["model"], r["error"] or r["status"])] += 1
    failure_text = (
        "\n".join(
            f"- {m}: {n} {reason} cases." for (m, reason), n in failure_counts.items()
        )
        or "No recorded comparison failures."
    )
    activation_text = "\n".join(
        f"| {r['model']} | {r['cases']}/{r['planned']} | {r['fixture_valid']}/3 | {r['real_valid']}/2 | {r['native_reads']} |"
        for r in summary["activation"]
    )
    activation_failures = "\n".join(
        f"- {r['model']}, {f['query_id']}: {f['error'] or f['status']}."
        for r in summary["activation"]
        for f in r["failures"]
    )
    route_text = "\n".join(
        f"- {r['model']}: {r['routes']} routes; mean gold retention {r['mean_gold_retention']:.3f}."
        if r["routes"]
        else f"- {r['model']}: no eligible router calls."
        for r in summary["router"]
    )
    text = f"""# Local Ollama follow-up — real Deep Agents inference

The user authorized local Ollama testing instead of OpenRouter. These results are a separate provider/model protocol; they do not complete the original OpenRouter acceptance milestone.

**Assessment: conditional go for a local feasibility study.** Real native metadata discovery, local inference and strict tool boundaries are exercised. Saved format and activation failures remain limitations for unattended expansion; this is not evidence of successful skill execution or general method superiority.

Both installed model digests, server version, quantization and settings are pinned in `manifests/ollama/`. The native Deep Agents graph, metadata and tool isolation are retained. The local protocol adds one fixed UUID-format clarification to the system instruction and tool schema, uniformly across both models and all methods. Initial canaries returned the native s- prefix and were rejected; their failures remain saved. No model output is repaired. There are five fixed SkillRet pilot queries, N=50, top-five shortlists, gold-preserving random libraries and the same display order. No new final-holdout queries are consumed. No model downloads or paid API calls occur.

| Local model | Method | Valid/planned | Exact matches/valid | Candidate recall (queries) | Mean attempted workflow seconds |
|---|---|---:|---:|---:|---:|
{table_text}

Conditional accuracy excludes invalid completions; valid/planned exposes missing/failed cases. Detailed recall/F1, token counts and failures are in OLLAMA_RESULTS.csv and append-only `results/ollama/`. Logical domain-vector usage includes its query-only router. Cold model loading and prompt processing are included in wall time; these are not pure generation-speed measurements. Five queries are too few for a general model ranking.

Latency and token means include invalid completed attempts as well as valid ones; gated cases without inference are excluded. Physical totals include all canaries, comparison/routing calls and activation attempts, including failures.

The domain arm is attempted for a query only when listing, BM25 and vector selectors returned valid IDs. This gate and format failures yield different query subsets, so the conditional averages are diagnostics, not paired method comparisons. No failed output is repaired or sampled again for accuracy.

Post-hoc router retention against labels:

{route_text}

Labels are evaluated after routing and never enter router state. Single/multiple-positive denominators are in `OLLAMA_SINGLE_MULTI.csv`.

Separate activation results:

| Model | Recorded/planned | Valid fixtures | Valid SkillRet cases | Native reads |
|---|---:|---:|---:|---:|
{activation_text}

{activation_failures}

Actual native read_file calls demonstrate disclosure, not execution of skill instructions. Each model plans three hand-authored fixtures and two real SkillRet queries. One fixture canary precedes each model's comparison.

Native reads count only actual ToolMessage results, deduplicated by tool-call ID. A graph that fails after a read can return no structured read list; its already-saved model-message history retains the completed native tool results. `evidence/ollama/derived_native_reads.json` recovers those events without changing raw rows. Requested reads rejected by a guard are never counted as reads. Observed real SkillRet native reads: **{sum(r['real_native_reads'] for r in summary['activation'])}**; valid real activation workflows: **{sum(r['real_valid'] for r in summary['activation'])}/{2*len(latest)} planned**. A subsequent denied pagination or out-of-candidate request still makes that activation workflow a failure.

Recorded comparison failures:

{failure_text}

Canary history, failure query IDs and exact run IDs are in `ollama_summary.json`; failed canaries block comparison for that configuration.

Total local generation attempts including preserved failures: {summary['requests_total_including_preserved_attempts']}; actual HTTP requests: {summary['http_requests']}; completed responses with usage: {summary['completed_requests']}. The initial adapter argument error occurred before HTTP. Local API charge: USD 0 / CAD 0; electricity and hardware cost were not measured. Original OpenRouter cost ledger remains separate.

Settings: temperature=0, seed=42, think=false, 32768 context, max output 512 (router 96), sequential calls, 90-second timeout, no automatic retries, persistent 100-request cap. The request uses a conservative full-message/tool UTF-8 byte bound before inference; no context is silently truncated. Returned usage and thinking behavior are checked. Local inference requires no OpenRouter key or paid opt-in; the original paid route retains all guards.

Request UUIDs are client-assigned trace identifiers; the local response does not supply a provider completion ID. Model identity is recorded by installed digest and returned model tag.

Input/output token counts come from returned usage. Ollama did not expose separate cached-token or reasoning-token counters here; zeros in the shared per-case accumulator are placeholders for unavailable detail, not measured cache/reasoning totals. The summary marks those counters null. Thinking was explicitly disabled in the wire request and no thinking content was returned.

Run `python pilot.py ollama-smoke`, `python pilot.py ollama-run`, then `python pilot.py ollama-report` in the project venv. Add `--model qwen3.5:4b-mlx` or `--model qwen3.5:9b-mlx` to restrict a run. Exact matching cases are reused; failure traces remain.

API references: [Ollama chat](https://docs.ollama.com/api/chat), [tool calling](https://docs.ollama.com/capabilities/tool-calling), [ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama). Installed interfaces and raw request/response hooks were used to verify actual behavior.
"""
    (ROOT / "reports/OLLAMA_TRIAL_REPORT.md").write_text(text)
    return summary


def native_read_events(record):
    """Recover completed native reads even when a later guard aborts the graph."""
    events = {}
    requested = {}
    for call in record["calls"]:
        for message in call["messages"]:
            data = message["data"]
            if message["type"] == "ai":
                requested.update({t["id"]: t for t in data.get("tool_calls", [])})
            if message["type"] == "tool" and data.get("name") == "read_file":
                key = data["tool_call_id"]
                request = requested[key]
                if (
                    request["name"] != "read_file"
                    or data.get("status", "success") != "success"
                ):
                    continue
                events[key] = {
                    "tool_call_id": key,
                    "args": request["args"],
                    "output": data,
                    "source": "saved_native_tool_message",
                }
    for read in record["reads"]:
        key = read["output"]["tool_call_id"]
        if key in events:
            assert (
                events[key]["args"] == read["args"]
                and events[key]["output"]["content"] == read["output"]["content"]
            )
        events[key] = {
            "tool_call_id": key,
            **read,
            "source": "structured_native_read_trace",
        }
    return list(events.values())
