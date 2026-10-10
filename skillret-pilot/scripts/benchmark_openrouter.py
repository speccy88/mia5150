"""Balanced three-query scaling benchmark using the existing native runner.

Same model settings/prompts, strict outputs, persistent budget, no retries.
N=50 evidence is reused explicitly. No additional activation calls are needed.
"""

from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import (
    ROOT,
    config,
    read_json,
    save_json,
    rows,
    append_row,
    digest,
    make_nested_catalogs,
    bm25_rank,
    retrieval_metrics,
    selection_metrics,
)
from agent_runner import (
    GuardedModel,
    Blocked,
    Budget,
    require_permission,
    run_agent,
    sanitize,
)
from trial import (
    experiment_data,
    vectors,
    configured_preflight,
    smoke,
    route,
    usage_totals,
    run_config,
)
from scripts.run_openrouter_models import MODELS

DEST = "results/openrouter/scaling"
SIZES = [50, 250, 1000]
METHODS = ["listing", "bm25", "vector", "domain_vector"]


def select_candidates(
    method, library, query, qid, metadata, v, indices, qindices, major=None
):
    """Retrieve within this query's library without exposing reference labels.

    A predicted category with no candidates remains empty. Falling back to the
    global vector index would hide the cost of a wrong routing decision.
    """
    if method == "listing":
        return library
    if method == "bm25":
        return [
            s
            for s, _ in bm25_rank(
                [metadata[s]["metadata"] for s in library], library, [query], 5
            )[0]
        ]
    pool = [s for s in library if method == "vector" or metadata[s]["major"] == major]
    if not pool:
        return []
    scores = v["docs"][[indices[s] for s in pool]] @ v["queries"][qindices[qid]]
    return [pool[i] for i in np.argsort(-scores, kind="stable")[:5]]


def build_plan(refresh=False):
    configs = {
        label: config(str(ROOT / "manifests/configs" / f"{label}.json"))
        for label in MODELS
    }
    pins = {
        label: configured_preflight(cfg)
        if refresh
        else read_json(cfg["llm"]["pin_manifest"])
        for label, cfg in configs.items()
    }
    spec = {
        "query_ids": read_json("manifests/live_query_ids.json")[:3],
        "sizes": SIZES,
        "methods": METHODS,
        "models": MODELS,
        "protocol": "Same native selector and per-query domain gate; three fixed queries chosen before new results to fit remaining request cap.",
        "sources": {label: run_config(configs[label], pins[label]) for label in MODELS},
        "runner_sha256": digest((ROOT / "scripts/benchmark_openrouter.py").read_text()),
    }
    return configs, pins, spec, digest(spec)


def execute():
    require_permission()
    if Budget().totals()["unresolved_reserved_usd"]:
        raise Blocked("unresolved_reservation; reconcile before continuing")
    configs, pins, spec, run_id = build_plan(True)
    for label in MODELS:
        if not smoke(True, configs[label]).get("canary_correct"):
            raise Blocked(label + ": canary_failed")
    vectors()
    by, qby, gold, metadata, v, indices, qindices = experiment_data()
    mby = {m["id"]: m for m in metadata}
    save_json("manifests/openrouter_scaling.json", {"run_id": run_id, **spec})
    existing = {
        r["case_key"]: r for r in rows(DEST + "/trial.jsonl") if r["run_id"] == run_id
    }
    source = {}
    route_cache = {}
    for label in MODELS:
        old_id = digest(spec["sources"][label])
        source[label] = {
            (r["query_id"], r["N"], r["strategy"]): r
            for r in rows(f"results/openrouter/{label}/trial.jsonl")
            if r["run_id"] == old_id
        }
        for r in rows(f"results/openrouter/{label}/routes.jsonl"):
            if r["case_key"] == digest([old_id, r["query_id"], "router"]):
                route_cache[label, r["query_id"]] = {**r, "source_reused": True}
    for r in rows(DEST + "/routes.jsonl"):
        if r["run_id"] == run_id:
            route_cache[r["model"], r["query_id"]] = r
    paused = {r["model"] for r in existing.values() if r.get("halt_model")}
    catalogs = []
    for qid in spec["query_ids"]:
        nested = make_nested_catalogs(list(by), gold[qid], SIZES, qid)
        for n in SIZES:
            library = list(nested[n])
            catalogs.append({"query_id": qid, "N": n, "ordered_ids": library})
            save_json("manifests/openrouter_scaling_catalogs.json", catalogs)
            for label in MODELS:
                cfg = configs[label]
                pin = pins[label]
                for method in METHODS:
                    key = digest([run_id, label, qid, n, method])
                    if key in existing:
                        continue
                    old = source[label].get((qid, n, method))
                    if old is not None:
                        row = {
                            **old,
                            "run_id": run_id,
                            "case_key": key,
                            "model": label,
                            "source_case_key": old["case_key"],
                            "source_run_id": old["run_id"],
                            "source_reused": True,
                        }
                        append_row(DEST + "/trial.jsonl", row)
                        existing[key] = row
                        continue
                    started = time.monotonic()
                    selector = None
                    routed = None
                    row = {
                        "run_id": run_id,
                        "case_key": key,
                        "model": label,
                        "query_id": qid,
                        "N": n,
                        "strategy": method,
                        "mode": "live",
                        "status": "blocked",
                        "error_category": None,
                        "attempt_count": 0,
                        "candidate_ids": [],
                        "returned_ids": [],
                        "evaluator_only_gold": sorted(gold[qid]),
                        "source_reused": False,
                    }
                    try:
                        if label in paused:
                            raise Blocked("provider_paused_after_failure")
                        if method == "domain_vector":
                            prior = [
                                existing.get(digest([run_id, label, qid, n, m]), {})
                                for m in METHODS[:3]
                            ]
                            if any(r.get("status") != "valid" for r in prior):
                                raise Blocked("selector_prerequisite_failed")
                            if (label, qid) not in route_cache:
                                rkey = digest([run_id, label, qid, "router"])
                                router = GuardedModel(
                                    pin=pin,
                                    cfg=cfg,
                                    case_id=rkey,
                                    max_output=96,
                                    allowed={"route_domain"},
                                )
                                rstart = time.monotonic()
                                try:
                                    routed = {
                                        "status": "valid",
                                        "major": route(qby[qid]["query"], router),
                                    }
                                except Blocked as exc:
                                    routed = {
                                        "status": "blocked",
                                        "error_category": str(exc),
                                    }
                                routed.update(
                                    run_id=run_id,
                                    model=label,
                                    query_id=qid,
                                    case_key=rkey,
                                    calls=sanitize(router.calls),
                                    wall_seconds=time.monotonic() - rstart,
                                    source_reused=False,
                                )
                                append_row(DEST + "/routes.jsonl", routed)
                                route_cache[label, qid] = routed
                            routed = route_cache[label, qid]
                            if routed.get("status", "valid") != "valid":
                                raise Blocked(routed["error_category"])
                            pool = {
                                s for s in library if mby[s]["major"] == routed["major"]
                            }
                            row.update(
                                predicted_major=routed["major"],
                                routing_retention=len(pool & gold[qid])
                                / len(gold[qid]),
                                route_case_key=routed["case_key"],
                            )
                        ranked = select_candidates(
                            method,
                            library,
                            qby[qid]["query"],
                            qid,
                            mby,
                            v,
                            indices,
                            qindices,
                            routed["major"] if routed else None,
                        )
                        chosen = set(ranked)
                        display = [s for s in library if s in chosen]
                        row.update(
                            candidate_ids=display,
                            retrieval_ranking=ranked,
                            candidate_recall=len(chosen & gold[qid]) / len(gold[qid]),
                            candidate_completeness=float(gold[qid] <= chosen),
                            first_stage=retrieval_metrics(
                                ranked, gold[qid], n if method == "listing" else 5
                            ),
                        )
                        if not display:
                            row["status"] = "empty_domain"
                            raise Blocked("empty_domain")
                        selector = GuardedModel(
                            pin=pin, cfg=cfg, case_id=key, max_output=512
                        )
                        decision = run_agent(
                            qby[qid]["query"], [by[s] for s in display], selector
                        )
                        row.update(
                            status=decision["status"],
                            decision=decision,
                            returned_ids=decision["selected"],
                            selection=selection_metrics(
                                decision["selected"], gold[qid]
                            ),
                            attempt_count=len(selector.calls),
                            selector_usage=usage_totals(selector.calls),
                            logical_usage=usage_totals(
                                selector.calls + (routed["calls"] if routed else [])
                            ),
                            logical_workflow_seconds=sum(
                                c.get("latency_seconds", 0) for c in selector.calls
                            )
                            + (routed["wall_seconds"] if routed else 0),
                        )
                        if decision["status"] != "valid":
                            row["error_category"] = decision["status"]
                    except Blocked as exc:
                        error = str(exc)
                        row["error_category"] = error
                        if error == "context_limit":
                            row["status"] = "context_limit"
                        if selector is not None:
                            row.update(
                                attempt_count=len(selector.calls),
                                failed_calls=sanitize(selector.calls),
                            )
                        benign = {
                            "provider_paused_after_failure",
                            "selector_prerequisite_failed",
                            "router_malformed",
                            "router_unknown_category",
                            "empty_domain",
                            "context_limit",
                            "budget_limit",
                            "request_cap",
                        }
                        if error not in benign:
                            row["halt_model"] = True
                            paused.add(label)
                    row["workflow_wall_seconds"] = time.monotonic() - started
                    append_row(DEST + "/trial.jsonl", sanitize(row))
                    existing[key] = row
                    print(
                        json.dumps(
                            {
                                k: row.get(k)
                                for k in [
                                    "model",
                                    "query_id",
                                    "N",
                                    "strategy",
                                    "status",
                                    "error_category",
                                ]
                            }
                        ),
                        flush=True,
                    )
                    if Budget().totals()["unresolved_reserved_usd"]:
                        raise Blocked(
                            "unresolved_reservation; current failure retained; reconcile before continuing"
                        )
                    if row.get("error_category") in {"budget_limit", "request_cap"}:
                        raise Blocked(row["error_category"])
    return {
        "run_id": run_id,
        "saved": len(existing),
        "planned": 180,
        "valid": sum(r["status"] == "valid" for r in existing.values()),
        "budget": Budget().totals(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true")
    args = p.parse_args()
    if not args.live:
        _, _, spec, rid = build_plan()
        print(
            json.dumps(
                {
                    "run_id": rid,
                    "query_ids": spec["query_ids"],
                    "planned": 180,
                    "budget": Budget().totals(),
                },
                indent=2,
            )
        )
        return 0
    try:
        result = execute()
        code = 0 if result["valid"] == result["planned"] else 2
    except Exception as exc:
        result = {
            "status": "blocked",
            "error_type": type(exc).__name__,
            "error": sanitize(str(exc)),
        }
        code = 2
    append_row(
        "evidence/openrouter_scaling_stages.jsonl",
        {"time": time.time(), "result": result, "exit_code": code},
    )
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
