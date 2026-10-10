"""Offline Jev preparation and cached-response ingestion; hosted execution is disabled.

This module imports no network or credential APIs. Preparation is not an API run.
The exact rubric is frozen before evaluating any Jev response. Prior source contract:
mia5150-local/round2/probe_contract.py (typed Score, zero-based continuous score).
Actual provider contract/context/version must be checked after the baseline gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "skillret-jev-offline-v1"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
FUTURE_CREDENTIAL_ENV = "TYPESAFE_API_KEY"  # A name only; never read in this module.
RRF_K = 60
POOL_SIZE = 20
QUESTION = (
    "How well does the following agent skill satisfy the user’s request? "
    "Consider functionality, required capabilities, constraints, and whether "
    "the skill would actually help accomplish the request.\n\nCandidate skill:\n"
)
RUBRIC = [
    "Irrelevant — does not meaningfully satisfy the request.",
    "Weak match — shares some topic or functionality but would not adequately solve the request.",
    "Partial match — useful for part of the request but incomplete or missing an important requirement.",
    "Strong match — directly addresses the request with only minor limitations.",
    "Excellent match — directly and comprehensively matches the requested capability.",
]
PHASE_BLOCK = (
    "Hosted Jev execution is unavailable in this preparation-only implementation. "
    "No credential was read and no request was sent. Before implementing execution, "
    "validate the full official baseline and verify the explicit experiment budget, "
    "current provider price/context/model contract, FX/fees buffer, and existing credit. "
    "The only future credential source is TYPESAFE_API_KEY. Never load saved secrets."
)


class ContractError(ValueError):
    """A source, preparation, or cached-response contract failed."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("Duplicate JSON object key")
        result[key] = value
    return result


def loads(text: str) -> Any:
    def reject_constant(_: str) -> None:
        raise ContractError("Nonfinite JSON number")
    return json.loads(text, object_pairs_hook=no_duplicate_keys, parse_constant=reject_constant)


def read_json(path: Path) -> Any:
    return loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = loads(line)
                if not isinstance(row, dict):
                    raise ContractError("JSONL rows must be objects")
                rows.append(row)
    return rows


def write_new(path: Path, value: Any) -> None:
    """Idempotent byte-identical resume; never replace a prior artifact."""
    data = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ContractError("Output exists with different content; use a new directory")


def identifier(row: dict, preferred: str) -> str:
    value = row.get(preferred, row.get("id"))
    if preferred in row and "id" in row and row[preferred] != row["id"]:
        raise ContractError("Conflicting identifier aliases")
    if not isinstance(value, str) or not value:
        raise ContractError("Identifiers must be nonempty strings")
    return value


def finite(value: Any, label: str, minimum: float | None = None,
           maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(label + " must be finite numeric")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ContractError(label + " outside permitted range")
    return float(value)


def token_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(label + " must be a nonnegative integer")
    return value


def index_rows(rows: list[dict], preferred: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = identifier(row, preferred)
        if key in result:
            raise ContractError("Duplicate source identifier")
        result[key] = row
    return result


def select_ids(query_ids: list[str], count: int = 200) -> list[str]:
    if len(set(query_ids)) != len(query_ids):
        raise ContractError("Duplicate query IDs")
    return sorted(query_ids, key=lambda q: (hashlib.sha256(q.encode("utf-8")).hexdigest(), q))[:count]


def document(skill: dict) -> str:
    values = [skill.get(field) for field in ("name", "description", "skill_md")]
    if any(not isinstance(value, str) for value in values):
        raise ContractError("Full name, description and skill_md string fields are required")
    if not values[2].strip():
        raise ContractError("Empty skill_md is not full-text evidence")
    return " | ".join(values)


def dense_candidates(row: dict) -> list[dict]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < POOL_SIZE:
        raise ContractError("Dense row requires at least twenty ranked candidates")
    result = []
    for rank, candidate in enumerate(candidates, start=1):
        skill_id = identifier(candidate, "skill_id")
        score = finite(candidate.get("score"), "dense score")
        result.append({"skill_id": skill_id, "dense_score": score, "dense_rank": rank})
    if len({c["skill_id"] for c in result}) != len(result):
        raise ContractError("Dense ranking contains duplicate skills")
    if any(a["dense_score"] < b["dense_score"] for a, b in zip(result, result[1:])):
        raise ContractError("Dense candidates must already be score-descending")
    return result


def make_request(query: str, candidates: list[dict], model: str) -> dict:
    return {"model": model, "state": {"task": query}, "questions": {
        "s" + str(c["dense_rank"] - 1): {
            "type": "score", "instructions": {"question": QUESTION + c["document"]},
            "criteria": RUBRIC.copy(),
        } for c in candidates}}


def build_batches(query_id: str, query: str, pool: list[dict], skills: dict[str, dict],
                  model: str, context_limit: int, reserved_output_tokens: int,
                  overhead_tokens: int = 2048) -> tuple[list[dict], list[dict]]:
    """Greedy dense-order batches with full text; a single overflow rejects the query.

    UTF-8 serialized bytes + overhead is a deliberately pessimistic local estimate,
    not the provider tokenizer or a verified provider context bound. Paid execution
    must validate both. No content is truncated and no request is sent here.
    """
    if len(pool) != POOL_SIZE or len({c["skill_id"] for c in pool}) != POOL_SIZE:
        raise ContractError("Exactly twenty distinct candidates are required")
    if not isinstance(query, str) or not query or not isinstance(model, str) or not model:
        raise ContractError("Original query and model must be nonempty strings")
    if min(context_limit, reserved_output_tokens, overhead_tokens) < 0 or context_limit <= reserved_output_tokens + overhead_tokens:
        raise ContractError("Invalid conservative context budget")
    prepared = []
    for candidate in pool:
        if candidate["skill_id"] not in skills:
            raise ContractError("Dense candidate missing from full-text catalogue")
        text = document(skills[candidate["skill_id"]])
        prepared.append({**candidate, "document": text})
    batches, current = [], []
    def fits(items: list[dict]) -> bool:
        size = len(canonical(make_request(query, items, model))) + overhead_tokens
        return size + reserved_output_tokens <= context_limit
    for candidate in prepared:
        if not fits([candidate]):
            raise ContractError("single_candidate_exceeds_conservative_context_no_truncation")
        if current and not fits(current + [candidate]):
            batches.append(current)
            current = []
        current.append(candidate)
    if current:
        batches.append(current)
    packets = []
    for i, candidates in enumerate(batches):
        request = make_request(query, candidates, model)
        request_sha = digest(request)
        packets.append({
            "query_id": query_id, "batch_index": i, "request_sha256": request_sha,
            "candidate_ids": [c["skill_id"] for c in candidates],
            "question_ids": ["s" + str(c["dense_rank"] - 1) for c in candidates],
            "request_file": "requests/" + request_sha + ".json",
            "response_cache_file": "raw_responses/" + request_sha + ".json",
            "serialized_utf8_bytes": len(canonical(request)),
            "estimated_input_token_upper_bound": len(canonical(request)) + overhead_tokens,
            "request": request,
        })
    counts = [{**{k: c[k] for k in ("skill_id", "dense_rank", "dense_score")},
               "original_characters": len(c["document"]),
               "prepared_characters": len(c["document"]),
               "original_utf8_bytes": len(c["document"].encode("utf-8")),
               "prepared_utf8_bytes": len(c["document"].encode("utf-8")),
               "original_exact_tokens": None, "prepared_exact_tokens": None,
               "document_sha256": hashlib.sha256(c["document"].encode("utf-8")).hexdigest(),
               "truncated": False} for c in prepared]
    return packets, counts


def prepare(args: argparse.Namespace) -> dict:
    if args.model_version is not None and (not args.model_version or "latest" in args.model_version.lower()):
        raise ContractError("Expected model version must be a stable versioned ID")
    queries = index_rows(read_jsonl(args.queries), "query_id")
    skills = index_rows(read_jsonl(args.skills), "skill_id")
    dense = index_rows(read_jsonl(args.dense), "query_id")
    pilot_ids = select_ids(list(queries), 200)
    smoke_ids = pilot_ids[:10]
    selected = {"pilot": pilot_ids, "smoke": smoke_ids,
                "full": select_ids(list(queries), len(queries))}[args.scope]
    if not selected:
        raise ContractError("No queries selected")
    rows = []
    for query_id in selected:
        if query_id not in dense:
            raise ContractError("Selected query missing from dense top50")
        query = queries[query_id].get("query")
        candidates = dense_candidates(dense[query_id])
        if len(candidates) < 50:
            raise ContractError("Preparation requires dense top50 to preserve candidate-coverage evidence")
        row = {"query_id": query_id, "query": query, "dense_top20": candidates[:20],
               "dense_top50": candidates[:50], "batches": [], "status": "prepared_not_run"}
        try:
            packets, counts = build_batches(query_id, query, candidates[:20], skills, args.model,
                                            args.context_limit, args.reserved_output_tokens,
                                            args.overhead_tokens)
            row["candidate_text_audit"] = counts
            for packet in packets:
                request = packet.pop("request")
                write_new(args.output_dir / packet["request_file"], request)
                row["batches"].append(packet)
        except ContractError as exc:
            row.update(status="blocked_no_truncation", reason=str(exc))
        rows.append(row)
    manifest = {
        "schema_version": SCHEMA_VERSION, "phase": "preparation_only_no_hosted_execution",
        "network_calls": 0, "credentials_accessed": False, "scope": args.scope,
        "sources": {kind: {"sha256": file_hash(path), "filename": path.name}
                    for kind, path in (("queries", args.queries), ("skills", args.skills), ("dense", args.dense))},
        "counts": {"queries": len(queries), "skills": len(skills), "dense_queries": len(dense),
                   "selected": len(selected), "prepared": sum(r["status"] == "prepared_not_run" for r in rows),
                   "blocked": sum(r["status"] != "prepared_not_run" for r in rows)},
        "selection": "SHA256(query_id UTF-8), ascending hex digest then query_id; smoke is first10 of pilot200",
        "rubric": RUBRIC, "question_prefix": QUESTION, "rubric_sha256": digest(RUBRIC),
        "model_requested": args.model,
        "expected_model_version": args.model_version or (args.model if "latest" not in args.model.lower() else None),
        "version_policy": "Every response records a non-alias actual model; one actual version per ingested comparison.",
        "endpoint_future_only": ENDPOINT, "future_credential_variable": FUTURE_CREDENTIAL_ENV,
        "context": {"limit_supplied": args.context_limit, "verified_provider_limit": False,
                    "reserved_output_tokens": args.reserved_output_tokens, "overhead_tokens": args.overhead_tokens,
                    "estimator": "serialized request UTF-8 bytes plus overhead; provisional conservative estimate",
                    "truncation_policy": "none; reject whole query if a single full candidate cannot fit"},
        "ranking": "Jev native score desc, confidence desc when present (missing=0), original dense rank asc",
        "score_mapping": "Five ordered criteria; native continuous score 0..4; displayed relevance=native+1",
        "fusion": {"k": RRF_K, "weights": [1, 1], "ranks_start_at": 1,
                   "formula": "1/(60+r_dense)+1/(60+r_jev)", "tie_break": "original dense rank"},
        "candidate_pool": "exact frozen dense top20; no candidate additions/removals in valid B or C",
        "raw_cache_contract": {"filename": "<request_sha256>.json", "fields": [
            "schema_version", "request_sha256", "status_code", "raw_response", "response_sha256", "elapsed_seconds"],
            "raw_response": "Unmodified UTF-8 response body string; save before parsing; no headers/credentials",
            "response_sha256": "SHA256 of raw_response UTF-8 bytes", "overwrite": "forbidden"},
        "execution_block": PHASE_BLOCK, "rows": rows,
    }
    manifest["plan_sha256"] = digest(manifest)
    write_new(args.output_dir / "pilot_query_ids.json", pilot_ids)
    write_new(args.output_dir / "smoke_query_ids.json", smoke_ids)
    write_new(args.output_dir / "preparation_manifest.json", manifest)
    return {"plan_sha256": manifest["plan_sha256"], **manifest["counts"], "network_calls": 0,
            "status": "prepared_only_not_executed"}


def parse_response(response: dict, batch: dict, expected_model: str | None,
                   context_limit: int) -> tuple[list[dict], dict]:
    model = response.get("model")
    if not isinstance(model, str) or not model or model == "jev-latest" or "latest" in model.lower():
        raise ContractError("A stable actual model version is required in every response")
    if expected_model is not None and model != expected_model:
        raise ContractError("Actual model version differs from the frozen version")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ContractError("Missing response usage")
    input_tokens = token_count(usage.get("input_tokens"), "input_tokens")
    if input_tokens > context_limit:
        raise ContractError("Actual input token usage exceeds supplied context limit")
    output_tokens = usage.get("output_tokens")
    if output_tokens is not None:
        token_count(output_tokens, "output_tokens")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(batch["question_ids"]):
        raise ContractError("Missing, duplicate or unexpected score answers")
    scored = []
    for question_id, skill_id in zip(batch["question_ids"], batch["candidate_ids"]):
        answer = answers[question_id]
        if not isinstance(answer, dict) or answer.get("type") != "score":
            raise ContractError("Expected a typed Score answer")
        score = finite(answer.get("score"), "native score", 0, 4)
        confidence = answer.get("confidence")
        if confidence is not None:
            confidence = finite(confidence, "confidence", 0, 1)
        scored.append({"skill_id": skill_id, "native_score": score,
                       "relevance_score": score + 1, "confidence": confidence})
    return scored, {"actual_model_version": model, "input_tokens": input_tokens,
                    "output_tokens": output_tokens, "raw_usage": usage}


def parse_cache(path: Path, batch: dict, expected_model: str | None,
                context_limit: int) -> tuple[list[dict], dict]:
    envelope = read_json(path)
    if envelope.get("schema_version") != SCHEMA_VERSION or envelope.get("request_sha256") != batch["request_sha256"]:
        raise ContractError("Cached response does not match frozen request/schema")
    if envelope.get("status_code") != 200:
        raise ContractError("Cached response is not HTTP 200")
    raw = envelope.get("raw_response")
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 1024 * 1024 or hashlib.sha256(raw.encode("utf-8")).hexdigest() != envelope.get("response_sha256"):
        raise ContractError("Raw response SHA256 mismatch")
    elapsed = finite(envelope.get("elapsed_seconds"), "API elapsed_seconds", 0)
    scored, usage = parse_response(loads(raw), batch, expected_model, context_limit)
    return scored, {**usage, "elapsed_seconds": elapsed, "request_sha256": batch["request_sha256"],
                    "response_sha256": envelope["response_sha256"]}


def rank_and_fuse(pool: list[dict], scored: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    dense_ids = [c["skill_id"] for c in pool]
    if len(dense_ids) != POOL_SIZE or len(set(dense_ids)) != POOL_SIZE:
        raise ContractError("Expected twenty distinct dense candidates")
    if len(scored) != POOL_SIZE or {s["skill_id"] for s in scored} != set(dense_ids):
        raise ContractError("Incomplete or changed Jev candidate pool")
    dense_rank = {skill_id: rank for rank, skill_id in enumerate(dense_ids, 1)}
    ordered = sorted(scored, key=lambda s: (-s["native_score"], -(s["confidence"] or 0.0), dense_rank[s["skill_id"]]))
    jev = [s["skill_id"] for s in ordered]
    jev_rank = {skill_id: rank for rank, skill_id in enumerate(jev, 1)}
    values = {sid: 1 / (RRF_K + dense_rank[sid]) + 1 / (RRF_K + jev_rank[sid]) for sid in dense_ids}
    rrf = sorted(dense_ids, key=lambda sid: (-values[sid], dense_rank[sid]))
    details = [{**s, "dense_rank": dense_rank[s["skill_id"]], "jev_rank": jev_rank[s["skill_id"]],
                "rrf60_score": values[s["skill_id"]]} for s in ordered]
    return jev, rrf, details


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    x = (len(ordered) - 1) * p
    lo = int(x)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (x - lo)


def ingest(args: argparse.Namespace) -> dict:
    manifest = read_json(args.plan)
    claimed = manifest.pop("plan_sha256", None)
    if manifest.get("schema_version") != SCHEMA_VERSION or claimed != digest(manifest):
        raise ContractError("Preparation manifest hash/schema mismatch")
    if manifest["rubric"] != RUBRIC or manifest["question_prefix"] != QUESTION:
        raise ContractError("Frozen question or rubric changed")
    frozen_version = manifest.get("expected_model_version")
    rows, successful_calls, cache_seen = [], {}, set()
    jev_rankings, rrf_rankings, dense_rankings = {}, {}, {}
    for item in manifest["rows"]:
        qid = item["query_id"]
        dense_rankings[qid] = [c["skill_id"] for c in item["dense_top20"]]
        jev_rankings[qid], rrf_rankings[qid] = [], []
        row = {"query_id": qid, "status": "incomplete", "scores": [], "batches": []}
        scores, row_usage = [], []
        if item["status"] != "prepared_not_run":
            row["status"] = "preparation_blocked"
            rows.append(row)
            continue
        for batch in item["batches"]:
            request = read_json(args.plan.parent / batch["request_file"])
            if digest(request) != batch["request_sha256"]:
                raise ContractError("Frozen request file changed")
            cache_path = args.responses_dir / (batch["request_sha256"] + ".json")
            if not cache_path.exists():
                row["batches"].append({"request_sha256": batch["request_sha256"], "status": "missing_cache"})
                continue
            cache_seen.add(batch["request_sha256"])
            try:
                values, usage = parse_cache(cache_path, batch, frozen_version, manifest["context"]["limit_supplied"])
                if frozen_version is None:
                    frozen_version = usage["actual_model_version"]
                scores.extend(values)
                row_usage.append(usage)
                successful_calls[batch["request_sha256"]] = usage
                row["batches"].append({"request_sha256": batch["request_sha256"], "status": "validated", **usage})
            except (ContractError, KeyError, TypeError, json.JSONDecodeError):
                # Never serialize raw exception messages, headers or credential material.
                row["batches"].append({"request_sha256": batch["request_sha256"], "status": "invalid_cache"})
        if len(row_usage) == len(item["batches"]):
            jev, rrf, details = rank_and_fuse(item["dense_top20"], scores)
            jev_rankings[qid], rrf_rankings[qid] = jev, rrf
            row.update(status="complete", scores=details,
                       api_elapsed_seconds=sum(u["elapsed_seconds"] for u in row_usage),
                       candidate_pool_identical=True)
        rows.append(row)
    completed = sum(r["status"] == "complete" for r in rows)
    common = {"schema_version": SCHEMA_VERSION, "plan_sha256": claimed,
              "scope": manifest["scope"], "planned_queries": len(rows), "complete_queries": completed,
              "actual_model_version": frozen_version, "failed_query_policy": "empty B/C rankings preserve selected denominator",
              "claim": "Offline ingestion of actual supplied caches; no hosted calls made by this script.",
              "network_calls_this_process": 0}
    times = [r["api_elapsed_seconds"] for r in rows if r["status"] == "complete"]
    usage = {**common, "observed_cache_files": len(cache_seen), "validated_cached_responses": len(successful_calls),
             "invalid_cached_responses": len(cache_seen) - len(successful_calls),
             "known_input_tokens_validated_responses": sum(u["input_tokens"] for u in successful_calls.values()),
             "known_output_tokens_validated_responses": sum(u["output_tokens"] or 0 for u in successful_calls.values()),
             "output_tokens_reported_for_all_validated": all(u["output_tokens"] is not None for u in successful_calls.values()) if successful_calls else None,
             "cost": None, "cost_status": "not_inferred; provider billing reconciliation required",
             "usage_limit": "Validated caches only; missing/invalid/uncertain requests may have additional billable usage.",
             "api_latency_completed_queries": {"median_seconds": percentile(times, 0.5), "p95_seconds": percentile(times, 0.95),
                  "definition": "sum of originally observed batch durations per completed query; cache lookup/service latency not measured"},
             "retrieval_latency": "not included; read baseline artifact", "responses": list(successful_calls.values())}
    write_new(args.output_dir / "dense_top20.json", {**common, "system": "A_dense_top20", "rankings": dense_rankings})
    write_new(args.output_dir / "jev_rerank.json", {**common, "system": "B_jev", "rankings": jev_rankings, "rows": rows})
    write_new(args.output_dir / "rrf60_fusion.json", {**common, "system": "C_rrf60", "k": RRF_K, "rankings": rrf_rankings})
    write_new(args.output_dir / "jev_usage.json", usage)
    checkpoint = {**common, "rows": [{"query_id": r["query_id"], "status": r["status"], "batches": r["batches"]} for r in rows]}
    write_new(args.output_dir / "ingestion_checkpoint.json", checkpoint)
    return {"planned_queries": len(rows), "complete_queries": completed, "network_calls": 0,
            "status": "cached_responses_ingested" if completed else "no_complete_jev_queries"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv.insert(0, "prepare")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Freeze full-text requests locally; zero network/key access")
    for field in ("queries", "skills", "dense", "output-dir"):
        prep.add_argument("--" + field, type=Path, required=True)
    prep.add_argument("--scope", choices=("smoke", "pilot", "full"), default="pilot")
    prep.add_argument("--model", default="jev-latest")
    prep.add_argument("--model-version", default=None, help="Expected actual version after an authorized smoke; never discovered here")
    prep.add_argument("--context-limit", type=int, required=True, help="Explicit provisional bound; provider verification still required")
    prep.add_argument("--reserved-output-tokens", type=int, default=2048)
    prep.add_argument("--overhead-tokens", type=int, default=2048)
    imp = commands.add_parser("ingest", help="Strictly ingest already saved responses; never contacts provider")
    imp.add_argument("--plan", type=Path, required=True)
    imp.add_argument("--responses-dir", type=Path, required=True)
    imp.add_argument("--output-dir", type=Path, required=True, help="New snapshot directory on every changed cache state")
    commands.add_parser("execute", help="Unavailable in this phase; fails before any credential access")
    args = parser.parse_args(argv)
    if args.command == "execute":
        print(PHASE_BLOCK, file=sys.stderr)
        return 2
    try:
        result = prepare(args) if args.command == "prepare" else ingest(args)
    except (ContractError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print("Offline operation failed: " + type(exc).__name__ + ". No network call or credential access occurred.", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
