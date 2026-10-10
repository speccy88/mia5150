"""Sequential, explicitly enabled execution of an immutable hosted Jev plan.

No credential discovery or API call occurs on import or in default status mode.
Receipt configuration supplies evidence references; it never contains a key.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

import hosted_jev_client as hosted
import run_jev_rerank as prep

MAX_SECONDS = 6 * 60 * 60
SCOPE_SIZE = {"smoke": 10, "pilot": 200, "full": 4392}


class StudyStop(hosted.GateError):
    """A runner-owned, safe-to-display reason code."""


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(prep.canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_plan(path: Path) -> dict:
    plan = prep.read_json(path)
    hosted.require(isinstance(plan, dict), "Plan must be an object")
    hosted.require(plan.get("plan_sha256") == prep.digest(
        {k: v for k, v in plan.items() if k != "plan_sha256"}), "Plan SHA256 mismatch")
    hosted.require(plan.get("schema_version") == prep.SCHEMA_VERSION, "Unsupported plan schema")
    rows = plan.get("rows", [])
    hosted.require(len(rows) == SCOPE_SIZE.get(plan.get("scope")), "Frozen scope size differs")
    ids = [row["query_id"] for row in rows]
    hosted.require(len(ids) == len(set(ids)), "Duplicate selected query")
    hosted.require(plan.get("rubric") == prep.RUBRIC and plan.get("question_prefix") == prep.QUESTION,
                   "Frozen question or rubric differs")
    return plan


def request_path(plan_path: Path, packet: dict) -> Path:
    base = plan_path.parent.resolve()
    path = (base / packet["request_file"]).resolve()
    hosted.require(path.is_relative_to(base), "Request path escapes plan directory")
    return path


def validate_requests(plan_path: Path, plan: dict) -> None:
    """Validate the whole scope before the first paid call, without dropping rows."""
    if any(row.get("status") != "prepared_not_run" for row in plan["rows"]):
        raise StudyStop("preparation_blocked_selected_queries")
    for row in plan["rows"]:
        hosted.require(bool(row.get("batches")), "Prepared query has no batches")
        pool = row["dense_top20"]
        hosted.require(len(pool) == 20 and len({c["skill_id"] for c in pool}) == 20,
                       "Prepared pool must contain twenty unique skills")
        expected = [("s" + str(i), c["skill_id"]) for i, c in enumerate(pool)]
        actual = []
        for packet in row["batches"]:
            hosted.require(packet.get("query_id") == row["query_id"], "Packet query mismatch")
            question_ids, candidates = packet["question_ids"], packet["candidate_ids"]
            hosted.require(len(question_ids) == len(candidates), "Packet mapping length differs")
            actual.extend(zip(question_ids, candidates))
            request = prep.read_json(request_path(plan_path, packet))
            hosted.require(prep.digest(request) == packet["request_sha256"], "Request SHA256 mismatch")
            hosted.HostedJevClient._check_request_text(request)
        hosted.require(actual == expected, "Batches do not partition the fixed top twenty")


def load_gates(path: Path) -> hosted.GateBundle:
    """Read only named receipt references; never inspect environment or key files."""
    config = prep.read_json(path)
    allowed = {field.name for field in fields(hosted.GateBundle)}
    hosted.require(isinstance(config, dict) and set(config) <= allowed, "Unexpected gate configuration fields")
    refs = {}
    for name, ref in config.items():
        if ref is None:
            refs[name] = None
            continue
        hosted.require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Invalid receipt reference")
        target = Path(ref["path"])
        if not target.is_absolute():
            target = path.parent / target
        refs[name] = hosted.Artifact(target.resolve(), ref["sha256"])
    return hosted.GateBundle(**refs)


def ledger_summary(ledger: hosted.SpendLedger) -> dict:
    rows = ledger.rows()  # Verifies the existing hash chain; never creates or repairs it.
    reserved, settled, released = {}, {}, set()
    for row in rows:
        attempt = row.get("attempt_id")
        if row["event"] == "reserved":
            hosted.require(attempt not in reserved, "Duplicate ledger reservation")
            stamp = row.get("utc_unix")
            hosted.require(type(stamp) in (int, float) and math.isfinite(stamp) and stamp >= 0,
                           "Ledger attempt timestamp invalid")
            reserved[attempt] = row
        elif row["event"] == "settled":
            hosted.require(attempt in reserved and attempt not in settled and attempt not in released,
                           "Ledger settlement is not unique")
            settled[attempt] = row
        elif row["event"] == "documented_no_execution":
            hosted.require(attempt in reserved and attempt not in settled and attempt not in released,
                           "Ledger release is not unique")
            released.add(attempt)
    unresolved = {key: row for key, row in reserved.items() if key not in settled and key not in released}
    total = lambda values, key: str(sum((hosted.dec(row[key]) for row in values), Decimal(0)))
    return {
        "ledger_events": len(rows),
        "ledger_head_sha256": rows[-1]["event_sha256"] if rows else None,
        "paid_api_attempts_reserved": len(reserved),
        "paid_responses_settled": len(settled),
        "documented_unexecuted_attempts": len(released),
        "unresolved_attempts": len(unresolved),
        "alias_cache_reuse_events": sum(row["event"] == "cache_reused" for row in rows),
        "settled_usd": total(settled.values(), "actual_usd"),
        "settled_cad_with_fee_buffer": total(settled.values(), "actual_cad"),
        "held_usd": total(unresolved.values(), "reserve_usd"),
        "held_cad_with_fee_buffer": total(unresolved.values(), "reserve_cad"),
        "known_input_tokens_settled": sum(row["input_tokens"] for row in settled.values()),
        "known_output_tokens_settled": sum(row.get("output_tokens") or 0 for row in settled.values()),
        "first_attempt_unix": min((row["utc_unix"] for row in reserved.values()), default=None),
        "cost_definition": "Unique ledger attempts across all scopes; cache reuse adds no cost. CAD includes recorded FX/fee/buffer, excludes any experiment fixed fee. Reservations may precede an unobserved transport attempt; held amounts are not claimed as settled charges.",
    }


def cached_provenance(root: Path, packet: dict, request: dict, evidence: dict,
                      expected_model: str | None, context_limit: int) -> tuple[list[dict], dict]:
    """Read-only settlement/raw-evidence validation, using the client's validators.

    A self-consistent envelope alone is not evidence that this experiment paid
    for a response. Never call client.request() here: missing provenance must
    fail before a code path capable of transport, including after receipt expiry.
    """
    request_hash = packet["request_sha256"]
    cache_path = root / "raw_responses" / (request_hash + ".json")
    scores, usage = prep.parse_cache(cache_path, packet, expected_model, context_limit)
    envelope = prep.read_json(cache_path)
    original_hash = envelope.get("original_transport_request_sha256")
    hosted.require(isinstance(original_hash, str) and re.fullmatch(r"[0-9a-f]{64}", original_hash),
                   "Cached response lacks original transport identity")
    hosted.require(envelope.get("expected_request_sha256") == request_hash,
                   "Cache expected request identity differs")
    original_path = root / "raw_responses" / (original_hash + ".json")
    original = prep.read_json(original_path)
    original_packet = {**packet, "request_sha256": original_hash}
    prep.parse_cache(original_path, original_packet, usage["actual_model_version"], context_limit)
    attempt_id = original.get("attempt_id")
    hosted.require(isinstance(attempt_id, str) and re.fullmatch(original_hash + r"-[1-3]", attempt_id),
                   "Original cached attempt identity differs")
    settled = evidence["settled"].get(original_hash, [])
    hosted.require(len(settled) == 1, "Cached response lacks a unique ledger settlement")
    settlement = settled[0]
    model = usage["actual_model_version"]
    response_hash = usage["response_sha256"]
    hosted.require(settlement.get("attempt_id") == attempt_id
                   and settlement.get("actual_model_version") == model
                   and settlement.get("response_sha256") == response_hash,
                   "Cache model, attempt or response differs from settlement")
    reserved = evidence["reserved"].get(attempt_id, [])
    saved = evidence["response_saved"].get(attempt_id, [])
    hosted.require(len(reserved) == len(saved) == 1
                   and reserved[0].get("request_sha256") == original_hash
                   and reserved[0].get("actual_model_pin") == model
                   and saved[0].get("request_sha256") == original_hash
                   and saved[0].get("response_sha256") == response_hash and saved[0].get("status_code") == 200,
                   "Original reservation or saved response evidence differs")
    request_bytes = (root / "requests" / (original_hash + ".json")).read_bytes()
    hosted.require(hosted.hash_bytes(request_bytes) == original_hash, "Original transport request bytes changed")
    original_request = prep.loads(request_bytes.decode("utf-8"))
    normalize = hosted.HostedJevClient._normalized_payload
    hosted.require(normalize(original_request, model) == normalize(request, model),
                   "Cached payload differs beyond its model alias")
    body = (root / "attempts" / (attempt_id + ".body")).read_bytes()
    meta = prep.read_json(root / "attempts" / (attempt_id + ".json"))
    hosted.require(hosted.hash_bytes(body) == response_hash
                   and body == original["raw_response"].encode("utf-8")
                   and body == envelope["raw_response"].encode("utf-8")
                   and meta.get("attempt_id") == attempt_id and meta.get("request_sha256") == original_hash
                   and meta.get("response_sha256") == response_hash and meta.get("status_code") == 200,
                   "Original raw response or transport metadata differs")
    for cached in (original, envelope):
        hosted.require(cached.get("attempt_id") == attempt_id and cached.get("actual_model_version") == model
                       and cached.get("input_tokens") == usage["input_tokens"] == settlement.get("input_tokens")
                       and cached.get("output_tokens") == usage["output_tokens"] == settlement.get("output_tokens")
                       and cached.get("elapsed_seconds") == usage["elapsed_seconds"] == settlement.get("elapsed_seconds")
                       and hosted.dec(cached.get("cost_usd")) == hosted.dec(settlement.get("actual_usd"))
                       and hosted.dec(cached.get("all_in_cost_cad_with_buffer")) == hosted.dec(settlement.get("actual_cad")),
                       "Cached accounting differs from the original settlement")
    if original_hash != request_hash:
        semantic = prep.digest(normalize(request, model))
        reused = evidence["cache_reused"].get(request_hash, [])
        hosted.require(len(reused) == 1 and reused[0].get("original_transport_request_sha256") == original_hash
                       and reused[0].get("actual_model_version") == model
                       and reused[0].get("response_sha256") == response_hash
                       and reused[0].get("normalized_payload_sha256") == semantic
                       and reused[0].get("additional_api_calls") == 0,
                       "Aliased cache lacks its verified ledger reuse event")
        index = prep.read_json(root / "semantic_cache" / (semantic + ".json"))
        hosted.require(index.get("original_transport_request_sha256") == original_hash
                       and index.get("actual_model_version") == model and index.get("response_sha256") == response_hash
                       and index.get("normalized_payload_sha256") == semantic
                       and index.get("batch_binding") == {"candidate_ids": packet["candidate_ids"],
                                                          "question_ids": packet["question_ids"]},
                       "Aliased cache semantic evidence differs")
    return scores, usage


def cache_state(root: Path, plan: dict, plan_path: Path) -> list[dict]:
    """Only ledger-backed caches with raw evidence can mark a query complete."""
    ledger_rows = hosted.SpendLedger(root).rows()
    evidence = {event: {} for event in ("reserved", "response_saved", "settled", "cache_reused")}
    for event in ledger_rows:
        kind = event["event"]
        if kind in evidence:
            key = event.get("attempt_id" if kind in ("reserved", "response_saved") else "request_sha256")
            evidence[kind].setdefault(key, []).append(event)
    state, version = [], plan.get("expected_model_version")
    for row in plan["rows"]:
        values, batches = [], []
        for packet in row["batches"]:
            status = "missing_cache"
            path = root / "raw_responses" / (packet["request_sha256"] + ".json")
            if path.exists():
                request = prep.read_json(request_path(plan_path, packet))
                scores, usage = cached_provenance(root, packet, request, evidence, version,
                                                  plan["context"]["limit_supplied"])
                version = version or usage["actual_model_version"]
                values.extend(scores)
                status = "validated"
            batches.append({"request_sha256": packet["request_sha256"], "status": status})
        complete = bool(batches) and all(item["status"] == "validated" for item in batches)
        if complete:
            prep.rank_and_fuse(row["dense_top20"], values)
        state.append({"query_id": row["query_id"], "status": "complete" if complete else "incomplete",
                      "batches": batches})
    return state


def status(plan_path: Path, run_dir: Path) -> dict:
    plan = read_plan(plan_path)
    account = ledger_summary(hosted.SpendLedger(run_dir))
    return {"status": "disabled_status_only", "scope": plan["scope"], "plan_sha256": plan["plan_sha256"],
            "selected_queries": len(plan["rows"]),
            "prepared_queries": sum(row.get("status") == "prepared_not_run" for row in plan["rows"]),
            "blocked_queries": sum(row.get("status") != "prepared_not_run" for row in plan["rows"]),
            "prepared_batches": sum(len(row["batches"]) for row in plan["rows"]),
            "network_calls_this_process": 0, "credentials_accessed": False, "ledger": account}


def run_study(plan_path: Path, client: hosted.HostedJevClient, *, execute: bool = False,
              max_queries: int | None = None, max_seconds: float = MAX_SECONDS,
              clock=time.time, emit=None) -> dict:
    """Run the entire frozen scope; max_queries bounds newly incomplete queries only.

    A bounded invocation is explicitly interrupted. Every checkpoint and ingestion
    snapshot retains the original selected denominator. Receipts are supplied by
    the caller and must be valid under HostedJevClient; no approval is inferred.
    """
    plan_path = Path(plan_path)
    if not execute:
        return status(plan_path, client.root)
    hosted.require(client.enabled is True, "Hosted client disabled by default")
    hosted.require(type(max_seconds) in (int, float) and 0 < max_seconds <= MAX_SECONDS,
                   "Deadline must be positive and no more than six hours")
    hosted.require(max_queries is None or type(max_queries) is int and max_queries > 0,
                   "max_queries must be a positive integer")
    plan = read_plan(plan_path)
    validate_requests(plan_path, plan)
    # The study checks the deadline before each single-attempt client call. Never
    # allow an internal retry to cross this boundary without another time check.
    hosted.require(client.gates is not None and client.gates.contract.read().get("max_attempts", 1) == 1,
                   "Study runner requires one transport attempt per client request")
    root = client.root
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "study_runner.lock"
    try:
        lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise StudyStop("runner_locked_manual_recovery_required") from None
    with os.fdopen(lock_fd, "wb") as stream:
        stream.write(b"Do not remove until the prior study runner and API request are confirmed inactive.\n")
        stream.flush()
        os.fsync(stream.fileno())
    invocation = uuid.uuid4().hex
    progress_dir = root / "study_progress" / plan["plan_sha256"] / invocation
    state = [{"query_id": row["query_id"], "status": "incomplete", "batches": []} for row in plan["rows"]]
    result, sequence, newly_visited, initial_attempts = {}, 0, 0, None
    started = clock()
    original_transport = client.transport
    original_timeout = client.timeout
    deadline_reached_at_transport = False

    def bounded_transport(method, url, body, key, timeout):
        nonlocal deadline_reached_at_transport
        # Receipt validation or filesystem work may cross the deadline after the
        # outer loop check. Recheck immediately before invoking the transport.
        if method == "POST":
            first = ledger_summary(client.ledger)["first_attempt_unix"]
            remaining = first + max_seconds - clock() if first is not None else max_seconds
            if remaining <= 0:
                deadline_reached_at_transport = True
                raise StudyStop("deadline_before_transport")
            timeout = min(timeout, remaining)
        return original_transport(method, url, body, key, timeout)

    client.transport = bounded_transport

    def checkpoint(phase, reason=None, error_type=None):
        nonlocal sequence, result
        account = ledger_summary(client.ledger)
        first = account["first_attempt_unix"]
        result = {"schema_version": 1, "invocation_id": invocation, "plan_sha256": plan["plan_sha256"],
                  "scope": plan["scope"], "status": phase, "reason": reason, "error_type": error_type,
                  "selected_queries": len(state), "complete_queries": sum(row["status"] == "complete" for row in state),
                  "selected_query_ids": [row["query_id"] for row in state], "rows": state,
                  "newly_incomplete_queries_visited": newly_visited, "max_queries_this_invocation": max_queries,
                  "ledger": account, "paid_attempts_reserved_this_invocation": (
                      account["paid_api_attempts_reserved"] - initial_attempts if initial_attempts is not None else 0),
                  "updated_at_unix": clock(), "invocation_started_at_unix": started,
                  "deadline_unix": first + max_seconds if first is not None else None,
                  "maximum_cumulative_seconds": max_seconds,
                  "denominator_policy": "All selected IDs retained; incomplete B/C rankings remain empty in snapshots."}
        hosted.write_immutable(progress_dir / (str(sequence).zfill(6) + ".json"), prep.canonical(result) + b"\n")
        atomic_json(root / "study_progress" / plan["plan_sha256"] / "latest.json", result)
        sequence += 1
        if emit:
            emit({key: result[key] for key in ("status", "reason", "error_type", "scope", "selected_queries",
                  "complete_queries", "paid_attempts_reserved_this_invocation", "ledger", "deadline_unix")})

    try:
        account = ledger_summary(client.ledger)
        initial_attempts = account["paid_api_attempts_reserved"]
        if account["unresolved_attempts"]:
            raise StudyStop("unresolved_prior_attempt_manual_reconciliation_required")
        state = cache_state(root, plan, plan_path)
        checkpoint("running")
        for index, row in enumerate(plan["rows"]):
            if state[index]["status"] == "complete":
                continue
            if max_queries is not None and newly_visited >= max_queries:
                raise StudyStop("max_queries_checkpoint_limit")
            newly_visited += 1
            for packet in row["batches"]:
                account = ledger_summary(client.ledger)
                if account["unresolved_attempts"]:
                    raise StudyStop("unresolved_prior_attempt_manual_reconciliation_required")
                first = account["first_attempt_unix"]
                now = clock()
                if first is not None:
                    if now < first:
                        raise StudyStop("clock_precedes_first_attempt")
                    remaining = first + max_seconds - now
                    if remaining <= 0:
                        raise StudyStop("cumulative_six_hour_deadline")
                    client.timeout = min(client.timeout, remaining)
                request = prep.read_json(request_path(plan_path, packet))
                client.request(packet, request, plan=plan)
                # Per-batch progress includes a durable per-query checkpoint;
                # trust persisted caches, never only the client's return value.
                state[index] = cache_state(root, {**plan, "rows": [row]}, plan_path)[0]
                checkpoint("running")
            hosted.require(state[index]["status"] == "complete", "Query cache incomplete after client return")
        checkpoint("completed")
    except (Exception, KeyboardInterrupt) as exc:
        # Exception strings may contain request text or transport details. Only
        # runner-owned reason codes and the exception class leave this process.
        checkpoint("interrupted", "deadline_before_transport" if deadline_reached_at_transport else
                   (str(exc) if isinstance(exc, StudyStop) else "operation_failed"),
                   type(exc).__name__)
    finally:
        try:
            snapshot = root / "study_snapshots" / plan["plan_sha256"] / invocation
            # Ingestion validates envelope structure but deliberately supports
            # externally supplied responses. This execution runner additionally
            # requires local spend/raw provenance before publishing a snapshot.
            cache_state(root, plan, plan_path)
            summary = prep.ingest(argparse.Namespace(plan=plan_path, responses_dir=root / "raw_responses",
                                                     output_dir=snapshot))
            result["snapshot_dir"] = str(snapshot)
            result["snapshot"] = summary
            result["complete_queries"] = summary["complete_queries"]
            result["rows"] = prep.read_json(snapshot / "ingestion_checkpoint.json")["rows"]
            if result.get("status") == "completed" and summary["complete_queries"] != len(plan["rows"]):
                result.update(status="interrupted", reason="snapshot_incomplete", error_type=None)
            hosted.write_immutable(progress_dir / "final.json", prep.canonical(result) + b"\n")
            atomic_json(root / "study_progress" / plan["plan_sha256"] / "latest.json", result)
        except Exception as exc:
            result.update(status="interrupted", reason="snapshot_failed", snapshot_error_type=type(exc).__name__)
            result["snapshot_status"] = "blocked_no_unverified_snapshot_published"
            atomic_json(root / "study_progress" / plan["plan_sha256"] / "latest.json", result)
        finally:
            client.transport = original_transport
            client.timeout = original_timeout
            lock.unlink()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog, width=88))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path, help="One persistent experiment ledger/cache across all scopes")
    parser.add_argument("--gates", type=Path, help="JSON mapping GateBundle fields to path/SHA256 references")
    parser.add_argument("--execute", action="store_true", help="Explicitly enable sequential paid calls after all existing client gates")
    parser.add_argument("--max-queries", type=int, help="Stop after this many previously incomplete queries; retain the full denominator")
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS, help="Cumulative deadline from first ledger reservation; at most 21600")
    args = parser.parse_args(argv)
    try:
        if not args.execute:
            output = status(args.plan, args.run_dir)
        else:
            if args.gates is None:
                raise StudyStop("gate_configuration_required")
            client = hosted.HostedJevClient(args.run_dir, gates=load_gates(args.gates), enabled=True)
            output = run_study(args.plan, client, execute=True, max_queries=args.max_queries,
                               max_seconds=args.max_seconds,
                               emit=lambda value: print(json.dumps(value, sort_keys=True), flush=True))
        # Keep query IDs, per-query state and payloads in artifacts, not stdout.
        print(json.dumps({k: v for k, v in output.items() if k not in ("rows", "selected_query_ids")}, sort_keys=True), flush=True)
        return 0 if output["status"] in ("completed", "disabled_status_only") else 2
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, StudyStop) else "operation_failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
