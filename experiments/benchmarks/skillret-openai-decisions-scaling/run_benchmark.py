"""Freeze, execute and export the matched SkillRet Decisions experiment."""

from __future__ import annotations

import argparse
import asyncio
import csv
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
import sqlite3
import time
from collections import Counter, deque
from collections.abc import Iterable
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from pathlib import Path
import re
from typing import Any

import httpx

from decisions_reranker import (
    CACHE_WRITE_NANO_USD,
    CANDIDATE_COUNT,
    ENDPOINT,
    LONG_CONTEXT_THRESHOLD,
    MAX_REQUEST_NANO_USD,
    MODEL,
    NANO_USD_PER_DOLLAR,
    InvalidDecision,
    Usage,
    build_payload,
    canonical_json,
    parse_probabilities,
    request_digest,
    reranked_candidate_ids,
    sanitize,
)

ROOT = Path(__file__).resolve().parent
EXPECTED_CASES = 12_600
HARD_LIMIT_NANO_USD = 15 * NANO_USD_PER_DOLLAR
SYNTHETIC_CASE_ID = "synthetic/compatibility"
USAGE_FIELDS = tuple(Usage.__dataclass_fields__)
# These client errors reject a request without inference. The attempt cap also
# counts them, including every explicit retry of a rate-limited request.
REJECTED_HTTP_STATUSES = frozenset({400, 401, 403, 404, 405, 413, 415, 422, 429})
FATAL_HTTP_STATUSES = frozenset({401, 403, 404, 405})
QUOTA_ERROR_CODES = frozenset({"insufficient_quota", "billing_hard_limit_reached"})


class RunBlocked(RuntimeError):
    """A persistent budget, configuration or smoke gate stopped live work."""


@dataclass(frozen=True)
class Request:
    request_hash: str
    encoded: bytes
    candidate_ids: list[str]
    kind: str


@dataclass(frozen=True)
class Case:
    case_id: str
    query_id: str
    request_hash: str
    candidates: list[dict[str, Any]]
    retrieval_latency_ms: float


@dataclass(frozen=True)
class Plan:
    cases: dict[str, Case]
    requests: dict[str, Request]
    owners: dict[str, str]
    smoke_case_ids: list[str]
    qrels: dict[str, list[str]]
    manifest_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as stream:
        if path.suffix == ".gz":
            with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as output:
                for row in rows:
                    output.write(canonical_json(row) + b"\n")
        else:
            for row in rows:
                stream.write(canonical_json(row) + b"\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def usd(nano_usd: int | None) -> float | None:
    return None if nano_usd is None else nano_usd / NANO_USD_PER_DOLLAR


def read_plan(root: Path, api_key: str = "") -> Plan:
    manifest_path = root / "results" / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_paths = (
        "rubric.json",
        "config.json",
        "results/catalog_memberships.jsonl.gz",
        "results/candidate_pools.jsonl.gz",
        "results/qrels.json",
    )
    checksums = manifest.get("sha256", {})
    if manifest.get("stage") != "full" or not set(required_paths) <= checksums.keys():
        raise RunBlocked("Live planning requires the complete frozen factorial manifest.")
    experiment = root.resolve()
    for name, expected in checksums.items():
        path = (root / name).resolve()
        if not path.is_relative_to(experiment) or sha256_file(path) != expected:
            raise RunBlocked(f"Frozen input checksum mismatch for {name}.")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if (
        config["decisions"]["endpoint"] != ENDPOINT
        or config["decisions"]["model"] != MODEL
        or config["decisions"]["question_type"] != "predicate"
        or config["candidate_count"] != CANDIDATE_COUNT
    ):
        raise RunBlocked("The frozen configuration does not match the Decisions implementation.")

    rubric = json.loads((root / "rubric.json").read_text(encoding="utf-8"))
    qrels = json.loads((root / "results" / "qrels.json").read_text(encoding="utf-8"))
    cases: dict[str, Case] = {}
    requests: dict[str, Request] = {}
    owners: dict[str, str] = {}
    with gzip.open(root / "results" / "candidate_pools.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            case = json.loads(line)
            case_id = case["case_id"]
            candidates = case["candidates"]
            payload = build_payload(case["query"], candidates, rubric)
            if sanitize(payload, api_key) != payload:
                raise RunBlocked("A candidate payload matched a credential pattern.")
            encoded = canonical_json(payload)
            digest = request_digest(encoded)
            if case_id == SYNTHETIC_CASE_ID or case_id in cases:
                raise RunBlocked("Duplicate or reserved case identifier in the frozen manifest.")
            cases[case_id] = Case(
                case_id=case_id,
                query_id=str(case["query_id"]),
                request_hash=digest,
                candidates=[{"id": item["id"], "rank": item["rank"]} for item in candidates],
                retrieval_latency_ms=case["retrieval_latency_ms"],
            )
            if digest not in requests:
                requests[digest] = Request(
                    digest, encoded, [question["name"] for question in payload["questions"]], "benchmark"
                )
                owners[digest] = case_id

    if len(cases) != EXPECTED_CASES:
        raise RunBlocked(f"Frozen inputs must contain exactly {EXPECTED_CASES:,} cases.")
    if manifest["unique_candidate_payloads"] != len(requests):
        raise RunBlocked("Exact canonical payload count differs from the frozen manifest.")
    smoke_case_ids = manifest["smoke_case_ids"]
    if (
        len(smoke_case_ids) != 10
        or not set(smoke_case_ids) <= cases.keys()
        or len({cases[case_id].request_hash for case_id in smoke_case_ids}) != 10
    ):
        raise RunBlocked("Frozen inputs must identify 10 smoke cases with distinct payloads.")
    if any(not qrels.get(case.query_id) for case in cases.values()):
        raise RunBlocked("A frozen query is missing positive relevance judgments.")

    return Plan(cases, requests, owners, smoke_case_ids, qrels, sha256_file(manifest_path))


def synthetic_request(rubric_path: Path) -> Request:
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    candidates = [
        {
            "id": f"synthetic_{index:02d}",
            "metadata": (
                "Compute the sum of a column in a CSV file."
                if index == 0
                else f"Identify terminology in sample plant category {index}."
            ),
        }
        for index in range(CANDIDATE_COUNT)
    ]
    payload = build_payload("Compute the sum of the values in a CSV column.", candidates, rubric)
    encoded = canonical_json(payload)
    return Request(request_digest(encoded), encoded, [item["id"] for item in candidates], "synthetic")


@contextmanager
def exclusive_run(path: Path):
    with path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RunBlocked("Another process is using this experiment ledger.") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class Ledger:
    """Persist reservations before dispatch and settle them atomically with evidence."""

    def __init__(self, path: Path, manifest_sha256: str, regional: bool):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id INTEGER PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    started_at_utc TEXT NOT NULL,
                    finished_at_utc TEXT,
                    reserve_nano_usd INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    http_status INTEGER,
                    request_id TEXT,
                    api_latency_ms REAL,
                    rerank_wall_ms REAL,
                    response_body TEXT,
                    probabilities TEXT,
                    usage TEXT,
                    known_cost_nano_usd INTEGER,
                    budgeted_cost_nano_usd INTEGER,
                    failure TEXT
                );
                CREATE INDEX IF NOT EXISTS attempts_request_hash ON attempts(request_hash);
                CREATE TABLE IF NOT EXISTS run_log (
                    event_id INTEGER PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    event TEXT NOT NULL,
                    details TEXT NOT NULL
                );
                """
            )
            with self.db:
                for key, value in {
                    "manifest_sha256": manifest_sha256,
                    "endpoint": ENDPOINT,
                    "model": MODEL,
                    "regional_processing": str(regional),
                }.items():
                    self.db.execute("INSERT OR IGNORE INTO metadata VALUES (?, ?)", (key, value))
                    previous = self.db.execute(
                        "SELECT value FROM metadata WHERE key = ?", (key,)
                    ).fetchone()["value"]
                    if previous != value:
                        raise RunBlocked(f"Ledger provenance differs for {key}.")
        except BaseException:
            self.db.close()
            raise

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *_: object) -> None:
        self.db.close()

    def log(self, event: str, **details: Any) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO run_log (timestamp_utc, event, details) VALUES (?, ?, ?)",
                (utc_now(), event, canonical_json(details).decode("utf-8")),
            )

    def totals(self) -> dict[str, int]:
        row = self.db.execute(
            """
            SELECT
                COUNT(*) AS reservations,
                COUNT(finished_at_utc) AS completed_attempts,
                COALESCE(SUM(known_cost_nano_usd), 0) AS known_cost_nano_usd,
                COALESCE(SUM(budgeted_cost_nano_usd), 0) AS budgeted_cost_nano_usd,
                COALESCE(SUM(CASE WHEN known_cost_nano_usd IS NULL
                    THEN reserve_nano_usd ELSE 0 END), 0) AS held_nano_usd,
                COALESCE(SUM(CASE WHEN finished_at_utc IS NULL
                    THEN reserve_nano_usd ELSE 0 END), 0) AS in_flight_nano_usd
            FROM attempts
            """
        ).fetchone()
        return dict(row)

    def reserve(
        self, request: Request, case_id: str, phase: str, budget: int, max_attempts: int,
        wait_for_active: bool = False,
    ) -> int | None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            totals = self.totals()
            if totals["reservations"] >= max_attempts:
                raise RunBlocked("The configured HTTP attempt cap has been reached.")
            committed = totals["budgeted_cost_nano_usd"] + totals["held_nano_usd"]
            if committed + MAX_REQUEST_NANO_USD > budget:
                if wait_for_active:
                    return None
                raise RunBlocked("A full request reservation would exceed the configured USD budget.")
            cursor = self.db.execute(
                """
                INSERT INTO attempts (
                    request_hash, case_id, kind, phase, started_at_utc, reserve_nano_usd, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'reserved')
                """,
                (request.request_hash, case_id, request.kind, phase, utc_now(), MAX_REQUEST_NANO_USD),
            )
            return cursor.lastrowid
        finally:
            self.db.commit()

    def settle(self, attempt_id: int, **values: Any) -> None:
        allowed = {
            "status", "http_status", "request_id", "api_latency_ms", "rerank_wall_ms",
            "response_body", "probabilities", "usage", "known_cost_nano_usd",
            "budgeted_cost_nano_usd", "failure",
        }
        if set(values) != allowed:
            raise ValueError("The settled attempt record has unexpected fields.")
        for key in ("response_body", "probabilities", "usage"):
            values[key] = canonical_json(values[key]).decode("utf-8")
        values["finished_at_utc"] = utc_now()
        assignments = ", ".join(f"{name} = :{name}" for name in values)
        with self.db:
            cursor = self.db.execute(
                f"UPDATE attempts SET {assignments} WHERE attempt_id = :attempt_id AND status = 'reserved'",
                {**values, "attempt_id": attempt_id},
            )
            if cursor.rowcount != 1:
                raise RuntimeError("An attempt was already settled or has no reservation.")

    def latest(self, digest: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM attempts WHERE request_hash = ? ORDER BY attempt_id DESC LIMIT 1",
            (digest,),
        ).fetchone()

    def attempts(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM attempts ORDER BY attempt_id").fetchall()

    def count_rate_limited(self, digest: str) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM attempts WHERE request_hash = ? AND http_status = 429",
            (digest,),
        ).fetchone()[0]


class RateLimiter:
    """Space request start times and delay all workers on provider rate-limit signals."""

    def __init__(self, requests_per_minute: int):
        self.spacing = 60 / requests_per_minute
        self.next_start = 0.0
        self.paused_until = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                delay = max(self.next_start, self.paused_until) - now
                if delay <= 0:
                    self.next_start = now + self.spacing
                    return
                await asyncio.sleep(delay)

    def pause(self, seconds: float) -> None:
        self.paused_until = max(self.paused_until, time.monotonic() + seconds)


def provider_wait(headers: httpx.Headers, fallback: float) -> float:
    candidates = [fallback]
    with suppress(ValueError):
        candidates.append(float(headers.get("retry-after-ms", "")) / 1000)
    retry_after = headers.get("retry-after", "")
    if retry_after:
        try:
            candidates.append(float(retry_after))
        except ValueError:
            with suppress(TypeError, ValueError, OverflowError):
                date = parsedate_to_datetime(retry_after)
                candidates.append((date - datetime.now(timezone.utc)).total_seconds())
    for suffix in ("requests", "tokens", "project-requests", "project-tokens"):
        if headers.get(f"x-ratelimit-remaining-{suffix}") == "0":
            reset = headers.get(f"x-ratelimit-reset-{suffix}", "")
            seconds = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
            pieces = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", reset)
            candidates.append(sum(float(number) * seconds[unit] for number, unit in pieces))
    finite = [value for value in candidates if 0 <= value < float("inf")]
    return max(finite, default=fallback)


@dataclass(frozen=True)
class ExecutionLimits:
    budget: int
    concurrency: int
    requests_per_minute: int
    max_http_attempts: int
    rate_limit_retries: int
    timeout_seconds: float
    regional_processing: bool


class Executor:
    def __init__(
        self, ledger: Ledger, client: httpx.AsyncClient, limits: ExecutionLimits, api_key: str
    ):
        self.ledger = ledger
        self.client = client
        self.limits = limits
        self.api_key = api_key
        self.rate_limiter = RateLimiter(limits.requests_per_minute)
        self.settled = asyncio.Event()
        self.active_attempts = 0
        self.transport_errors = 0
        self.exhausted_rate_limit_retries = 0
        self.previous_attempt_id = self.ledger.db.execute(
            "SELECT COALESCE(MAX(attempt_id), 0) FROM attempts"
        ).fetchone()[0]
        self.run_stop: str | None = None

    async def attempt(self, request: Request, case_id: str, phase: str, retry: int) -> sqlite3.Row:
        await self.rate_limiter.wait()
        while True:
            if self.run_stop:
                raise RunBlocked(self.run_stop)
            attempt_id = self.ledger.reserve(
                request, case_id, phase, self.limits.budget, self.limits.max_http_attempts,
                wait_for_active=self.active_attempts > 0,
            )
            if attempt_id is not None:
                break
            self.settled.clear()
            await self.settled.wait()
            await self.rate_limiter.wait()

        self.active_attempts += 1
        try:
            return await self.dispatch(attempt_id, request, retry)
        finally:
            self.active_attempts -= 1
            self.settled.set()

    async def dispatch(self, attempt_id: int, request: Request, retry: int) -> sqlite3.Row:
        started = time.perf_counter_ns()
        status = None
        request_id = None
        body = None
        probabilities = None
        usage = None
        cost = None
        budgeted_cost = None
        failure = None
        try:
            async with asyncio.timeout(self.limits.timeout_seconds):
                response = await self.client.post(
                    ENDPOINT,
                    content=request.encoded,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
            api_latency_ms = (time.perf_counter_ns() - started) / 1e6
            status = response.status_code
            request_id = sanitize(response.headers.get("x-request-id"), self.api_key)
            try:
                body = response.json()
            except ValueError:
                body = response.text

            error_object = body.get("error") if isinstance(body, dict) else None
            error_code = error_object.get("code") if isinstance(error_object, dict) else None
            quota_error = status == 429 and isinstance(error_code, str) and error_code in QUOTA_ERROR_CODES
            if status == 429 and not quota_error:
                self.rate_limiter.pause(provider_wait(response.headers, min(60, 2 ** (retry + 1))))
            elif status < 300:
                self.rate_limiter.pause(provider_wait(response.headers, 0))

            try:
                usage = Usage.from_response(body)
                cost = usage.cost_nano_usd(self.limits.regional_processing)
                budgeted_cost = usage.cost_nano_usd(regional=True)
            except InvalidDecision as error:
                if status < 300:
                    failure = str(error)
                elif status in REJECTED_HTTP_STATUSES:
                    cost = 0
                    budgeted_cost = 0

            if 200 <= status < 300:
                try:
                    parsed = parse_probabilities(body, request.candidate_ids)
                except InvalidDecision as error:
                    failure = str(error)
                else:
                    if failure is None:
                        probabilities = parsed
            else:
                failure = "http_429_insufficient_quota" if quota_error else f"http_{status}"
        except httpx.ProxyError:
            api_latency_ms = (time.perf_counter_ns() - started) / 1e6
            failure = "proxy_error"
            # CONNECT was rejected before TLS or a request to the API was established.
            cost = 0
            budgeted_cost = 0
        except (httpx.TransportError, TimeoutError):
            api_latency_ms = (time.perf_counter_ns() - started) / 1e6
            failure = "transport_error"

        self.ledger.settle(
            attempt_id,
            status="success" if probabilities is not None else "failed",
            http_status=status,
            request_id=request_id,
            api_latency_ms=api_latency_ms,
            rerank_wall_ms=(time.perf_counter_ns() - started) / 1e6,
            response_body=sanitize(body, self.api_key),
            probabilities=probabilities,
            usage=asdict(usage) if usage is not None else None,
            known_cost_nano_usd=cost,
            budgeted_cost_nano_usd=budgeted_cost,
            failure=failure,
        )
        self.transport_errors = self.transport_errors + 1 if failure == "transport_error" else 0
        if status in FATAL_HTTP_STATUSES or failure in {
            "proxy_error", "response_model_mismatch", "http_429_insufficient_quota",
        }:
            self.run_stop = f"Decisions stopped after {failure}."
        if self.transport_errors >= 3:
            self.run_stop = "Decisions stopped after three consecutive transport failures."
        return self.ledger.latest(request.request_hash)

    def may_dispatch(self, request: Request, previous: sqlite3.Row | None) -> bool:
        if previous is None:
            return True
        if previous["known_cost_nano_usd"] != 0 or previous["status"] != "failed":
            return False
        if previous["failure"] == "http_429":
            return self.ledger.count_rate_limited(request.request_hash) <= self.limits.rate_limit_retries
        return previous["attempt_id"] <= self.previous_attempt_id and (
            previous["http_status"] in FATAL_HTTP_STATUSES
            or previous["failure"] in {"proxy_error", "http_429_insufficient_quota"}
        )

    async def score(self, request: Request, case_id: str, phase: str) -> sqlite3.Row:
        previous = self.ledger.latest(request.request_hash)
        if not self.may_dispatch(request, previous):
            return previous

        retry = self.ledger.count_rate_limited(request.request_hash) if previous else 0
        while True:
            row = await self.attempt(request, case_id, phase, retry)
            if row["http_status"] != 429 or not self.may_dispatch(request, row):
                if row["failure"] == "http_429":
                    self.exhausted_rate_limit_retries += 1
                else:
                    self.exhausted_rate_limit_retries = 0
                if self.exhausted_rate_limit_retries >= 3:
                    self.run_stop = "Decisions stopped after rate-limit retries were exhausted on three consecutive requests."
                return row
            retry += 1

    async def score_remaining(self, plan: Plan) -> None:
        pending = deque(
            request for request in plan.requests.values()
            if self.may_dispatch(request, self.ledger.latest(request.request_hash))
        )

        async def worker() -> None:
            while pending and self.run_stop is None:
                request = pending.popleft()
                try:
                    await self.score(request, plan.owners[request.request_hash], "full")
                except RunBlocked as error:
                    self.run_stop = str(error)
                if self.ledger.totals()["completed_attempts"] % 100 == 0:
                    totals = self.ledger.totals()
                    print(
                        f"Settled {totals['completed_attempts']:,} attempts, "
                        f"known estimated cost ${usd(totals['known_cost_nano_usd']):.4f}.",
                        flush=True,
                    )

        await asyncio.gather(*(worker() for _ in range(self.limits.concurrency)))
        if self.run_stop:
            raise RunBlocked(self.run_stop)


def smoke_report(plan: Plan, executor: Executor, smoke_rows: list[sqlite3.Row]) -> dict[str, Any]:
    observations = []
    total_smoke_bytes = 0
    total_smoke_tokens = 0
    for case_id, row in zip(plan.smoke_case_ids, smoke_rows, strict=False):
        case = plan.cases[case_id]
        relevant = set(plan.qrels[case.query_id])
        original = [candidate["id"] for candidate in case.candidates]
        observation = {
            "case_id": case_id,
            "request_hash": case.request_hash,
            "attempt_id": row["attempt_id"],
            "status": row["status"],
            "failure": row["failure"],
            "baseline_recall_at_5": len(relevant.intersection(original[:5])) / len(relevant),
            "candidate_recall_at_20": len(relevant.intersection(original)) / len(relevant),
            "reranked_recall_at_5": None,
            "reranked_top_5": None,
            "usage": None,
        }
        usage = json.loads(row["usage"]) if row["usage"] is not None else None
        if usage is not None:
            total_smoke_tokens += usage["input_tokens"]
            total_smoke_bytes += len(plan.requests[case.request_hash].encoded)
            observation["usage"] = usage
        if row["status"] == "success":
            probabilities = json.loads(row["probabilities"])
            reranked = reranked_candidate_ids(case.candidates, probabilities)
            if set(reranked) != set(original) or len(reranked) != CANDIDATE_COUNT:
                raise RunBlocked("A smoke ranking did not preserve the frozen candidate IDs.")
            recall = len(relevant.intersection(reranked[:5])) / len(relevant)
            if recall > observation["candidate_recall_at_20"]:
                raise RunBlocked("A smoke ranking exceeded its candidate recall ceiling.")
            observation.update(
                reranked_recall_at_5=recall, reranked_top_5=reranked[:5],
            )
        observations.append(observation)

    remaining = [
        request for request in plan.requests.values()
        if executor.may_dispatch(request, executor.ledger.latest(request.request_hash))
    ]
    totals = executor.ledger.totals()
    complete = len(observations) == len(plan.smoke_case_ids)
    valid_count = sum(row["status"] == "success" for row in smoke_rows)
    compatible = complete and valid_count > 0 and all(
        row["usage"] is not None and (row["status"] == "success" or row["failure"] == "predicate_refusal")
        for row in observations
    )
    projected_nano_usd = None
    guarded_projected_nano_usd = None
    if compatible:
        projected_nano_usd = totals["known_cost_nano_usd"] + totals["held_nano_usd"]
        guarded_projected_nano_usd = totals["budgeted_cost_nano_usd"] + totals["held_nano_usd"]
        # Use observed token density and the byte length of each remaining payload.
        # Project no cache-read discounts and price every token as a cache write.
        for request in remaining:
            size = len(request.encoded)
            tokens = (total_smoke_tokens * size + total_smoke_bytes - 1) // total_smoke_bytes
            estimate = tokens * CACHE_WRITE_NANO_USD
            if tokens > LONG_CONTEXT_THRESHOLD:
                estimate *= 2
            upper_estimate = (estimate * 11 + 9) // 10
            projected_nano_usd += upper_estimate if executor.limits.regional_processing else estimate
            guarded_projected_nano_usd += upper_estimate

    return {
        "case_count": len(plan.smoke_case_ids),
        "valid_case_count": valid_count,
        "observations": observations,
        "unique_benchmark_requests": len(plan.requests),
        "remaining_unique_requests": len(remaining),
        "observed_smoke_input_tokens": total_smoke_tokens,
        "observed_smoke_payload_bytes": total_smoke_bytes,
        "projected_total_cost_usd": usd(projected_nano_usd),
        "guarded_projected_total_cost_usd": usd(guarded_projected_nano_usd),
        "known_estimated_cost_usd": usd(totals["known_cost_nano_usd"]),
        "held_reservations_usd": usd(totals["held_nano_usd"]),
        "budget_usd": usd(executor.limits.budget),
        "compatible_responses": compatible,
        "gate_passed": compatible and guarded_projected_nano_usd <= executor.limits.budget,
    }


async def execute_live(
    plan: Plan,
    synthetic: Request,
    ledger: Ledger,
    results: Path,
    limits: ExecutionLimits,
    api_key: str,
    smoke_only: bool,
) -> str:
    connection_limits = httpx.Limits(
        max_connections=limits.concurrency, max_keepalive_connections=limits.concurrency,
    )
    async with httpx.AsyncClient(
        timeout=limits.timeout_seconds, limits=connection_limits, follow_redirects=False,
    ) as client:
        executor = Executor(ledger, client, limits, api_key)
        result = await executor.score(synthetic, SYNTHETIC_CASE_ID, "synthetic")
        ledger.log(
            "synthetic_check", status=result["status"], attempt_id=result["attempt_id"],
            failure=result["failure"],
        )
        if result["status"] != "success":
            raise RunBlocked(f"The synthetic compatibility check failed ({result['failure'] or result['status']}).")

        smoke_rows = []
        for case_id in plan.smoke_case_ids:
            case = plan.cases[case_id]
            smoke_rows.append(
                await executor.score(plan.requests[case.request_hash], case_id, "smoke")
            )
            if executor.run_stop:
                break
        report = smoke_report(plan, executor, smoke_rows)
        write_json(results / "smoke_test.json", report)
        ledger.log(
            "smoke_check", valid_case_count=report["valid_case_count"],
            projected_total_cost_usd=report["projected_total_cost_usd"],
            gate_passed=report["gate_passed"],
        )
        if not report["compatible_responses"]:
            raise RunBlocked("The frozen smoke test did not establish response and usage compatibility.")
        if not report["gate_passed"]:
            raise RunBlocked("The smoke-based projection exceeds the configured USD budget.")
        if smoke_only:
            return "smoke_passed"
        print(
            f"Smoke gate passed with {len(plan.requests):,} unique benchmark requests "
            f"and projected total cost ${report['projected_total_cost_usd']:.4f}.",
            flush=True,
        )
        await executor.score_remaining(plan)
    return "complete"


def raw_attempt(row: sqlite3.Row) -> dict[str, Any]:
    fields = (
        "attempt_id", "request_hash", "case_id", "kind", "phase", "status", "started_at_utc",
        "finished_at_utc", "http_status", "request_id", "api_latency_ms", "rerank_wall_ms", "failure",
    )
    return {
        **{field: row[field] for field in fields},
        "body": json.loads(row["response_body"]) if row["response_body"] is not None else None,
        "usage": json.loads(row["usage"]) if row["usage"] is not None else None,
        "estimated_cost_usd": usd(row["known_cost_nano_usd"]),
        "budgeted_cost_usd": usd(row["budgeted_cost_nano_usd"]),
        "reserved_cost_usd": usd(row["reserve_nano_usd"]),
    }


def ledger_events(rows: list[sqlite3.Row]) -> Iterable[dict[str, Any]]:
    events = []
    for row in rows:
        identity = {key: row[key] for key in ("attempt_id", "request_hash", "case_id", "kind", "phase")}
        reservation = {"reserved_cost_usd": usd(row["reserve_nano_usd"])}
        events.append({
            **identity, **reservation, "event": "reserved", "timestamp_utc": row["started_at_utc"],
        })
        if row["finished_at_utc"] is not None:
            events.append({
                **identity, **reservation, "event": "settled", "timestamp_utc": row["finished_at_utc"],
                "status": row["status"], "http_status": row["http_status"], "failure": row["failure"],
                "estimated_cost_usd": usd(row["known_cost_nano_usd"]),
                "budgeted_cost_usd": usd(row["budgeted_cost_nano_usd"]),
                "retained_reserve_usd": (
                    usd(row["reserve_nano_usd"]) if row["known_cost_nano_usd"] is None else 0.0
                ),
            })
    return sorted(events, key=lambda row: (row["timestamp_utc"], row["attempt_id"], row["event"]))


def write_usage(path: Path, rows: list[sqlite3.Row]) -> None:
    fields = (
        "attempt_id", "request_hash", "case_id", "kind", "phase", "status", "started_at_utc",
        "finished_at_utc", "http_status", "request_id", "api_latency_ms", "rerank_wall_ms", "failure",
        *USAGE_FIELDS, "provider_prompt_cache_hit", "estimated_cost_usd", "budgeted_cost_usd",
        "reserved_cost_usd",
    )
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            attempt = raw_attempt(row)
            usage = attempt["usage"]
            writer.writerow({
                **attempt,
                **(usage or {}),
                "provider_prompt_cache_hit": usage["cached_input_tokens"] > 0 if usage else None,
            })
    temporary.replace(path)


def logical_outcomes(plan: Plan, ledger: Ledger) -> Iterable[dict[str, Any]]:
    for case in plan.cases.values():
        started = time.perf_counter_ns()
        row = ledger.latest(case.request_hash)
        lookup_ms = (time.perf_counter_ns() - started) / 1e6
        outcome = {
            "case_id": case.case_id,
            "request_hash": case.request_hash,
            "status": "unattempted",
            "cache_status": "none",
            "probabilities": None,
            "response_reference": None,
            "response_model": None,
            "request_id": None,
            "http_status": None,
            "api_latency_ms": None,
            "source_api_latency_ms": None,
            "rerank_wall_ms": None,
            "end_to_end_ms": None,
            "usage": None,
            "source_usage": None,
            "estimated_cost_usd": None,
            "timestamp_utc": None,
            "failure": None,
        }
        if row is None:
            yield outcome
            continue

        fresh = row["case_id"] == case.case_id and row["status"] != "reserved"
        cache_hit = not fresh and row["status"] == "success"
        probabilities = None
        ordering_ms = 0.0
        if row["status"] == "success":
            ordering_started = time.perf_counter_ns()
            probabilities = json.loads(row["probabilities"])
            reranked_candidate_ids(case.candidates, probabilities)
            ordering_ms = (time.perf_counter_ns() - ordering_started) / 1e6
        if fresh:
            rerank_wall_ms = row["rerank_wall_ms"] + ordering_ms
        elif cache_hit:
            rerank_wall_ms = lookup_ms + ordering_ms
        else:
            rerank_wall_ms = None
        body = json.loads(row["response_body"]) if row["response_body"] is not None else None
        usage = json.loads(row["usage"]) if row["usage"] is not None else None
        outcome.update(
            status="failed" if row["status"] == "reserved" else row["status"],
            cache_status="fresh" if fresh else "local_cache_hit" if cache_hit else "none",
            probabilities=probabilities,
            response_reference=row["attempt_id"] if row["status"] != "reserved" else None,
            response_model=body.get("model") if isinstance(body, dict) else None,
            request_id=row["request_id"],
            http_status=row["http_status"],
            api_latency_ms=row["api_latency_ms"] if fresh else None,
            source_api_latency_ms=row["api_latency_ms"],
            rerank_wall_ms=rerank_wall_ms,
            end_to_end_ms=(
                case.retrieval_latency_ms + rerank_wall_ms if rerank_wall_ms is not None else None
            ),
            usage=usage if fresh else None,
            source_usage=usage,
            estimated_cost_usd=usd(row["known_cost_nano_usd"]) if fresh else 0.0 if cache_hit else None,
            timestamp_utc=row["finished_at_utc"] or row["started_at_utc"],
            failure="dispatch_or_response_unknown" if row["status"] == "reserved" else row["failure"],
        )
        yield outcome


def export(
    results: Path,
    plan: Plan,
    synthetic: Request,
    ledger: Ledger,
    limits: ExecutionLimits,
    state: str,
    reason: str | None,
) -> dict[str, Any]:
    requests = (synthetic, *plan.requests.values())
    atomic_jsonl(
        results / "raw_requests.jsonl.gz",
        ({
            "request_hash": request.request_hash,
            "kind": request.kind,
            "payload": json.loads(request.encoded),
        } for request in requests),
    )
    rows = ledger.attempts()
    atomic_jsonl(
        results / "raw_responses.jsonl.gz",
        (raw_attempt(row) for row in rows if row["status"] != "reserved"),
    )
    atomic_jsonl(results / "cost_ledger.jsonl", ledger_events(rows))
    write_usage(results / "api_usage.csv", rows)
    atomic_jsonl(
        results / "failures.jsonl",
        ({
            "attempt_id": row["attempt_id"],
            "case_id": row["case_id"],
            "request_hash": row["request_hash"],
            "kind": row["kind"],
            "status": row["status"],
            "failure": row["failure"] or "dispatch_or_response_unknown",
            "http_status": row["http_status"],
            "timestamp_utc": row["finished_at_utc"] or row["started_at_utc"],
        } for row in rows if row["status"] != "success"),
    )
    events = ledger.db.execute("SELECT * FROM run_log ORDER BY event_id")
    atomic_jsonl(
        results / "run_log.jsonl",
        ({
            "event_id": row["event_id"], "timestamp_utc": row["timestamp_utc"],
            "event": row["event"], **json.loads(row["details"]),
        } for row in events),
    )
    outcomes = list(logical_outcomes(plan, ledger))
    atomic_jsonl(results / "decision_outcomes.jsonl.gz", outcomes)

    outcome_counts = Counter(row["status"] for row in outcomes)
    cache_counts = Counter(row["cache_status"] for row in outcomes)
    totals = ledger.totals()
    usages = [json.loads(row["usage"]) for row in rows if row["usage"] not in (None, "null")]
    sdk_version = None
    with suppress(importlib.metadata.PackageNotFoundError):
        sdk_version = importlib.metadata.version("openai")
    final = {
        "state": "complete_with_failures" if state == "complete" and outcome_counts["failed"] else state,
        "reason": reason,
        "logical_experiment_cases": len(plan.cases),
        "unique_benchmark_requests": len(plan.requests),
        "unique_benchmark_requests_reserved": len({
            row["request_hash"] for row in rows if row["kind"] == "benchmark"
        }),
        "unique_benchmark_requests_completed_successfully": len({
            row["request_hash"] for row in rows
            if row["kind"] == "benchmark" and row["status"] == "success"
        }),
        "ledger_reservations": totals["reservations"],
        "completed_http_attempt_records": totals["completed_attempts"],
        "http_attempts_with_response": sum(row["http_status"] is not None for row in rows),
        "unknown_dispatches": totals["reservations"] - totals["completed_attempts"],
        "failed_attempt_records": sum(row["status"] == "failed" for row in rows),
        "provider_prompt_cache_hits": sum(usage["cached_input_tokens"] > 0 for usage in usages),
        "local_cache_hits": cache_counts["local_cache_hit"],
        "successful_logical_cases": outcome_counts["success"],
        "failed_logical_cases": outcome_counts["failed"],
        "unattempted_logical_cases": outcome_counts["unattempted"],
        "known_estimated_api_cost_usd": usd(totals["known_cost_nano_usd"]),
        "budgeted_api_cost_usd": usd(totals["budgeted_cost_nano_usd"]),
        "held_api_reservations_usd": usd(totals["held_nano_usd"]),
        "budget_usd": usd(limits.budget),
        "regional_processing_price_enabled": limits.regional_processing,
        "token_totals": {field: sum(usage[field] for usage in usages) for field in USAGE_FIELDS},
        "transport": "httpx REST",
        "httpx_version": httpx.__version__,
        "openai_sdk_installed_version": sdk_version,
        "manifest_sha256": plan.manifest_sha256,
    }
    write_json(results / "run_state.json", final)
    return final


def parse_limits(args: argparse.Namespace) -> ExecutionLimits:
    try:
        budget = Decimal(args.budget_usd) * NANO_USD_PER_DOLLAR
        exact_budget = int(budget)
    except (InvalidOperation, OverflowError, ValueError) as error:
        raise RunBlocked("The USD budget must be a finite positive dollar amount.") from error
    if not 0 < exact_budget <= HARD_LIMIT_NANO_USD or budget != exact_budget:
        raise RunBlocked("The USD budget must be positive, at most $15, and precise to 9 decimal places.")
    if (
        not 1 <= args.concurrency <= 64
        or args.requests_per_minute < 1
        or args.max_http_attempts < 1
        or not 0 <= args.rate_limit_retries <= 10
        or not 0 < args.timeout_seconds < float("inf")
    ):
        raise RunBlocked("Execution limits are invalid.")
    return ExecutionLimits(
        exact_budget, args.concurrency, args.requests_per_minute, args.max_http_attempts,
        args.rate_limit_retries, args.timeout_seconds, args.regional_processing,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"), help="Plan offline or execute live inference.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--budget-usd", default="15")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--requests-per-minute", type=int, default=300)
    parser.add_argument("--max-http-attempts", type=int, default=15_000)
    parser.add_argument("--rate-limit-retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=45)
    parser.add_argument("--regional-processing", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    try:
        limits = parse_limits(args)
        api_key = os.environ.get("OPENAI_API_KEY", "") if args.mode == "run" else ""
        if args.mode == "run" and not api_key:
            raise RunBlocked("Live execution requires OPENAI_API_KEY in the process environment.")
        plan = read_plan(args.root, api_key)
        synthetic = synthetic_request(args.root / "rubric.json")
        results = args.root / "results"
        runtime = results / ".decisions-runtime"
        runtime.mkdir(exist_ok=True, mode=0o700)
        ledger_path = runtime / "decisions.sqlite3"
        exported_ledger = results / "cost_ledger.jsonl"
        if not ledger_path.exists() and exported_ledger.exists() and exported_ledger.stat().st_size:
            raise RunBlocked("Restore the original runtime ledger before resuming a recorded run.")
        with (
            exclusive_run(runtime / "run.lock"),
            Ledger(ledger_path, plan.manifest_sha256, limits.regional_processing) as ledger,
        ):
            ledger.log(
                "invocation", mode=args.mode, manifest_sha256=plan.manifest_sha256,
                concurrency=limits.concurrency, requests_per_minute=limits.requests_per_minute,
                max_http_attempts=limits.max_http_attempts, budget_usd=usd(limits.budget),
                rate_limit_retries=limits.rate_limit_retries, timeout_seconds=limits.timeout_seconds,
                regional_processing=limits.regional_processing,
            )
            state, reason = "planned", None
            try:
                if args.mode == "run":
                    state = asyncio.run(
                        execute_live(plan, synthetic, ledger, results, limits, api_key, args.smoke_only)
                    )
                elif ledger.totals()["reservations"]:
                    state = "offline_snapshot"
            except RunBlocked as error:
                state, reason = "blocked", str(error)
            except KeyboardInterrupt:
                state, reason = "interrupted", "Execution was interrupted."
            finally:
                ledger.log("invocation_finished", state=state, reason=reason)
                final = export(results, plan, synthetic, ledger, limits, state, reason)
            print(json.dumps(final, indent=2), flush=True)
        return 1 if state in {"blocked", "interrupted"} else 0
    except (OSError, RunBlocked, ValueError) as error:
        print(f"Benchmark stopped. {sanitize(str(error))}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
