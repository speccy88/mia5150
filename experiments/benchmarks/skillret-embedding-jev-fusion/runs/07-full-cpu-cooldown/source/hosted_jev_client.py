"""Disabled-by-default Jev transport, with evidence gates and a durable spend ledger.

This module has no CLI and performs no I/O at import. It does not discover keys,
purchase credit, or retry ambiguous outcomes. See HOSTED_CLIENT.md for receipts.
The offline preparation/ingestion runner intentionally remains independent.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
import urllib.error
import urllib.request

import run_jev_rerank as prep

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODELS_ENDPOINT = "https://api.typesafe.ai/v1/models"
EXPERIMENT = "newskillret-embedding-jev-fusion"
MAX_CAD = Decimal("10")
PREFIX = "Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: "
SOURCE_HASHES = {
    "skills": "1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81",
    "queries": "4d07a35f40e597f8468c6dab480c95a1f9b40ecbde80274081ba2e7c643d0f6d",
    "qrels": "edbfae0b8109de460d54e27e6235b1b47316f34956744db0571ad1b4ddb860af",
}


class GateError(RuntimeError):
    """No request may be sent until the named gate is satisfied."""


class AmbiguousRequest(GateError):
    """Execution or billing is uncertain; reservation is retained, never resent."""


def dec(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise GateError("Boolean is not a monetary amount")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise GateError("Invalid monetary amount") from None
    if not result.is_finite() or result < 0:
        raise GateError("Money must be finite and nonnegative")
    return result


def require(condition: bool, label: str) -> None:
    if not condition:
        raise GateError(label)


def integer(value: Any, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, "Invalid token/count integer")
    return value


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        require(path.read_bytes() == data, "Immutable artifact differs; manual review required")


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str

    def read(self) -> Any:
        require(bool(re.fullmatch(r"[0-9a-f]{64}", self.sha256)), "Missing artifact SHA256")
        data = Path(self.path).read_bytes()
        require(hash_bytes(data) == self.sha256, "Gate artifact SHA256 mismatch")
        return prep.loads(data.decode("utf-8"))


@dataclass(frozen=True)
class GateBundle:
    baseline_gate: Artifact
    baseline_result: Artifact
    approval: Artifact
    pricing: Artifact
    fx: Artifact
    credit: Artifact
    contract: Artifact
    dense: Artifact
    model_pin: Artifact | None = None
    expansion: Artifact | None = None


@dataclass(frozen=True)
class HTTPResult:
    status: int
    body: bytes
    request_id: str | None = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a bearer credential to another URL.


def stdlib_transport(method: str, url: str, body: bytes | None,
                     key: str, timeout: float) -> HTTPResult:
    """Exactly one HTTP attempt. No SDK retry policy, redirects or proxy env lookup."""
    require(url in (ENDPOINT, MODELS_ENDPOINT), "Endpoint not allowlisted")
    request = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json",
        "Accept": "application/json", "User-Agent": "skillret-jev-study/1",
    })
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc  # Preserve error bytes before any status interpretation.
    with response:
        return HTTPResult(response.status, response.read(), response.headers.get("x-typesafe-request-id"))


class SpendLedger:
    """Single-writer hash-chained JSONL; fsync before request, lock spans settlement.

    A crash leaves a reservation and possibly the lock. Never remove either
    automatically. An incomplete tail fails closed rather than being repaired.
    """
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.path = self.directory / "budget_ledger.jsonl"
        self.lock = self.directory / "budget_ledger.lock"

    @contextmanager
    def locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise GateError("Experiment ledger locked; resolve prior process before proceeding") from None
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(b"Manual recovery only after confirming no request remains active.\n")
                stream.flush()
                os.fsync(stream.fileno())
            yield self
        finally:
            self.lock.unlink()

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        data = self.path.read_bytes()
        require(not data or data.endswith(b"\n"), "Ledger incomplete tail; manual reconciliation required")
        rows, previous = [], "0" * 64
        for line in data.splitlines():
            row = prep.loads(line.decode("utf-8"))
            expected = row.pop("event_sha256", None)
            require(row.get("previous_sha256") == previous and row.get("sequence") == len(rows),
                    "Ledger chain/order mismatch")
            require(row.get("experiment_id") == EXPERIMENT and prep.digest(row) == expected,
                    "Ledger integrity or experiment mismatch")
            row["event_sha256"] = expected
            rows.append(row)
            previous = expected
        return rows

    def append(self, event: dict) -> dict:
        require(self.lock.exists(), "Ledger write requires lock")
        rows = self.rows()
        row = {**event, "sequence": len(rows), "experiment_id": EXPERIMENT,
               "previous_sha256": rows[-1]["event_sha256"] if rows else "0" * 64}
        row["event_sha256"] = prep.digest(row)
        encoded = prep.canonical(row) + b"\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return row

    def commitments(self) -> tuple[Decimal, Decimal]:
        held: dict[str, tuple[Decimal, Decimal]] = {}
        for row in self.rows():
            attempt = row.get("attempt_id")
            if row["event"] == "reserved":
                require(attempt not in held, "Duplicate reservation")
                held[attempt] = (dec(row["reserve_cad"]), dec(row["reserve_usd"]))
            elif row["event"] == "settled":
                require(attempt in held, "Settlement without reservation")
                held[attempt] = (dec(row["actual_cad"]), dec(row["actual_usd"]))
            elif row["event"] == "documented_no_execution":
                require(attempt in held, "Release without reservation")
                held[attempt] = (Decimal(0), Decimal(0))
        return (sum((x[0] for x in held.values()), Decimal(0)),
                sum((x[1] for x in held.values()), Decimal(0)))


class HostedJevClient:
    def __init__(self, root_dir: Path, gates: GateBundle | None = None, *, enabled: bool = False,
                 transport: Callable = stdlib_transport, clock: Callable = time.time,
                 sleeper: Callable = time.sleep, timeout: float = 120):
        self.root = Path(root_dir)
        self.gates, self.enabled = gates, enabled
        self.transport, self.clock, self.sleeper = transport, clock, sleeper
        require(0 < timeout <= 300, "Timeout must be in (0, 300] seconds")
        self.timeout = timeout
        self.ledger = SpendLedger(self.root)
        self._baseline_cache: tuple | None = None

    def _fresh(self, artifact: Artifact, age: float) -> dict:
        receipt = artifact.read()
        require(isinstance(receipt, dict), "Receipt must be an object")
        require(receipt.get("verified") is True, "Receipt is unverified")
        stamp = receipt.get("verified_at_unix")
        require(type(stamp) in (int, float) and 0 <= self.clock() - stamp <= age,
                "Receipt is stale or has a future timestamp")
        require(isinstance(receipt.get("source"), str) and bool(receipt["source"]),
                "Receipt source is required")
        return receipt

    def _baseline(self) -> dict:
        g = self.gates
        require(g is not None, "Gate bundle absent")
        # Re-read after a local change; avoid hashing a large baseline on every call.
        refs = (g.baseline_gate, g.baseline_result, g.dense)
        identity = tuple((str(a.path), a.sha256, Path(a.path).stat().st_size,
                          Path(a.path).stat().st_mtime_ns) for a in refs)
        if self._baseline_cache and self._baseline_cache[0] == identity:
            return self._baseline_cache[1]
        gate, result = g.baseline_gate.read(), g.baseline_result.read()
        require(gate.get("status") == "validated" and result.get("baseline_gate") == "validated"
                and result.get("status") == "completed", "Full baseline not validated")
        require((gate.get("full_query_count"), gate.get("full_skill_count")) == (4392, 6006),
                "Baseline full counts differ")
        require(dec(gate.get("max_abs_ndcg_deviation")) <= Decimal("0.005")
                and dec(result.get("max_abs_ndcg_deviation")) <= Decimal("0.005"),
                "NDCG discrepancy requires investigation")
        require(gate.get("contract_sha256") == result.get("cache_contract_sha256"),
                "Baseline contract mismatch")
        protocol = result.get("protocol", {})
        require(protocol.get("test_file_sha256") == SOURCE_HASHES, "Dataset identity mismatch")
        require(protocol.get("model_revision") == "67f1594c46c40e13a5eb2f4092cc85c98c9ed32d"
                and protocol.get("dataset_revision") == "6583d7d2ed07644d0fb8938ed8178f3a7dc42a12"
                and protocol.get("official_code_revision") == "0cc508e6bfcfff1fe6352d90ff5957ba38a64bba",
                "Baseline model/source revisions differ")
        require(hash_bytes(json.dumps(protocol, sort_keys=True).encode()) == gate.get("contract_sha256"),
                "Baseline protocol content hash differs")
        require(protocol.get("query_prefix") == PREFIX and protocol.get("max_seq_length") == 32768
                and protocol.get("normalize_embeddings") is True and protocol.get("top_k") == 50,
                "Baseline encoding protocol mismatch")
        require(protocol.get("effective_rope_theta") == 1000000,
                "Baseline effective RoPE theta must be 1000000")
        require(all(isinstance(protocol.get(name), str) and re.fullmatch(r"[0-9a-f]{64}", protocol[name])
                    for name in ("document_text_sha256", "query_text_sha256")),
                "Baseline transformed document/query text hashes missing")
        evaluation = result.get("evaluation", {})
        require(evaluation.get("query_count") == 4392 and evaluation.get("missing_query_count") == 0,
                "Baseline evaluated scope incomplete")
        aggregate = evaluation.get("aggregate", {})
        require(all(abs(dec(aggregate.get("NDCG@" + str(k))) - target) <= Decimal("0.005")
                    for k, target in ((5, Decimal(".7902")), (10, Decimal(".8112")), (15, Decimal(".8187")))),
                "Baseline aggregate NDCG fails target tolerance")
        rankings = result.get("rankings", {})
        require(len(rankings) == 4392 and all(len(v) == len(set(v)) == 50 for v in rankings.values()),
                "Baseline top50 incomplete")
        dense_bytes = Path(g.dense.path).read_bytes()
        require(hash_bytes(dense_bytes) == g.dense.sha256, "Dense file SHA256 mismatch")
        dense_rows = [prep.loads(line) for line in dense_bytes.decode("utf-8").splitlines() if line.strip()]
        require(len(dense_rows) == 4392, "Dense row count incomplete")
        dense = prep.index_rows(dense_rows, "query_id")
        require(set(dense) == set(rankings), "Dense and baseline query IDs differ")
        for query_id, row in dense.items():
            candidates = prep.dense_candidates(row)
            require([c["skill_id"] for c in candidates] == rankings[query_id], "Dense ranks differ")
        self._baseline_cache = (identity, result)
        return result

    def _base_gates(self) -> dict:
        require(self.enabled is True, "Hosted client disabled by default")
        baseline = self._baseline()  # Never inspect a credential before all gates below.
        g = self.gates
        approval = g.approval.read()
        require(approval.get("approved") is True and approval.get("experiment_id") == EXPERIMENT
                and approval.get("currency") == "CAD" and approval.get("all_in") is True
                and approval.get("replaces_old_limit") is True
                and isinstance(approval.get("human_confirmation"), str)
                and bool(approval["human_confirmation"]), "Explicit all-in approval receipt absent")
        cap = dec(approval.get("total_cap"))
        require(0 < cap <= MAX_CAD, "Approval exceeds CAD10 experiment ceiling")
        return {"baseline": baseline, "cap": cap}

    def _gates(self, paid: bool) -> dict:
        base = self._base_gates()
        baseline, cap = base["baseline"], base["cap"]
        g = self.gates
        price = self._fresh(g.pricing, 86400)
        fx = self._fresh(g.fx, 172800)
        credit = self._fresh(g.credit, 3600)
        contract = self._fresh(g.contract, 30 * 86400)
        require(price.get("currency") == "USD" and price.get("unit") == "per_million_tokens",
                "Unsupported pricing units")
        input_rate, output_rate = dec(price.get("input")), dec(price.get("output"))
        require(input_rate > 0, "Input price verification required")
        require(fx.get("base") == "USD" and fx.get("quote") == "CAD", "FX pair mismatch")
        rate, fees, buffer = dec(fx.get("rate")), dec(fx.get("fee_fraction")), dec(fx.get("buffer_fraction"))
        fixed = dec(fx.get("fixed_fee_cad"))
        require(rate > 0 and buffer >= Decimal("0.10") and fx.get("fees_verified") is True,
                "Verified FX fees and at least ten-percent buffer required")
        require(credit.get("currency") == "USD" and credit.get("read_only") is True
                and credit.get("no_topup") is True, "Existing read-only credit receipt required")
        available = dec(credit.get("available"))
        require(available > 0, "No verified existing credit")
        require(contract.get("endpoint") == ENDPOINT and contract.get("models_endpoint") == MODELS_ENDPOINT
                and contract.get("models_no_charge") is True
                and contract.get("credential_env") == "TYPESAFE_API_KEY"
                and contract.get("rubric_sha256") == prep.digest(prep.RUBRIC), "Provider contract mismatch")
        total_context = integer(contract.get("total_context_tokens"), 1)
        single_context = integer(contract.get("state_plus_question_tokens"), 1)
        require(total_context <= 64000 and single_context <= 32000, "Unverified context enlargement")
        require(contract.get("token_estimator") == "utf8_bytes_plus_overhead_conservative_estimate",
                "Conservative context estimator must be documented")
        reserve_input = integer(contract.get("max_billable_input_tokens"), total_context)
        require(isinstance(contract.get("reserve_allowance_basis"), str)
                and bool(contract["reserve_allowance_basis"]), "Full-context reservation allowance needs an explanation")
        retries = contract.get("documented_no_execution_statuses", {})
        require(isinstance(retries, dict) and all(str(k).isdigit() and 400 <= int(k) <= 599
                and isinstance(v, str) and v.startswith("https://") for k, v in retries.items()),
                "Retry statuses require explicit no-execution/no-billing documentation")
        stable = None
        if paid:
            require(g.model_pin is not None, "Stable model pin absent")
            pin = self._fresh(g.model_pin, 86400)
            stable = pin.get("actual_model")
            require(isinstance(stable, str) and bool(re.fullmatch(r"jev-\d+\.\d+\.\d+", stable)),
                    "Actual model must be a stable Jev version")
            require(pin.get("requested_alias") == "jev-latest" and price.get("model") == stable,
                    "Price or alias version mismatch")
            model_record = Artifact(self.root / "model_list_receipt.json", pin.get("model_list_receipt_sha256", "")).read()
            require(model_record.get("status_code") == 200
                    and model_record.get("baseline_sha256") == g.baseline_result.sha256,
                    "Model-list receipt must follow this baseline")
            require(type(model_record.get("verified_at_unix")) in (int, float)
                    and 0 <= self.clock() - model_record["verified_at_unix"] <= 86400,
                    "Model-list receipt is stale")
            raw = (self.root / "model_list.body").read_bytes()
            require(hash_bytes(raw) == model_record.get("response_sha256"), "Model list raw bytes differ")
            models = prep.loads(raw.decode("utf-8")).get("models")
            require(isinstance(models, list) and any(m.get("name") == "jev-latest" for m in models),
                    "jev-latest absent from verified list")
            require(pin.get("alias_resolution_verified") is True and bool(pin.get("source")),
                    "Alias resolution requires evidence; model list alone may expose only aliases")
        return dict(cap=cap, price=price, input_rate=input_rate, output_rate=output_rate, fx=rate,
                    multiplier=(1 + fees) * (1 + buffer), fixed=fixed, available=available,
                    contract=contract, model=stable, baseline=baseline, retry_statuses=retries,
                    reserve_input=reserve_input)

    @staticmethod
    def _credential() -> str:
        # This is the ONLY credential access. No get(), fallback, file, SDK or key logging.
        try:
            key = os.environ["TYPESAFE_API_KEY"]
        except KeyError:
            raise GateError("TYPESAFE_API_KEY missing; no request sent") from None
        require(isinstance(key, str) and bool(key.strip()), "TYPESAFE_API_KEY empty; no request sent")
        return key

    def _call(self, method: str, url: str, body: bytes | None, key: str) -> HTTPResult:
        return self.transport(method, url, body, key, self.timeout)

    def model_list(self) -> dict:
        """Authenticated read-only lookup after baseline and explicit budget approval."""
        self._base_gates()
        with self.ledger.locked():
            receipt_path = self.root / "model_list_receipt.json"
            if receipt_path.exists():
                receipt = prep.read_json(receipt_path)
                raw = (self.root / "model_list.body").read_bytes()
                require(hash_bytes(raw) == receipt.get("response_sha256"), "Model-list cache corrupt")
                require(receipt.get("baseline_sha256") == self.gates.baseline_result.sha256,
                        "Model-list cache baseline differs")
                require(receipt.get("status_code") == 200, "Cached model-list lookup failed")
                require(type(receipt.get("verified_at_unix")) in (int, float)
                        and 0 <= self.clock() - receipt["verified_at_unix"] <= 86400, "Cached model-list receipt is stale")
                require(isinstance(prep.loads(raw.decode("utf-8")).get("models"), list),
                        "Cached model-list response contract changed")
                return receipt
            started = self.clock()
            result = self._call("GET", MODELS_ENDPOINT, None, self._credential())
            elapsed = max(0, self.clock() - started)
            write_immutable(self.root / "model_list.body", result.body)  # Before parse.
            receipt = {"status_code": result.status, "response_sha256": hash_bytes(result.body),
                       "elapsed_seconds": elapsed, "verified_at_unix": self.clock(),
                       "baseline_sha256": self.gates.baseline_result.sha256,
                       "provider_request_id": self._safe_request_id(result.request_id)}
            write_immutable(receipt_path, prep.canonical(receipt) + b"\n")
            require(result.status == 200, "Model-list lookup failed; raw bytes saved")
            require(isinstance(prep.loads(result.body.decode("utf-8")).get("models"), list),
                    "Model-list response contract changed")
            return receipt

    @staticmethod
    def _safe_request_id(value: Any) -> str | None:
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value) else None

    def _cost(self, input_tokens: int, output_tokens: int, gates: dict) -> tuple[Decimal, Decimal]:
        usd = (Decimal(input_tokens) * gates["input_rate"] + Decimal(output_tokens) * gates["output_rate"]) / 1000000
        return usd * gates["fx"] * gates["multiplier"], usd

    def _validate_packet(self, packet: dict, request: dict, plan: dict, gates: dict) -> tuple[int, int]:
        unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
        require(prep.digest(unsigned) == plan.get("plan_sha256"), "Preparation manifest changed")
        require(prep.digest(request) == packet.get("request_sha256"), "Exact request hash mismatch")
        require(plan.get("rubric_sha256") == prep.digest(prep.RUBRIC), "Prepared rubric mismatch")
        require(plan.get("sources", {}).get("dense", {}).get("sha256") == self.gates.dense.sha256,
                "Prepared dense pool differs from validated baseline")
        for kind in ("skills", "queries"):
            require(plan.get("sources", {}).get(kind, {}).get("sha256") == SOURCE_HASHES[kind],
                    "Prepared source data differs from official source")
        require(plan.get("counts", {}).get("queries") == 4392 and plan["counts"].get("skills") == 6006,
                "Prepared source counts differ")
        expected_size = {"smoke": 10, "pilot": 200, "full": 4392}.get(plan.get("scope"))
        require(expected_size is not None and len(plan.get("rows", [])) == expected_size,
                "Prepared scope size mismatch")
        expected_ids = prep.select_ids(list(gates["baseline"]["rankings"]), expected_size)
        require([row["query_id"] for row in plan["rows"]] == expected_ids, "Prepared IDs not deterministic scope")
        matched = [row for row in plan["rows"] if row["query_id"] == packet.get("query_id")]
        require(len(matched) == 1, "Packet query absent or duplicated")
        row = matched[0]
        require(row.get("status") == "prepared_not_run" and packet in row.get("batches", []),
                "Packet absent or query blocked")
        require([c["skill_id"] for c in row["dense_top20"]] == gates["baseline"]["rankings"][row["query_id"]][:20],
                "Candidate pool differs from baseline top20")
        require([c["dense_rank"] for c in row["dense_top20"]] == list(range(1, 21)),
                "Candidate dense-rank mapping is invalid")
        expected_pairs = [("s" + str(c["dense_rank"] - 1), c["skill_id"]) for c in row["dense_top20"]]
        actual_pairs = []
        for batch in row["batches"]:
            qids, cids = batch.get("question_ids", []), batch.get("candidate_ids", [])
            require(len(qids) == len(cids) == len(set(qids)) == len(set(cids)) and bool(qids),
                    "Candidate/question ID lengths or uniqueness differ")
            actual_pairs.extend(zip(qids, cids))
        require(actual_pairs == expected_pairs, "Prepared batches do not partition dense top20 with exact ID mapping")
        require(set(request) == {"state", "questions", "model"}, "Unexpected request fields")
        require(request.get("state") == {"task": row.get("query")}, "Original task state mismatch")
        questions = request.get("questions", {})
        require(set(questions) == set(packet.get("question_ids", [])) and 1 <= len(questions) <= 20,
                "Question IDs differ from prepared batch")
        audit = {item["skill_id"]: item for item in row.get("candidate_text_audit", [])}
        mapping = dict(expected_pairs)
        for question_id, question in questions.items():
            require(set(question) == {"type", "criteria", "instructions"}
                    and question.get("type") == "score" and question.get("criteria") == prep.RUBRIC
                    and set(question.get("instructions", {})) == {"question"}
                    and question["instructions"]["question"].startswith(prep.QUESTION), "Fixed rubric changed")
            text = question["instructions"]["question"][len(prep.QUESTION):]
            evidence = audit.get(mapping[question_id], {})
            require(evidence.get("truncated") is False
                    and evidence.get("document_sha256") == hash_bytes(text.encode("utf-8"))
                    and evidence.get("original_characters") == evidence.get("prepared_characters") == len(text)
                    and evidence.get("original_utf8_bytes") == evidence.get("prepared_utf8_bytes") == len(text.encode("utf-8")),
                    "Full candidate text differs from its prepared audit")
        require(request.get("model") in ("jev-latest", gates["model"]), "Request model differs from pin")
        require(plan.get("expected_model_version") in (None, gates["model"]), "Prepared model version differs")
        context = plan.get("context", {})
        overhead = integer(context.get("overhead_tokens"), 1)
        output = integer(context.get("reserved_output_tokens"), 1)
        estimates = prep.estimate_request_context(request, overhead_tokens=overhead)
        input_bound = estimates["estimated_input_token_upper_bound"]
        require(input_bound == packet.get("estimated_input_token_upper_bound"), "Input bound differs")
        require(input_bound + output <= gates["contract"]["total_context_tokens"], "Total context overflow")
        for single in estimates["per_question_estimated_pair_tokens"].values():
            require(single + output <= gates["contract"]["state_plus_question_tokens"],
                    "State-plus-question context overflow; full text must not be silently truncated")
        if plan["scope"] != "smoke":
            require(self.gates.expansion is not None, "Pilot/full expansion requires measured projection")
            expansion = self._fresh(self.gates.expansion, 86400)
            require(expansion.get("plan_sha256") == plan["plan_sha256"]
                    and expansion.get("based_on_actual_smoke_usage") is True
                    and integer(expansion.get("completed_smoke_queries")) == 10
                    and dec(expansion.get("projected_total_all_in_cad")) <= gates["cap"]
                    and dec(expansion.get("projected_total_runtime_seconds")) <= Decimal(21600),
                    "Measured expansion exceeds budget/runtime or has wrong scope")
        return input_bound, output

    @staticmethod
    def _normalized_payload(request: dict, actual_model: str) -> dict:
        """Only model alias is normalized; every other payload field is preserved."""
        return {**request, "model": actual_model}

    def _reuse_alias_cache(self, packet: dict, request: dict, plan: dict, gates: dict,
                          rows: list[dict], cache_path: Path) -> dict | None:
        semantic = prep.digest(self._normalized_payload(request, gates["model"]))
        index_path = self.root / "semantic_cache" / (semantic + ".json")
        if not index_path.exists():
            return None
        index = prep.read_json(index_path)
        original_hash = index.get("original_transport_request_sha256")
        require(isinstance(original_hash, str) and bool(re.fullmatch(r"[0-9a-f]{64}", original_hash)),
                "Invalid semantic-cache source request hash")
        require(index.get("normalized_payload_sha256") == semantic and index.get("actual_model_version") == gates["model"],
                "Semantic cache model or payload identity differs")
        require(index.get("batch_binding") == {"candidate_ids": packet["candidate_ids"], "question_ids": packet["question_ids"]},
                "Alias cache candidate/question mapping differs")
        original_bytes = (self.root / "requests" / (original_hash + ".json")).read_bytes()
        require(hash_bytes(original_bytes) == original_hash, "Original transport request bytes changed")
        original_request = prep.loads(original_bytes.decode("utf-8"))
        require(self._normalized_payload(original_request, gates["model"]) == self._normalized_payload(request, gates["model"]),
                "Alias cache differs beyond the model alias")
        settled = [r for r in rows if r.get("request_sha256") == original_hash and r["event"] == "settled"]
        require(len(settled) == 1 and settled[0].get("actual_model_version") == gates["model"]
                and settled[0].get("response_sha256") == index.get("response_sha256"),
                "Original alias request has no verified settlement; never resend")
        original_path = self.root / "raw_responses" / (original_hash + ".json")
        original_packet = {**packet, "request_sha256": original_hash}
        prep.parse_cache(original_path, original_packet, gates["model"], gates["contract"]["total_context_tokens"])
        original = prep.read_json(original_path)
        require(isinstance(original.get("attempt_id"), str)
                and re.fullmatch(original_hash + r"-[1-3]", original["attempt_id"]),
                "Original cache attempt identity is invalid")
        original_body = (self.root / "attempts" / (original["attempt_id"] + ".body")).read_bytes()
        require(hash_bytes(original_body) == index.get("response_sha256")
                and original_body == original["raw_response"].encode("utf-8"), "Original raw response bytes differ")
        envelope = {**original, "request_sha256": packet["request_sha256"],
                    "expected_request_sha256": packet["request_sha256"],
                    "original_transport_request_sha256": original_hash,
                    "cache_reuse": {"kind": "verified_model_alias_to_pin_payload_equivalence",
                                    "normalized_payload_sha256": semantic,
                                    "original_requested_model": original_request["model"],
                                    "target_requested_model": request["model"],
                                    "actual_model_version": gates["model"], "additional_api_calls": 0,
                                    "additional_input_tokens_billed": 0, "additional_output_tokens_billed": 0,
                                    "additional_cost_usd": "0", "additional_all_in_cost_cad": "0",
                                    "original_usage_and_latency_preserved": True}}
        write_immutable(cache_path, prep.canonical(envelope) + b"\n")
        self.ledger.append({"event": "cache_reused", "request_sha256": packet["request_sha256"],
                            "original_transport_request_sha256": original_hash, "plan_sha256": plan["plan_sha256"],
                            "normalized_payload_sha256": semantic, "actual_model_version": gates["model"],
                            "response_sha256": index["response_sha256"], "utc_unix": self.clock(), "additional_api_calls": 0})
        return envelope

    def request(self, packet: dict, request: dict, *, plan: dict) -> dict:
        """Execute one frozen batch, or return its validated cache; never partially rank.

        Returns the raw-response envelope consumed by run_jev_rerank.ingest().
        Paid context/runtime projection and every attempt remain inspectable on disk.
        """
        gates = self._gates(paid=True)
        input_bound, output_bound = self._validate_packet(packet, request, plan, gates)
        request_hash = packet["request_sha256"]
        cache_path = self.root / "raw_responses" / (request_hash + ".json")
        max_attempts = integer(gates["contract"].get("max_attempts", 1), 1)
        require(max_attempts <= 3, "At most three explicitly safe attempts permitted")
        with self.ledger.locked():
            rows = self.ledger.rows()
            require({r.get("actual_model_pin") for r in rows if r["event"] == "reserved"} <= {gates["model"]},
                    "Experiment's pinned actual model cannot change across requests")
            history = [row for row in rows if row.get("request_sha256") == request_hash]
            if any(row["event"] in ("settled", "cache_reused") for row in history):
                require(cache_path.exists(), "Settled response cache missing; never resend")
                prep.parse_cache(cache_path, packet, gates["model"], gates["contract"]["total_context_tokens"])
                return prep.read_json(cache_path)
            reused = self._reuse_alias_cache(packet, request, plan, gates, rows, cache_path)
            if reused is not None:
                return reused
            resolved = {r["attempt_id"] for r in rows if r["event"] in ("settled", "documented_no_execution")}
            require(not any(r["event"] == "reserved" and r["attempt_id"] not in resolved for r in rows),
                    "Experiment has an unsettled prior request; ambiguous reserve retained and resend blocked")
            released = {r["attempt_id"] for r in history if r["event"] == "documented_no_execution"}
            require(not any(r["event"] == "reserved" and r["attempt_id"] not in released for r in history),
                    "Unsettled prior request; ambiguous reserve retained and resend blocked")
            prior_attempts = sum(r["event"] == "reserved" for r in history)
            for attempt in range(prior_attempts + 1, max_attempts + 1):
                # Validate timestamps again after lock acquisition/backoff before touching a key.
                gates = self._gates(paid=True)
                # A deliberately large reservation allowance, not a claim about native tokenization.
                reserve_cad, reserve_usd = self._cost(gates["reserve_input"], output_bound, gates)
                used_cad, used_usd = self.ledger.commitments()
                require(used_cad + reserve_cad + gates["fixed"] <= gates["cap"], "CAD10 all-in budget would be exceeded")
                require(used_usd + reserve_usd <= gates["available"], "Verified existing credit insufficient")
                key = self._credential()  # Missing key: fail without a request or misleading spend reservation.
                attempt_id = request_hash + "-" + str(attempt)
                event = {"request_sha256": request_hash, "attempt_id": attempt_id,
                         "plan_sha256": plan["plan_sha256"], "utc_unix": self.clock()}
                write_immutable(self.root / "requests" / (request_hash + ".json"), prep.canonical(request))
                self.ledger.append({**event, "event": "reserved", "reserve_cad": str(reserve_cad),
                                    "reserve_usd": str(reserve_usd), "input_token_reserve_allowance": gates["reserve_input"],
                                    "estimated_input_tokens_for_packing": input_bound,
                                    "output_token_bound": output_bound, "actual_model_pin": gates["model"],
                                    "requested_model": request["model"],
                                    "pricing_sha256": self.gates.pricing.sha256, "fx_sha256": self.gates.fx.sha256})
                started = self.clock()
                try:
                    response = self._call("POST", ENDPOINT, prep.canonical(request), key)
                except BaseException as exc:
                    self.ledger.append({**event, "event": "ambiguous_blocked", "error_type": type(exc).__name__,
                                        "elapsed_seconds": max(0, self.clock() - started)})
                    raise AmbiguousRequest("Attempt did not complete; reserve retained, no automatic retry") from None
                elapsed = max(0, self.clock() - started)
                body_path = self.root / "attempts" / (attempt_id + ".body")
                write_immutable(body_path, response.body)  # Unmodified bytes, fsynced BEFORE decode/parse.
                meta = {**event, "status_code": response.status, "response_sha256": hash_bytes(response.body),
                        "elapsed_seconds": elapsed, "provider_request_id": self._safe_request_id(response.request_id)}
                write_immutable(self.root / "attempts" / (attempt_id + ".json"), prep.canonical(meta) + b"\n")
                self.ledger.append({**meta, "event": "response_saved"})
                if str(response.status) in gates["retry_statuses"]:
                    self.ledger.append({**event, "event": "documented_no_execution", "status_code": response.status,
                                        "documentation": gates["retry_statuses"][str(response.status)]})
                    if attempt < max_attempts:
                        self.sleeper(min(2 ** (attempt - 1), 8))
                    continue
                try:
                    require(response.status == 200, "Non-200 paid result is not safe to retry")
                    raw = response.body.decode("utf-8")
                    parsed = prep.loads(raw)
                    _, usage = prep.parse_response(parsed, packet, gates["model"], gates["contract"]["total_context_tokens"])
                    require(usage["input_tokens"] <= gates["reserve_input"], "Actual input exceeds held token allowance")
                    output_tokens = usage["output_tokens"]
                    require(output_tokens is not None or gates["output_rate"] == 0,
                            "Billable output token usage missing")
                    if output_tokens is not None:
                        require(output_tokens <= output_bound or gates["output_rate"] == 0,
                                "Billable output exceeds reservation bound")
                    actual_cad, actual_usd = self._cost(usage["input_tokens"], output_tokens or 0, gates)
                    require(actual_cad <= reserve_cad and actual_usd <= reserve_usd, "Cost exceeded held reservation")
                    envelope = {"schema_version": prep.SCHEMA_VERSION, "request_sha256": request_hash,
                                "expected_request_sha256": request_hash, "original_transport_request_sha256": request_hash,
                                "status_code": response.status, "raw_response": raw,
                                "response_sha256": hash_bytes(response.body), "elapsed_seconds": elapsed,
                                "provider_request_id": meta["provider_request_id"], "attempt_id": attempt_id,
                                "actual_model_version": usage["actual_model_version"],
                                "input_tokens": usage["input_tokens"], "output_tokens": output_tokens,
                                "cost_usd": str(actual_usd), "all_in_cost_cad_with_buffer": str(actual_cad)}
                    write_immutable(cache_path, prep.canonical(envelope) + b"\n")
                    semantic = prep.digest(self._normalized_payload(request, gates["model"]))
                    write_immutable(self.root / "semantic_cache" / (semantic + ".json"), prep.canonical({
                        "normalized_payload_sha256": semantic, "actual_model_version": gates["model"],
                        "original_transport_request_sha256": request_hash, "response_sha256": hash_bytes(response.body),
                        "batch_binding": {"candidate_ids": packet["candidate_ids"], "question_ids": packet["question_ids"]}
                    }) + b"\n")
                    self.ledger.append({**event, "event": "settled", "actual_cad": str(actual_cad),
                                        "actual_usd": str(actual_usd), "input_tokens": usage["input_tokens"],
                                        "output_tokens": output_tokens, "actual_model_version": gates["model"],
                                        "elapsed_seconds": elapsed, "response_sha256": hash_bytes(response.body)})
                    return envelope
                except Exception as exc:
                    self.ledger.append({**event, "event": "ambiguous_blocked", "error_type": type(exc).__name__,
                                        "response_sha256": hash_bytes(response.body)})
                    raise AmbiguousRequest("Response saved but validation/billing uncertain; do not resend") from None
            raise GateError("Documented safe retry attempts exhausted; no completed paid response")
