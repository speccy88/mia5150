"""Synthetic offline tests. Network and environment credential access are mocked."""
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hosted_jev_client as client
import run_jev_rerank as prep


NOW = 1801700000.0


class DeniedEnvironment(dict):
    def __getitem__(self, key):
        raise AssertionError("Gate failure must not inspect credentials")


class HostedClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.shared.name)
        cls.query_ids = ["q" + str(i) for i in range(4392)]
        cls.ranks = [str(i) for i in range(50)]
        protocol = {"test_file_sha256": client.SOURCE_HASHES, "query_prefix": client.PREFIX,
                    "max_seq_length": 32768, "normalize_embeddings": True, "top_k": 50,
                    "effective_rope_theta": 1000000, "document_text_sha256": "a" * 64, "query_text_sha256": "b" * 64,
                    "model_revision": "67f1594c46c40e13a5eb2f4092cc85c98c9ed32d",
                    "dataset_revision": "6583d7d2ed07644d0fb8938ed8178f3a7dc42a12",
                    "official_code_revision": "0cc508e6bfcfff1fe6352d90ff5957ba38a64bba"}
        contract_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        cls.result = {"status": "completed", "baseline_gate": "validated", "max_abs_ndcg_deviation": 0,
                      "cache_contract_sha256": contract_hash, "protocol": protocol,
                      "evaluation": {"query_count": 4392, "missing_query_count": 0,
                                     "aggregate": {"NDCG@5": .7902, "NDCG@10": .8112, "NDCG@15": .8187}},
                      "rankings": {qid: cls.ranks for qid in cls.query_ids}}
        cls.gate = {"status": "validated", "max_abs_ndcg_deviation": 0, "full_query_count": 4392,
                    "full_skill_count": 6006, "contract_sha256": contract_hash}
        cls.baseline_gate = cls.artifact(cls.folder / "gate.json", cls.gate)
        cls.baseline_result = cls.artifact(cls.folder / "baseline.json", cls.result)
        dense_path = cls.folder / "dense.jsonl"
        candidates = [{"skill_id": str(i), "score": 1 - i / 100} for i in range(50)]
        dense_path.write_bytes(b"".join(prep.canonical({"query_id": qid, "candidates": candidates}) + b"\n"
                                       for qid in cls.query_ids))
        cls.dense = client.Artifact(dense_path, prep.file_hash(dense_path))

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    @staticmethod
    def artifact(path, value):
        path.write_bytes(prep.canonical(value) + b"\n")
        return client.Artifact(path, prep.file_hash(path))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.calls = []
        self.sleep_calls = []
        self.common = {"verified": True, "verified_at_unix": NOW, "source": "https://example.test/offline-fixture"}
        self.approval = {"approved": True, "experiment_id": client.EXPERIMENT, "currency": "CAD",
                         "all_in": True, "replaces_old_limit": True, "total_cap": "10",
                         "human_confirmation": "Synthetic offline fixture only"}
        self.price = {**self.common, "currency": "USD", "unit": "per_million_tokens",
                      "input": ".042", "output": "0", "model": "jev-1.13.0"}
        self.fx = {**self.common, "base": "USD", "quote": "CAD", "rate": "1.40",
                   "fee_fraction": ".03", "buffer_fraction": ".10", "fixed_fee_cad": "0",
                   "fees_verified": True}
        self.credit = {**self.common, "currency": "USD", "available": "5", "read_only": True, "no_topup": True}
        self.contract = {**self.common, "endpoint": client.ENDPOINT, "models_endpoint": client.MODELS_ENDPOINT,
                         "models_no_charge": True, "credential_env": "TYPESAFE_API_KEY",
                         "rubric_sha256": prep.digest(prep.RUBRIC), "total_context_tokens": 64000,
                         "state_plus_question_tokens": 32000,
                         "token_estimator": "utf8_bytes_plus_overhead_conservative_estimate",
                         "max_billable_input_tokens": 65536,
                         "reserve_allowance_basis": "Synthetic conservative full-context allowance; not tokenizer proof",
                         "documented_no_execution_statuses": {}, "max_attempts": 1}
        self.gates = client.GateBundle(
            baseline_gate=self.baseline_gate, baseline_result=self.baseline_result, dense=self.dense,
            approval=self.artifact(self.path / "approval.json", self.approval),
            pricing=self.artifact(self.path / "price.json", self.price),
            fx=self.artifact(self.path / "fx.json", self.fx),
            credit=self.artifact(self.path / "credit.json", self.credit),
            contract=self.artifact(self.path / "contract.json", self.contract))
        self.run_dir = self.path / "run"
        self.run_dir.mkdir()
        model_raw = prep.canonical({"models": [{"name": "jev-latest"}, {"name": "jev-1.13.0"}]})
        (self.run_dir / "model_list.body").write_bytes(model_raw)
        model_receipt = {"status_code": 200, "response_sha256": client.hash_bytes(model_raw),
                         "verified_at_unix": NOW,
                         "baseline_sha256": self.baseline_result.sha256}
        model_art = self.artifact(self.run_dir / "model_list_receipt.json", model_receipt)
        pin = {**self.common, "actual_model": "jev-1.13.0", "requested_alias": "jev-latest",
               "alias_resolution_verified": True, "model_list_receipt_sha256": model_art.sha256}
        self.gates = replace(self.gates, model_pin=self.artifact(self.path / "pin.json", pin))
        self.plan, self.requests = self.make_plan()
        self.packet = self.plan["rows"][0]["batches"][0]
        self.request = self.requests[self.packet["request_sha256"]]
        self.response_data = {"model": "jev-1.13.0", "usage": {"input_tokens": 100, "output_tokens": 20},
                              "answers": {qid: {"type": "score", "score": 3.5, "confidence": 7 / 12,
                                                "legend": {str(i): text for i, text in enumerate(prep.RUBRIC)},
                                                "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.5, "4": 0.5}}
                                          for qid in self.packet["question_ids"]}}
        self.response = client.HTTPResult(200, prep.canonical(self.response_data), "req_test_1")

    def make_plan(self, scope="smoke", model="jev-latest", *, skills=None, oversize_policy=None):
        if skills is None:
            skills = {str(i): {"name": "Skill " + str(i), "description": "Capability",
                               "skill_md": "# Skill\nEntire source text"} for i in range(50)}
        pool = [{"skill_id": str(i), "dense_rank": i + 1, "dense_score": 1 - i / 100} for i in range(20)]
        rows, requests = [], {}
        for qid in prep.select_ids(self.query_ids, {"smoke": 10, "pilot": 200}[scope]):
            batches, audit = prep.build_batches(qid, "Original task " + qid, pool, skills, model, 64000, 2048,
                                                oversize_policy=oversize_policy or "preserve_and_block")
            for batch in batches:
                requests[batch["request_sha256"]] = batch.pop("request")
            rows.append({"query_id": qid, "query": "Original task " + qid, "status": "prepared_not_run",
                         "dense_top20": pool, "batches": batches, "candidate_text_audit": audit})
        manifest = {"sources": {"dense": {"sha256": self.dense.sha256},
                                "skills": {"sha256": client.SOURCE_HASHES["skills"]},
                                "queries": {"sha256": client.SOURCE_HASHES["queries"]}},
                    "rubric_sha256": prep.digest(prep.RUBRIC), "counts": {"queries": 4392, "skills": 6006},
                    "context": {"overhead_tokens": 2048, "reserved_output_tokens": 2048},
                    "scope": scope, "expected_model_version": None, "rows": rows}
        if oversize_policy is not None:
            frozen = prep.oversize_policy_contract(oversize_policy)
            manifest.update(rubric=prep.RUBRIC, question_prefix=prep.QUESTION, model_requested=model)
            manifest["context"].update(
                limit_supplied=64000, pair_limit_supplied=32000, oversize_policy=oversize_policy,
                truncation_policy=frozen, truncation_policy_sha256=prep.digest(frozen),
                estimator_version=prep.CONTEXT_ESTIMATOR_VERSION, native_tokenizer_verified=False)
            manifest["counts"].update(
                selected=len(rows), prepared=len(rows), blocked=0,
                prepared_batches=sum(len(r["batches"]) for r in rows),
                prepared_candidate_pairs=20 * len(rows),
                queries_with_truncation=sum(any(c["truncated"] for c in r["candidate_text_audit"]) for r in rows),
                truncated_candidate_pairs=sum(c["truncated"] for r in rows for c in r["candidate_text_audit"]))
        manifest["plan_sha256"] = prep.digest(manifest)
        return manifest, requests

    def prepare_truncation_fixture(self):
        """Replace the official source pin only inside a fully synthetic test fixture."""
        skills = {str(i): {"skill_id": str(i), "name": "Skill " + str(i),
                           "description": "Capability", "skill_md": "# Skill\nEntire source text"}
                  for i in range(6006)}
        skills["0"]["skill_md"] = "Body with Unicode é and 漢.\n" * 2200
        source = self.path / "synthetic-skills.jsonl"
        source.write_bytes(b"".join(prep.canonical(s) + b"\n" for s in skills.values()))
        source_artifact = client.Artifact(source, prep.file_hash(source))
        replacement = {**client.SOURCE_HASHES, "skills": source_artifact.sha256}
        pin_patch = patch.object(client, "SOURCE_HASHES", replacement)
        pin_patch.start()
        self.addCleanup(pin_patch.stop)
        result = json.loads(json.dumps(self.result))
        result["protocol"]["test_file_sha256"] = replacement
        contract_hash = client.hash_bytes(json.dumps(result["protocol"], sort_keys=True).encode())
        result["cache_contract_sha256"] = contract_hash
        self.change_gate("baseline_result", result)
        self.change_gate("baseline_gate", {**self.gate, "contract_sha256": contract_hash})
        receipt = prep.read_json(self.run_dir / "model_list_receipt.json")
        receipt["baseline_sha256"] = self.gates.baseline_result.sha256
        model_artifact = self.artifact(self.run_dir / "model_list_receipt.json", receipt)
        pin = self.gates.model_pin.read()
        pin["model_list_receipt_sha256"] = model_artifact.sha256
        self.change_gate("model_pin", pin)
        self.gates = replace(self.gates, skills_source=source_artifact)
        self.contract.update(
            approved_oversize_policy="truncate_skill_md_right",
            approved_truncation_policy_sha256=prep.digest(prep.oversize_policy_contract("truncate_skill_md_right")))
        self.change_gate("contract", self.contract)
        self.plan, self.requests = self.make_plan(skills=skills, oversize_policy="truncate_skill_md_right")
        self.packet = self.plan["rows"][0]["batches"][0]
        self.request = self.requests[self.packet["request_sha256"]]
        self.response_data["answers"] = {qid: {"type": "score", "score": 3.5, "confidence": 7 / 12,
                                              "legend": {str(i): text for i, text in enumerate(prep.RUBRIC)},
                                              "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.5, "4": 0.5}}
                                         for qid in self.packet["question_ids"]}
        self.response = client.HTTPResult(200, prep.canonical(self.response_data))

    def rehash_plan(self):
        self.plan["plan_sha256"] = prep.digest({k: v for k, v in self.plan.items() if k != "plan_sha256"})

    def rehash_request(self):
        self.packet["request_sha256"] = prep.digest(self.request)
        self.rehash_plan()

    def transport(self, *args):
        self.calls.append(args)
        # Reservation is durable before the mocked POST sees a request.
        if args[0] == "POST":
            self.assertTrue(any(r["event"] == "reserved" for r in self.subject.ledger.rows()))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    def make_client(self, **kwargs):
        config = dict(root_dir=self.run_dir, gates=self.gates, enabled=True,
                      transport=self.transport, clock=lambda: NOW, sleeper=self.sleep_calls.append)
        config.update(kwargs)
        self.subject = client.HostedJevClient(**config)
        return self.subject

    def send(self):
        return self.subject.request(self.packet, self.request, plan=self.plan)

    def change_gate(self, name, value):
        self.gates = replace(self.gates, **{name: self.artifact(self.path / (name + "-changed.json"), value)})

    def promotional_credit(self, **changes):
        return {**self.common, "basis": "user_reported_promotional_credit_cap",
                "human_attestation": "Synthetic fixture: user reports USD5 signup credit; refill is off.",
                "attested_at_unix": NOW,
                "currency": "USD", "consumption_cap": "5", "available_balance_verified": False,
                "no_topup": True, "automatic_refill_enabled": False, **changes}

    def prepare_recovery_fixture(self):
        """Synthetic reboot and reservation; patch the one real-ID constant only here."""
        patcher = patch.object(client, "REVIEWED_RECOVERY_REQUEST_SHA256", self.packet["request_sha256"])
        patcher.start()
        self.addCleanup(patcher.stop)
        self.make_client()
        gates = self.subject._gates(paid=True)
        cad, usd = self.subject._cost(65536, 2048, gates)
        self.original_id = self.packet["request_sha256"] + "-1"
        self.replacement_id = self.packet["request_sha256"] + "-2"
        self.original_request_path = self.run_dir / "requests" / (self.packet["request_sha256"] + ".json")
        client.write_immutable(self.original_request_path, prep.canonical(self.request))
        with self.subject.ledger.locked():
            self.original_reservation = self.subject.ledger.append({
                "event": "reserved", "request_sha256": self.packet["request_sha256"],
                "attempt_id": self.original_id, "plan_sha256": self.plan["plan_sha256"],
                "utc_unix": NOW - 120, "reserve_cad": str(cad), "reserve_usd": str(usd),
                "input_token_reserve_allowance": 65536, "output_token_bound": 2048,
                "actual_model_pin": "jev-1.13.0", "requested_model": self.request["model"]})
        self.reboot_proof = {"kind": "wsl_reboot_observation", "observed_at_unix": NOW - 30,
                             "uptime_seconds": 17, "boot_id": "synthetic-reboot-id",
                             "matching_live_process_count": 0}
        proof = self.artifact(self.run_dir / "reboot-proof.json", self.reboot_proof)
        self.recovery_receipt = {**self.common, "schema_version": client.RECOVERY_POLICY_VERSION,
                                 "experiment_id": client.EXPERIMENT, "reviewed_by": "synthetic offline reviewer",
                                 "request_sha256": self.packet["request_sha256"],
                                 "original_attempt_id": self.original_id, "replacement_attempt_id": self.replacement_id,
                                 "actual_model_version": "jev-1.13.0", "plan_sha256": self.plan["plan_sha256"],
                                 "original_reservation_event_sha256": self.original_reservation["event_sha256"],
                                 "retain_original_reservation": True, "additional_attempts": 1,
                                 "no_automatic_retry": True,
                                 "reboot_proof": {"path": proof.path.name, "sha256": proof.sha256}}
        self.change_gate("recovery", self.recovery_receipt)
        self.make_client()

    def test_default_disabled_never_reads_environment_or_network(self):
        subject = client.HostedJevClient(self.run_dir, self.gates, transport=self.transport)
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "disabled"):
                subject.model_list()
            with self.assertRaisesRegex(client.GateError, "disabled"):
                subject.request(self.packet, self.request, plan=self.plan)
        self.assertEqual(self.calls, [])

    def test_recovery_receipt_is_read_only_and_absence_keeps_unknown_blocked(self):
        self.prepare_recovery_fixture()
        before = self.subject.ledger.path.read_bytes()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            self.assertEqual(self.subject.reviewed_unresolved_attempt_ids(), {self.original_id})
        self.assertEqual(self.subject.ledger.path.read_bytes(), before)
        self.make_client(gates=replace(self.gates, recovery=None))
        with patch.object(client.os, "environ", DeniedEnvironment()):
            self.assertEqual(self.subject.reviewed_unresolved_attempt_ids(), set())
            with self.assertRaisesRegex(client.GateError, "unsettled"):
                self.send()
        self.assertFalse(self.calls)

    def test_forged_stale_or_mismatched_recovery_receipts_fail_before_key(self):
        self.prepare_recovery_fixture()
        original = self.gates.recovery
        variants = [{"verified_at_unix": NOW - 86401}, {"original_attempt_id": "a" * 64 + "-1"},
                    {"replacement_attempt_id": self.packet["request_sha256"] + "-3"},
                    {"original_reservation_event_sha256": "b" * 64}, {"actual_model_version": "jev-9.9.9"},
                    {"plan_sha256": "c" * 64}, {"retain_original_reservation": False}, {"additional_attempts": 2}]
        for changes in variants:
            with self.subTest(fields=list(changes)):
                self.change_gate("recovery", {**self.recovery_receipt, **changes})
                self.make_client()
                with patch.object(client.os, "environ", DeniedEnvironment()):
                    with self.assertRaises(client.GateError):
                        self.send()
        self.make_client(gates=replace(self.gates, recovery=client.Artifact(original.path, "f" * 64)))
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "SHA256"):
                self.send()
        self.assertFalse(self.calls)

    def test_recovery_rejects_changed_or_nonreboot_process_proof(self):
        self.prepare_recovery_fixture()
        proof_path = self.run_dir / "reboot-proof.json"
        proof_path.write_bytes(proof_path.read_bytes() + b' ')
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "SHA256"):
                self.send()
        for fields in ({"uptime_seconds": 1000}, {"matching_live_process_count": 1}, {"boot_id": ""}):
            with self.subTest(fields=list(fields)):
                artifact = self.artifact(proof_path, {**self.reboot_proof, **fields})
                self.change_gate("recovery", {**self.recovery_receipt,
                                              "reboot_proof": {"path": proof_path.name, "sha256": artifact.sha256}})
                self.make_client()
                with patch.object(client.os, "environ", DeniedEnvironment()):
                    with self.assertRaises(client.GateError):
                        self.send()
        self.assertFalse(self.calls)

    def test_recovery_requires_original_bytes_and_absent_original_response(self):
        self.prepare_recovery_fixture()
        original_bytes = self.original_request_path.read_bytes()
        for replacement in (None, original_bytes + b' '):
            if replacement is None:
                self.original_request_path.unlink()
            else:
                self.original_request_path.write_bytes(replacement)
            with patch.object(client.os, "environ", DeniedEnvironment()):
                with self.assertRaisesRegex(client.GateError, "exact original request"):
                    self.send()
        self.original_request_path.write_bytes(original_bytes)
        for suffix in (".body", ".json"):
            target = self.run_dir / "attempts" / (self.original_id + suffix)
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(b'{}')
            with patch.object(client.os, "environ", DeniedEnvironment()):
                with self.assertRaisesRegex(client.GateError, "response evidence exists"):
                    self.send()
            target.unlink()
        self.assertFalse(self.calls)

    def test_reviewed_attempt_two_settles_once_and_original_reserve_remains_held(self):
        self.prepare_recovery_fixture()
        self.assertEqual(self.contract["max_attempts"], 1)
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            result = self.send()
        self.assertEqual(result["attempt_id"], self.replacement_id)
        self.assertEqual(result["reviewed_recovery_of_attempt_id"], self.original_id)
        self.assertEqual(result["recovery_receipt_sha256"], self.gates.recovery.sha256)
        rows = self.subject.ledger.rows()
        self.assertEqual([r["event"] for r in rows if r.get("attempt_id") == self.original_id], ["reserved"])
        replacement = next(r for r in rows if r["event"] == "reserved" and r["attempt_id"] == self.replacement_id)
        self.assertEqual(replacement["recovery_receipt_sha256"], self.gates.recovery.sha256)
        self.assertEqual(self.subject.ledger.commitments()[1],
                         Decimal(self.original_reservation["reserve_usd"]) + Decimal(".0000042"))
        with patch.object(client.os, "environ", DeniedEnvironment()):
            self.assertEqual(self.send(), result)
        self.assertEqual(len(self.calls), 1)
        self.packet = self.plan["rows"][1]["batches"][0]
        self.request = self.requests[self.packet["request_sha256"]]
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            next_result = self.send()
        self.assertTrue(next_result["attempt_id"].endswith('-1'))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.subject.reviewed_unresolved_attempt_ids(), {self.original_id})

    def test_recovery_must_finish_before_new_distinct_request(self):
        self.prepare_recovery_fixture()
        self.packet = self.plan["rows"][1]["batches"][0]
        self.request = self.requests[self.packet["request_sha256"]]
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "replacement must settle"):
                self.send()
        self.assertFalse(self.calls)

    def test_second_ambiguity_blocks_every_further_paid_attempt(self):
        self.prepare_recovery_fixture()
        self.response = TimeoutError("synthetic second ambiguity")
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            with self.assertRaises(client.AmbiguousRequest):
                self.send()
        self.assertEqual(self.subject.ledger.commitments()[1], 2 * Decimal(self.original_reservation["reserve_usd"]))
        for another in (False, True):
            if another:
                self.packet = self.plan["rows"][1]["batches"][0]
                self.request = self.requests[self.packet["request_sha256"]]
            with patch.object(client.os, "environ", DeniedEnvironment()):
                with self.assertRaisesRegex(client.GateError, "unsettled"):
                    self.send()
        self.assertEqual(len(self.calls), 1)

    def test_recovery_budget_check_counts_original_and_replacement_reserves(self):
        self.prepare_recovery_fixture()
        self.change_gate("approval", {**self.approval, "total_cap": str(Decimal(self.original_reservation["reserve_cad"]) * Decimal("1.5"))})
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "budget"):
                self.send()
        self.assertEqual(len(self.subject.ledger.rows()), 1)
        self.assertFalse(self.calls)
        self.change_gate("approval", self.approval)
        self.change_gate("credit", self.promotional_credit(
            consumption_cap=str(Decimal(self.original_reservation["reserve_usd"]) * Decimal("1.5"))))
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "USD experiment consumption cap"):
                self.send()
        self.assertEqual(len(self.subject.ledger.rows()), 1)
        self.assertFalse(self.calls)

    def test_baseline_absent_or_unvalidated_never_reads_environment(self):
        self.make_client(gates=None)
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaises(client.GateError):
                self.send()
        self.change_gate("baseline_gate", {**self.gate, "status": "investigation_required"})
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "not validated"):
                self.send()

    def test_full_counts_and_hash_gate_precede_credentials(self):
        self.change_gate("baseline_gate", {**self.gate, "full_query_count": 10})
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "full counts"):
                self.send()
        self.make_client(gates=replace(self.gates, baseline_gate=client.Artifact(self.baseline_gate.path, "a" * 64)))
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "SHA256"):
                self.send()

    def test_wrong_effective_rope_blocks_even_if_claimed_metrics_match(self):
        result = json.loads(json.dumps(self.result))
        result["protocol"]["effective_rope_theta"] = 10000
        changed_hash = client.hash_bytes(json.dumps(result["protocol"], sort_keys=True).encode())
        result["cache_contract_sha256"] = changed_hash
        self.change_gate("baseline_result", result)
        self.change_gate("baseline_gate", {**self.gate, "contract_sha256": changed_hash})
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "RoPE theta"):
                self.send()
    def test_approval_price_fx_credit_contract_and_pin_are_required(self):
        changes = [("approval", {**self.approval, "approved": False}),
                   ("pricing", {**self.price, "verified": False}),
                   ("fx", {**self.fx, "buffer_fraction": "0"}),
                   ("credit", {**self.credit, "verified_at_unix": NOW - 4000}),
                   ("contract", {**self.contract, "token_estimator": "unverified"})]
        original = self.gates
        for name, value in changes:
            with self.subTest(gate=name):
                self.gates = original
                self.change_gate(name, value)
                self.make_client()
                with patch.object(client.os, "environ", DeniedEnvironment()):
                    with self.assertRaises(client.GateError):
                        self.send()
        self.make_client(gates=replace(original, model_pin=None))
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "pin absent"):
                self.send()
        self.assertFalse(self.calls)

    def test_success_saves_raw_before_parse_and_resume_avoids_paid_duplicate(self):
        self.make_client()
        # prepare writes canonical sorted JSON; question keys are then lexicographic.
        self.request = prep.loads(prep.canonical(self.request).decode("utf-8"))
        original = prep.parse_response
        def assert_saved(*args, **kwargs):
            self.assertTrue(list((self.run_dir / "attempts").glob("*.body")))
            return original(*args, **kwargs)
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}), \
             patch.object(prep, "parse_response", side_effect=assert_saved):
            first = self.send()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            second = self.send()
        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(first["actual_model_version"], "jev-1.13.0")
        self.assertEqual(first["input_tokens"], 100)
        rows = self.subject.ledger.rows()
        self.assertEqual([r["event"] for r in rows], ["reserved", "response_saved", "settled"])
        self.assertEqual(rows[0]["input_token_reserve_allowance"], 65536)
        self.assertGreater(rows[0]["input_token_reserve_allowance"], rows[0]["estimated_input_tokens_for_packing"])
        self.assertEqual(self.subject.ledger.commitments()[1], Decimal(".0000042"))
        for path in self.run_dir.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"FAKE_OFFLINE_KEY", path.read_bytes())

    def test_timeout_retains_reservation_and_never_retries_even_on_resume(self):
        self.make_client()
        self.response = TimeoutError("may have reached provider; secret text must not be persisted")
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            with self.assertRaises(client.AmbiguousRequest):
                self.send()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "resend blocked"):
                self.send()
        self.assertEqual(len(self.calls), 1)
        self.assertGreater(self.subject.ledger.commitments()[0], 0)
        self.assertNotIn("may have reached", self.subject.ledger.path.read_text())

    def test_ambiguous_outcome_blocks_other_queries_as_well_as_duplicate(self):
        self.make_client()
        self.response = TimeoutError("synthetic")
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            with self.assertRaises(client.AmbiguousRequest):
                self.send()
        self.packet = self.plan["rows"][1]["batches"][0]
        self.request = self.requests[self.packet["request_sha256"]]
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "Experiment has an unsettled"):
                self.send()
        self.assertEqual(len(self.calls), 1)

    def test_429_529_and_500_default_are_ambiguous_not_retried(self):
        for status in (429, 529, 500):
            with self.subTest(status=status):
                self.make_client(root_dir=self.path / ("status" + str(status)))
                # Model evidence remains identical in each separate synthetic run.
                self.subject.root.mkdir()
                for name in ("model_list.body", "model_list_receipt.json"):
                    (self.subject.root / name).write_bytes((self.run_dir / name).read_bytes())
                self.response = client.HTTPResult(status, b'{"error":"synthetic"}')
                with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
                    with self.assertRaises(client.AmbiguousRequest):
                        self.send()
                self.assertGreater(self.subject.ledger.commitments()[0], 0)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.sleep_calls, [])

    def test_invalid_usage_or_changed_model_saved_and_blocked(self):
        self.make_client()
        changed = {**self.response_data, "model": "jev-1.14.0"}
        self.response = client.HTTPResult(200, prep.canonical(changed))
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            with self.assertRaises(client.AmbiguousRequest):
                self.send()
        self.assertEqual(list((self.run_dir / "attempts").glob("*.body"))[0].read_bytes(), self.response.body)
        self.assertFalse(list((self.run_dir / "raw_responses").glob("*.json")))
        self.assertGreater(self.subject.ledger.commitments()[0], 0)

    def test_request_manifest_and_context_mutations_precede_credentials(self):
        self.make_client()
        self.request["state"]["task"] = "tampered"
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "request hash"):
                self.send()

    def test_rehashed_malformed_candidate_mapping_precedes_credentials(self):
        self.make_client()
        ids = self.packet["candidate_ids"]
        ids[0], ids[1] = ids[1], ids[0]
        self.plan.pop("plan_sha256")
        self.plan["plan_sha256"] = prep.digest(self.plan)
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "exact ID mapping"):
                self.send()
        self.assertFalse(self.calls)

    def test_alias_smoke_then_stable_pilot_reuses_original_paid_response(self):
        self.make_client()
        original_hash = self.packet["request_sha256"]
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            first = self.send()
        committed = self.subject.ledger.commitments()
        self.plan, self.requests = self.make_plan(scope="pilot", model="jev-1.13.0")
        self.packet = self.plan["rows"][0]["batches"][0]
        self.request = self.requests[self.packet["request_sha256"]]
        expansion = {**self.common, "plan_sha256": self.plan["plan_sha256"],
                     "based_on_actual_smoke_usage": True, "completed_smoke_queries": 10,
                     "projected_total_all_in_cad": "5", "projected_total_runtime_seconds": 600}
        self.subject.gates = replace(self.gates, expansion=self.artifact(self.path / "synthetic-expansion.json", expansion))
        with patch.object(client.os, "environ", DeniedEnvironment()):
            reused = self.send()
            replay = self.send()
        self.assertEqual(reused, replay)
        self.assertNotEqual(reused["request_sha256"], original_hash)
        self.assertEqual(reused["original_transport_request_sha256"], original_hash)
        self.assertEqual(reused["raw_response"], first["raw_response"])
        self.assertEqual(reused["response_sha256"], first["response_sha256"])
        self.assertEqual(reused["cache_reuse"]["additional_api_calls"], 0)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.subject.ledger.commitments(), committed)
        self.assertEqual(len(list((self.run_dir / "attempts").glob("*.body"))), 1)
        self.assertEqual(prep.file_hash(self.run_dir / "requests" / (original_hash + ".json")), original_hash)
        prep.parse_cache(self.run_dir / "raw_responses" / (self.packet["request_sha256"] + ".json"),
                         self.packet, "jev-1.13.0", 64000)

    def test_usage_above_packing_estimate_can_settle_within_full_allowance(self):
        self.make_client()
        actual = self.packet["estimated_input_token_upper_bound"] + 1
        self.response_data["usage"]["input_tokens"] = actual
        self.response = client.HTTPResult(200, prep.canonical(self.response_data))
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            result = self.send()
        self.assertEqual(result["input_tokens"], actual)

    def test_state_plus_one_question_context_rejected_before_key(self):
        self.contract["state_plus_question_tokens"] = 100
        self.change_gate("contract", self.contract)
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "State-plus-question context"):
                self.send()

    def test_input_usage_null_is_unknown_cost_and_retains_reserve(self):
        self.make_client()
        self.response_data["usage"]["input_tokens"] = None
        self.response = client.HTTPResult(200, prep.canonical(self.response_data))
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            with self.assertRaises(client.AmbiguousRequest):
                self.send()
        self.assertGreater(self.subject.ledger.commitments()[0], 0)
        self.assertTrue(list((self.run_dir / "attempts").glob("*.body")))

    def test_budget_and_credit_checked_before_key(self):
        self.change_gate("approval", {**self.approval, "total_cap": ".00000001"})
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "budget"):
                self.send()
        self.assertEqual(self.subject.ledger.rows(), [])

    def test_promotional_cap_and_explicit_planning_fees_accept_mock_without_balance_claim(self):
        self.change_gate("credit", self.promotional_credit())
        self.change_gate("fx", {**self.fx, "fees_verified": False,
                                "fee_basis": "planning_allowance", "source_verified": True})
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            gates = self.subject._gates(paid=True)
        self.assertEqual(gates["usd_cap"], Decimal("5"))
        self.assertNotIn("available", gates)
        self.assertEqual(gates["multiplier"], Decimal("1.03") * Decimal("1.10"))
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            self.send()
        reservation = self.subject.ledger.rows()[0]
        self.assertEqual(reservation["credit_basis"], "user_reported_promotional_credit_cap")
        self.assertEqual(reservation["usd_consumption_cap"], "5")
        self.assertEqual(reservation["credit_receipt_sha256"], self.gates.credit.sha256)
        self.assertEqual(len(self.calls), 1)

    def test_promotional_cap_unknown_or_enabled_refill_blocks_before_key(self):
        for value in (None, True, "false", 0, "missing"):
            with self.subTest(value=value):
                receipt = self.promotional_credit(automatic_refill_enabled=value)
                if value == "missing":
                    receipt.pop("automatic_refill_enabled")
                self.change_gate("credit", receipt)
                self.make_client()
                with patch.object(client.os, "environ", DeniedEnvironment()):
                    with self.assertRaisesRegex(client.GateError, "automatic refill"):
                        self.send()
                self.assertEqual(self.subject.ledger.rows(), [])
        self.assertFalse(self.calls)

    def test_promotional_receipt_cannot_claim_balance_or_exceed_attested_cap(self):
        invalid = [{"available": "5"}, {"available_balance": "5"}, {"balance": "5"},
                   {"available_balance_verified": True}, {"human_attestation": " "},
                   {"consumption_cap": "5.01"}, {"consumption_cap": "0"},
                   {"currency": "CAD"}, {"no_topup": False}]
        for fields in invalid:
            with self.subTest(fields=list(fields)):
                self.change_gate("credit", self.promotional_credit(**fields))
                self.make_client()
                with patch.object(client.os, "environ", DeniedEnvironment()):
                    with self.assertRaises(client.GateError):
                        self.send()
        self.assertFalse(self.calls)

    def test_unverified_fees_need_labeled_nonnegative_allowance_and_verified_source(self):
        base = {**self.fx, "fees_verified": False, "fee_basis": "planning_allowance", "source_verified": True}
        for fields in ({"source_verified": False}, {"source_verified": None}, {"fee_basis": "verified_fees"},
                       {"fee_fraction": "-0.01"}, {"buffer_fraction": "0.09"}, {"fees_verified": None}):
            with self.subTest(fields=list(fields)):
                self.change_gate("fx", {**base, **fields})
                self.make_client()
                with patch.object(client.os, "environ", DeniedEnvironment()):
                    with self.assertRaises(client.GateError):
                        self.send()
        self.assertFalse(self.calls)

    def test_promotional_cap_and_prior_spend_persist_across_client_and_receipt_changes(self):
        # First full-context reserve fits; a second reserve plus first actual spend does not.
        self.change_gate("credit", self.promotional_credit(consumption_cap="0.002754"))
        self.make_client()
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            self.send()
        self.assertEqual(self.subject.ledger.commitments()[1], Decimal("0.0000042"))
        self.packet = self.plan["rows"][1]["batches"][0]
        self.request = self.requests[self.packet["request_sha256"]]
        self.change_gate("credit", self.promotional_credit(consumption_cap="5"))
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "USD experiment consumption cap"):
                self.send()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(sum(r["event"] == "reserved" for r in self.subject.ledger.rows()), 1)

    def test_legacy_verified_balance_receipt_remains_compatible(self):
        self.assertNotIn("automatic_refill_enabled", self.credit)
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            gates = self.subject._gates(paid=True)
        self.assertEqual(gates["credit_basis"], "verified_read_only_balance")
        self.assertEqual(gates["usd_cap"], Decimal("5"))

    def test_promotional_attestation_does_not_expire_as_live_balance_or_self_refresh(self):
        self.change_gate("credit", self.promotional_credit(attested_at_unix=NOW - 864000,
                                                           verified_at_unix=NOW - 864000))
        before = self.gates.credit.path.read_bytes()
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            gates = self.subject._gates(paid=True)
        self.assertEqual(gates["usd_cap"], Decimal("5"))
        self.assertEqual(self.gates.credit.path.read_bytes(), before)

    def test_only_named_environment_key_is_read_no_fallback(self):
        self.make_client()
        seen = []
        class Capture(dict):
            def __getitem__(self, key):
                seen.append(key)
                raise KeyError(key)
        with patch.object(client.os, "environ", Capture()):
            with self.assertRaisesRegex(client.GateError, "TYPESAFE_API_KEY missing"):
                self.send()
        self.assertEqual(seen, ["TYPESAFE_API_KEY"])
        self.assertFalse(self.calls)
        self.assertEqual(self.subject.ledger.rows(), [])

    def test_null_output_allowed_only_when_free_and_recorded_unknown(self):
        self.make_client()
        self.response_data["usage"]["output_tokens"] = None
        self.response = client.HTTPResult(200, prep.canonical(self.response_data))
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            result = self.send()
        self.assertIsNone(result["output_tokens"])

    def test_incomplete_score_distribution_is_saved_but_never_settled_or_retried(self):
        self.make_client()
        self.response_data["answers"][self.packet["question_ids"][0]].pop("probabilities")
        self.response = client.HTTPResult(200, prep.canonical(self.response_data))
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            with self.assertRaises(client.AmbiguousRequest):
                self.send()
        self.assertEqual(len(self.calls), 1)
        bodies = list((self.run_dir / "attempts").glob("*.body"))
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0].read_bytes(), self.response.body)
        self.assertGreater(self.subject.ledger.commitments()[0], 0)
        self.assertNotIn("settled", [row["event"] for row in self.subject.ledger.rows()])
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "unsettled|resend blocked"):
                self.send()
        self.assertEqual(len(self.calls), 1)

    def test_rounded_score_response_keeps_native_values_and_cached_raw_evidence(self):
        self.make_client()
        for answer in self.response_data["answers"].values():
            answer.update(score=2.32, confidence=.73,
                          probabilities={"0": 0, "1": 0, "2": .76, "3": .15, "4": .09})
        self.response = client.HTTPResult(200, prep.canonical(self.response_data))
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            result = self.send()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            self.assertEqual(self.send(), result)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["raw_response"].encode(), self.response.body)
        scored, usage = prep.parse_response(prep.loads(result["raw_response"]), self.packet, "jev-1.13.0", 64000)
        self.assertEqual(scored[0]["native_score"], 2.32)
        self.assertEqual(scored[0]["confidence"], .73)
        self.assertEqual(scored[0]["response_validation"]["mode"], "cent")
        self.assertEqual(self.subject.ledger.commitments()[1], Decimal(".0000042"))

    def test_native_arithmetic_inconsistency_settles_once_and_reuses_unchanged_raw_cache(self):
        self.make_client()
        for answer in self.response_data["answers"].values():
            answer.update(score=1.87, confidence=.84,
                          probabilities={"0": 0, "1": .15, "2": .83, "3": .02, "4": 0})
        self.response = client.HTTPResult(200, prep.canonical(self.response_data))
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            result = self.send()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            self.assertEqual(self.send(), result)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["raw_response"].encode(), self.response.body)
        scored, usage = prep.parse_cache(self.run_dir / "raw_responses" / (self.packet["request_sha256"] + ".json"),
                                        self.packet, "jev-1.13.0", 64000)
        self.assertTrue(all(row["native_score"] == 1.87 and row["confidence"] == .84 for row in scored))
        self.assertTrue(all(row["response_validation"]["policy"] == prep.SCORE_RESPONSE_POLICY
                            and row["response_validation"]["mode"] == "inconsistent" for row in scored))
        self.assertTrue(all(not row["response_validation"]["checks"]["joint_nearest_cent_feasible"] for row in scored))
        events = self.subject.ledger.rows()
        self.assertEqual(sum(row["event"] == "settled" for row in events), 1)
        self.assertNotIn("ambiguous_blocked", [row["event"] for row in events])
        self.assertEqual(self.subject.ledger.commitments()[1], Decimal(".0000042"))

    def test_exclusive_lock_and_partial_ledger_fail_closed(self):
        self.make_client()
        with self.subject.ledger.locked():
            with self.assertRaisesRegex(client.GateError, "locked"):
                with self.subject.ledger.locked():
                    self.fail("should not acquire")
        self.subject.ledger.path.write_bytes(b'{"incomplete":')
        with self.assertRaisesRegex(client.GateError, "incomplete tail"):
            self.subject.ledger.rows()

    def test_model_list_after_baseline_approval_without_pricing_or_credit(self):
        absent = client.Artifact(self.path / "must-not-read.json", "a" * 64)
        self.make_client(gates=replace(self.gates, model_pin=None, pricing=absent, fx=absent, credit=absent, contract=absent),
                         root_dir=self.path / "fresh-model-list")
        self.response = client.HTTPResult(200, b'{"models":[{"name":"jev-latest"}]}', "list_123")
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            receipt = self.subject.model_list()
        self.assertEqual(self.calls[0][:3], ("GET", client.MODELS_ENDPOINT, None))
        self.assertEqual(receipt["response_sha256"], client.hash_bytes(self.response.body))
        self.assertEqual(self.subject.ledger.rows(), [])

    def test_documented_no_execution_retry_is_bounded_and_retained_in_journal(self):
        self.change_gate("contract", {**self.contract, "max_attempts": 2,
                                      "documented_no_execution_statuses": {"429": "https://example.test/TEST-ONLY-no-execution-guarantee"}})
        self.make_client()
        good = self.response
        def two_attempts(*args):
            self.calls.append(args)
            return client.HTTPResult(429, b'{"error":"synthetic_not_executed"}') if len(self.calls) == 1 else good
        self.subject.transport = two_attempts
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            self.send()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.sleep_calls, [1])
        self.assertEqual(len(list((self.run_dir / "attempts").glob("*.body"))), 2)
        self.assertEqual(self.subject.ledger.commitments()[1], Decimal(".0000042"))

    def test_truncation_recomputed_from_pinned_source_accepts_mock_and_caches_source(self):
        self.prepare_truncation_fixture()
        self.make_client()
        audit = self.plan["rows"][0]["candidate_text_audit"][0]
        self.assertTrue(audit["truncated"])
        self.assertGreater(audit["original_characters"], audit["prepared_characters"])
        with patch.object(client.os, "environ", {"TYPESAFE_API_KEY": "FAKE_OFFLINE_KEY"}):
            first = self.send()
        source = self.gates.skills_source.path
        original_read = Path.read_bytes
        def do_not_reread_skills(path):
            self.assertNotEqual(path, source, "Unchanged source should remain cached")
            return original_read(path)
        with patch.object(client.os, "environ", DeniedEnvironment()), \
             patch.object(Path, "read_bytes", do_not_reread_skills):
            second = self.send()
        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 1)

    def test_truncation_missing_contract_policy_or_hash_blocks_before_key(self):
        self.prepare_truncation_fixture()
        for field in ("approved_oversize_policy", "approved_truncation_policy_sha256"):
            with self.subTest(field=field):
                self.change_gate("contract", {k: v for k, v in self.contract.items() if k != field})
                self.make_client()
                with patch.object(client.os, "environ", DeniedEnvironment()):
                    with self.assertRaisesRegex(client.GateError, "approved truncation policy"):
                        self.send()
        self.assertFalse(self.calls)

    def test_truncation_missing_source_or_wrong_pin_blocks_before_key(self):
        self.prepare_truncation_fixture()
        for source in (None, client.Artifact(self.gates.skills_source.path, "a" * 64)):
            with self.subTest(source_absent=source is None):
                self.make_client(gates=replace(self.gates, skills_source=source))
                with patch.object(client.os, "environ", DeniedEnvironment()):
                    with self.assertRaisesRegex(client.GateError, "skills.source|skills source"):
                        self.send()
        self.assertFalse(self.calls)

    def test_truncation_source_change_after_cache_fails_hash_before_key(self):
        self.prepare_truncation_fixture()
        self.make_client()
        gates = self.subject._gates(paid=True)
        self.subject._validate_packet(self.packet, self.request, self.plan, gates)
        with self.gates.skills_source.path.open("ab") as stream:
            stream.write(b"\n")
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "skills source SHA256"):
                self.send()
        self.assertFalse(self.calls)

    def test_rehashed_truncation_audit_alteration_blocks_before_key(self):
        self.prepare_truncation_fixture()
        self.plan["rows"][0]["candidate_text_audit"][0]["removed_skill_md_characters"] += 1
        self.rehash_plan()
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "audit differs from recomputed"):
                self.send()
        self.assertFalse(self.calls)

    def test_rehashed_wrong_truncation_prefix_and_audit_cannot_replace_original(self):
        self.prepare_truncation_fixture()
        question = self.request["questions"]["s0"]["instructions"]
        original = question["question"]
        # Equal-length modified body keeps all packing estimates valid.
        offset = len(prep.QUESTION) + len("Skill 0 | Capability | ")
        question["question"] = original[:offset] + "X" + original[offset + 1:]
        changed_text = question["question"][len(prep.QUESTION):]
        audit = self.plan["rows"][0]["candidate_text_audit"][0]
        changed_hash = client.hash_bytes(changed_text.encode("utf-8"))
        audit["document_sha256"] = audit["prepared_document_sha256"] = changed_hash
        self.rehash_request()
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "audit differs from recomputed"):
                self.send()
        self.assertFalse(self.calls)

    def test_rehashed_truncation_rubric_policy_counts_and_budget_changes_fail_before_key(self):
        self.prepare_truncation_fixture()
        original = prep.canonical(self.plan)
        for change, label in (
                (lambda p: p.__setitem__("question_prefix", "Changed prefix"), "fixed rubric or question prefix"),
                (lambda p: p["context"]["truncation_policy"].__setitem__("algorithm", "Keep last suffix"), "frozen algorithm"),
                (lambda p: p["counts"].__setitem__("truncated_candidate_pairs", 0), "aggregate counts"),
                (lambda p: p["context"].__setitem__("pair_limit_supplied", 32001), "context budget")):
            with self.subTest(label=label):
                self.plan = prep.loads(original.decode())
                self.packet = self.plan["rows"][0]["batches"][0]
                change(self.plan)
                self.rehash_plan()
                self.make_client()
                with patch.object(client.os, "environ", DeniedEnvironment()):
                    with self.assertRaisesRegex(client.GateError, label):
                        self.send()
        self.assertFalse(self.calls)

    def test_truncation_cannot_be_relabelled_as_legacy_untruncated_plan(self):
        self.prepare_truncation_fixture()
        self.plan["context"].pop("oversize_policy")
        self.rehash_plan()
        self.make_client()
        with patch.object(client.os, "environ", DeniedEnvironment()):
            with self.assertRaisesRegex(client.GateError, "explicit prepared policy"):
                self.send()

    def test_suspected_credential_text_blocks_before_key_without_logging_value(self):
        examples = ["ghp_" + "ABC012" * 6, "github_pat_" + "DEF345" * 6,
                    "-----BEGIN PRIVATE KEY-----", "-----BEGIN RSA PRIVATE KEY-----"]
        for token in examples:
            with self.subTest(kind=token[:4]):
                self.plan, self.requests = self.make_plan()
                self.packet = self.plan["rows"][0]["batches"][0]
                self.request = self.requests[self.packet["request_sha256"]]
                self.request["state"]["task"] = self.plan["rows"][0]["query"] = "Fixture " + token
                self.rehash_request()
                self.make_client()
                with patch.object(client.os, "environ", DeniedEnvironment()):
                    with self.assertRaisesRegex(client.GateError, "Private credential-shaped text audit") as caught:
                        self.send()
                self.assertNotIn(token, str(caught.exception))
                self.assertIn(client.hash_bytes(token.encode()), str(caught.exception))
                self.assertIn("request.state.task", str(caught.exception))
        self.assertFalse(self.calls)
        self.assertEqual(self.subject.ledger.rows(), [])

    def test_all_x_documentation_placeholders_are_allowed_without_transformation(self):
        request = {"state": {"task": "Example ghp_" + "x" * 40},
                   "questions": {"s0": {"instructions": {"question": "github_pat_" + "X_" * 20}}}}
        before = prep.canonical(request)
        with patch.object(client.os, "environ", DeniedEnvironment()):
            client.HostedJevClient._check_request_text(request)
        self.assertEqual(prep.canonical(request), before)


if __name__ == "__main__":
    unittest.main()
