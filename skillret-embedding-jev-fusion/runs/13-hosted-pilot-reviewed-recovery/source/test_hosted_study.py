"""Runner tests use synthetic data and injected transport; no real API/key access."""
import contextlib
from dataclasses import replace
from decimal import Decimal
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import hosted_jev_client as hosted
import run_hosted_study as runner
import run_jev_rerank as prep
import test_hosted_jev_client as fixtures
from test_hosted_jev_client import DeniedEnvironment, NOW


class StudyRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.HostedClientTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.HostedClientTests.tearDownClass()

    def setUp(self):
        self.fixture = fixtures.HostedClientTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.plan = f.plan
        self.plan.update(schema_version=prep.SCHEMA_VERSION, rubric=prep.RUBRIC, question_prefix=prep.QUESTION)
        self.plan["context"]["limit_supplied"] = 64000
        self.plan_path = f.path / "plan" / "manifest.json"
        self.plan_path.parent.mkdir()
        for row in self.plan["rows"]:
            for packet in row["batches"]:
                path = self.plan_path.parent / packet["request_file"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(prep.canonical(f.requests[packet["request_sha256"]]))
        self.save_plan()
        self.calls = []
        self.now = NOW
        self.client = hosted.HostedJevClient(f.run_dir, gates=f.gates, enabled=True,
                                             transport=self.transport, clock=lambda: self.now)
        environment = patch.object(hosted.os, "environ", {"TYPESAFE_API_KEY": "synthetic-offline-test"})
        environment.start()
        self.addCleanup(environment.stop)

    def save_plan(self):
        self.plan["plan_sha256"] = prep.digest({k: v for k, v in self.plan.items() if k != "plan_sha256"})
        self.plan_path.write_bytes(prep.canonical(self.plan))

    def transport(self, method, url, body, key, timeout):
        self.calls.append((method, url))
        request = prep.loads(body.decode())
        return hosted.HTTPResult(200, prep.canonical({
            "model": "jev-1.13.0", "usage": {"input_tokens": 100, "output_tokens": 20},
            "answers": {qid: {"type": "score", "score": 3.5, "confidence": 7 / 12,
                              "legend": {str(i): text for i, text in enumerate(prep.RUBRIC)},
                              "probabilities": {"0": 0., "1": 0., "2": 0., "3": .5, "4": .5}}
                        for qid in request["questions"]}}))

    def run_study(self, **kwargs):
        return runner.run_study(self.plan_path, self.client, execute=True, clock=lambda: self.now, **kwargs)

    def reviewed_loss_fixture(self):
        """Synthetic reboot proof for exactly one lost first batch; no live state."""
        packet = self.plan["rows"][0]["batches"][0]
        request_hash = packet["request_sha256"]
        pinned = patch.object(hosted, "REVIEWED_RECOVERY_REQUEST_SHA256", request_hash)
        pinned.start()
        self.addCleanup(pinned.stop)
        original_id = request_hash + "-1"
        with self.client.ledger.locked():
            original = self.client.ledger.append({
                "event": "reserved", "request_sha256": request_hash, "attempt_id": original_id,
                "plan_sha256": self.plan["plan_sha256"], "actual_model_pin": "jev-1.13.0",
                "utc_unix": NOW - 100, "reserve_usd": ".00275", "reserve_cad": ".004"})
        request_path = self.client.root / "requests" / (request_hash + ".json")
        hosted.write_immutable(request_path, prep.canonical(self.fixture.requests[request_hash]))
        proof_dir = self.client.root / "recovery"
        proof_dir.mkdir()
        proof = self.fixture.artifact(proof_dir / "proof.json", {
            "kind": "wsl_reboot_observation", "observed_at_unix": NOW - 10,
            "uptime_seconds": 20, "boot_id": "synthetic-new-boot", "matching_live_process_count": 0})
        receipt = self.fixture.artifact(proof_dir / "review.json", {
            "schema_version": hosted.RECOVERY_POLICY_VERSION, "experiment_id": hosted.EXPERIMENT,
            "verified": True, "verified_at_unix": NOW, "source": "offline synthetic recovery fixture",
            "reviewed_by": "offline unittest", "request_sha256": request_hash,
            "original_attempt_id": original_id, "replacement_attempt_id": request_hash + "-2",
            "actual_model_version": "jev-1.13.0", "plan_sha256": self.plan["plan_sha256"],
            "original_reservation_event_sha256": original["event_sha256"],
            "retain_original_reservation": True, "additional_attempts": 1, "no_automatic_retry": True,
            "reboot_proof": {"path": "recovery/proof.json", "sha256": proof.sha256}})
        self.client.gates = replace(self.client.gates, recovery=receipt)
        return original_id

    def test_default_status_never_reads_environment_or_transport(self):
        with patch.object(hosted.os, "environ", DeniedEnvironment()):
            output = runner.run_study(self.plan_path, self.client)
        self.assertEqual(output["status"], "disabled_status_only")
        self.assertEqual(output["selected_queries"], 10)
        self.assertFalse(self.calls)
        self.assertFalse((self.client.root / "budget_ledger.jsonl").exists())

    def test_partial_resume_preserves_denominator_and_never_rebills_cache(self):
        first = self.run_study(max_queries=1)
        self.assertEqual((first["status"], first["complete_queries"], first["selected_queries"]), ("interrupted", 1, 10))
        snapshot = prep.read_json(Path(first["snapshot_dir"]) / "jev_rerank.json")
        self.assertEqual(len(snapshot["rankings"]), 10)
        self.assertEqual(sum(bool(r) for r in snapshot["rankings"].values()), 1)
        second = self.run_study(max_queries=1)
        self.assertEqual(second["complete_queries"], 2)
        final = self.run_study()
        self.assertEqual((final["status"], final["complete_queries"]), ("completed", 10))
        self.assertEqual(len(self.calls), 10)
        self.assertEqual(final["ledger"]["paid_api_attempts_reserved"], 10)
        self.assertEqual(final["ledger"]["known_input_tokens_settled"], 1000)
        self.assertNotEqual(first["snapshot_dir"], final["snapshot_dir"])
        with patch.object(hosted.os, "environ", DeniedEnvironment()):
            repeated = self.run_study()
        self.assertEqual(repeated["paid_attempts_reserved_this_invocation"], 0)
        self.assertEqual(len(self.calls), 10)
        self.assertEqual(repeated["ledger"]["settled_usd"], final["ledger"]["settled_usd"])

    def test_copied_cache_without_ledger_cannot_complete_or_publish_snapshot(self):
        self.assertEqual(self.run_study()["status"], "completed")
        copied = self.fixture.path / "orphan-cache"
        shutil.copytree(self.client.root / "raw_responses", copied / "raw_responses")
        self.client.root = copied
        self.client.ledger = hosted.SpendLedger(copied)
        with patch.object(hosted.os, "environ", DeniedEnvironment()):
            result = self.run_study()
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["complete_queries"], 0)
        self.assertEqual(result["selected_queries"], 10)
        self.assertEqual(result["ledger"]["paid_api_attempts_reserved"], 0)
        self.assertEqual(result["snapshot_status"], "blocked_no_unverified_snapshot_published")
        self.assertNotIn("snapshot_dir", result)
        self.assertEqual(len(self.calls), 10)

    def test_settled_cache_with_changed_raw_attempt_is_blocked(self):
        self.run_study(max_queries=1)
        packet = self.plan["rows"][0]["batches"][0]
        cache = prep.read_json(self.client.root / "raw_responses" / (packet["request_sha256"] + ".json"))
        (self.client.root / "attempts" / (cache["attempt_id"] + ".body")).write_bytes(b"changed")
        with patch.object(hosted.os, "environ", DeniedEnvironment()):
            result = self.run_study()
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["complete_queries"], 0)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn("snapshot_dir", result)

    def test_verified_alias_cache_resumes_without_network_or_double_spend(self):
        original = self.run_study()
        self.plan, requests = self.fixture.make_plan(model="jev-1.13.0")
        self.plan.update(schema_version=prep.SCHEMA_VERSION, rubric=prep.RUBRIC, question_prefix=prep.QUESTION)
        self.plan["context"]["limit_supplied"] = 64000
        self.plan["expected_model_version"] = "jev-1.13.0"
        self.plan_path = self.fixture.path / "pinned-plan" / "manifest.json"
        self.plan_path.parent.mkdir()
        for row in self.plan["rows"]:
            for packet in row["batches"]:
                path = self.plan_path.parent / packet["request_file"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(prep.canonical(requests[packet["request_sha256"]]))
        self.save_plan()
        with patch.object(hosted.os, "environ", DeniedEnvironment()):
            aliased = self.run_study()
            resumed = self.run_study()
        self.assertEqual(aliased["status"], "completed")
        self.assertEqual(resumed["complete_queries"], 10)
        self.assertEqual(resumed["ledger"]["alias_cache_reuse_events"], 10)
        self.assertEqual(resumed["ledger"]["settled_usd"], original["ledger"]["settled_usd"])
        self.assertEqual(len(self.calls), 10)
        packet = self.plan["rows"][0]["batches"][0]
        semantic = prep.digest(hosted.HostedJevClient._normalized_payload(
            requests[packet["request_sha256"]], "jev-1.13.0"))
        index_path = self.client.root / "semantic_cache" / (semantic + ".json")
        index = prep.read_json(index_path)
        index["response_sha256"] = "0" * 64
        index_path.write_bytes(prep.canonical(index))
        with patch.object(hosted.os, "environ", DeniedEnvironment()):
            tampered = self.run_study()
        self.assertEqual(tampered["status"], "interrupted")
        self.assertNotIn("snapshot_dir", tampered)
        self.assertEqual(len(self.calls), 10)

    def test_cumulative_deadline_survives_process_resume(self):
        self.run_study(max_queries=1)
        self.now = NOW + 21601
        result = self.run_study()
        self.assertEqual(result["reason"], "cumulative_six_hour_deadline")
        self.assertEqual(result["complete_queries"], 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["deadline_unix"], NOW + 21600)

    def test_ambiguous_response_stops_and_keeps_reserve_without_error_text(self):
        def failed(*args):
            self.calls.append("failed")
            raise RuntimeError("sensitive-response-text-must-not-escape")
        self.client.transport = failed
        result = self.run_study()
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["error_type"], "AmbiguousRequest")
        self.assertEqual(result["ledger"]["unresolved_attempts"], 1)
        self.assertNotIn("sensitive-response", json.dumps(result))
        resumed = self.run_study()
        self.assertEqual(resumed["reason"], "unresolved_prior_attempt_manual_reconciliation_required")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(resumed["selected_queries"], 10)

    def test_reviewed_original_remains_unknown_held_while_scope_completes(self):
        original_id = self.reviewed_loss_fixture()
        result = self.run_study()
        self.assertEqual((result["status"], result["complete_queries"], result["selected_queries"]), ("completed", 10, 10))
        self.assertEqual(result["reviewed_unknowns"], [original_id])
        self.assertEqual(result["ledger"]["unresolved_attempts"], 1)
        self.assertEqual(result["ledger"]["unreviewed_unresolved_attempts"], 0)
        self.assertEqual(result["ledger"]["paid_api_attempts_reserved"], 11)
        self.assertEqual(result["ledger"]["paid_responses_settled"], 10)
        self.assertEqual(Decimal(result["ledger"]["held_usd"]), Decimal(".00275"))
        self.assertEqual(Decimal(result["ledger"]["held_cad_with_fee_buffer"]), Decimal(".004"))
        self.assertEqual(result["deadline_unix"], NOW - 100 + 21600)
        self.assertEqual(len(self.calls), 10)
        self.assertEqual(self.client.gates.contract.read()["max_attempts"], 1)
        originals = [row for row in self.client.ledger.rows() if row.get("attempt_id") == original_id]
        self.assertEqual([row["event"] for row in originals], ["reserved"])
        with patch.object(hosted.os, "environ", DeniedEnvironment()):
            resumed = self.run_study()
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["paid_attempts_reserved_this_invocation"], 0)
        self.assertEqual(resumed["ledger"]["held_usd"], result["ledger"]["held_usd"])

    def test_reviewed_original_does_not_exempt_lost_replacement(self):
        original_id = self.reviewed_loss_fixture()
        def failed(*args):
            self.calls.append("replacement-lost")
            raise TimeoutError("synthetic")
        self.client.transport = failed
        first = self.run_study()
        self.assertEqual(first["status"], "interrupted")
        self.assertEqual(first["reviewed_unknowns"], [original_id])
        self.assertEqual(first["ledger"]["unresolved_attempts"], 2)
        self.assertEqual(first["ledger"]["unreviewed_unresolved_attempts"], 1)
        with patch.object(hosted.os, "environ", DeniedEnvironment()):
            second = self.run_study()
        self.assertEqual(second["reason"], "unresolved_prior_attempt_manual_reconciliation_required")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(second["ledger"]["paid_api_attempts_reserved"], 2)

    def test_reviewed_original_does_not_exempt_another_unknown_request(self):
        self.reviewed_loss_fixture()
        other_hash = self.plan["rows"][-1]["batches"][0]["request_sha256"]
        with self.client.ledger.locked():
            self.client.ledger.append({"event": "reserved", "request_sha256": other_hash,
                                      "attempt_id": other_hash + "-1", "utc_unix": NOW - 50,
                                      "reserve_usd": ".003", "reserve_cad": ".005"})
        with patch.object(hosted.os, "environ", DeniedEnvironment()):
            result = self.run_study()
        self.assertEqual(result["reason"], "unresolved_prior_attempt_manual_reconciliation_required")
        self.assertEqual(result["ledger"]["reviewed_unknown_attempts"], 1)
        self.assertEqual(result["ledger"]["unreviewed_unresolved_attempts"], 1)
        self.assertEqual(Decimal(result["ledger"]["held_usd"]), Decimal(".00575"))
        self.assertFalse(self.calls)

    def test_changed_recovery_receipt_blocks_without_credential_access(self):
        self.reviewed_loss_fixture()
        self.client.gates.recovery.path.write_bytes(b"{}")
        with patch.object(hosted.os, "environ", DeniedEnvironment()):
            result = self.run_study()
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["error_type"], "GateError")
        self.assertEqual(result["reviewed_unknowns"], [])
        self.assertEqual(result["ledger"]["unreviewed_unresolved_attempts"], 1)
        self.assertEqual(result["ledger"]["recovery_review_error_type"], "GateError")
        self.assertFalse(self.calls)

    def test_deadline_crossed_during_gate_checks_stops_before_transport(self):
        self.run_study(max_queries=1)
        self.now = NOW + .1
        original = self.client._gates
        def slow_gates(*args, **kwargs):
            value = original(*args, **kwargs)
            self.now = NOW + .9
            return value
        with patch.object(self.client, "_gates", side_effect=slow_gates):
            result = self.run_study(max_seconds=.5)
        self.assertEqual(result["reason"], "deadline_before_transport")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["ledger"]["unresolved_attempts"], 1)
        self.assertEqual(result["complete_queries"], 1)

    def test_any_blocked_query_prevents_whole_scope_execution(self):
        self.plan["rows"][-1]["status"] = "blocked_context_or_contract"
        self.save_plan()
        with self.assertRaisesRegex(runner.StudyStop, "preparation_blocked"):
            self.run_study()
        self.assertFalse(self.calls)

    def test_later_corrupt_request_is_caught_before_first_paid_call(self):
        last = self.plan["rows"][-1]["batches"][-1]
        (self.plan_path.parent / last["request_file"]).write_text("{}")
        with self.assertRaises(hosted.GateError):
            self.run_study()
        self.assertFalse(self.calls)

    def test_existing_study_lock_is_not_removed(self):
        lock = self.client.root / "study_runner.lock"
        lock.write_text("manual recovery")
        with self.assertRaisesRegex(runner.StudyStop, "runner_locked"):
            self.run_study()
        self.assertTrue(lock.exists())
        self.assertFalse(self.calls)

    def test_settled_missing_cache_never_resends(self):
        self.run_study(max_queries=1)
        first = self.plan["rows"][0]["batches"][0]
        (self.client.root / "raw_responses" / (first["request_sha256"] + ".json")).unlink()
        result = self.run_study()
        self.assertEqual(result["error_type"], "GateError")
        self.assertEqual(len(self.calls), 1)

    def test_ledger_reuse_events_do_not_duplicate_charges(self):
        self.run_study(max_queries=1)
        before = runner.ledger_summary(self.client.ledger)
        with self.client.ledger.locked():
            self.client.ledger.append({"event": "cache_reused", "additional_api_calls": 0})
        after = runner.ledger_summary(self.client.ledger)
        self.assertEqual(before["settled_usd"], after["settled_usd"])
        self.assertEqual(after["paid_api_attempts_reserved"], 1)
        self.assertEqual(after["alias_cache_reuse_events"], 1)

    def test_cli_requires_execute_and_gates(self):
        with contextlib.redirect_stdout(io.StringIO()) as out, patch.object(hosted.os, "environ", DeniedEnvironment()):
            code = runner.main(["--plan", str(self.plan_path), "--run-dir", str(self.client.root)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["status"], "disabled_status_only")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = runner.main(["--plan", str(self.plan_path), "--run-dir", str(self.client.root), "--execute"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(err.getvalue())["reason"], "gate_configuration_required")
        self.assertFalse(self.calls)

    def test_gate_reference_config_does_not_accept_key_fields(self):
        path = self.fixture.path / "gate-config.json"
        path.write_text('{"api_key":"never-accepted"}')
        with self.assertRaises(hosted.GateError):
            runner.load_gates(path)

    def test_deadline_cannot_be_extended_beyond_six_hours(self):
        with self.assertRaises(hosted.GateError):
            self.run_study(max_seconds=21601)
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
