"""Runner tests use synthetic data and injected transport; no real API/key access."""
import contextlib
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
