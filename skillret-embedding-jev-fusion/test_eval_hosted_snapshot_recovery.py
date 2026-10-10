"""Synthetic-only recovery accounting: no network, credential or runtime reads."""
from copy import deepcopy
from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import eval_hosted_snapshot as subject


class RecoveryAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {"plan_sha256": "pilot", "rows": [{"batches": [
            {"request_sha256": "paid"}, {"request_sha256": "unknown"}]}]}
        self.rows = [
            self.reserve("smoke", "smoke-a", 80, "smoke", "1", ".5"),
            self.settle("smoke", "smoke-a", 80, "1", ".5", 4),
            self.reserve("paid", "paid-a", 110, "pilot", "1", ".5"),
            self.settle("paid", "paid-a", 110, "1", ".5", 5),
            self.reserve("unknown", "unknown-a", 160, "pilot", ".5", ".25"),
        ]
        for index, row in enumerate(self.rows):
            row["event_sha256"] = "event-" + str(index)
        self.interrupted = self.root / "study_progress" / "pilot" / "interrupted"
        self.write(self.interrupted / "000000.json", self.checkpoint("interrupted", 100, 100, 0))
        self.last_path = self.interrupted / "000001.json"
        self.write(self.last_path, self.checkpoint("interrupted", 100, 150, 146))
        self.final = self.checkpoint("finished", 210, 230, 200)
        self.final.update(status="completed", ledger=subject.accounting(self.rows))
        self.final_path = self.root / "study_progress" / "pilot" / "finished" / "final.json"
        self.write(self.final_path, self.final, mtime=235)
        self.proof_path = self.root / "recovery-after-wsl-restart" / "reboot.json"
        self.write(self.proof_path, {"kind": "wsl_reboot_observation", "boot_id": "synthetic-boot",
                                     "matching_live_process_count": 0,
                                     "observed_at_unix": 206, "uptime_seconds": 6,
                                     "confirmed_new_wsl_boot_unix": 200})
        self.recovery = {"schema_version": "reviewed-wsl-reboot-recovery-v1",
                         "original_invocation_id": "interrupted", "plan_sha256": "pilot",
                         "original_invocation_started_at_unix": 100,
                         "observed_last_action_unix": 160,
                         "interruption_lower_bound_unix": 160,
                         "interruption_upper_bound_unix": 200,
                         "confirmed_new_wsl_boot_unix": 200,
                         "original_attempt_id": "unknown-a", "request_sha256": "unknown",
                         "original_reservation_event_sha256": "event-4",
                         "reboot_proof": {"path": "recovery-after-wsl-restart/reboot.json",
                                           "sha256": subject.sha(self.proof_path)}}
        self.recovery_path = self.proof_path.with_name("recovery.json")
        self.write(self.recovery_path, self.recovery, mtime=9999999)

    @staticmethod
    def reserve(request, attempt, stamp, plan, cad, usd):
        return {"event": "reserved", "request_sha256": request, "attempt_id": attempt,
                "utc_unix": stamp, "plan_sha256": plan, "reserve_cad": cad, "reserve_usd": usd}

    @staticmethod
    def settle(request, attempt, stamp, cad, usd, elapsed):
        return {"event": "settled", "request_sha256": request, "attempt_id": attempt,
                "utc_unix": stamp, "actual_cad": cad, "actual_usd": usd,
                "input_tokens": 100, "output_tokens": None, "elapsed_seconds": elapsed}

    @staticmethod
    def checkpoint(invocation, start, updated, complete):
        return {"invocation_id": invocation, "plan_sha256": "pilot", "status": "running",
                "invocation_started_at_unix": start, "updated_at_unix": updated,
                "complete_queries": complete}

    @staticmethod
    def write(path, value, mtime=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        if mtime is not None:
            os.utime(path, (mtime, mtime))

    def timing(self):
        return subject.invocation_timing(self.root, "pilot", self.rows, 235)[0]

    def checkpoint_accounting(self, rows=None):
        with patch.object(subject, "SpendLedger") as ledger:
            ledger.return_value.rows.return_value = self.rows if rows is None else rows
            result = subject.checkpoint_accounting(self.plan, self.root, self.root / "synthetic-snapshot",
                                                  self.final, self.final_path)
            ledger.assert_called_once_with(self.root)
            return result

    def test_missing_final_uses_checkpoint_and_reviewed_bounds_not_mtime(self):
        account, scope, timing = self.checkpoint_accounting()
        interrupted, finished = timing["scope_invocations"]
        self.assertFalse(interrupted["final_receipt_present"])
        self.assertEqual(interrupted["evidence_status"], "partial_missing_final")
        self.assertEqual(interrupted["last_checkpoint_status"], "running")
        self.assertEqual(interrupted["last_checkpoint_complete_queries"], 146)
        self.assertEqual(interrupted["wall_seconds_lower_bound"], 60)
        self.assertEqual(interrupted["wall_seconds_upper_bound"], 100)
        self.assertNotIn("exit", interrupted)
        self.assertTrue(finished["final_receipt_present"])
        self.assertEqual(timing["scope_invocation_wall_seconds_lower_bound"], 85)
        self.assertEqual(timing["scope_invocation_wall_seconds_upper_bound"], 125)
        self.assertIsNone(timing["scope_invocation_wall_seconds"])
        self.assertEqual(timing["scope_invocations_missing_final"], 1)
        self.assertEqual(timing["new_paid_api_elapsed_in_scope_invocations_seconds"], 5)
        self.assertEqual(timing["scope_non_api_wall_seconds_lower_bound"], 0)
        self.assertEqual(timing["scope_wall_minus_recorded_settled_api_seconds_lower_bound"], 80)
        self.assertEqual(timing["experiment_elapsed_including_gaps_seconds"], 155)
        self.assertEqual(timing["cumulative_six_hour_deadline_unix"], 80 + 21600)
        self.assertEqual(account["unresolved_attempts"], 1)
        self.assertEqual(scope["paid_responses_settled"], 1)

    def test_missing_recovery_keeps_observed_work_and_unknown_upper(self):
        self.recovery_path.unlink()
        account, scope, timing = self.checkpoint_accounting()
        self.assertEqual(timing["scope_invocation_wall_seconds_lower_bound"], 75)
        self.assertIsNone(timing["scope_invocation_wall_seconds_upper_bound"])
        result = subject.projection({"query_count": 200, "complete_hosted_queries": 200},
                                    scope, account, timing, 4392)
        self.assertEqual(result["status"], "descriptive_projection_not_execution_authorization")
        self.assertIsNone(result["projected_additional_wall_seconds"])
        self.assertIsNone(result["projected_experiment_wall_seconds_upper_bound"])
        self.assertGreater(result["projected_experiment_wall_seconds_lower_bound"], 155)

    def test_mutable_latest_is_not_an_invocation_receipt(self):
        self.write(self.interrupted.parent / "latest.json", {"invalid": "mutable"})
        self.assertEqual(len(self.timing()), 2)

    def test_recovery_cannot_precede_last_checkpoint(self):
        self.recovery["interruption_lower_bound_unix"] = 149
        self.write(self.recovery_path, self.recovery)
        with self.assertRaisesRegex(ValueError, "Recovery bounds"):
            self.timing()

    def test_recovery_cannot_extend_past_confirmed_boot(self):
        self.recovery["interruption_upper_bound_unix"] = 201
        self.write(self.recovery_path, self.recovery)
        with self.assertRaisesRegex(ValueError, "Recovery bounds"):
            self.timing()

    def test_recovery_requires_same_reservation_hash(self):
        for key in ("request_sha256", "original_reservation_event_sha256"):
            with self.subTest(key=key):
                edited = {**self.recovery, key: "different"}
                self.write(self.recovery_path, edited)
                with self.assertRaisesRegex(ValueError, "reservation request or event hash"):
                    self.timing()

    def test_recovery_requires_snapshot_reservation(self):
        with self.assertRaisesRegex(ValueError, "unique snapshot reservation"):
            subject.invocation_timing(self.root, "pilot", self.rows[:-1], 235)

    def test_recovery_requires_proof_hash(self):
        self.write(self.proof_path, {"modified": True})
        with self.assertRaisesRegex(ValueError, "proof hash differs"):
            self.timing()

    def test_hash_matching_empty_reboot_proof_is_insufficient(self):
        self.write(self.proof_path, {})
        self.recovery["reboot_proof"]["sha256"] = subject.sha(self.proof_path)
        self.write(self.recovery_path, self.recovery)
        with self.assertRaisesRegex(ValueError, "proof is incomplete"):
            self.timing()

    def test_boot_bound_must_match_observation_minus_uptime(self):
        proof = subject.read(self.proof_path)
        proof["uptime_seconds"] = 7
        self.write(self.proof_path, proof)
        self.recovery["reboot_proof"]["sha256"] = subject.sha(self.proof_path)
        self.write(self.recovery_path, self.recovery)
        with self.assertRaisesRegex(ValueError, "boot bound differs"):
            self.timing()

    def test_historical_snapshot_does_not_use_later_reboot_as_end(self):
        invocations, _ = subject.invocation_timing(self.root, "pilot", self.rows[:-1], 155)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(invocations[0]["wall_seconds_lower_bound"], 50)
        self.assertIsNone(invocations[0]["wall_seconds_upper_bound"])

    def test_final_updated_before_ingestion_does_not_leak_later_end(self):
        self.write(self.final_path.with_name("000000.json"), self.checkpoint("finished", 210, 230, 200))
        invocations, _ = subject.invocation_timing(self.root, "pilot", self.rows, 232)
        self.assertFalse(invocations[-1]["final_receipt_present"])
        self.assertEqual(invocations[-1]["wall_seconds_lower_bound"], 20)
        self.assertIsNone(invocations[-1]["wall_seconds_upper_bound"])

    def test_recovery_cannot_escape_run_directory(self):
        self.recovery["reboot_proof"]["path"] = "../outside.json"
        self.write(self.recovery_path, self.recovery)
        with self.assertRaisesRegex(ValueError, "escapes experiment directory"):
            self.timing()

    def test_checkpoint_identity_and_start_are_consistent(self):
        for key, value in (("invocation_id", "other"), ("invocation_started_at_unix", 99)):
            with self.subTest(key=key):
                receipt = self.checkpoint("interrupted", 100, 150, 146)
                receipt[key] = value
                self.write(self.last_path, receipt)
                with self.assertRaisesRegex(ValueError, "Immutable checkpoint invocation"):
                    self.timing()

    def test_invalid_or_reversed_checkpoint_times_fail(self):
        for value in (float("nan"), float("inf"), True, -1, 99):
            with self.subTest(value=value):
                self.write(self.last_path, self.checkpoint("interrupted", 100, value, 146))
                with self.assertRaises(ValueError):
                    self.timing()

    def test_duplicate_recovery_receipts_are_ambiguous(self):
        self.write(self.root / "recovery-other" / "recovery.json", self.recovery)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.timing()

    def test_post_snapshot_ledger_event_does_not_change_usage(self):
        extra = self.reserve("future", "future-a", 999, "pilot", "40", "20")
        extra["event_sha256"] = "future-event"
        account, _, timing = self.checkpoint_accounting([*self.rows, extra])
        self.assertEqual(account["ledger_events"], 5)
        self.assertEqual(Decimal(account["held_cad_with_fee_buffer"]), Decimal(".5"))
        self.assertEqual(timing["cumulative_six_hour_deadline_unix"], 21680)

    def test_future_invocation_does_not_enter_earlier_snapshot(self):
        path = self.root / "study_progress" / "pilot" / "future" / "000000.json"
        self.write(path, self.checkpoint("future", 300, 301, 0))
        self.assertEqual(len(self.timing()), 2)

    def test_experiment_elapsed_and_deadline_start_at_first_ledger_attempt(self):
        prior = self.checkpoint("earlier", 70, 75, 0)
        prior["plan_sha256"] = "smoke"
        self.write(self.root / "study_progress" / "smoke" / "earlier" / "final.json", prior, mtime=76)
        _, _, timing = self.checkpoint_accounting()
        self.assertEqual(timing["experiment_elapsed_start_unix"], 80)
        self.assertEqual(timing["experiment_elapsed_including_gaps_seconds"], 155)
        self.assertEqual(timing["cumulative_six_hour_deadline_unix"], 21680)

    def test_complete_projection_keeps_unknown_hold_once_and_is_not_authorization(self):
        account, scope, timing = self.checkpoint_accounting()
        original = deepcopy((account, scope, timing))
        result = subject.projection({"query_count": 200, "complete_hosted_queries": 200},
                                    scope, account, timing, 400)
        self.assertEqual(result["status"], "descriptive_projection_not_execution_authorization")
        self.assertFalse(result["execution_authorized"])
        self.assertEqual(Decimal(result["projected_additional_usage_cad_with_recorded_fee_buffer"]), Decimal("1"))
        self.assertEqual(Decimal(result["projected_total_settled_rate_cad_lower_bound"]), Decimal("3"))
        self.assertEqual(Decimal(result["projected_total_planning_commitment_cad_upper_bound"]), Decimal("3.5"))
        self.assertEqual(Decimal(result["projected_total_settled_rate_usd_lower_bound"]), Decimal("1.5"))
        self.assertEqual(Decimal(result["projected_total_planning_commitment_usd_upper_bound"]), Decimal("1.75"))
        self.assertEqual(result["projected_experiment_wall_seconds_lower_bound"], 240)
        self.assertEqual(result["projected_experiment_wall_seconds_upper_bound"], 280)
        self.assertEqual(result["projected_experiment_wall_seconds_including_elapsed_gaps"], 280)
        self.assertEqual((account, scope, timing), original)

    def test_replacement_settlement_does_not_resolve_original_hold(self):
        replacement = self.reserve("unknown", "replacement-a", 220, "pilot", ".5", ".25")
        replacement["event_sha256"] = "replacement-reserved"
        settled = self.settle("unknown", "replacement-a", 220, ".2", ".1", 2)
        settled["event_sha256"] = "replacement-settled"
        account = subject.accounting([*self.rows, replacement, settled])
        self.assertEqual(account["unresolved_attempts"], 1)
        self.assertEqual(Decimal(account["held_usd"]), Decimal(".25"))
        self.assertEqual(account["paid_responses_settled"], 3)

    def test_partial_fixed_scope_still_blocks_projection(self):
        account, scope, timing = self.checkpoint_accounting()
        for complete in (0, 199):
            result = subject.projection({"query_count": 200, "complete_hosted_queries": complete},
                                        scope, account, timing, 4392)
            self.assertEqual(result["status"], "unavailable")

    def test_uninterrupted_final_retains_legacy_scalar_timing(self):
        self.last_path.unlink()
        (self.interrupted / "000000.json").unlink()
        account, scope, timing = self.checkpoint_accounting()
        self.assertEqual(timing["scope_invocation_wall_seconds"], 25)
        self.assertEqual(timing["scope_invocation_wall_seconds_lower_bound"], 25)
        self.assertEqual(timing["scope_invocation_wall_seconds_upper_bound"], 25)
        self.assertEqual(timing["scope_invocations_missing_final"], 0)


if __name__ == "__main__":
    unittest.main()
