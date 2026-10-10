"""Synthetic reporting-only recovery tests: no API, key or actual report output."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import actual_hosted_report as report


class DeniedEnvironment(dict):
    def __getitem__(self, key):
        raise AssertionError("Credential/environment access is forbidden")
    def get(self, key, default=None):
        raise AssertionError("Credential/environment access is forbidden")
    def __contains__(self, key):
        raise AssertionError("Credential/environment access is forbidden")


class RecoveryReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = "a" * 64
        self.original = self.request + "-1"
        self.replacement = self.request + "-2"
        self.rows = [
            {"event": "reserved", "attempt_id": self.original, "request_sha256": self.request,
             "event_sha256": "b" * 64, "reserve_usd": "0.003", "reserve_cad": "0.006"},
            {"event": "reserved", "attempt_id": self.replacement, "request_sha256": self.request},
            {"event": "response_saved", "attempt_id": self.replacement},
            {"event": "settled", "attempt_id": self.replacement},
            {"event": "cache_reused", "request_sha256": "c" * 64},
        ]
        self.receipt = {"schema_version": "reviewed-wsl-reboot-recovery-v1", "verified": True,
            "additional_attempts": 1, "retain_original_reservation": True,
            "request_bytes_and_candidate_pool_unchanged": True,
            "use_first_replacement_response_regardless_of_scores": True,
            "possible_duplicate_billing_disclosed": True,
            "request_sha256": self.request, "original_attempt_id": self.original,
            "replacement_attempt_id": self.replacement,
            "original_reservation_event_sha256": "b" * 64,
            "original_unknown_reservation_usd": "0.003", "original_unknown_reservation_cad": "0.006",
            "plan_sha256": "d" * 64, "original_invocation_id": "interrupted",
            "interruption_lower_bound_unix": 100.0, "interruption_upper_bound_unix": 200.0}
        self.path = self.root / "recovery.json"
        self.path.write_text(json.dumps(self.receipt), encoding="utf-8")

    def test_attempts_responses_and_holds_are_distinct(self):
        with patch("os.environ", DeniedEnvironment()):
            value = report.response_observability(self.rows)
        self.assertEqual(value["physical_attempt_reservations"], 2)
        self.assertEqual(value["attempts_with_preserved_http_response"], 1)
        self.assertEqual(value["attempts_with_settled_usage"], 1)
        self.assertEqual(value["unresolved_reservations"], 1)
        self.assertEqual(value["attempts_without_preserved_response"], 1)

    def test_recovery_keeps_original_hold_and_missing_final(self):
        with patch("os.environ", DeniedEnvironment()):
            value = report.recovery_summary(self.path, self.rows, self.root)
        self.assertTrue(value["original_reservation_still_unresolved"])
        self.assertEqual(value["original_unknown_held_usd"], "0.003")
        self.assertFalse(value["original_final_receipt_present"])
        self.assertFalse(value["original_http_response_preserved_in_ledger"])
        self.assertTrue(value["replacement_usage_settled"])
        self.assertTrue(value["one_possible_duplicate_execution_and_charge"])

    def test_unbound_receipt_is_rejected(self):
        self.receipt["original_unknown_reservation_usd"] = "0"
        self.path.write_text(json.dumps(self.receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "held allowance differs"):
            report.recovery_summary(self.path, self.rows, self.root)

    def test_third_attempt_is_not_silently_covered(self):
        self.rows.append({"event": "reserved", "attempt_id": self.request + "-3", "request_sha256": self.request})
        with self.assertRaisesRegex(ValueError, "one reviewed replacement"):
            report.recovery_summary(self.path, self.rows, self.root)

    def timing_document(self, upper=200.0):
        return {"wall_time": {"scope_invocation_wall_seconds": None,
            "scope_invocation_wall_seconds_lower_bound": 100.0,
            "scope_invocation_wall_seconds_upper_bound": upper,
            "new_paid_api_elapsed_in_scope_invocations_seconds": 20.0,
            "scope_non_api_wall_seconds": None,
            "scope_non_api_wall_seconds_lower_bound": 80.0,
            "scope_non_api_wall_seconds_upper_bound": upper - 20 if upper is not None else None,
            "experiment_elapsed_including_gaps_seconds": 1000.0,
            "scope_invocations_missing_final": 1, "definition": "Synthetic timing bounds."},
            "projection": {"status": "descriptive_projection_not_execution_authorization", "target_queries": 4392,
                "projected_additional_wall_seconds_lower_bound": 2000.0,
                "projected_additional_wall_seconds_upper_bound": 4000.0 if upper is not None else None,
                "projected_experiment_wall_seconds_lower_bound": 3000.0,
                "projected_experiment_wall_seconds_upper_bound": 5000.0 if upper is not None else None,
                "projected_total_settled_rate_cad_lower_bound": "2.0",
                "projected_total_planning_commitment_cad_upper_bound": "2.006",
                "projected_total_settled_rate_usd_lower_bound": "1.0",
                "projected_total_planning_commitment_usd_upper_bound": "1.003",
                "held_unknown_usd": "0.003", "held_unknown_cad_with_fee_buffer": "0.006"}}

    def test_uncertain_times_and_unknown_cost_are_disclosed(self):
        with patch("os.environ", DeniedEnvironment()):
            text = "\n".join(report.timing_projection_lines(self.timing_document()))
        self.assertIn("100.000 to 200.000", text)
        self.assertIn("without a final receipt: 1", text)
        self.assertIn("not necessarily pure local overhead", text)
        self.assertIn("unknown held allowance once", text)
        self.assertIn("not statistical cost intervals or known charges", text)
        self.assertIn("does not authorize expansion", text)

    def test_unknown_timing_upper_is_not_formatted_as_zero(self):
        text = "\n".join(report.timing_projection_lines(self.timing_document(None)))
        self.assertIn("100.000 to unknown", text)
        self.assertIn("2000.0 to unknown", text)


if __name__ == "__main__":
    unittest.main()
