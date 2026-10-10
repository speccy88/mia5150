"""Unit controls only: these are NOT benchmark results or API integration tests."""

import math
import os
from pathlib import Path
import stat
import tempfile
import unittest

from reference_metrics import make_nested_catalogs, retrieval_metrics, selection_metrics
from setup_local_key import write_local_env


class MetricControls(unittest.TestCase):
    def test_multi_positive_recall(self):
        m = retrieval_metrics(["x", "b", "a"], {"a", "b"}, 2)
        self.assertEqual(m["recall_at_2"], 0.5)
        self.assertEqual(m["mrr_at_2"], 0.5)
        self.assertEqual(m["completeness_at_2"], 0.0)
        self.assertAlmostEqual(
            m["ndcg_at_2"], (1 / math.log2(3)) / (1 + 1 / math.log2(3))
        )

    def test_exact_retrieval(self):
        m = retrieval_metrics(["b", "a", "x"], {"a", "b"}, 2)
        self.assertEqual(m["recall_at_2"], 1.0)
        self.assertEqual(m["completeness_at_2"], 1.0)
        self.assertEqual(m["ndcg_at_2"], 1.0)

    def test_empty_prediction_is_not_success(self):
        self.assertEqual(selection_metrics([], {"a"})["exact_set_match"], 0.0)
        self.assertEqual(selection_metrics([], {"a"})["f1"], 0.0)
        self.assertEqual(retrieval_metrics([], {"a"}, 5)["recall_at_5"], 0.0)

    def test_unknown_ids_are_false_positives(self):
        m = selection_metrics(["a", "unknown"], {"a"})
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["exact_set_match"], 0.0)

    def test_selection_duplicates_flagged(self):
        m = selection_metrics(["a", "a"], {"a"})
        self.assertEqual(m["duplicate_predictions"], 1)
        self.assertEqual(m["precision"], 1.0)
        # Strict validity is a separate gate; this function is explicitly set-based.

    def test_retrieval_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            retrieval_metrics(["a", "a"], {"a"}, 2)

    def test_gold_and_k_validation(self):
        with self.assertRaises(ValueError):
            retrieval_metrics(["a"], set(), 1)
        with self.assertRaises(ValueError):
            retrieval_metrics(["a"], {"a"}, 0)
        with self.assertRaises(ValueError):
            selection_metrics(["a"], set())

    def test_nested_sets_and_full_gold(self):
        catalogs = make_nested_catalogs(
            [str(i) for i in range(20)], {"2", "7"}, [5, 10, 20], "q1"
        )
        self.assertEqual([len(catalogs[n]) for n in [5, 10, 20]], [5, 10, 20])
        self.assertLessEqual(set(catalogs[5]), set(catalogs[10]))
        self.assertLessEqual(set(catalogs[10]), set(catalogs[20]))
        for catalog in catalogs.values():
            self.assertLessEqual({"2", "7"}, set(catalog))

    def test_order_invariant_and_reproducible(self):
        ids = [str(i) for i in range(20)]
        a = make_nested_catalogs(ids, {"2", "7"}, [5, 10], "q1", 42)
        b = make_nested_catalogs(list(reversed(ids)), ["7", "2"], [10, 5], "q1", 42)
        self.assertEqual(a, b)
        self.assertEqual(tuple(x for x in a[10] if x in a[5]), a[5])

    def test_bad_catalogs_rejected(self):
        with self.assertRaises(ValueError):
            make_nested_catalogs(["a", "a"], {"a"}, [1], "q")
        with self.assertRaises(ValueError):
            make_nested_catalogs(["a", "b"], {"a", "b"}, [1], "q")
        with self.assertRaises(ValueError):
            make_nested_catalogs(["a", "b"], {"c"}, [2], "q")


class LocalKeyHelperControls(unittest.TestCase):
    def test_private_new_file_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            write_local_env(path, "TEST_ONLY_not_a_real_key", False)
            self.assertIn("ALLOW_PAID_RUNS=0", path.read_text())
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                write_local_env(path, "another_TEST_ONLY_key", True)
            self.assertIn("ALLOW_PAID_RUNS=0", path.read_text())

    def test_permission_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            write_local_env(path, "TEST_ONLY_not_a_real_key", True)
            self.assertIn("ALLOW_PAID_RUNS=1", path.read_text())

    def test_reject_multiline_and_empty_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            for key in ["", "short", "onekey\nEVIL=1", "unquoted#comment-key"]:
                with self.assertRaises(ValueError):
                    write_local_env(path, key)
                self.assertFalse(path.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX symlink control")
    def test_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as folder:
            real = Path(folder) / "target"
            real.write_text("preserve")
            path = Path(folder) / ".env"
            path.symlink_to(real)
            with self.assertRaises(OSError):
                write_local_env(path, "TEST_ONLY_not_a_real_key", True)
            self.assertEqual(real.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
