"""Boundary checks for label counting and the saved-report update."""

import unittest

from golden_skills import START, analyze_rows, update_report


def query(query_id, skills):
    return {"id": query_id, "skill_ids": skills, "k": len(skills)}


def qrel(query_id, skill_id, relevance=1):
    return {"query_id": query_id, "skill_id": skill_id, "relevance": relevance}


class GoldenSkillsTests(unittest.TestCase):
    def test_zero_queries_negative_labels_and_duplicate_qrels(self):
        result = analyze_rows(
            [query("q0", []), query("q1", ["s1"]), query("q2", ["s1", "s2"])],
            [qrel("q0", "s0", 0), qrel("q0", "s2", -1),
             qrel("q1", "s1"), qrel("q1", "s1"),
             qrel("q2", "s1"), qrel("q2", "s2", 2)],
        )
        self.assertEqual(result["distribution"], {0: 1, 1: 1, 2: 1})
        self.assertEqual(result["queries"], 3)
        self.assertEqual(result["distinct_golden_pairs"], 3)
        self.assertEqual(result["duplicate_positive_qrel_rows"], 1)
        self.assertEqual(result["mean_golden_skills"], 1)

    def test_orphan_qrel_is_rejected_even_if_nonpositive(self):
        with self.assertRaisesRegex(ValueError, "unknown query"):
            analyze_rows([query("q", [])], [qrel("unknown", "s", 0)])

    def test_duplicate_query_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate query"):
            analyze_rows([query("q", []), query("q", [])], [])

    def test_skill_ids_must_match_qrels(self):
        with self.assertRaisesRegex(ValueError, "skill_ids disagrees"):
            analyze_rows([query("q", ["s1"])], [qrel("q", "s2")])

    def test_k_must_match_golden_set(self):
        row = query("q", ["s"])
        row["k"] = 2
        with self.assertRaisesRegex(ValueError, "k disagrees"):
            analyze_rows([row], [qrel("q", "s")])

    def test_duplicate_query_skill_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate skill ID"):
            analyze_rows([query("q", ["s", "s"])], [qrel("q", "s")])

    def test_empty_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            analyze_rows([], [])

    def test_nonfinite_relevance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite number"):
            analyze_rows([query("q", [])], [qrel("q", "s", float("nan"))])

    def test_report_update_is_idempotent_and_preserves_other_sections(self):
        split = analyze_rows([query("q", ["s"])], [qrel("q", "s")])
        stats = {"revision": "pinned", "splits": {"train": split, "test": split},
                 "combined": {"queries": 2, "distribution": {1: 2}}}
        original = '<div class="nav"></div><p>Preserved</p><!-- Section 6: Qrels -->'
        updated = update_report(original, stats)
        self.assertEqual(update_report(updated, stats), updated)
        self.assertEqual(updated.count(START), 1)
        self.assertIn('<p>Preserved</p>', updated)
        self.assertIn('Golden skills', updated)


if __name__ == "__main__":
    unittest.main()
