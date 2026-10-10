"""Weight-free contracts. No model weights, Torch imports, or inference."""
import unittest
from types import SimpleNamespace

from decision_adapters import (
    _QueryFitMixin, GLiNERAdapter, VerdictONNXAdapter, softmax, pair_state,
)


class CharacterBudget(_QueryFitMixin):
    def __init__(self, budget=20, truncate=False):
        self.max_length = budget
        self.truncate_query = truncate

    def _measure_pair(self, query, candidate):
        return 4 + len(query) + len(candidate)


class AdapterContracts(unittest.TestCase):
    def test_softmax_stability(self):
        values = softmax([10000, 9999])
        self.assertAlmostEqual(sum(values), 1.0)
        self.assertGreater(values[0], values[1])

    def test_softmax_rejects_invalid(self):
        for values, temperature in [([], 1), ([float("nan")], 1), ([1], 0)]:
            with self.assertRaises(ValueError):
                softmax(values, temperature)

    def test_exact_verdict_markers(self):
        prompt = VerdictONNXAdapter.prompt("Find flights", "Compare airline schedules")
        self.assertEqual(prompt.count("<<LABEL>>"), 3)
        self.assertEqual(prompt.count("<<SEP>>"), 1)
        self.assertIn("insufficient evidence<<SEP>>Question:", prompt)
        self.assertIn(pair_state("Find flights", "Compare airline schedules"), prompt)

    def test_lossless_unchanged(self):
        pairs = [("query", "skill")]
        prepared, audit = CharacterBudget().prepare_pairs(pairs)
        self.assertEqual(prepared, pairs)
        self.assertFalse(audit[0]["truncated"])
        self.assertTrue(audit[0]["candidate_preserved"])

    def test_overflow_requires_explicit_option(self):
        with self.assertRaisesRegex(ValueError, "truncate_query=True"):
            CharacterBudget().prepare_pairs([("x" * 100, "skill")])

    def test_query_only_prefix_is_preserved(self):
        original = "abcdefghijklmnopqrstuvwxyz"
        prepared, audit = CharacterBudget(truncate=True).prepare_pairs([(original, "skill")])
        self.assertEqual(prepared, [(original[:11], "skill")])
        self.assertEqual(audit[0]["query_chars_original"], 26)
        self.assertEqual(audit[0]["query_chars_used"], 11)
        self.assertEqual(audit[0]["tokens"], 20)
        self.assertTrue(audit[0]["query_truncated"])

    def test_candidate_cannot_be_silently_shortened(self):
        with self.assertRaisesRegex(ValueError, "candidate truncation is forbidden"):
            CharacterBudget(truncate=True).prepare_pairs([("query", "x" * 30)])

    def test_unicode_query_prefix(self):
        query = "é漢🙂" * 10
        prepared, audit = CharacterBudget(truncate=True).prepare_pairs([(query, "skill")])
        self.assertEqual(prepared[0][0], query[:11])
        self.assertTrue(audit[0]["candidate_preserved"])

    def test_gliner_uses_strict_public_collator(self):
        calls = []

        class Mask:
            shape = (1, 42)

            def __getitem__(self, index):
                self.index = index
                return self

            def sum(self):
                return 37

        def collate(rows, **kwargs):
            calls.append((rows, kwargs))
            return SimpleNamespace(attention_mask=Mask())

        adapter = GLiNERAdapter.__new__(GLiNERAdapter)
        processor = SimpleNamespace(collate_fn_inference=collate)
        adapter.classifier = SimpleNamespace(model=SimpleNamespace(processor=processor))
        adapter.compiled = SimpleNamespace(build=lambda: {"schema": "fixed"})
        self.assertEqual(adapter._measure_pair("query", "entire candidate"), 37)
        rows, kwargs = calls[0]
        self.assertEqual(rows[0][0], pair_state("query", "entire candidate"))
        self.assertEqual(kwargs, {"max_len": None, "error_policy": "raise"})


if __name__ == "__main__":
    unittest.main()
