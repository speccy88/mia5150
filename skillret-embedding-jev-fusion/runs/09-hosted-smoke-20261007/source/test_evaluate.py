"""Offline tests, including direct pytrec_eval parity when installed."""

import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

import evaluate as ev

try:
    import pytrec_eval
except ImportError:
    pytrec_eval = None


class MetricTests(unittest.TestCase):
    def test_graded_linear_ndcg_and_binary_metrics(self):
        result = ev.evaluate({"q": {"a": 2, "b": 1, "c": 1}}, {"q": ["b", "x", "a"]})
        row = result["per_query"]["q"]
        dcg = 1 + 2 / math.log2(4)
        idcg = 2 + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(row["NDCG@5"], dcg / idcg)
        self.assertAlmostEqual(row["Recall@5"], 2 / 3)
        self.assertAlmostEqual(row["MAP@5"], (1 + 2 / 3) / 3)
        self.assertEqual(row["Precision@5"], 2 / 5)
        self.assertEqual(row["Completeness@5"], 0)
        self.assertEqual(row["MRR@5"], 1)

    def test_map_denominator_is_all_relevant_even_above_cutoff(self):
        labels = {"q": {str(i): 1 for i in range(8)}}
        row = ev.evaluate(labels, {"q": [str(i) for i in range(5)]})["per_query"]["q"]
        self.assertEqual(row["MAP@5"], 5 / 8)
        self.assertEqual(row["Precision@10"], 0.5)

    def test_missing_and_empty_queries_use_fixed_denominator(self):
        labels = {"a": {"r": 1}, "b": {"s": 1}, "c": {"t": 1}}
        result = ev.evaluate(labels, {"a": ["r"], "b": []})
        self.assertEqual(result["aggregate"]["Recall@5"], 1 / 3)
        self.assertEqual(result["query_count"], 3)
        self.assertEqual(result["missing_query_count"], 1)
        self.assertEqual(result["empty_rate"], 2 / 3)

    def test_no_relevant_documents_scores_zero(self):
        row = ev.evaluate({"q": {"a": 0, "b": -1}}, {"q": ["a", "b"]})["per_query"]["q"]
        self.assertTrue(all(value == 0 for value in row.values()))

    def test_candidate_coverage_macro_is_not_pair_micro(self):
        result = ev.evaluate({"a": {"x": 1}, "b": {"u": 1, "v": 1, "w": 1}},
                             {"a": ["x"], "b": []})
        coverage = result["candidate_coverage"]["20"]
        self.assertEqual(coverage["macro_recall"], 0.5)
        self.assertEqual(coverage["relevant_pair_micro"], 0.25)
        self.assertEqual(coverage["relevant_pairs_found"], 1)
        self.assertEqual(coverage["relevant_pairs_total"], 4)

    def test_mrr_cutoffs(self):
        row = ev.evaluate({"q": {"hit": 2}}, {"q": [str(i) for i in range(7)] + ["hit"]})["per_query"]["q"]
        self.assertEqual(row["MRR@5"], 0)
        self.assertEqual(row["MRR@10"], 1 / 8)

    def test_duplicate_or_unknown_ids_rejected(self):
        for rankings in ({"q": ["a", "a"]}, {"extra": ["a"]}, {"q": [1, "1"]}):
            with self.assertRaises(ValueError):
                ev.evaluate({"q": {"a": 1}}, rankings)
        with self.assertRaises(ValueError):
            ev.evaluate({}, {})
        with self.assertRaises(ValueError):
            ev.evaluate({1: {"a": 1}, "1": {"a": 1}}, {})

    @unittest.skipUnless(pytrec_eval, "pytrec_eval is not installed")
    def test_randomized_pytrec_eval_parity(self):
        rng = random.Random(817)
        documents = [f"d{i}" for i in range(70)]
        labels, rankings = {}, {}
        for index in range(100):
            qid = f"q{index}"
            labels[qid] = {sid: rng.choice([0, 1, 2, 3]) for sid in rng.sample(documents, rng.randrange(1, 30))}
            rankings[qid] = rng.sample(documents, rng.randrange(0, 65))
        observed = ev.evaluate(labels, rankings)
        evaluator = pytrec_eval.RelevanceEvaluator(labels, {"ndcg_cut.5,10,15", "map_cut.5,10,15", "recall.5,10,15", "P.5,10,15"})
        reference = evaluator.evaluate({qid: {sid: float(len(rank) - index) for index, sid in enumerate(rank)}
                                        for qid, rank in rankings.items()})
        names = {"NDCG": "ndcg_cut", "Recall": "recall", "MAP": "map_cut", "Precision": "P"}
        for qid, row in observed["per_query"].items():
            for k in ev.CUTOFFS:
                for name, trec_name in names.items():
                    with self.subTest(qid=qid, metric=name, k=k):
                        self.assertAlmostEqual(row[f"{name}@{k}"], reference.get(qid, {}).get(f"{trec_name}_{k}", 0.0), places=13)


class StudyTests(unittest.TestCase):
    def test_hash_subsets_are_order_invariant_and_nested(self):
        ids = [f"q{i}" for i in range(240)]
        expected = sorted(ids, key=lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())
        result = ev.build_study_subsets(list(reversed(ids)))
        self.assertEqual(result["study_200"], expected[:200])
        self.assertEqual(result["smoke_10"], expected[:10])
        self.assertEqual(ev.deterministic_subset(["é", "ø", "漢字"], 0), [])

    def test_jev_ties_confidence_then_dense_rank(self):
        dense = ["a", "b", "c", "d"]
        scores = {"a": 3, "b": 4, "c": 3, "d": 3}
        confidence = {"a": 0.5, "b": 0.1, "c": 0.8, "d": 0.8}
        self.assertEqual(ev.rank_jev(dense, scores, confidence), ["b", "c", "d", "a"])
        with self.assertRaises(ValueError):
            ev.rank_jev(dense, scores, {**confidence, "a": float("nan")})
        with self.assertRaises(ValueError):
            ev.rank_jev(dense, {"a": 2}, confidence)

    def test_fixed_rrf_pool_and_dense_tie_break(self):
        self.assertEqual(ev.rrf60(["a", "b", "c"], ["c", "b", "a"]), ["a", "c", "b"])
        with self.assertRaises(ValueError):
            ev.rrf60(["a", "b"], ["b", "c"])
        with self.assertRaises(ValueError):
            ev.rrf60(["a", "b"], ["a", "a", "b"])

    def test_bootstrap_identical_results_and_strict_pairs(self):
        a = ev.evaluate({"q1": {"a": 1}, "q2": {"b": 1}}, {"q1": ["a"]})
        result = ev.paired_bootstrap(a, a, n_resamples=80, seed=19)
        for metric in result["metrics"].values():
            self.assertEqual((metric["delta"], metric["ci_low"], metric["ci_high"]), (0, 0, 0))
        different = ev.evaluate({"q3": {"b": 1}}, {})
        with self.assertRaises(ValueError):
            ev.paired_bootstrap(a, different, n_resamples=5)

    def test_bootstrap_reproducibility_and_delta_direction(self):
        labels = {f"q{i}": {"hit": 1} for i in range(20)}
        a = ev.evaluate(labels, {})
        c = ev.evaluate(labels, {qid: ["hit"] for qid in list(labels)[:10]})
        first = ev.paired_bootstrap(a, c, n_resamples=300, seed=31415)
        second = ev.paired_bootstrap(a, c, n_resamples=300, seed=31415)
        self.assertEqual(first, second)
        metric = first["metrics"]["Recall@5"]
        self.assertEqual(metric["delta"], 0.5)
        self.assertLess(metric["ci_low"], 0.5)
        self.assertGreater(metric["ci_high"], 0.5)

    def test_cli_round_trip_and_explicit_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qrels = root / "qrels.jsonl"
            qrels.write_text('{"query_id":"a","skill_id":"x","relevance":2}\n'
                             '{"query_id":"b","skill_id":"y","relevance":1}\n', encoding="utf-8")
            rankings, treatment, subset, output = (root / name for name in ("a.json", "c.json", "ids.json", "out.json"))
            rankings.write_text(json.dumps({"rankings": {"a": ["z"], "b": ["y"]}}), encoding="utf-8")
            treatment.write_text(json.dumps({"rankings": {"a": ["x"], "b": ["y"]}}), encoding="utf-8")
            subset.write_text('["a"]', encoding="utf-8")
            ev.main(["--qrels", str(qrels), "--rankings", str(rankings), "--compare", str(treatment),
                     "--query-ids", str(subset), "--output", str(output), "--bootstrap-resamples", "20"])
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["query_count"], 1)
            self.assertEqual(result["comparison"]["paired_bootstrap"]["metrics"]["NDCG@10"]["delta"], 1)


if __name__ == "__main__":
    unittest.main()
