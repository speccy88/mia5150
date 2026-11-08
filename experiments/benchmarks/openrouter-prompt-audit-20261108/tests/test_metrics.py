import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prompt_audit', STUDY / 'audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class RankingMetricTests(unittest.TestCase):
    def test_single_gold_rank_two(self):
        scores = audit.score_ranking(['x', 'g'], ['g'])
        self.assertEqual(scores['recall@1'], 0)
        self.assertEqual(scores['recall@5'], 1)
        self.assertEqual(scores['precision@5'], .2)
        self.assertEqual(scores['mrr@5'], .5)
        self.assertEqual(scores['map@5'], .5)
        self.assertAlmostEqual(scores['ndcg@5'], 1/math.log2(3))
        self.assertAlmostEqual(scores['f1@5'], 1/3)
        self.assertEqual(scores['exact_set_match'], 0)

    def test_fixed_k_short_list(self):
        scores = audit.score_ranking(['g'], ['g'])
        self.assertEqual(scores['top1_accuracy'], 1)
        self.assertEqual(scores['exact_set_match'], 1)
        self.assertEqual(scores['precision@10'], .1)
        self.assertAlmostEqual(scores['f1@10'], 2/11)

    def test_invalid_slots_not_removed(self):
        scores = audit.score_ranking(['invalid', 'g'], ['g'], valid_ids={'g'})
        self.assertEqual(scores['mrr@3'], .5)
        self.assertEqual(scores['top1_accuracy'], 0)
        self.assertEqual(scores['output_slot_validity'], .5)
        self.assertEqual(scores['exact_set_match'], 0)
        self.assertEqual(audit.score_ranking([None,'g'],['g'])['mrr@3'],.5)

    def test_duplicate_slots_not_removed_or_credited_twice(self):
        scores = audit.score_ranking(['a','a','b'], ['a','b'])
        self.assertEqual(scores['recall@3'], 1)
        self.assertEqual(scores['precision@3'], 2/3)
        self.assertEqual(scores['duplicate_slot_rate'], 1/3)
        self.assertAlmostEqual(scores['map@3'], (1+2/3)/2)
        self.assertEqual(scores['exact_set_match'], 1)

    def test_multi_gold(self):
        scores = audit.score_ranking(['a','x','b','c'], ['a','b','c'])
        self.assertAlmostEqual(scores['recall@3'], 2/3)
        self.assertEqual(scores['hit@3'], 1)
        self.assertEqual(scores['completeness@3'], 0)
        self.assertAlmostEqual(scores['map@3'], (1+2/3)/3)
        self.assertAlmostEqual(scores['ndcg@3'], (1+1/math.log2(4))/(1+1/math.log2(3)+1/math.log2(4)))
        self.assertEqual(scores['completeness@5'], 1)

    def test_empty_completed_output_is_zero_not_missing(self):
        scores = audit.score_ranking([], ['g'])
        self.assertEqual(scores['recall@10'], 0)
        self.assertEqual(scores['exact_set_match'], 0)
        self.assertIsNone(scores['output_slot_validity'])

    def test_empty_gold_rejected(self):
        with self.assertRaises(ValueError): audit.score_ranking(['x'], [])

    def test_missing_and_excluded_have_no_scores(self):
        rows, _, _, _, _ = audit.load_rows()
        for row in rows:
            if row['status'] != 'completed':
                self.assertTrue(all(s is None for s in row['scores'].values()))
                self.assertIsNone(row['latency_s'])
                self.assertIsNone(row['estimated_cost_usd'])
                self.assertIsNone(row['actual_cost_usd'])

    def test_insertion_bound_does_not_promote_gold(self):
        for n in range(1, 9):
            upper = audit.score_ranking(['x', 'g'], ['g'])
            lower = audit.score_ranking([None]*n+['x','g'], ['g'])
            for name in audit.METRICS:
                self.assertLessEqual(lower[name], upper[name])


class BootstrapTests(unittest.TestCase):
    def test_deterministic(self):
        a = audit.cluster_bootstrap([0,1,0,1], ['a','a','b','b'], replicates=500)
        b = audit.cluster_bootstrap([0,1,0,1], ['a','a','b','b'], replicates=500)
        np.testing.assert_array_equal(a,b)

    def test_retains_whole_query_cluster(self):
        # Every query has mean .5 across sizes, so clustered CI must collapse.
        low, high = audit.cluster_bootstrap([0,1,0,1], ['a','a','b','b'], replicates=500)
        self.assertEqual(low[0], .5)
        self.assertEqual(high[0], .5)

    def test_identical_paired_differences_are_zero(self):
        low, high = audit.cluster_bootstrap([0]*6, ['a','b','c']*2, replicates=500)
        self.assertEqual(low[0],0)
        self.assertEqual(high[0],0)

    def test_empty_returns_na(self):
        self.assertEqual(audit.cluster_bootstrap([],[]),(None,None))

    def test_mismatched_ids_rejected(self):
        with self.assertRaises(ValueError): audit.cluster_bootstrap([0,1],['a'])


class FrozenEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.cfg, cls.queries, cls.registry, cls.issues = audit.load_rows()

    def test_source_hash(self):
        self.assertEqual(audit.sha256(audit.RAW/'details.csv'), '16a27b1af18d212ca1a0f89b6e5c519ac6ecb97a0d0fd7cf37cbe7c2146b7ab0')

    def test_duplicate_evaluation_key_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            for name in ('run_config.json','test_queries.json'):
                (path/name).write_bytes((audit.RAW/name).read_bytes())
            with (audit.RAW/'details.csv').open(newline='') as fh:
                reader=csv.DictReader(fh)
                first=next(reader)
                fields=reader.fieldnames
            with (path/'details.csv').open('w',newline='') as fh:
                writer=csv.DictWriter(fh,fieldnames=fields)
                writer.writeheader();writer.writerow(first);writer.writerow(first)
            with self.assertRaisesRegex(ValueError,'Duplicate evaluation key'):
                audit.load_rows(path)

    def test_frozen_input_manifest(self):
        expected=json.loads((STUDY/'source/input_manifest.json').read_text())
        for name, digest in expected.items():
            self.assertEqual(audit.sha256(audit.ROOT/name),digest)

    def test_independent_counts(self):
        self.assertEqual(len(self.rows),1260)
        self.assertEqual(sum(r['source_line'] is not None for r in self.rows),1224)
        self.assertEqual(sum(r['status']=='completed' for r in self.rows),1184)
        self.assertEqual(sum(r['status']=='context_excluded_not_attempted' for r in self.rows),40)
        self.assertEqual(sum(r['status']=='unattempted_missing' for r in self.rows),36)
        self.assertEqual(sum(r['status']=='completed' and r['n_invalid']>0 for r in self.rows),72)
        self.assertEqual(sum(r['status']=='completed' and not r['ranking'] for r in self.rows),6)

    def test_nine_models_and_cost(self):
        self.assertEqual(len(self.cfg['models']),9)
        self.assertAlmostEqual(sum(r['estimated_cost_usd'] or 0 for r in self.rows),35.140929,places=6)
        self.assertTrue(all(r['actual_cost_usd'] is None for r in self.rows))
        self.assertEqual(sum(r['input_tokens'] or 0 for r in self.rows),67394826)
        self.assertEqual(sum(r['output_tokens'] or 0 for r in self.rows),476745)

    def test_expected_parser_issues(self):
        self.assertEqual(self.issues['valid_list_count_mismatch'],1)
        self.assertEqual(self.issues['duplicate_in_saved_ranking'],0)
        self.assertFalse(any(k.startswith('legacy_metric_difference') for k in self.issues))
        self.assertEqual(self.issues['recorded_cost_recompute_mismatch'],0)

    def test_single_gold_identities_every_row(self):
        for row in self.rows:
            for scores in row['scores'].values():
                if scores is None: continue
                for k in audit.K_VALUES:
                    self.assertEqual(scores[f'recall@{k}'],scores[f'hit@{k}'])
                    self.assertEqual(scores[f'recall@{k}'],scores[f'completeness@{k}'])
                    self.assertEqual(scores[f'map@{k}'],scores[f'mrr@{k}'])
                    self.assertAlmostEqual(scores[f'precision@{k}'],scores[f'recall@{k}']/k)
                    self.assertAlmostEqual(scores[f'f1@{k}'],2*scores[f'recall@{k}']/(k+1))

    def test_empty_group_is_missing_not_zero(self):
        s=audit.summarize([], 'retained_ranking',replicates=100)
        self.assertEqual(s['n_scored'],0)
        self.assertIsNone(s['recall@5'])
        self.assertIsNone(s['estimated_cost_usd'])
        self.assertIsNone(s['latency_p95_s'])


if __name__=='__main__': unittest.main()
