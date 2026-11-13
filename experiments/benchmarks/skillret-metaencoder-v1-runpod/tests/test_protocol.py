"""Run on RunPod only. These are fixture tests, not benchmark outcomes.

PYTHONPATH=src python -m unittest discover -s tests
"""
import unittest
from unittest.mock import patch
from metrics_existing import evaluate, deterministic_subset
import runner


class ProtocolTests(unittest.TestCase):
    def test_canonical_document_and_original_query(self):
        data = {
            'skills': [{'id': 'skill-a', 'name': ' Name ',
                        'description': ' Description ', 'skill_md': ' # Body\n '}],
            'queries': [{'id': 'query-a', 'query': 'Original request',
                         'labels': 'MUST NOT BE ENCODED'}],
        }
        sid, documents, qid, queries = runner.texts(data)
        self.assertEqual(documents, ['Name | Description | # Body'])
        self.assertEqual(queries, ['Original request'])
        self.assertEqual(sid, ['skill-a'])
        self.assertEqual(qid, ['query-a'])

    def test_chunking_preserves_characters(self):
        text = 'é alpha\nβ beta | gamma'
        with patch.object(runner, 'token_count', side_effect=lambda m,t,i: len(t)+2):
            pieces = runner.chunks(None, text, 'fixed instruction', 8)
        self.assertEqual(''.join(pieces), text)
        self.assertTrue(all(len(piece)+2 <= 8 for piece in pieces))
        self.assertGreater(len(pieces), 1)

    def test_no_chunk_when_within_limit(self):
        with patch.object(runner, 'token_count', return_value=8):
            self.assertEqual(runner.chunks(None, 'unchanged', 'fixed', 8), ['unchanged'])

    def test_query_denominator_retains_missing(self):
        result = evaluate({'q1': {'a': 1}, 'q2': {'b': 1}}, {'q1': ['a']})
        self.assertEqual(result['query_count'], 2)
        self.assertEqual(result['missing_query_count'], 1)
        self.assertEqual(result['aggregate']['Recall@10'], 0.5)
        self.assertEqual(result['per_query']['q2']['NDCG@10'], 0.0)

    def test_precision_and_map_denominators(self):
        scores = evaluate({'q': {'a': 1, 'b': 1}}, {'q': ['a']})['aggregate']
        self.assertEqual(scores['Precision@5'], 0.2)
        self.assertEqual(scores['MAP@5'], 0.5)
        self.assertEqual(scores['Completeness@5'], 0.0)

    def test_duplicate_ranking_ids_rejected(self):
        with self.assertRaises(ValueError):
            evaluate({'q': {'a': 1}}, {'q': ['a', 'a']})

    def test_unknown_query_rejected(self):
        with self.assertRaises(ValueError):
            evaluate({'q': {'a': 1}}, {'other': ['a']})

    def test_fixed_pilot_independent_of_input_order(self):
        ids = ['q-a', 'q-b', 'q-c', 'q-d']
        self.assertEqual(deterministic_subset(ids, 2), deterministic_subset(ids[::-1], 2))


if __name__ == '__main__':
    unittest.main()
