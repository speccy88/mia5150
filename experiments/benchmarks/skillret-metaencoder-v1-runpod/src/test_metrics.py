"""Run on RunPod only, alongside runner.py and metrics_existing.py."""
import unittest
from metrics_existing import evaluate
class MetricTests(unittest.TestCase):
    def test_full_denominator(self):
        x=evaluate({'a':{'x':1},'b':{'y':1}},{'a':['x']})
        self.assertEqual(x['aggregate']['Recall@10'],.5)
        self.assertEqual(x['aggregate']['MRR@10'],.5)
        self.assertEqual(x['aggregate']['Precision@5'],.1)
        self.assertEqual(x['missing_query_count'],1)
    def test_multi(self):
        x=evaluate({'a':{'x':1,'y':1}},{'a':['x']})['aggregate']
        self.assertEqual(x['Completeness@10'],0)
        self.assertEqual(x['MAP@5'],.5)
if __name__=='__main__':unittest.main()
