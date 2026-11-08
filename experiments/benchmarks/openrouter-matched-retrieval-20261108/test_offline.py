import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('offline', HERE/'run_offline.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class OfflineTest(unittest.TestCase):
    def test_metrics_cutoffs_and_order(self):
        self.assertEqual(m.single_target_metrics(['a','b','c'],['b'],1)['mrr'],0)
        self.assertEqual(m.single_target_metrics(['a','b','c'],['b'],5)['mrr'],.5)
        self.assertEqual(m.single_target_metrics(['a','b','c'],['b'],5)['recall'],1)
        self.assertEqual(m.single_target_metrics(['a','b','c'],['b'],5)['precision'],.2)
        self.assertEqual(m.single_target_metrics([],['b'],5)['completeness'],0)
        self.assertEqual(m.single_target_metrics(['a','b'],['a','c'],5)['recall'],1)

    def test_name_equivalence_is_one_target(self):
        result=m.single_target_metrics(['duplicate','gold'],['duplicate','gold'],5)
        self.assertEqual(result['recall'],1)
        self.assertEqual(result['precision'],.2)
        self.assertEqual(result['mrr'],1)
        self.assertEqual(result['ndcg'],1)
        self.assertEqual(m.single_target_metrics(['x','duplicate','gold'],['duplicate','gold'],5)['mrr'],.5)

    def test_bm25_and_ties(self):
        bm = m.BM25(['alpha','beta beta','gamma'])
        self.assertEqual(m.rank(bm.scores('beta'),['c','b','a'])[0],'b')
        self.assertEqual(m.rank(bm.scores('missing'),['c','b','a']),['a','b','c'])
        self.assertTrue(np.isfinite(bm.scores('beta')).all())

    def test_rrf_fixed_depth(self):
        self.assertEqual(m.rrf(['a','b'],['b','a']),['a','b'])
        self.assertNotIn('b',m.rrf(['a','b'],['a','b'],depth=1))

    def test_original_ast_matches_independent_builder(self):
        skills = [{'id':str(i),'name':f'skill{i}'} for i in range(75)]
        qs = [{'gold_skill':'skill10'}, {'gold_skill':'skill3'}]
        source=m.ROOT/'benchmark_openrouter/benchmark_models_testplan_openrouter.py'
        for n in [10,20,50,75]:
            a=m.source_builder(source,skills,qs,n)
            b=m.literal_builder(skills,qs,n)
            self.assertEqual(a,b)
            self.assertEqual(len(a),n)
            self.assertTrue({'skill3','skill10'} <= {x['name'] for x in a})
            self.assertEqual(a,m.source_builder(source,skills,qs,n))

    def test_source_order_changes_library(self):
        skills = [{'id':str(i),'name':f'skill{i}'} for i in range(75)]
        qs = [{'gold_skill':'skill10'}, {'gold_skill':'skill3'}]
        self.assertNotEqual(m.literal_builder(skills,qs,50),m.literal_builder(skills[::-1],qs,50))

    def test_metadata_view_cap(self):
        self.assertEqual(m.text_view({'name':' A ','description':'word\nnext'}),'A | word next')
        self.assertEqual(len(m.text_view({'name':'A','description':'x'*2000})),1028)

    def test_hash_guard_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('wrong')
            with self.assertRaises(ValueError):m.require_hash(p,m.CORPUS_SHA)

    def test_reranker_gate(self):
        import sys
        sys.path.insert(0,str(HERE))
        from evaluate_saved_reranker import validate_response
        plan={'query':'q','catalog_sha256':'a','candidate_order_sha256':'b',
              'candidate_text_sha256':'c','candidate_ids':['x','y']}
        response={**plan,'model_id':'Jev','schema_version':'v1','status':'ok',
                  'ordered_ids':['y','x'],'api_latency_seconds':1.,'reported_cost_usd':None}
        self.assertEqual(validate_response(response,plan,'Jev','v1'),['y','x'])
        for update in [{'query':'different'},{'ordered_ids':['outside']},
                       {'ordered_ids':['x','x']},{'status':'failed'},
                       {'reported_cost_usd':-1},{'reported_cost_usd':float('nan')},
                       {'model_id':'different'}]:
            with self.assertRaises(ValueError):validate_response({**response,**update},plan,'Jev','v1')

    def test_unknown_reranker_outcome_has_null_metrics(self):
        import sys
        sys.path.insert(0,str(HERE))
        from evaluate_saved_reranker import evaluate
        plan={'query':'q','catalog_sha256':'a','candidate_order_sha256':'b','candidate_text_sha256':'c',
              'candidate_ids':['x'],'source_method':'bm25','size':50,'query_index':0}
        raw={'method':'bm25','size':50,'query_index':0,'gold_id':'x','name_equivalent_ids':['x']}
        record={**plan,'model_id':'Jev','schema_version':'v1','status':'unknown','ordered_ids':[],
                'api_latency_seconds':None,'reported_cost_usd':None}
        quality,attempts,coverage=evaluate([record],[plan],[raw],'Jev','v1')
        self.assertTrue(all(row['recall'] is None for row in quality))
        self.assertEqual(coverage['unknown'],1)
        self.assertEqual(coverage['unknown_cost_records'],1)

    def test_saved_results_rescore(self):
        p=HERE/'results/per_case.jsonl'
        if not p.exists():self.skipTest('Full run not yet present')
        with p.open() as handle:
            rows=[json.loads(x) for x in handle]
        self.assertEqual(len(rows),560)
        keys={(r['method'],r['size'],r['query_index']) for r in rows}
        self.assertEqual(len(keys),560)
        catalogs={c['size']:c for c in json.loads((HERE/'results/catalogs.json').read_text())}
        for r in rows:
            self.assertEqual(len(set(r['ranking20'])),len(r['ranking20']))
            self.assertTrue(set(r['ranking20'])<=set(catalogs[r['size']]['ordered_ids']))
            self.assertEqual(r['catalog_sha256'],catalogs[r['size']]['ordered_sha256'])
            self.assertIsNone(r['final_selection'])
            self.assertEqual(r['candidate_recall20'],float(r['gold_id'] in r['ranking20']))
        import csv
        with (HERE/'results/summary.csv').open() as handle:
            summary=list(csv.DictReader(handle))
        self.assertEqual(len(summary),112)
        for cell in summary:
            sample=[r for r in rows if r['method']==cell['method'] and r['size']==int(cell['size'])]
            self.assertEqual(len(sample),20)
            for key in ['recall','hit','precision','f1','mrr','map','ndcg','completeness']:
                value=sum(m.single_target_metrics(r['ranking20'], [r['gold_id']] if cell['label_contract']=='uuid_single_target' else r['name_equivalent_ids'],int(cell['k']))[key] for r in sample)/20
                self.assertAlmostEqual(float(cell[key]),value,places=12)
        plans=json.loads((HERE/'results/jev-rerank-plan.json').read_text())
        for request in plans['requests']:
            self.assertFalse(any('gold' in k for k in request))
        self.assertEqual(plans['network_calls'],0)

if __name__ == '__main__':unittest.main()
