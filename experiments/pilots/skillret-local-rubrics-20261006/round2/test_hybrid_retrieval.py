import json,tempfile,unittest,os
from pathlib import Path
from hybrid_retrieval import *
POOL=[str(i) for i in range(20)]
HEADERS={d:{'id':d,'name':'Skill '+d,'description':'Public dummy evidence'} for d in POOL}
QUERY={'id':'dummy-query','query':'dummy task'}
def response():return {'model':MODEL,'usage':{'input_tokens':100},'answers':{f's{i}':{'type':'score','score':1.5} for i in range(20)}}
class CoreTests(unittest.TestCase):
    def test_equal_scores_and_fusion_ties_preserve_dense(self):
        self.assertEqual(score_order(POOL,[1]*20),POOL)
        self.assertEqual(rrf60(POOL,POOL),POOL)
        self.assertLess(rrf60(POOL,POOL[::-1]).index('0'),rrf60(POOL,POOL[::-1]).index('19'))
    def test_missing_answers_and_nonfinite_rejected(self):
        r=response();del r['answers']['s3']
        with self.assertRaises(ValueError):parse_scores(r,POOL)
        for value in [float('nan'),float('inf'),True,4]:
            r=response();r['answers']['s3']['score']=value
            with self.assertRaises(ValueError):parse_scores(r,POOL)
    def test_candidate_pool_cannot_change(self):
        with self.assertRaises(ValueError):rrf60(POOL,POOL[:-1]+['other'])
    def test_failure_serves_baseline_but_benchmark_empty(self):
        class Failure:
            def rank(self,*args):raise RuntimeError('DO-NOT-EXPOSE')
        r=HybridRetriever(CachedRetriever({QUERY['id']:POOL}),HEADERS,Failure()).search(QUERY)
        self.assertEqual(r.served,POOL);self.assertEqual(r.strict_benchmark,[])
        self.assertEqual(r.status,'baseline_fallback');self.assertNotIn('DO-NOT-EXPOSE',str(r))
    def test_cached_replay_never_needs_transport(self):
        rows=[{'id':QUERY['id'],'candidate_ids':POOL,'status':'ok','scores':[1]*20,'input_tokens':100}]
        r=HybridRetriever(CachedRetriever({QUERY['id']:POOL}),HEADERS,CachedJev(rows)).search(QUERY)
        self.assertEqual(r.served,POOL);self.assertEqual(r.status,'ok');self.assertEqual(r.metadata['source'],'cached')
    def test_labels_not_accepted_as_query_fields(self):
        with self.assertRaises(ValueError):HybridRetriever(None,None,None).search({**QUERY,'skill_ids':['gold']})
@unittest.skipUnless(os.name=='posix','WSL locking')
class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'budget-ledger.json'
        self.path.write_text(json.dumps({'cap_usd':'4','committed_usd':'0.75','external_spend_usd':'1','runs':{}}))
    def tearDown(self):self.temp.cleanup()
    def test_shared_cap_and_no_implicit_refund(self):
        with SharedBudget(self.path,'dummy',0.1,2) as budget:
            budget.before_call('a')
            with self.assertRaises(ValueError):budget.before_call('a')
            budget.before_call('b')
            with self.assertRaises(ValueError):budget.before_call('c')
        self.assertEqual(json.loads(self.path.read_text())['committed_usd'],'0.85')
        with self.assertRaises(ValueError):
            with SharedBudget(self.path,'dummy',0.1,2):pass
        with self.assertRaises(ValueError):
            with SharedBudget(self.path,'over',3,1):pass
    def test_reservation_persisted_before_transport_and_stops_on_failure(self):
        calls=[]
        def transport(packet):
            ledger=json.loads(self.path.read_text());self.assertEqual(ledger['runs']['test']['attempt_ids'],['dummy-query'])
            calls.append(1);raise ValueError('DO-NOT-EXPOSE')
        with SharedBudget(self.path,'test',0.1,2) as budget:
            adapter=JevScoreAdapter(transport,budget);pipeline=HybridRetriever(CachedRetriever({'dummy-query':POOL,'second':POOL}),HEADERS,adapter)
            r=pipeline.search(QUERY);self.assertEqual(r.served,POOL);self.assertEqual(r.strict_benchmark,[])
            pipeline.search({'id':'second','query':'dummy'});self.assertEqual(len(calls),1)
    def test_underfunded_token_bound_rejected(self):
        with self.assertRaises(ValueError):SharedBudget(self.path,'bad',0.001,8)
    def test_review_pause_blocks_new_reservations(self):
        ledger=json.loads(self.path.read_text());ledger['paid_tests_paused']=True;self.path.write_text(json.dumps(ledger))
        with self.assertRaises(ValueError):
            with SharedBudget(self.path,'paused',0.1,1):pass
        self.assertEqual(json.loads(self.path.read_text())['committed_usd'],'0.75')
