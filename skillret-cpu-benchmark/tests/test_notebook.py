"""Synthetic analysis fixtures only; no actual result scan or inference."""
import csv, importlib.util, json, os, tempfile, unittest
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('notebook_builder',ROOT/'src/build_notebook.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class MetricTests(unittest.TestCase):
 def test_all_nine_metrics_and_candidate_depth(self):
  m=b.metric_values(['x','a','b','y','z','c'],{'a','b','c'})
  self.assertEqual(set(m),set(b.METRICS));self.assertEqual(m['mrr5'],.5);self.assertEqual(m['recall5'],2/3)
  self.assertEqual(m['complete5'],0);self.assertEqual(m['candidate_recall'],1)
 def test_recall_differs_from_hit(self):
  m=b.metric_values(['a','b'],{'a','b'})
  self.assertEqual(m['recall1'],.5);self.assertEqual(m['hit1'],1);self.assertEqual(m['ndcg5'],1)
 def test_invalid_gold(self):
  with self.assertRaises(ValueError):b.metric_values(['a'],set())
 def test_bootstrap_deterministic(self):
  a=b.bootstrap([.25,.25,.25],100)
  self.assertEqual(a['ci95_low'],.25);self.assertEqual(a['ci95_high'],.25)
  self.assertEqual(b.bootstrap([-.1,.2,.8],100),b.bootstrap([-.1,.2,.8],100))

class CohortTests(unittest.TestCase):
 def setUp(self):
  self.context=dict(target_count=4,checkpoint_count=2,frozen=['q1','q2','q3','q4'],sizes=[10,100,1000,6006])
  self.retrieval={};self.decisions={}
  for model in b.RETRIEVAL_MODELS:
   methods=['bm25'] if model=='bm25' else ['vector','hybrid','domain_vector']
   self.retrieval[model]=pd.DataFrame([dict(model=model,method=m,catalog_size=n,query_id=q) for m in methods for n in self.context['sizes'] for q in self.context['frozen']])
  for count in [2,4]:
   for model in b.DECISION_MODELS:
    self.decisions[count,model]=pd.DataFrame([dict(model=model,method='hybrid_rerank',catalog_size=n,query_id=q) for n in self.context['sizes'] for q in self.context['frozen'][:count]])
 def test_largest_common(self):self.assertEqual(b.choose_common(self.context,self.retrieval,self.decisions)[0],4)
 def test_one_missing_larger_falls_back(self):
  self.decisions[4,'laya-multilingual']=self.decisions[4,'laya-multilingual'].iloc[:-1]
  self.assertEqual(b.choose_common(self.context,self.retrieval,self.decisions)[0],2)
 def test_missing_retriever_blocks_primary(self):
  self.retrieval['e5-small']=pd.DataFrame()
  self.assertIsNone(b.choose_common(self.context,self.retrieval,self.decisions)[0])
 def test_no_common_never_mixes(self):
  self.decisions[4,'laya-multilingual']=pd.DataFrame();self.decisions[2,'julia-1']=pd.DataFrame()
  self.assertIsNone(b.choose_common(self.context,self.retrieval,self.decisions)[0])
 def test_duplicate_grid_rejected(self):
  f=self.decisions[4,'julia-1'];f=pd.concat([f,f.iloc[:1]],ignore_index=True)
  self.assertFalse(b.exact_grid(f,'julia-1',['hybrid_rerank'],self.context['sizes'],self.context['frozen']))

class RawTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);(self.root/'results').mkdir()
  self.path=self.root/'results/model_per_query.csv'
  self.context=dict(gold={'q1':{'a','b'}},catalog={'a','b','x'},sizes=[10],baselines={('q1',10):('a','x','b')},hashes={})
  self.row=dict(model='model',method='hybrid_rerank',catalog_size=10,query_id='q1',gold_count=2,ranked_skill_ids=json.dumps(['x','a','b']),shortlist_mrr5=1,**b.metric_values(['x','a','b'],{'a','b'}))
 def tearDown(self):self.temp.cleanup()
 def write(self):
  with self.path.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(self.row));w.writeheader();w.writerow(self.row)
  receipt=self.path.with_name('model_run.json');receipt.write_text('{"status":"completed"}')
  t=self.path.stat().st_mtime_ns+1_000_000;os.utime(receipt,ns=(t,t));return receipt
 def test_metric_and_baseline_recompute(self):
  self.write();frame,_,_=b.read_quality(self.root,self.path,self.context,'unit')
  self.assertEqual(frame.mrr5_delta.iloc[0],-.5);self.assertEqual(frame.mrr5_headroom.iloc[0],0)
 def test_corrupt_candidate_metric(self):
  self.row['candidate_recall']=.5;self.write()
  with self.assertRaisesRegex(ValueError,'candidate_recall'):b.read_quality(self.root,self.path,self.context,'unit')
 def test_changed_shortlist(self):
  self.row.update(ranked_skill_ids=json.dumps(['a','b']),**b.metric_values(['a','b'],{'a','b'}));self.write()
  with self.assertRaisesRegex(ValueError,'shortlist differs'):b.read_quality(self.root,self.path,self.context,'unit')
 def test_growing_csv_rejected(self):
  receipt=self.write();t=receipt.stat().st_mtime_ns+1_000_000;os.utime(self.path,ns=(t,t))
  self.assertIsNone(b.completed_result(self.root,self.path,{})[0])
 def test_jsonl_unicode_line_separator_is_not_new_record(self):
  p=self.root/'text.jsonl';p.write_text(json.dumps({'text':'first\u2028second'},ensure_ascii=False)+'\n')
  self.assertEqual(list(b.jsonl_rows(p)),[{'text':'first\u2028second'}])

class NotebookTests(unittest.TestCase):
 def test_schema_and_code_syntax(self):
  n=b.make_notebook();b.validate_notebook(n)
  s='\n'.join(c['source'] for c in n['cells'])
  self.assertNotIn('import common',s);self.assertNotIn('import embeddings',s);self.assertIn('retrieval64',s)
 def test_timing_exact_pairs(self):
  f=pd.DataFrame([dict(query_id=q,repeat=r) for q in ['q1','q2'] for r in range(2)])
  self.assertTrue(b.exact_timing(f,['q1','q2'],2));self.assertFalse(b.exact_timing(f.iloc[:-1],['q1','q2'],2))
 def test_executor_order_and_honesty(self):
  n=dict(metadata={},cells=[b.markdown('title'),b.markdown('methods'),b.code("context={'hashes':{}}\nevidence={'selected':None}\nsummary_text='Synthetic execution check'\nprint('before')\ndisplay(7)\nprint('after')")])
  with tempfile.TemporaryDirectory() as d:ns=b.execute_notebook(n,Path(d))
  self.assertEqual([o['output_type'] for o in n['cells'][2]['outputs']],['stream','display_data','stream'])
  self.assertIn('No Jupyter kernel was started',n['metadata']['execution_note'])

if __name__=='__main__':unittest.main()

class Common48Tests(unittest.TestCase):
 def make_panel(self):
  context=dict(target_count=64,checkpoint_count=32,intermediate_counts=[48],frozen=[f'q{i:03d}' for i in range(64)],sizes=[10,100,1000,6006])
  retrieval={};decisions={}
  for model in b.RETRIEVAL_MODELS:
   methods=['bm25'] if model=='bm25' else ['vector','hybrid','domain_vector']
   retrieval[model]=pd.DataFrame([dict(model=model,method=m,catalog_size=n,query_id=q) for m in methods for n in context['sizes'] for q in context['frozen']])
  for count in [32,48,64]:
   for model in b.DECISION_MODELS:
    decisions[count,model]=pd.DataFrame([dict(model=model,method='hybrid_rerank',catalog_size=n,query_id=q) for n in context['sizes'] for q in context['frozen'][:count]])
  return context,retrieval,decisions
 def test_prefer64_when_all_complete(self):
  c,r,d=self.make_panel();self.assertEqual(b.choose_common(c,r,d)[0],64)
 def test_choose48_when64_missing_one_model(self):
  c,r,d=self.make_panel();d[64,'laya-multilingual']=pd.DataFrame();self.assertEqual(b.choose_common(c,r,d)[0],48)
 def test_choose32_when48_grid_not_exact(self):
  c,r,d=self.make_panel();d[64,'laya-multilingual']=pd.DataFrame();d[48,'julia-1']=d[48,'julia-1'].iloc[:-1]
  self.assertEqual(b.choose_common(c,r,d)[0],32)
 def test_wrong48_ids_cannot_substitute_for_prefix(self):
  c,r,d=self.make_panel();d[64,'laya-multilingual']=pd.DataFrame();d[48,'julia-1'].loc[0,'query_id']='q063'
  self.assertEqual(b.choose_common(c,r,d)[0],32)
