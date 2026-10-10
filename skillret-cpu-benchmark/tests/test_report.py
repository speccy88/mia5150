"""Arithmetic/contract unit fixtures are not benchmark observations."""
from pathlib import Path
import sys,tempfile,os,json,hashlib,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import build_report as br
from report_fresh import rank_metrics,exact_grid

class ReportTests(unittest.TestCase):
 def test_percentiles(self):
  self.assertEqual(br.percentile([0,10],.95),9.5);self.assertIsNone(br.percentile([],.5))
 def test_independent_metrics(self):
  r=rank_metrics(['x','a','b'],['a','b']);self.assertEqual(r['mrr5'],.5);self.assertEqual(r['recall5'],1);self.assertEqual(r['hit1'],0)
 def test_grid_requires_every_condition(self):
  run={'rows':[{'method':m,'catalog_size':s,'query_id':q} for m in ['vector','hybrid','domain_vector'] for s in br.SIZES for q in ['a','b']]}
  self.assertTrue(exact_grid(run,('vector','hybrid','domain_vector'),['a','b']));run['rows'].pop();self.assertFalse(exact_grid(run,('vector','hybrid','domain_vector'),['a','b']))
 def test_empty_aggregation_is_empty(self):self.assertEqual(br.aggregate_sql([]),[])
 def test_restoration_exception_requires_exact_bytes(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'results').mkdir();csv=root/'results/a_per_query.csv';run=root/'results/a_run.json';csv.write_bytes(b'unit fixture');run.write_text('{}');os.utime(run,(100,100));os.utime(csv,(200,200))
   self.assertFalse(br.completed_csv_is_current(root,csv,run))
   item={'original_file':'results/a_per_query.csv','original_bytes':csv.stat().st_size,'original_sha256':hashlib.sha256(csv.read_bytes()).hexdigest()};(root/'rank_restore_manifest.json').write_text(json.dumps({'files':[item]}))
   self.assertTrue(br.completed_csv_is_current(root,csv,run));csv.write_bytes(b'changed');self.assertFalse(br.completed_csv_is_current(root,csv,run))
 def test_historical_values_are_labeled(self):
  rows=br.historical_rows(br.read_json(ROOT/'docs/retained_observations.json'));self.assertEqual(len(rows),10);self.assertTrue(all('raw unavailable' in r['evidence_status'] for r in rows))
if __name__=='__main__':unittest.main()

class CommonFallbackTests(unittest.TestCase):
 def test_priority_and_all_model_gate(self):
  from report_fresh import choose_common_size
  done={n:{m:True for m in br.DECISIONS} for n in (64,48,32)}
  self.assertEqual(choose_common_size(done),64)
  done[64]['julia-1']=False
  self.assertEqual(choose_common_size(done),48)
  done[48]['laya-multilingual']=False
  self.assertEqual(choose_common_size(done),32)
  self.assertIsNone(choose_common_size(done,False))
  done[32]['verdict-151m-onnx']=False
  self.assertIsNone(choose_common_size(done))
