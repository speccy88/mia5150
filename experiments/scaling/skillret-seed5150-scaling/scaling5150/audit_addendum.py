"""Post-run source/provenance audit. CPU-only; no inference or networking."""
import hashlib,importlib.util,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'scaling5150'
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=read(HERE/'manifest.json');gold=read(HERE/'evaluator-qrels.json');ids=list(read(HERE/'documents.json'));source=HERE/'reviewed-source/reference_metrics.py'
 # This exact pure-stdlib file was read/reviewed before this offline import.
 spec=importlib.util.spec_from_file_location('reviewed_reference_metrics',source);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
 membership=0;order_different=0
 for c in m['cases']:
  catalog=ref.make_nested_catalogs(ids,gold[c['query_id']],[c['size']],c['query_id'],seed=5150)[c['size']];assert set(catalog)==set(c['skill_ids']);membership+=1;order_different+=list(catalog)!=c['skill_ids']
 rows=read(HERE/'retrieval-full/results.json')['rows']
 comparisons=0
 from harness import evaluate
 for r in rows:
  q=r['query_id'];ev=evaluate({q:gold[q]},{q:r['ranking']})['aggregate']
  for k in [5,10]:
   v=ref.retrieval_metrics(r['ranking'],gold[q],k)
   for old,new in [('recall','Recall'),('precision','Precision'),('ndcg','NDCG'),('completeness','Completeness')]:assert abs(v[f'{old}_at_{k}']-ev[f'{new}@{k}'])<1e-12;comparisons+=1
   if k==5:assert abs(v['mrr_at_5']-ev['MRR@5'])<1e-12;comparisons+=1
 models=[];work=[]
 for p in sorted(HERE.glob('index-*/manifest.json')):
  x=read(p);role=x['role'];model_path=ROOT/'models'/role/'manifest.json'
  if model_path.exists():
   mm=read(model_path);models.append({'role':role,'repository':mm['repository'],'revision':mm['revision'],'manifest_sha256':digest(model_path),'weight_sha256':mm.get('files',{}).get('model.safetensors',{}).get('sha256')})
  else:
   mp=HERE/'model-provenance'/('eg2-model-manifest.json' if role.startswith('eg2') else 'jina-model-manifest.json');mm=read(mp);first=mm[0];models.append({'role':role,'repository':first['url'].split('huggingface.co/')[1].split('/resolve/')[0],'revision':first['revision'],'manifest_sha256':digest(mp),'weight_sha256':next((t['sha256'] for t in mm if t['file']=='model.safetensors'),None)})
  if 'new_document_forwards' in x:work.append({'role':role,'stage':'full_cached_refresh','new_docs':x['new_document_forwards'],'new_queries':x['new_query_forwards'],'reused_docs':x['reused_document_vectors'],'reused_queries':x['reused_query_vectors'],'startup_seconds':x['startup_seconds'],'document_seconds':sum(x['changed_stats'].get('batch_seconds',[])),'truncated':x['changed_stats'].get('truncated_count')})
  else:
   for stage,prefix in [('pilot','embedding-pilot-'),('full','index-')]:
    z=read(HERE/(prefix+role)/'manifest.json');calls=[json.loads(t) for t in (HERE/(prefix+role)/'calls.jsonl').read_text().splitlines()];work.append({'role':role,'stage':stage,'new_docs':sum(not c['query'] and c['id']!='warmup' for c in calls),'new_queries':sum(c['query'] for c in calls),'warmup_forwards':sum(c['id']=='warmup' for c in calls),'reused_docs':z['reused_document_vectors'],'reused_queries':z['reused_query_vectors'],'document_seconds':z['document_forward_seconds'],'query_seconds_known':sum(t for t in z['query_seconds'] if t is not None),'startup_seconds':z['startup_seconds'],'truncated':z['truncated_new_inputs']})
 result={'repository_review':'speccy88/mia5150 at041f2ce read after experimental execution; no exact-proposal-reproduction claim','source_git_blobs':{'project_proposal.tex':'4df0f466cbf6c928e4f653887b47469341a9b72f','reference_metrics.py':'f3caea1b642dc36d89567830d8e97d2eec859e09'},'source_local_sha256':{p.name:digest(p) for p in (HERE/'reviewed-source').iterdir() if p.is_file()},'reference_membership_matches':membership,'reference_display_order_differences':order_different,'reference_metric_scalar_comparisons':comparisons,'models':models,'embedding_work':work,'total_new_document_forwards':sum(x['new_docs'] for x in work),'total_new_query_forwards':sum(x['new_queries'] for x in work),'warmup_note':'One warmup for each of4 cached-model refreshes and4 EG2/Jina pilot/full loads:8 total, separate from new document/query forwards.','protocol_sha256':read(HERE/'protocol.json')['protocol_sha256'],'source_data_sha256':m['source_hashes'],'library_sha256':m['library_sha256'],'text_sha256':m['text_sha256']};(HERE/'report/source-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
