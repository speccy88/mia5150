"""Independent artifact verification; no model imports for saved-score checks."""
from common import *

def metrics_reference(gold,rankings):
 old=module('historical_metrics',OLD/'evaluate.py');a=old.evaluate({q:{s:1 for s in g} for q,g in gold.items()},rankings);b=evaluate(gold,rankings)
 for k,v in a['aggregate'].items():
  if k in b['aggregate']:assert abs(v-b['aggregate'][k])<1e-12
 for k in (20,50):assert abs(a['aggregate'][f'Coverage@{k}']-b['aggregate'][f'Recall@{k}'])<1e-12
 top={qid:(rank[0] if rank else None) for qid,rank in rankings.items()};hits=[int(top.get(qid) in labels) for qid,labels in gold.items()];assert abs(sum(hits)/len(hits)-b['aggregate']['Hit@1'])<1e-12;assert abs(math.fsum(h/len(labels) for h,labels in zip(hits,gold.values()))/len(hits)-b['aggregate']['Recall@1'])<1e-12
 for qid,expected in a['per_query'].items():
  actual=b['per_query'][qid]
  for key,value in expected.items():
   key=key.replace('Coverage@','Recall@')
   if key in actual:assert abs(value-actual[key])<1e-12,(qid,key)
  h=int(top.get(qid) in gold[qid]);assert actual['Hit@1']==h and actual['Recall@1']==h/len(gold[qid])
 return b

def main():
 s,q,g=data('test');base=read(OLD/'results/baseline_skillret_06b.json');ref=metrics_reference(g,base['rankings']);checked=[];skill_ids={x['id'] for x in s};query_texts={x['id']:digest(PREFIX+x['query']) for x in q}
 assert len(skill_ids)==6006 and len(g)==4392 and sum(map(len,g.values()))==7187
 directory=STUDY/'results/final'
 if directory.exists():
  paths=set(directory.glob('*.jsonl'))|{Path(str(p)[:-3]) for p in directory.glob('*.jsonl.gz')}
  for path in sorted(paths):
   records=rows(path)
   if not records or 'ranking' not in records[0]:continue
   assert len(records)==4392 and {x['query_id'] for x in records}==set(g)
   for record in records:
    assert record['provenance']['instructed_query_text_sha256']==query_texts[record['query_id']]
    rank=record['ranking']
    if record['status']=='ok':
     assert len(rank)==50 and set(rank)<=skill_ids
     if 'candidates' in record:assert len(record['candidates'])==50 and set(rank)==set(record['candidates'])
    else:assert record['status']=='failed' and rank==[]
   r={x['query_id']:x['ranking'] for x in records};evaluation=metrics_reference(g,r);saved=read(path.with_suffix('.metrics.json'))
   for k,v in evaluation['aggregate'].items():assert abs(v-saved['aggregate'][k])<1e-12
   assert set(evaluation['per_query'])==set(saved['per_query'])
   for qid,actual in evaluation['per_query'].items():
    for k,v in actual.items():assert abs(v-saved['per_query'][qid][k])<1e-12,(path.name,qid,k)
   checked.append(path.name)
 if not checked:
  print('Archived baseline independently verified; no final challenger evidence is present.');return
 lock=read(STUDY/'FINAL_EVAL_LOCK.json')
 assert {f'{name}.jsonl' for name in ('reconstructed-base','adapted-query','released-reranker','trained-reranker','adapted-query-seed43','trained-reranker-seed43',lock['primary_condition'])}<=set(checked)
 from score_reuse import verify_official_records
 reuse_checks=verify_official_records(STUDY,lock)
 save(STUDY/'results/independent_verification.json',{'status':'passed','archived_baseline_ndcg10':ref['aggregate']['NDCG@10'],'complete_final_conditions_verified':checked,'offline_score_reuse_verified':reuse_checks,'method':'independent archived dependency-free TREC evaluator versus challenger metric implementation; all official query IDs, rankings, per-query metrics and aggregate metrics verified'})
 print('Independent verification passed',checked)
if __name__=='__main__':main()
