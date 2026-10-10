"""Evaluate only after frozen selective predictions have been saved."""
import hashlib,json,random,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness import qrels,evaluate
from selective_jev import PIN,fusion
from probe_contract import digest
HERE=ROOT/'round2';OUT=HERE/'selective-jev-8-v1'
def main():
 r=json.loads((OUT/'results.json').read_text());p=json.loads((HERE/'selective-probe-plan.json').read_text());assert p.pop('protocol_sha256')==PIN and digest(p)==PIN and r['protocol_sha256']==PIN
 ids=[x['id'] for x in p['rows']];assert [x['id'] for x in r['rows']]==ids and len(ids)==8
 packets={x['id']:x for x in json.loads((HERE/'prepared-selective-requests.json').read_text())};base={x['id']:x['candidate_ids'] for x in p['rows']}
 policy={};nofallback={}
 for row in r['rows']:
  q=row['id'];assert row['candidate_ids']==base[q]
  if row['status']=='ok':
   assert row['request_sha256']==digest(packets[q]['request'])
   assert row['jev_ranking']==[base[q][i] for i in sorted(range(20),key=lambda i:(-row['scores'][i],i))]
   assert row['ranking']==fusion(base[q],row['jev_ranking'])
  else:assert row['ranking']==base[q]
  policy[q]=row['ranking'];nofallback[q]=row['no_failure_fallback_ranking']
 # First evaluation of this round-disjoint panel occurs here, after predictions saved.
 labels=qrels('test');labels={q:labels[q] for q in ids}
 metrics={k:evaluate(labels,v) for k,v in {'SkillRet':base,'Selective RRF60':policy,'No failure fallback':nofallback}.items()}
 per=[]
 for row in r['rows']:
  q=row['id'];b=metrics['SkillRet']['per_query'][q]['NDCG@10'];v=metrics['Selective RRF60']['per_query'][q]['NDCG@10']
  per.append({'id':q,'status':row['status'],'baseline_ndcg10':b,'policy_ndcg10':v,'delta':v-b})
 ds=[x['delta'] for x in per];rng=random.Random(5150);boots=sorted(statistics.mean(rng.choices(ds,k=8)) for _ in range(10000));times=[row['api_seconds'] for row in r['rows'] if 'api_seconds' in row]
 result={'metrics':metrics,'per_query':per,'delta_ndcg10':statistics.mean(ds),'paired_bootstrap95':[boots[249],boots[9749]],'wins_ties_losses':[sum(x>1e-12 for x in ds),sum(abs(x)<=1e-12 for x in ds),sum(x< -1e-12 for x in ds)],'gate_trigger_count':sum(x['gate_triggered'] for x in p['rows']),'quota_excluded_count':sum(x['quota_excluded'] for x in p['rows']),'actual_calls':r['attempts'],'call_fraction':r['attempts']/8,'failed_call_count':sum(x['status']=='failed_or_uncertain' for x in r['rows']),'known_usage_usd':r['known_usage_usd'],'input_tokens':r['known_input_tokens'],'api_latency_seconds':{'median_per_call':statistics.median(times),'total':sum(times),'amortized_over_all8':sum(times)/8},'budget_committed_usd':'1.65','protocol_sha256':PIN,'results_sha256':hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),'caveat':'Fixed protocol before new predictions/labels;8 round-disjoint but historically exposed public queries. No genuine untouched final test. Latency is extra API wall time only; cached retrieval/lexical gate time not benchmarked here. No additional paid experiments authorized after this run pending review.'}
 (OUT/'evaluation.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2));print(json.dumps({k:{m:v['aggregate'][m] for m in ['NDCG@10','Completeness@5','Recall@20']} for k,v in metrics.items()},indent=2))
if __name__=='__main__':main()
