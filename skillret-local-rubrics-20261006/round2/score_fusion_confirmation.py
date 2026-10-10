"""Score only after all fixed-fusion outputs are saved; no network."""
import hashlib,json,random,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness import qrels,evaluate
from fusion_confirmation import PIN,fusion
from probe_contract import digest
HERE=ROOT/'round2';OUT=HERE/'fusion-confirmation-8-v1'
def main():
 r=json.loads((OUT/'results.json').read_text());p=json.loads((HERE/'fusion-confirmation-plan.json').read_text());assert p.pop('protocol_sha256')==PIN and digest(p)==PIN and r['protocol_sha256']==PIN
 ids=[q['id'] for q in p['queries']];assert [x['id'] for x in r['rows']]==ids and len(ids)==8
 packets={x['id']:x for x in json.loads((HERE/'prepared-fusion-confirmation.json').read_text())};base=p['candidate_ids'];pred={}
 for row in r['rows']:
  q=row['id'];assert row['candidate_ids']==base[q]
  if row['status']=='ok':
   assert row['request_sha256']==digest(packets[q]['request'])
   assert row['jev_ranking']==[base[q][i] for i in sorted(range(20),key=lambda i:(-row['scores'][i],i))]
   assert row['ranking']==fusion(base[q],row['jev_ranking'])
  else:assert row['ranking']==[]
  pred[q]=row['ranking']
 labels=qrels('test');labels={q:labels[q] for q in ids}
 oracle={q:sorted(base[q],key=lambda d:(-labels[q].get(d,0),base[q].index(d))) for q in ids}
 metrics={k:evaluate(labels,v) for k,v in {'SkillRet':base,'Fixed RRF60':pred,'Pool oracle':oracle}.items()};per=[]
 for q in ids:
  b=metrics['SkillRet']['per_query'][q]['NDCG@10'];v=metrics['Fixed RRF60']['per_query'][q]['NDCG@10'];gold={d for d,x in labels[q].items() if x>0}
  per.append({'id':q,'baseline_ndcg10':b,'fusion_ndcg10':v,'delta':v-b,'gold_count':len(gold),'pool_covered':len(gold&set(base[q]))})
 ds=[x['delta'] for x in per];rng=random.Random(5150);bs=sorted(statistics.mean(rng.choices(ds,k=8)) for _ in range(10000));times=[x['api_seconds'] for x in r['rows'] if 'api_seconds' in x]
 result={'metrics':metrics,'per_query':per,'delta_ndcg10':statistics.mean(ds),'paired_bootstrap95':[bs[249],bs[9749]],'wins_ties_losses':[sum(d>1e-12 for d in ds),sum(abs(d)<=1e-12 for d in ds),sum(d< -1e-12 for d in ds)],'calls':r['attempts'],'successes':sum(x['status']=='ok' for x in r['rows']),'known_usage_usd':r['known_usage_usd'],'input_tokens':r['known_input_tokens'],'median_api_seconds':statistics.median(times),'total_api_seconds':sum(times),'budget_committed_usd':'1.75','protocol_sha256':PIN,'results_sha256':hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),'caveat':'Fixed hypothesis/parameters/new eight-query cohort before inference and label review; still historically exposed public data. Tiny-sample interval is descriptive. API latency excludes cached retrieval, not end-to-end speed. No global-best claim.'}
 (OUT/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2));print(json.dumps({k:{m:v['aggregate'][m] for m in ['NDCG@10','Completeness@5','Recall@20']} for k,v in metrics.items()},indent=2))
if __name__=='__main__':main()
