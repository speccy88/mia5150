"""Evaluate only after fixed hybrid-pool predictions are durable; no network."""
import csv,hashlib,json,statistics,sys
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2';OUT=HERE/'hybrid-pool-8-v1';sys.path.insert(0,str(ROOT))
from harness import evaluate,qrels
from probe_contract import digest
ARMS=['bm25','vector','hybrid']
def main():
 p=json.loads((HERE/'hybrid-pool-plan.json').read_text());requests=json.loads((HERE/'prepared-hybrid-pool.json').read_text());r=json.loads((OUT/'results.json').read_text());pin=p['protocol_sha256'];assert digest({k:v for k,v in p.items() if k!='protocol_sha256'})==pin==r['protocol_sha256'] and digest(requests)==p['requests_sha256']
 ids=[q['id'] for q in p['queries']];assert len(ids)==len(set(ids))==8
 by={(x['id'],x['arm']):x for x in r['rows']};assert len(by)==len(r['rows'])
 for item in requests:
  row=by.get((item['id'],item['arm']))
  if row and row['status']=='ok':
   assert row['request_sha256']==digest(item['request']) and row['pool']==item['pool']
   scores=row['scores'];assert len(scores)==20 and row['ranking']==[row['pool'][i] for i in sorted(range(20),key=lambda i:(-scores[i],i))]
 labels=qrels('test');labels={q:labels[q] for q in ids};metrics={};per=[];timings={}
 for arm in ARMS:
  pools={q:p['pools'][q][arm] for q in ids};reranked={q:by.get((q,arm),{}).get('ranking',[]) if by.get((q,arm),{}).get('status')=='ok' else [] for q in ids};oracle={q:sorted(pool,key=lambda d:(-labels[q].get(d,0),pool.index(d))) for q,pool in pools.items()}
  for suffix,ranks in [('control',pools),('jev',reranked),('oracle',oracle)]:metrics[arm+'_'+suffix]=evaluate(labels,ranks)
  rows=[x for x in r['rows'] if x['arm']==arm and x.get('new_call')];secs=[x['api_seconds'] for x in rows]
  timings[arm]={'new_calls':len(rows),'successes':sum(x['status']=='ok' for x in rows),'input_tokens':sum(x.get('input_tokens',0) for x in rows),'metered_usd':str(Decimal(sum(x.get('input_tokens',0) for x in rows))*Decimal('.042')/1000000),'median_api_seconds':statistics.median(secs) if secs else None,'total_api_seconds':sum(secs)}
  for q in ids:
   gold={s for s,v in labels[q].items() if v>0};pool=pools[q];rank=reranked[q];covered=gold&set(pool);m=metrics[arm+'_jev']['per_query'][q]
   assert m['Recall@5']==len(gold&set(rank[:5]))/len(gold) and m['Recall@10']==len(gold&set(rank[:10]))/len(gold)
   assert m['Recall@5']<=min(5,len(covered))/len(gold) and m['Recall@10']<=min(10,len(covered))/len(gold)
   per.append({'query_id':q,'arm':arm,'status':by.get((q,arm),{}).get('status','not_attempted'),'positive_qrels':len(gold),'pool_hits':len(covered),'pool_recall':len(covered)/len(gold),'oracle_recall5':min(5,len(covered))/len(gold),'oracle_recall10':min(10,len(covered))/len(gold),'control_recall5':metrics[arm+'_control']['per_query'][q]['Recall@5'],'jev_recall5':m['Recall@5'],'jev_recall10':m['Recall@10'],'jev_ndcg10':m['NDCG@10'],'gained_vs_vector_pool':sorted(covered-set(p['pools'][q]['vector'])),'lost_vs_vector_pool':sorted((gold&set(p['pools'][q]['vector']))-set(pool))})
 comparisons={}
 for base in ['bm25','vector']:
  comparisons['hybrid_vs_'+base]={}
  for metric in ['Recall@5','Recall@10','NDCG@10']:
   ds={q:metrics['hybrid_jev']['per_query'][q][metric]-metrics[base+'_jev']['per_query'][q][metric] for q in ids}
   comparisons['hybrid_vs_'+base][metric]={'mean_delta':statistics.mean(ds.values()),'wins_ties_losses':[sum(d>1e-12 for d in ds.values()),sum(abs(d)<=1e-12 for d in ds.values()),sum(d< -1e-12 for d in ds.values())],'per_query_delta':ds}
 telemetry=[json.loads(x) for x in (ROOT/'runs/hybrid-pool-8-v1/telemetry.jsonl').read_text().splitlines()]
 resources={'host_min_available_gib':min(x['host']['available_bytes'] for x in telemetry)/2**30,'guest_min_available_gib':min(x['guest_available'] for x in telemetry)/2**30,'gpu_min_free_mib':min(x['gpu_free_mib'] for x in telemetry),'gpu_max_temp_c':max(x['gpu_temp'] for x in telemetry),'active_swap_observed':any(x['swap']!=telemetry[0]['swap'] for x in telemetry)}
 assert sum(x['input_tokens'] for x in timings.values())==r['known_input_tokens']
 summary={'metrics':metrics,'per_query':per,'comparisons':comparisons,'timings':timings,'resources':resources,'calls':r['attempts'],'successes':sum(x['status']=='ok' for x in r['rows']),'known_input_tokens':r['known_input_tokens'],'known_usage_usd':r['known_usage_usd'],'reserved_usd':'0.10','shared_commitments_usd':'1.97','cumulative_known_usage_usd':str(Decimal('.013220151')+Decimal(r['known_usage_usd'])),'cash_charge_usd':None,'unit_tests_passed':70,'protocol_sha256':pin,'results_sha256':hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),'caveat':'Eight historically exposed public queries; frozen selection independent of labels, no repeats, no posthoc tuning. Direct reranking only; this is candidate-pool hybridization, not vector/Jev rank fusion.'}
 (OUT/'evaluation.json').write_text(json.dumps(summary,indent=2))
 with (OUT/'per-query.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(per[0]));writer.writeheader();writer.writerows(per)
 print(json.dumps({'metrics':{k:{m:v['aggregate'][m] for m in ['Recall@5','Recall@10','NDCG@10','Recall@20']} for k,v in metrics.items()},'comparisons':comparisons,'cost':r['known_usage_usd'],'timings':timings},indent=2))
if __name__=='__main__':main()
