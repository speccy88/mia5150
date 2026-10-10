"""Offline audit and paired evaluation; never imports a network client."""
import json,sys,hashlib,random,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness import evaluate,qrels
from probe_contract import digest
PIN='0a2c3d1aa6de20d8b350ef68afacdac190eb986a09c989f77a605ff408b3036e'
def main():
 here=ROOT/'round2';out=here/'jev-typed-12-v1'
 p=json.loads((here/'exploration-freeze.json').read_text());assert p.pop('protocol_sha256')==PIN and digest(p)==PIN
 r=json.loads((out/'results.json').read_text());packets=json.loads((here/'prepared-public-requests.json').read_text())
 packets={x['id']:x for x in packets};ids=[q['id'] for q in p['queries']]
 assert len(r['rows'])==12 and [x['id'] for x in r['rows']]==ids and r['attempts']==12 and r['protocol_sha256']==PIN
 baseline=json.loads((ROOT/'results/skillret_embed_06-header-0.json').read_text())['rankings']
 ranks={'SkillRet':{},'Jev Noul':{},'Jev Score':{}}
 for row in r['rows']:
  q=row['id'];pool=p['candidate_ids'][q];assert row['candidate_ids']==pool==baseline[q][:20]
  assert row['request_sha256']==digest(packets[q]['request']) and row['model']=='jev-1.13.0'
  ranks['SkillRet'][q]=pool
  for kind,label in [('noul','Jev Noul'),('score','Jev Score')]:
   rr=row['rankings'][kind];assert len(rr)==20 and set(rr)==set(pool) and row['status']=='ok'
   ss=row['scores'][kind];assert len(ss)==20
   assert rr==[pool[i] for i in sorted(range(20),key=lambda i:(-ss[i],i))]
   ranks[label][q]=rr
 labels=qrels('test');labels={q:labels[q] for q in ids};metrics={k:evaluate(labels,v) for k,v in ranks.items()}
 rng=random.Random(5150);comparisons={}
 for kind in ('Jev Noul','Jev Score'):
  ds=[metrics[kind]['per_query'][q]['NDCG@10']-metrics['SkillRet']['per_query'][q]['NDCG@10'] for q in ids]
  bs=sorted(sum(rng.choices(ds,k=12))/12 for _ in range(10000))
  comparisons[kind]={'delta_ndcg10':statistics.mean(ds),'paired_bootstrap95':[bs[249],bs[9749]],'wins':sum(d>1e-12 for d in ds),'ties':sum(abs(d)<=1e-12 for d in ds),'losses':sum(d< -1e-12 for d in ds)}
 assert sum(x['input_tokens'] for x in r['rows'])==r['known_input_tokens']
 ledger=json.loads((here/'budget-ledger.json').read_text())
 result={'audit':'12 successful unique attempts, frozen requests/pools and score-derived rankings verified','metrics':metrics,'comparisons':comparisons,'latency_seconds':{'median':statistics.median(x['seconds'] for x in r['rows']),'min':min(x['seconds'] for x in r['rows']),'max':max(x['seconds'] for x in r['rows'])},'usage_tokens':r['known_input_tokens'],'known_usage_usd':r['known_usage_usd_at_pinned_rate'],'budget_ledger':ledger,'results_sha256':hashlib.sha256((out/'results.json').read_bytes()).hexdigest(),'caveats':'12 already-exposed development queries; two exploratory hypotheses, unadjusted bootstrap intervals; no blind test or general improvement claim. All methods share frozen top20 coverage. API latency excludes retrieval and is not a matched hardware comparison.'}
 (out/'evaluation.json').write_text(json.dumps(result,indent=2))
 lines=['# Jev typed API exploratory results','',result['audit'],'','| Method | NDCG@10 | Recall@10 | Completeness@5 |','|---|---:|---:|---:|']
 for k,m in metrics.items():
  a=m['aggregate'];lines.append(f"| {k} | {a['NDCG@10']:.6f} | {a['Recall@10']:.6f} | {a['Completeness@5']:.6f} |")
 lines+=['',json.dumps(comparisons,indent=2),'',result['caveats'],'',f"Known token usage: ${result['known_usage_usd']}. Keep $0.10 reserved plus $1 external, total $1.10 until billing reconciliation."]
 (out/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2));print(json.dumps({k:v['aggregate'] for k,v in metrics.items()},indent=2))
if __name__=='__main__':main()
