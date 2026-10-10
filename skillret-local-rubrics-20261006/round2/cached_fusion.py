"""One posthoc fixed RRF test; freeze future cohort before reading labels."""
import json,sys,hashlib,random,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness import evaluate,qrels
from probe_contract import digest
from jev_probe import PIN
HERE=ROOT/'round2'
def freeze():
 excluded={q['id'] for q in json.loads((ROOT/'manifests/scorer-queries.json').read_text())}
 queries=[json.loads(s) for s in (ROOT/'data/queries/test.jsonl').read_text().splitlines()]
 queries=[{'id':q['id'],'query':q['query']} for q in queries if q['id'] not in excluded]
 selected=sorted(queries,key=lambda q:hashlib.sha256(('mia5150-next-distinct-probe-v1:'+q['id']).encode()).hexdigest())[:8]
 obj={'queries':selected,'selection':'Lowest SHA256(salt + query_id), salt mia5150-next-distinct-probe-v1:, excluding all128 previous scorer queries; selection without qrels or results.','dataset_revision':'6583d7d2ed07644d0fb8938ed8178f3a7dc42a12','exposure':'Unused in current paid round and outside128 scorer panel; historically exposed public benchmark, including full4392 dense evaluation. NOT untouched/blind final test. No new confirmation labels inspected here.','future_inference_calls':0}
 obj['manifest_sha256']=digest(obj);path=HERE/'next-confirmation-8.json'
 if path.exists():assert json.loads(path.read_text())==obj
 else:path.write_text(json.dumps(obj,indent=2))
 return [q['id'] for q in selected]
def main():
 frozen=freeze() # Must precede loading qrels or inspecting outcomes.
 panel=json.loads((HERE/'exploration-freeze.json').read_text());assert panel.pop('protocol_sha256')==PIN and digest(panel)==PIN
 raw=json.loads((HERE/'jev-typed-12-v1/results.json').read_text());ids=[q['id'] for q in panel['queries']]
 dense=panel['candidate_ids'];jev={r['id']:r['rankings']['score'] for r in raw['rows']};fusion={}
 for q in ids:
  pool=dense[q];assert set(pool)==set(jev[q]) and len(pool)==20
  score={d:1/(60+i+1) for i,d in enumerate(pool)}
  for i,d in enumerate(jev[q]):score[d]+=1/(60+i+1)
  fusion[q]=sorted(pool,key=lambda d:(-score[d],pool.index(d)))
 labels=qrels('test');labels={q:labels[q] for q in ids}
 oracle={q:sorted(dense[q],key=lambda d:(-labels[q].get(d,0),dense[q].index(d))) for q in ids}
 methods={'SkillRet':dense,'Jev Score':jev,'RRF60 equal weight':fusion,'Pool oracle':oracle}
 metrics={name:evaluate(labels,r) for name,r in methods.items()};rows=[]
 for q in ids:
  gold={d:v for d,v in labels[q].items() if v>0}
  rows.append({'id':q,'gold_count':len(gold),'missing_gold_ids':[d for d in gold if d not in dense[q]],'ndcg10':{name:m['per_query'][q]['NDCG@10'] for name,m in metrics.items()},'gold_ranks':{name:{d:(r[q].index(d)+1 if d in r[q] else None) for d in gold} for name,r in methods.items() if name!='Pool oracle'}})
 delta=[r['ndcg10']['RRF60 equal weight']-r['ndcg10']['SkillRet'] for r in rows];rng=random.Random(5150);boots=sorted(statistics.mean(rng.choices(delta,k=12)) for _ in range(10000))
 result={'method':'Single posthoc equal-weight reciprocal rank fusion, k60; ties preserve dense rank. No parameter search.','metrics':metrics,'per_query':rows,'delta_ndcg10':statistics.mean(delta),'paired_bootstrap95':[boots[249],boots[9749]],'wins_ties_losses':[sum(d>1e-12 for d in delta),sum(abs(d)<=1e-12 for d in delta),sum(d< -1e-12 for d in delta)],'next_confirmation_ids':frozen,'caveat':'Posthoc on12 exposed public development queries after observing Jev outcomes. Oracle uses labels only for diagnosis, never as a deployable method. New panel is round-disjoint, not historically untouched. No inference or network calls.'}
 (HERE/'cached-fusion-evaluation.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','per_query')},indent=2));print(json.dumps({'ndcg10':{k:v['aggregate']['NDCG@10'] for k,v in metrics.items()},'per_query':rows},indent=2))
if __name__=='__main__':main()
