"""Independent stdlib-only recovery result audit. No inference or source writes."""
import json,csv,math,statistics,hashlib,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/processed';R=ROOT/'results'
jl=lambda p:[json.loads(line) for line in Path(p).open() if line.strip()]
jload=lambda p:json.loads(Path(p).read_text());csvrows=lambda p:list(csv.DictReader(Path(p).open()))
qrows=jl(D/'queries.jsonl');Q={v['query_id']:v for v in qrows};QIDS=[v['query_id'] for v in qrows];S={v['skill_id']:v for v in jl(D/'skills.jsonl')};LIDS=jload(D/'latency_query_ids.json');CAT={v['query_id']:v['candidate_skill_ids'] for v in jl(D/'catalogs.jsonl')};SIZES={10,100,1000,6006}
assert LIDS==QIDS[:32] and len(QIDS)==64

def same(a,b):assert math.isclose(float(a),float(b),abs_tol=2e-12,rel_tol=2e-12),(a,b)
def metric(ids,gold):
 g=set(gold);rel=[int(i in g) for i in ids[:5]];ideal=sum(1/math.log2(i+2) for i in range(min(5,len(g))))
 return {'mrr5':next((1/(i+1) for i,x in enumerate(rel) if x),0.),'recall1':(rel[0] if rel else 0)/len(g),'recall5':sum(rel)/len(g),'hit1':float(bool(rel and rel[0])),'hit5':float(any(rel)),'ndcg5':sum(x/math.log2(i+2) for i,x in enumerate(rel))/ideal,'complete5':float(sum(rel)==len(g)),'candidate_recall':len(set(ids)&g)/len(g),'candidate_hit':float(bool(set(ids)&g))}
def pct(values,q):
 v=sorted(values);a=(len(v)-1)*q;lo=math.floor(a);hi=math.ceil(a);return v[lo]+(v[hi]-v[lo])*(a-lo)
def validate_metrics(rows,n,methods,shortlists=None,scores=None):
 groups=collections.defaultdict(list);seen=set();checks=0
 for row in rows:
  qid=row['query_id'];size=int(row['catalog_size']);method=row['method'];key=(method,size,qid);assert key not in seen;seen.add(key)
  assert qid in QIDS[:n] and size in SIZES and method in methods
  ids=json.loads(row['ranked_skill_ids']);assert len(ids)==len(set(ids))<=min(size,20)
  assert set(ids)<=(set(S) if size==6006 else set(CAT[qid][:size]))
  if method!='domain_vector':assert len(ids)==min(size,20)
  if method=='domain_vector':
   major={S[i]['major'] for i in ids};assert len(major)==1;same(row['routing_hit'],bool(major&set(S[g]['major'] for g in Q[qid]['gold_skill_ids'])))
  if shortlists is not None:
   source=shortlists[qid,size];assert set(ids)==set(source);assert ids==sorted(source,key=lambda sid:-scores[qid,sid]['score']);same(row['shortlist_mrr5'],metric(source,Q[qid]['gold_skill_ids'])['mrr5'])
  assert int(row['gold_count'])==len(Q[qid]['gold_skill_ids']);m=metric(ids,Q[qid]['gold_skill_ids']);checks+=len(m)
  for name,val in m.items():same(row[name],val)
  groups[method,size].append(m)
 assert seen=={(m,size,qid) for m in methods for size in SIZES for qid in QIDS[:n]}
 return groups,checks

def validate_latency(path,methods,decision=False):
 groups=collections.defaultdict(list);seen=set()
 for row in csvrows(path):
  key=(row['method'],row['query_id'],int(row['repeat']));assert key not in seen;seen.add(key);assert int(row['catalog_size'])==6006
  assert float(row['total_ms'])>0
  if decision:
   assert row['shortlist_matches_frozen']=='True';assert json.loads(row['fresh_shortlist_ids'])==SHORTS[row['query_id'],6006]
   assert float(row['total_ms'])>=float(row['stage1_ms'])+float(row['rerank_ms'])-1e-6
  else:assert float(row['total_ms'])>=float(row['query_encode_ms'])>=0
  groups[row['method']].append(float(row['total_ms']))
 assert seen=={(m,qid,rep) for m in methods for qid in LIDS for rep in range(2)}
 return groups
SHORTS={(v['query_id'],v['catalog_size']):v['candidate_skill_ids'] for v in jl(R/'rerank_shortlists.jsonl')} if (R/'rerank_shortlists.jsonl').exists() else {}
reports=[]
for path in sorted((R/'retrieval64').glob('*_run.json')):
 slug=path.name[:-len('_run.json')];run=jload(path)
 if run.get('status')!='completed':continue
 methods={'bm25'} if slug=='bm25' else {'vector','hybrid','domain_vector'};rows=csvrows(path.with_name(slug+'_per_query.csv'));groups,checks=validate_metrics(rows,64,methods);lat=validate_latency(path.with_name(slug+'_latency.csv'),methods)
 for row in jload(path.with_name(slug+'_summary.json')):
  m,n=row['method'],row['catalog_size'];assert row['queries']==64
  for name in groups[m,n][0]:same(row[name],statistics.fmean(v[name] for v in groups[m,n]))
  if n==6006:
   vals=lat[m];same(row['p50_ms'],pct(vals,.5));same(row['p95_ms'],pct(vals,.95));same(row['mean_ms'],statistics.fmean(vals));same(row['serial_qps'],1000/statistics.fmean(vals))
  else:assert row['p50_ms'] is None and row['latency_queries']==0
 reports.append({'kind':'retrieval','model':slug,'queries':64,'rows':len(rows),'metric_checks':checks,'latency_rows':sum(map(len,lat.values())),'status':'passed'})
for n in [32,48,64]:
 for path in sorted((R/f'quality{n}').glob('*_run.json')):
  key=path.name[:-len('_run.json')];run=jload(path)
  if run.get('status')!='completed':continue
  pairrows=jl(R/'pair_scores'/f'{key}.jsonl');scores={(x['query_id'],x['skill_id']):x for x in pairrows};assert len(scores)==len(pairrows)
  needed={(q,s) for (q,size),ids in SHORTS.items() if q in QIDS[:n] for s in ids};assert needed<=scores.keys();assert run['unique_pair_count']==len(needed)
  for q,s in needed:
   row=scores[q,s];a=row['audit'];assert math.isfinite(row['score']) and 0<=row['score']<=1 and a['candidate_preserved'];assert a['tokens']<=run['configuration']['max_length'];assert a['query_chars_original']==len(Q[q]['text']);assert 0<=a['query_chars_used']<=a['query_chars_original'];assert a['query_truncated']==(a['query_chars_original']>a['query_chars_used'])
   if key=='verdict':same(a['temperature'],5.0069);assert row['score']+a['p_abstain']<=1+1e-12
  rows=csvrows(path.with_name(key+'_per_query.csv'));groups,checks=validate_metrics(rows,n,{'hybrid_rerank'},SHORTS,scores)
  for row in jload(path.with_name(key+'_summary.json')):
   assert row['queries']==n and row['p50_ms'] is None
   for name in groups[row['method'],row['catalog_size']][0]:same(row[name],statistics.fmean(v[name] for v in groups[row['method'],row['catalog_size']]))
  h=hashlib.sha256()
  for f in [ROOT/'src/run_decision.py',ROOT/'src/decision_adapters.py',ROOT/'docs/model_sources.json',D/'skills.jsonl',D/'queries.jsonl',R/'rerank_shortlists.jsonl']:h.update(f.read_bytes())
  h.update(json.dumps(run['configuration'],sort_keys=True).encode());meta=jload(R/'pair_scores'/f'{key}_metadata.json');assert h.hexdigest()==meta['fingerprint']==run['score_cache_fingerprint']
  reports.append({'kind':'decision_quality','model':key,'queries':n,'rows':len(rows),'pairs':len(needed),'metric_checks':checks,'query_shortened_pairs':sum(scores[p]['audit']['query_truncated'] for p in needed),'status':'passed','full_catalog_mrr5':statistics.fmean(v['mrr5'] for v in groups['hybrid_rerank',6006])})
for path in sorted((R/'timing32').glob('*_run.json')):
 key=path.name[:-len('_run.json')];run=jload(path)
 if run.get('status')!='completed':continue
 quality=jload(R/'quality32'/f'{key}_run.json');assert run['configuration']==quality['configuration'] and run['score_cache_fingerprint']==quality['score_cache_fingerprint'];assert run['query_count']==32 and run['latency_observations']==64 and run['latency_repeats']==2 and run['warm_weights'] and not run['encoding_and_score_caches']
 lat=validate_latency(path.with_name(key+'_latency.csv'),{'hybrid_rerank'},True)['hybrid_rerank'];reports.append({'kind':'decision_timing','model':key,'queries':32,'repeats':2,'rows':64,'p50_ms':pct(lat,.5),'p95_ms':pct(lat,.95),'status':'passed'})
print(json.dumps({'root':str(ROOT),'reports':reports},indent=2))
