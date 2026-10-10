"""Local SkillRet evaluation: label-free scorers and fixed-denominator metrics."""
import pathlib,json,hashlib,math,time,ast,typing,os
R=pathlib.Path(os.environ.get('MIA5150_ROOT',pathlib.Path(__file__).resolve().parent))
def load(subset,split):
 with (R/f'data/{subset}/{split}.jsonl').open() as f:return [json.loads(s) for s in f]
def save(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def view(s,kind):
 fields=['name','description']+(['skill_md'] if kind=='full' else [])
 return ' | '.join((s.get(k) or (s.get('description','') if k=='skill_md' else '')).strip() for k in fields)
def safe_queries(rows):return [{'id':x['id'],'query':x['query']} for x in rows]
def qrels(split):
 result={}
 for x in load('qrels',split):result.setdefault(x['query_id'],{})[x['skill_id']]=int(x['relevance'])
 return result
def evaluate(labels,rankings):
 import pytrec_eval
 fixed={q:{d:float(len(rankings.get(q,[]))-i) for i,d in enumerate(dict.fromkeys(rankings.get(q,[])))} for q in labels}
 ev=pytrec_eval.RelevanceEvaluator(labels,{'ndcg_cut.5,10,15','map_cut.5,10,15','recall.5,10,15,20,50,100','P.5,10,15'})
 raw=ev.evaluate(fixed); rows={}
 for q,gold in labels.items():
  r=list(dict.fromkeys(rankings.get(q,[])));gold={s for s,v in gold.items() if v>0};v=raw.get(q,{})
  out={}
  for k in (5,10,15):
   for name,key in [('NDCG','ndcg_cut'),('MAP','map_cut'),('Recall','recall'),('Precision','P')]:out[f'{name}@{k}']=v.get(f'{key}_{k}',0.)
  for k in (5,10,15,20,50,100):
   out[f'Completeness@{k}']=float(bool(gold) and gold.issubset(r[:k]));out[f'Recall@{k}']=len(gold.intersection(r[:k]))/len(gold) if gold else 0.
  out['MRR@5']=next((1/(i+1) for i,d in enumerate(r[:5]) if d in gold),0.)
  rows[q]=out
 return {'aggregate':{k:sum(v[k] for v in rows.values())/len(rows) for k in next(iter(rows.values()))},'per_query':rows,'query_count':len(labels),'empty_rate':sum(not rankings.get(q) for q in labels)/len(labels)}
def official_metrics(labels,rankings):
 import pytrec_eval
 tree=ast.parse((R/'official/skillret/eval.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='trec_eval')
 scope={'Dict':typing.Dict,'Tuple':typing.Tuple,'pytrec_eval':pytrec_eval};exec(compile(ast.Module(body=[node],type_ignores=[]),'official_trec_eval','exec'),scope)
 return scope['trec_eval'](labels,{q:{d:float(len(r)-i) for i,d in enumerate(r)} for q,r in rankings.items()})
def lexical(skills,queries,kind,top=100):
 import bm25s
 start=time.perf_counter();tokens=bm25s.tokenize([view(s,kind) for s in skills],stopwords='en',show_progress=False);index=bm25s.BM25();index.index(tokens,show_progress=False);index_seconds=time.perf_counter()-start
 result={};lat={};ids=[s['id'] for s in skills]
 for q in queries:
  start=time.perf_counter();t=bm25s.tokenize(q['query'],stopwords='en',show_progress=False)
  if not t.vocab:t=bm25s.tokenize('NONE',stopwords=[],show_progress=False)
  hits,scores=index.retrieve(t,corpus=ids,k=min(top,len(ids)),show_progress=False,n_threads=1)
  result[q['id']]=[str(d) for d,s in zip(hits[0],scores[0]) if s>0];lat[q['id']]=time.perf_counter()-start
 return result,{'index_seconds':index_seconds,'per_query_seconds':lat}
def rrf(channels,k=60):
 scores={}
 for ranking in channels:
  for i,d in enumerate(dict.fromkeys(ranking)):scores[d]=scores.get(d,0)+1/(k+i+1)
 return sorted(scores,key=lambda d:(-scores[d],d))
def negatives(gold,ranking,k=4):return [d for d in dict.fromkeys(ranking) if d not in gold][:k]
def mmr(ranking,relevance,vectors,weight=.8,k=5,gate=0.):
 selected=[];eligible=[d for d in ranking if relevance[d]>=gate]
 while eligible and len(selected)<k:
  d=max(eligible,key=lambda d:weight*relevance[d]-(1-weight)*max((float(vectors[d]@vectors[s]) for s in selected),default=0.));selected.append(d);eligible.remove(d)
 return selected
def set_metrics(selected,gold):
 s=set(selected);g=set(gold);p=len(s&g)/len(s) if s else 0.;r=len(s&g)/len(g) if g else 0.
 return {'exact':float(s==g),'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,'size':len(s)}
