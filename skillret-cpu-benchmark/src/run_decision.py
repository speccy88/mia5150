import sys,time,json,hashlib,resource
from common import *
from embeddings import Embedder
from decision_adapters import load_adapter
KEYS={'julia':'julia-1','gliner':'gliner','verdict':'verdict','laya':'laya'}
def run(key,mode,n):
 hardware=hw();c=Corpus();qs=c.queries[:n];qmap={x['query_id']:x for x in qs};shorts=[r for r in readjl(RESULTS/'rerank_shortlists.jsonl') if r['query_id'] in qmap];ml=512 if key=='verdict' else 1024
 cfg={'model':key,'max_length':ml,'truncate_query':True,'threads':4,'batch_size':8,'runtime_config':'ORT idle spinning off; Julia encoding/token caches off','candidate_text':'name plus first1024 description characters','score':'native binary relevance probability; Verdict abstention retained; stable shortlist tie ordering'}
 t=time.perf_counter();model=load_adapter(key,ROOT/'models'/KEYS[key],threads=4,max_length=ml,truncate_query=True);load_s=time.perf_counter()-t;model.score_pairs([('Find a skill for writing code','Example skill\nWrite and inspect code.')],batch_size=1)
 if mode=='quality':
  out=RESULTS/f'quality{n}';out.mkdir(exist_ok=True);cache=RESULTS/'pair_scores';cache.mkdir(exist_ok=True);p=cache/f'{key}.jsonl';mp=cache/f'{key}_metadata.json'
  h=hashlib.sha256()
  for f in [__file__,str(ROOT/'src/decision_adapters.py'),str(ROOT/'docs/model_sources.json'),str(DATA/'skills.jsonl'),str(DATA/'queries.jsonl'),str(RESULTS/'rerank_shortlists.jsonl')]:h.update(Path(f).read_bytes())
  h.update(json.dumps(cfg,sort_keys=True).encode());fp=h.hexdigest()
  if mp.exists():assert json.loads(mp.read_text())['fingerprint']==fp
  else:save(mp,{'fingerprint':fp,'configuration':cfg})
  scores={(x['query_id'],x['skill_id']):x for x in readjl(p)} if p.exists() else {};pairs=sorted({(r['query_id'],sid) for r in shorts for sid in r['candidate_skill_ids']},key=lambda x:(list(qmap).index(x[0]),x[1]));pending=[p for p in pairs if p not in scores];t=time.perf_counter()
  with p.open('a') as f:
   for start in range(0,len(pending),8):
    chunk=pending[start:start+8];values=model.score_pairs([(qmap[q]['text'],c.texts[c.index[s]]) for q,s in chunk],batch_size=8)
    assert len(values)==len(chunk)==len(model.last_audit), 'Incomplete scorer output'
    for (q,s),v,audit in zip(chunk,values,model.last_audit):
     assert np.isfinite(v) and 0<=v<=1;row={'query_id':q,'skill_id':s,'score':float(v),'audit':audit};scores[(q,s)]=row;f.write(json.dumps(row)+'\n')
    f.flush()
    if start%80==0:print(key,n,'pairs',min(start+8,len(pending)),len(pending),'seconds',round(time.perf_counter()-t,1),flush=True)
  rows=[]
  for r in shorts:
   q=qmap[r['query_id']];ids=r['candidate_skill_ids'];order=sorted(range(len(ids)),key=lambda i:-scores[(q['query_id'],ids[i])]['score']);ranked=[ids[i] for i in order]
   rows.append({'model':key,'method':'hybrid_rerank','catalog_size':r['catalog_size'],'query_id':q['query_id'],'gold_count':len(q['gold_skill_ids']),'query_group':'multi' if len(q['gold_skill_ids'])>1 else 'single',**metrics(ranked,q['gold_skill_ids']),'shortlist_mrr5':metrics(ids,q['gold_skill_ids'])['mrr5'],'unique_score_count':len({scores[(q['query_id'],s)]['score'] for s in ids}),'ranked_skill_ids':json.dumps(ranked)})
  writecsv(out/f'{key}_per_query.csv',rows);save(out/f'{key}_summary.json',summarize(rows,[]));extra={'new_pair_count':len(pending),'unique_pair_count':len(pairs),'new_pair_seconds':time.perf_counter()-t,'latency_intentionally_not_run':True}
 else:
  out=RESULTS/'timing32';out.mkdir(exist_ok=True);dense=Embedder('bge-small');docs=np.load(RESULTS/'embeddings/bge-small_docs.npy');dense.encode(['Warmup skill retrieval']);fullshort={r['query_id']:r['candidate_skill_ids'] for r in shorts if r['catalog_size']==6006};lat=[]
  for q in qs[:32]:
   for rep in range(2):
    t=time.perf_counter();v=dense.encode([q['text']],batch_size=1)[0];idx=rrf(c.full,c.bm.score(q['text'],c.full),docs@v);ids=[c.ids[int(i)] for i in idx];assert ids==fullshort[q['query_id']],f'Shortlist drift {q["query_id"]}'
    stage1=ms(t);t2=time.perf_counter();values=model.score_pairs([(q['text'],c.texts[c.index[s]]) for s in ids],batch_size=8);ranked=sorted(range(len(ids)),key=lambda i:-values[i]);rerank=ms(t2);total=ms(t)
    lat.append({'model':key,'method':'hybrid_rerank','catalog_size':6006,'query_id':q['query_id'],'repeat':rep,'stage1_ms':stage1,'rerank_ms':rerank,'total_ms':total,'shortlist_matches_frozen':True,'fresh_shortlist_ids':json.dumps(ids)});writecsv(out/f'{key}_latency.csv',lat)
   print(key,'timing',len(lat),'of64',flush=True)
  extra={'latency_observations':len(lat),'latency_query_count':32,'latency_repeats':2,'latency_catalog_sizes':[6006],'warm_weights':True,'encoding_and_score_caches':False,'stage1':'BGE + BM25 full catalog RRF top20'}
 receipt={'model':key,'status':'completed','mode':mode,'hardware':hardware,'configuration':cfg,'score_cache_fingerprint':json.loads((RESULTS/'pair_scores'/f'{key}_metadata.json').read_text())['fingerprint'],'load_seconds':load_s,'query_count':n,'catalog_sizes':c.sizes if mode=='quality' else [6006],'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra};save(out/f'{key}_run.json',receipt)
if __name__=='__main__':run(sys.argv[1],sys.argv[2],int(sys.argv[3]))
