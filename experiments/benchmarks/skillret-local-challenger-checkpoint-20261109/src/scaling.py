"""Exact historical secondary 20-query/eight-library scaling evaluation."""
from engine import *


def main():
 import shutil
 from evaluate_final import active_timing,encoder_times,lock,query_provenance
 from train_reranker import normalized_fusion
 from reranker import Renderer,score_sequences
 l=lock();scope=l['scaling'];panel_path=REPO/scope['panel_source'];pool_path=REPO/scope['pool_source']
 assert sha(panel_path)==scope['panel_sha256'] and sha(pool_path)==scope['pool_sha256']
 panel=read(panel_path);pools=read(pool_path);sizes=tuple(scope['library_sizes'])
 assert len(panel)==scope['query_count']==20 and sizes==SCALING_SIZES and set(pools)==set(map(str,sizes))
 shutil.copyfile(pool_path,STUDY/'manifests/scaling_pools.json')
 save(STUDY/'manifests/scaling_source.json',{**scope,'labels':'historical fixed single positive exact skill IDs',
      'biases':'exposed repeated 20 queries; golds at positions 1–20; smaller pools not fully nested; secondary evidence only'})
 qids=[f'panel-{i:02d}' for i in range(20)];texts=[PREFIX+x['query'] for x in panel]
 save(STUDY/'manifests/scaling_query_labels.json',[{'query_id':qid,**{k:x[k] for k in ('gold_id','gold_domain','gold_major')}} for qid,x in zip(qids,panel,strict=True)])
 b,a=[],[]
 for name,adapter in [('base',None),('adapted',CACHE/l['query']['relative_cache_path'])]:
  m,tok=load(adapter=adapter);v=encode_cache(m,tok,texts,qids,f'scaling-{name}-queries',32768)
  if name=='base':b=v
  else:a=v
  del m;torch.cuda.empty_cache()
 bt=encoder_times('scaling-base-queries');at=encoder_times('scaling-adapted-queries')
 docs,dids=load_vectors('test-documents');imap={sid:i for i,sid in enumerate(dids)}
 gold={qids[i]:{p['gold_id']} for i,p in enumerate(panel)}
 w=l['query']['mixture_weight'];out=[];dense={};dense_records=[]
 for name,weight in [('base',0),('adapted-query',1),('mixture-query',w)]:
  v=mix_query_vectors(b,a,weight);rankings=[]
  for n in sizes:
   ids=pools[str(n)]['ids'];assert len(ids)==len(set(ids))==n and set.union(*gold.values())<=set(ids)
   matrix=np.array(docs[[imap[s] for s in ids]]);ranks={}
   for i,qid in enumerate(qids):
    t=time.perf_counter();r,scores=retrieve(v[i],matrix,ids);search=time.perf_counter()-t
    enc=(bt[qid] if weight<1 else 0.)+(at[qid] if weight>0 else 0.)
    record={'condition':name,'library_size':n,'query_id':qid,'ranking':r,'scores':scores,'status':'ok',
            'timing':active_timing(enc,search),'provenance':query_provenance(l,panel[i]['query'],l['query']['adapter_sha256'] if weight>0 else None)}
    ranks[qid]=r;rankings.append(record);dense[name,n,qid]=record
   out.append({'condition':name,'library_size':n,'evaluation':evaluate(gold,ranks)})
  dense_records.extend(rankings)
 write_rows(STUDY/'results/scaling-rankings.jsonl',dense_records)
 cfg=l['reranker'];specs=scaling_rerankers(l['primary_condition'])
 assert scope['reranker_conditions']=={name:list(spec) for name,spec in specs.items()}
 for trained in (False,True):
  adapter=CACHE/cfg['relative_cache_path'] if trained else None
  m,tok=load('reranker',adapter=adapter);renderer=Renderer(tok,'test')
  for name,(candidate_source,is_trained) in specs.items():
   if is_trained!=trained:continue
   path=STUDY/'results'/f'scaling-{name}.jsonl'
   done={(r['library_size'],r['query_id']):r['ranking'] for r in iter_rows(path)} if path.exists() else {}
   for n in sizes:
    for i,qid in enumerate(qids):
     key=(n,qid)
     if key in done:continue
     first=dense[candidate_source,n,qid];candidates=first['ranking'];scores=first['scores']
     k=min(cfg['candidates'],len(candidates));sequences=[];owners=[];selection=[];rr=[];query=panel[i]['query']
     meta={'condition':name,'library_size':n,'query_id':qid,'candidates':candidates,'dense_scores':scores,
           'candidate_source':candidate_source,'provenance':query_provenance(l,query,
            l['query']['adapter_sha256'] if candidate_source!='base' and w>0 else None,
            cfg['adapter_sha256'] if trained else None,True)}
     try:
      t=time.perf_counter()
      for j,sid in enumerate(candidates[:k]):
       pieces=renderer.render(query,sid,cfg['budget'],cfg['full_coverage'])
       selection.append([{**span,'tokens':len(tokens),'input_sha256':digest(tokens)} for tokens,span in pieces])
       for tokens,_ in pieces:sequences.append(tokens);owners.append(j)
      render=time.perf_counter()-t;t=time.perf_counter();values=score_sequences(m,tok,sequences);rerank=time.perf_counter()-t
      rr=[max(v for v,j in zip(values,owners) if j==z) for z in range(k)]
      order=normalized_fusion(scores[:k],rr,cfg['fusion_weight']);ranking=[candidates[j] for j in order]+candidates[k:]
      record={**meta,'ranking':ranking,'reranker_scores':rr,'selections':selection,'status':'ok',
              'timing':active_timing(first['timing']['encode'],first['timing']['search'],render,rerank)}
     except Exception as e:
      append(path,{**meta,'ranking':[],'reranker_scores':rr,'selections':selection,'status':'failed','error_type':type(e).__name__})
      raise
     append(path,record);done[key]=ranking
    ranks={q:done[n,q] for q in qids}
    out.append({'condition':name,'library_size':n,'evaluation':evaluate(gold,ranks)})
    print(name,'scaling pool',n,'finished with',len(ranks),'fixed queries',flush=True)
   assert set(done)=={(n,q) for n in sizes for q in qids}
  del m,renderer;torch.cuda.empty_cache()
 save(STUDY/'results/scaling.json',out)
 print('Scaling complete',len(out),'pool/condition combinations of',len(qids),'queries',flush=True)


if __name__=='__main__':main()
