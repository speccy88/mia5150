"""Lossless portable vectors and CPU-only exact ranking/metric replay."""
from common import *
import argparse,shutil,gzip

VECTOR_COUNTS={'train-documents':10123,'train-queries':47455,'test-documents':6006,'test-base-queries':4392,'test-adapted-queries':4392,'test-adapted-queries-seed43':4392,'scaling-base-queries':20,'scaling-adapted-queries':20}

def expected_conditions(lock):
 result={'reconstructed-base','adapted-query','adapted-query-seed43','released-reranker','trained-reranker','trained-reranker-seed43','released-best-pipeline','released-reranker-unfused','trained-reranker-unfused','scaling'}
 if lock['query']['mixture_weight']!=1:result.add('mixture-query')
 if lock['secondary_query']['mixture_weight']!=1:result.add('mixture-query-seed43')
 if lock['primary_condition']=='trained-combined':result.update(('released-combined','trained-combined','trained-combined-seed43'))
 assert lock['primary_condition'] in result
 return result

def export():
 read(STUDY/'FINAL_EVAL_LOCK.json');arrays={};records={};panels={};folder=STUDY/'vectors';folder.mkdir(exist_ok=True)
 for name,count in VECTOR_COUNTS.items():
  p=CACHE/'vectors'/name
  meta=read(p/'contract.json');done=read(p/'complete.json');assert done['status']=='complete' and done['count']==count;assert sha(p/'vectors.npy')==done['array_sha256'];assert sha(p/'journal.jsonl')==done['journal_sha256'];a=np.load(p/'vectors.npy',mmap_mode='r');assert a.dtype==np.float32 and a.shape==(len(meta['ids']),1024);assert len(meta['ids'])==len(set(meta['ids']))==len(meta['text_hashes'])==count;assert np.isfinite(a).all();assert np.max(np.abs(np.linalg.norm(a,axis=1)-1))<1e-5
  receipts=folder/'receipts'/name;receipts.mkdir(parents=True,exist_ok=True)
  for filename in ('contract.json','complete.json','journal.jsonl'):shutil.copyfile(p/filename,receipts/filename)
  key=name.replace('-','_');arrays[key]=np.asarray(a);arrays[key+'_ids']=np.asarray(meta['ids']);arrays[key+'_text_sha256']=np.asarray(meta['text_hashes']);records[name]={'shape':list(a.shape),'dtype':'float32','array_sha256':done['array_sha256'],'contract_sha256':sha(p/'contract.json'),'effective_budget':meta['budget'],'adapter_tensor_sha256':meta.get('adapter_tensor_sha256')}
 selection=read(STUDY/'results/query/SELECTION_LOCK.json')
 secondary=read(STUDY/'results/SECONDARY_SEED_LOCK.json')['query']
 membership=read(STUDY/'manifests/split_membership.json')
 development_sources=[('selection','selection','selection','reload-selection',selection),('confirmation','confirmation','confirmation','confirmation-all-preregistered-mixtures',selection),('secondary_selection','selection','selection-secondary',f'secondary-seed43/selection-{secondary["best_step"]:05d}',secondary)]
 for label,part,stem,name,checkpoint in development_sources:
  source=CACHE/'query-selected'/f'{stem}-adapted.npy';ids=read(CACHE/'query-selected'/f'{stem}-ids.json')
  assert ids==membership[part+'_query_ids'] and len(ids)==512 and len(set(ids))==512
  vectors=np.load(source,mmap_mode='r',allow_pickle=False)
  receipt=STUDY/'results/query'/f'{name}.json';expected=read(receipt)['adapted_vectors_sha256']
  assert vectors.dtype==np.float32 and vectors.shape==(len(ids),1024)
  assert np.isfinite(vectors).all() and np.max(np.abs(np.linalg.norm(vectors,axis=1)-1))<1e-5
  assert digest(vectors.tobytes().hex())==expected
  key=f'dev_{label}_adapted_queries';arrays[key]=np.asarray(vectors);arrays[key+'_ids']=np.asarray(ids)
  panels[label]={'part':part,'array_key':key,'npy_sha256':sha(source),'vector_receipt_digest':expected,'ids':len(ids),'adapter_sha256':checkpoint['adapter_sha256'],'receipt':receipt.relative_to(STUDY).as_posix(),'receipt_sha256':sha(receipt)}
 for split in ('train','test'):
  s,q,g=data(split);pairs=[(qid,sid) for qid in sorted(g) for sid in sorted(g[qid])];arrays[split+'_qrel_query_ids']=np.array([x[0] for x in pairs]);arrays[split+'_qrel_skill_ids']=np.array([x[1] for x in pairs])
 panel=read(REPO/'experiments/benchmarks/skill-discovery-oauth-v1-20261103/data/test_queries.json');arrays['scaling_qrel_query_ids']=np.asarray([f'panel-{i:02d}' for i in range(len(panel))]);arrays['scaling_qrel_skill_ids']=np.asarray([x['gold_id'] for x in panel])
 target=folder/'dense-vectors-v1.npz';tmp=folder/'dense-vectors-v1.tmp.npz';np.savez_compressed(tmp,**arrays);tmp.replace(target);save(folder/'manifest.json',{'archive_sha256':sha(target),'bytes':target.stat().st_size,'schema':1,'arrays':records,'development_query_panels':panels,'dimension':1024,'normalization':'float32 L2','compression':'lossless ZIP, no quantization','lock_sha256':sha(STUDY/'FINAL_EVAL_LOCK.json')})
 print('Exported portable vectors',target.stat().st_size)

def replay_development(a,panels):
 docs=a['train_documents'];doc_ids=a['train_documents_ids'].tolist();anchors=a['train_queries'];anchor_row={q:i for i,q in enumerate(a['train_queries_ids'].tolist())}
 assert len(doc_ids)==len(set(doc_ids))==10123
 gold={}
 for q,s in zip(a['train_qrel_query_ids'].tolist(),a['train_qrel_skill_ids'].tolist()):gold.setdefault(q,set()).add(s)
 result={}
 for label,meta in panels.items():
  part=meta['part']
  key=meta['array_key'];adapt=a[key];ids=a[key+'_ids'].tolist();base=np.array([anchors[anchor_row[q]] for q in ids]);receipt=STUDY/meta['receipt']
  assert sha(receipt)==meta['receipt_sha256'] and digest(adapt.tobytes().hex())==meta['vector_receipt_digest']
  saved=read(receipt);assert saved['part']==part and len(ids)==len(set(ids))==512 and set(saved['mixtures'])==set(saved['rankings'])=={'0','0.25','0.5','0.75','1'}
  relevant={q:gold[q] for q in ids}
  for weight in (0,.25,.5,.75,1):
   mix=mix_query_vectors(base,adapt,weight)
   rankings={q:retrieve(v,docs,doc_ids)[0] for q,v in zip(ids,mix)}
   assert rankings==saved['rankings'][str(weight)],(part,weight)
   actual=evaluate(relevant,rankings)
   assert all(abs(v-saved['mixtures'][str(weight)]['aggregate'][m])<1e-12 for m,v in actual['aggregate'].items())
  result[label]={'queries':len(ids),'skills':len(doc_ids),'mixtures':5,'rankings_identical':len(ids)*5,'metrics_identical':True}
 return result

def replay_scaling(a,lock,docs,dids):
 scope=lock['scaling'];pool_path=STUDY/'manifests/scaling_pools.json';assert sha(pool_path)==scope['pool_sha256']
 pools=read(pool_path);imap={sid:i for i,sid in enumerate(dids)};qmap={qid:i for i,qid in enumerate(a['scaling_base_queries_ids'].tolist())}
 sb=a['scaling_base_queries'];sa=a['scaling_adapted_queries'];sg={str(q):{str(s)} for q,s in zip(a['scaling_qrel_query_ids'],a['scaling_qrel_skill_ids'],strict=True)}
 assert len(qmap)==len(sg)==scope['query_count']==20 and set(qmap)==set(sg) and a['scaling_adapted_queries_ids'].tolist()==list(qmap)
 labels=read(STUDY/'manifests/scaling_query_labels.json')
 assert len(labels)==20 and {row['query_id']:{row['gold_id']} for row in labels}==sg
 sizes={int(n):(p['ids'],np.array(docs[[imap[s] for s in p['ids']]])) for n,p in pools.items()}
 assert set(sizes)==set(scope['library_sizes'])==set(SCALING_SIZES)
 assert all(len(ids)==len(set(ids))==n for n,(ids,_) in sizes.items())
 q_hashes=dict(zip(qmap,a['scaling_base_queries_text_sha256'].tolist(),strict=True))
 assert a['scaling_adapted_queries_text_sha256'].tolist()==list(q_hashes.values())
 checks={};pool_candidates={};maxerr=0.;failed=0
 def add(name,n,qid,rank):
  group=checks.setdefault((name,n),{});assert qid not in group;group[qid]=rank
 weight={'base':0,'adapted-query':1,'mixture-query':lock['query']['mixture_weight']}
 for r in iter_rows(STUDY/'results/scaling-rankings.jsonl'):
  name=r['condition'];n=r['library_size'];qid=r['query_id'];i=qmap[qid];w=weight[name];v=mix_query_vectors(sb[i],sa[i],w)
  ids,matrix=sizes[n];rank,sc=retrieve(v,matrix,ids)
  assert r['status']=='ok' and rank==r['ranking'] and r['provenance']['instructed_query_text_sha256']==q_hashes[qid]
  maxerr=max(maxerr,max(abs(x-y) for x,y in zip(sc,r['scores'],strict=True)));add(name,n,qid,rank)
  pool_candidates[name,n,qid]=(rank,sc)
 cfg=lock['reranker'];w=cfg['fusion_weight']
 for name,(source,trained) in scope['reranker_conditions'].items():
  for r in iter_rows(STUDY/'results'/f'scaling-{name}.jsonl'):
   n=r['library_size'];qid=r['query_id'];dense_rank,dense_scores=pool_candidates[source,n,qid]
   assert dense_rank==r['candidates'] and source==r['candidate_source']
   assert max(abs(x-y) for x,y in zip(dense_scores,r['dense_scores'],strict=True))<1e-6
   provenance=r['provenance']
   assert provenance['instructed_query_text_sha256']==q_hashes[qid] and provenance['reranker_source_manifest_sha256']==lock['reranker_source_sha256']
   assert provenance['reranker_adapter_sha256']==(cfg['adapter_sha256'] if trained else None)
   if r['status']=='ok':
    k=min(cfg['candidates'],len(dense_rank));assert len(r['reranker_scores'])==len(r['selections'])==k
    # Independent rank-percentile implementation; dense input order breaks ties.
    rerank=sorted(range(k),key=lambda i:(-r['reranker_scores'][i],i));pos={i:j for j,i in enumerate(rerank)}
    order=sorted(range(k),key=lambda i:-((1-w)*(1-i/max(1,k-1))+w*(1-pos[i]/max(1,k-1))))
    rank=[dense_rank[j] for j in order]+dense_rank[k:]
   else:assert r['status']=='failed';failed+=1;rank=[]
   assert rank==r['ranking'];add(name,n,qid,rank)
 conditions=set(weight)|set(scope['reranker_conditions'])
 expected={(name,n) for name in conditions for n in sizes}
 assert set(checks)==expected and maxerr<1e-6
 measured=set()
 for r in read(STUDY/'results/scaling.json'):
  key=(r['condition'],r['library_size']);assert key not in measured;measured.add(key)
  rankings=checks[key];assert set(rankings)==set(sg);ev=evaluate(sg,rankings)
  for metric,v in ev['aggregate'].items():assert abs(v-r['evaluation']['aggregate'][metric])<1e-12
  assert ev['per_query']==r['evaluation']['per_query']
 assert measured==expected
 return {'rankings_identical':sum(map(len,checks.values())),'conditions':len(checks),'exact_pools':len(sizes),
         'failed_inference_in_fixed_denominator':failed,'metrics_identical':True,'embedding_max_score_error':maxerr}


def replay():
 p=STUDY/'vectors/dense-vectors-v1.npz';manifest=read(STUDY/'vectors/manifest.json');assert sha(p)==manifest['archive_sha256'];a=np.load(p,allow_pickle=False);docs=a['test_documents'];dids=a['test_documents_ids'].tolist();base=a['test_base_queries'];qids=a['test_base_queries_ids'].tolist();gold={q:set() for q in qids}
 for q,s in zip(a['test_qrel_query_ids'],a['test_qrel_skill_ids']):gold[str(q)].add(str(s))
 comparisons={};candidate_replay={};lock_path=STUDY/'FINAL_EVAL_LOCK.json';lock=read(lock_path);assert sha(lock_path)==manifest['lock_sha256']
 assert set(manifest['arrays'])==set(VECTOR_COUNTS) and len(dids)==len(set(dids))==6006 and len(qids)==len(set(qids))==4392 and all(gold.values()) and sum(map(len,gold.values()))==7187
 embedding_conditions=[('reconstructed-base',0,'test_adapted_queries'),('adapted-query',1,'test_adapted_queries'),('mixture-query',lock['query']['mixture_weight'],'test_adapted_queries')]
 if 'secondary_query' in lock:embedding_conditions += [('adapted-query-seed43',1,'test_adapted_queries_seed43'),('mixture-query-seed43',lock['secondary_query']['mixture_weight'],'test_adapted_queries_seed43')]
 for name,w,array_key in embedding_conditions:
  adapt=a[array_key] if array_key in a else None
  path=STUDY/'results/final'/f'{name}.jsonl'
  if not records_exist(path):continue
  if w>0:assert a[array_key+'_ids'].tolist()==qids
  saved={x['query_id']:x for x in rows(path)};r={};maxerr=0;candidate_replay[name]={}
  for i,q in enumerate(qids):
   v=mix_query_vectors(base[i],adapt[i] if w>0 else None,w)
   rank,score=retrieve(v,docs,dids);assert rank==saved[q]['ranking'],(name,q);maxerr=max(maxerr,max(abs(x-y) for x,y in zip(score,saved[q]['scores'])));r[q]=rank;candidate_replay[name][q]=(rank,score)
  assert maxerr<1e-6;metrics=evaluate(gold,r);expected=read(path.with_suffix('.metrics.json'))
  for k,v in metrics['aggregate'].items():assert abs(v-expected['aggregate'][k])<1e-12
  comparisons[name]={'rankings_identical':len(r),'max_score_error':maxerr,'metrics_identical':True}
 # Reranker replay uses saved local per-candidate scores; no teacher/model required.
 for name in ('released-reranker','trained-reranker','released-best-pipeline','released-reranker-unfused','trained-reranker-unfused','released-combined','trained-combined','trained-reranker-seed43','released-combined-seed43','trained-combined-seed43'):
  path=STUDY/'results/final'/f'{name}.jsonl'
  if not records_exist(path):continue
  cfg=lock['released_best_pipeline'] if name=='released-best-pipeline' else lock['secondary_reranker'] if name.endswith('-seed43') else lock['reranker'];k=cfg['candidates'];w=1. if name.endswith('-unfused') else cfg['fusion_weight'];rankings={}
  candidate_source='reconstructed-base'
  if 'combined' in name:
   secondary=name.endswith('-seed43');qcfg=lock['secondary_query'] if secondary else lock['query'];mix=qcfg['mixture_weight'];suffix='-seed43' if secondary else ''
   if mix>0:candidate_source=('adapted-query' if mix==1 else 'mixture-query')+suffix
  def fusion(ds,rs):
   n=len(ds);dr=np.empty(n);rr=np.empty(n);dr[np.argsort(-np.asarray(ds),kind='stable')]=np.arange(n);rr[np.argsort(-np.asarray(rs),kind='stable')]=np.arange(n);return np.argsort(-((1-w)*(1-dr/max(1,n-1))+w*(1-rr/max(1,n-1))),kind='stable')
  for r in rows(path):
   qid=r['query_id'];dense_rank,dense_scores=candidate_replay[candidate_source][qid]
   assert dense_rank==r['candidates'] and len(r['dense_scores'])==len(dense_scores)==50
   assert max(abs(x-y) for x,y in zip(dense_scores,r['dense_scores']))<1e-6
   rank=[] if r['status']!='ok' else [r['candidates'][i] for i in fusion(r['dense_scores'][:k],r['reranker_scores'][:k])]+r['candidates'][k:];assert rank==r['ranking'];rankings[r['query_id']]=rank
  assert len(rankings)==4392;ev=evaluate(gold,rankings);expected=read(path.with_suffix('.metrics.json'))
  for metric,v in ev['aggregate'].items():assert abs(v-expected['aggregate'][metric])<1e-12
  comparisons[name]={'dense_candidate_rankings_identical':4392,'saved_candidate_score_rankings_identical':4392,'metrics_identical':True,'note':'Candidate retrieval from vectors; reranker score replay without transformer inference'}
 comparisons['scaling']=replay_scaling(a,lock,docs,dids)
 assert set(comparisons)==expected_conditions(lock),(set(comparisons),expected_conditions(lock))
 development=replay_development(a,manifest['development_query_panels']);assert set(development)=={'selection','confirmation','secondary_selection'}
 from replay_development_reranker import replay as replay_reranker_panels
 reranking_development=replay_reranker_panels(a,manifest,lock)
 from score_reuse import verify_confirmation_records,verify_official_records
 reuse_checks=verify_official_records(STUDY,lock)
 confirmation_reuse=verify_confirmation_records(STUDY)
 assert confirmation_reuse==read(STUDY/'results/CONFIRMATION.json')['offline_score_reuse_verified']
 save(STUDY/'results/vector_replay.json',{'status':'passed','conditions':comparisons,'development_query_panels':development,'development_reranker_panels':reranking_development,'offline_score_reuse_verified':reuse_checks,'confirmation_score_reuse_verified':confirmation_reuse,'model_or_gpu_imports_required':False,'network_required':False});print(json.dumps({'conditions':comparisons,'development_query_panels':development,'development_reranker_panels':reranking_development},indent=2))
def hydrate(subset='train'):
 folder=STUDY/'vectors';manifest=read(folder/'manifest.json');archive=folder/'dense-vectors-v1.npz';assert sha(archive)==manifest['archive_sha256'];a=np.load(archive,allow_pickle=False);restored=[]
 for name,meta in manifest['arrays'].items():
  if subset!='all' and not name.startswith(subset+'-'):continue
  p=CACHE/'vectors'/name;p.mkdir(parents=True,exist_ok=True);target=p/'vectors.npy';receipts=folder/'receipts'/name;contract=read(receipts/'contract.json');done=read(receipts/'complete.json');key=name.replace('-','_');assert a[key+'_ids'].tolist()==contract['ids']
  if target.exists():assert sha(target)==done['array_sha256'],'Refusing to replace a different or incomplete existing vector array'
  else:
   tmp=p/'vectors.tmp.npy';np.save(tmp,a[key],allow_pickle=False);assert sha(tmp)==done['array_sha256'];tmp.replace(target)
  def copy_verified(filename):
   source=receipts/filename;dest=p/filename
   if dest.exists():assert sha(dest)==sha(source),'Refusing different existing receipt'
   else:
    tmp=p/(filename+'.tmp');shutil.copyfile(source,tmp);assert sha(tmp)==sha(source);tmp.replace(dest)
  copy_verified('contract.json');journal=p/'journal.jsonl'
  if not journal.exists():
   source=record_path(receipts/'journal.jsonl');opener=gzip.open if source.suffix=='.gz' else open;tmp=p/'journal.jsonl.tmp'
   with opener(source,'rb') as stream,tmp.open('wb') as dest:shutil.copyfileobj(stream,dest,1024*1024)
   assert sha(tmp)==done['journal_sha256'];tmp.replace(journal)
  assert sha(journal)==done['journal_sha256'];copy_verified('complete.json');restored.append(name)
 print('Hydrated verified immutable vector caches without inference',restored)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['export','replay','hydrate']);p.add_argument('--subset',choices=['train','test','all'],default='train');a=p.parse_args()
 if a.command=='hydrate':hydrate(a.subset)
 else:{'export':export,'replay':replay}[a.command]()
