"""Locked final inference, fixed denominator and complete per-query evidence."""
from engine import *
import argparse

def lock():
 p=STUDY/'FINAL_EVAL_LOCK.json';assert p.exists(),'Freeze the complete evaluation lock before new official inference';l=read(p);assert l['locked'] and l['test_query_count']==4392 and l['test_skill_count']==6006
 for component in l['adapters'].values():assert sha(CACHE/component['relative_cache_path']/'adapter_model.safetensors')==component['sha256']
 sources={'embedding_source_sha256':STUDY/'manifests/embedding_source.json','reranker_source_sha256':STUDY/'manifests/reranker_source.json','dataset_source_sha256':STUDY/'manifests/data_source.json','split_membership_sha256':STUDY/'manifests/split_membership.json','protocol_sha256':STUDY/'protocol.json','inference_config_sha256':STUDY/'INFERENCE_CONFIG.json','archived_protocol_sha256':OLD/'protocol.json','archived_reference_rankings_sha256':OLD/'results/baseline_skillret_06b.json','pipeline_selection_sha256':STUDY/'results/PIPELINE_SELECTION_LOCK.json','confirmation_sha256':STUDY/'results/CONFIRMATION.json','development_review_sha256':STUDY/'results/development_review.json','secondary_selection_sha256':STUDY/'results/SECONDARY_SEED_LOCK.json'}
 for key,source in sources.items():assert sha(source)==l[key],key
 assert sha(STUDY/'results/reranker/pilot_matched_comparison.json')==l['pilot_matched_audit_sha256']
 if l['primary_condition']=='trained-combined':assert sha(STUDY/'results/reranker/combined_selection_audit.json')==l['combined_selection_audit_sha256']
 reuse=l.get('shape_preserving_score_reuse')
 if reuse:assert sha(STUDY/reuse['audit_path'])==reuse['audit_sha256']
 for name,h in l['adapter_reload_receipts'].items():assert sha(STUDY/name)==h,name
 for name,h in l['source_sha256'].items():assert sha(SRC/name)==h,name
 for name,h in l['official_test_inputs'].items():assert sha(TRAIN_CACHE/name)==h,name
 return l

def encoder_times(name):
 return {r['id']:r['seconds'] for r in rows(CACHE/'vectors'/name/'journal.jsonl')}

def active_timing(encode,search,render=0.,rerank=0.,**extra):
 return {'encode':encode,'search':search,'render':render,'rerank':rerank,'active_component_sum':encode+search+render+rerank,**extra}

def query_provenance(l,text,embedding_adapter=None,reranker_adapter=None,uses_reranker=False):
 return {'instructed_query_text_sha256':digest(PREFIX+text),'embedding_source_manifest_sha256':l['embedding_source_sha256'],'embedding_adapter_sha256':embedding_adapter,**({'reranker_source_manifest_sha256':l['reranker_source_sha256'],'reranker_adapter_sha256':reranker_adapter} if uses_reranker else {})}

def baseline():
 l=lock();m,tok=load();s,q,g=data('test');texts=[fulltext(x) for x in s];qt=[PREFIX+x['query'] for x in q];old=read(OLD/'protocol.json');mod=module('old_baseline_protocol',OLD/'run_skillret_baseline.py');assert mod.text_digest(texts)==old['document_text_sha256'];assert mod.text_digest(qt)==old['query_text_sha256']
 counts=[len(tok.encode(t,truncation=False)) for t in texts];assert sum(n>32768 for n in counts)==2;save(STUDY/'results/reconstructed_test_input_audit.json',{'document_text_sha256':mod.text_digest(texts),'query_text_sha256':mod.text_digest(qt),'two_truncated_document_ids':[x['id'] for x,n in zip(s,counts) if n>32768],'document_tokens':summary(counts),'raw_token_counts':counts,'rule':'same historical right truncation at32768, no complete-source substitution'})
 docs=encode_cache(m,tok,texts,[x['id'] for x in s],'test-documents',32768);queries=encode_cache(m,tok,qt,[x['id'] for x in q],'test-base-queries',32768);ids=[x['id'] for x in s];arch=read(OLD/'results/baseline_skillret_06b.json');out=[];match=0;maxerr=0;bt=encoder_times('test-base-queries')
 for i,x in enumerate(q):
  t=time.perf_counter();r,scores=retrieve(queries[i],docs,ids);dt=time.perf_counter()-t;match+=int(r==arch['rankings'][x['id']]);maxerr=max(maxerr,max(abs(a-b) for a,b in zip(scores,arch['scores'][x['id']])));out.append({'query_id':x['id'],'ranking':r,'scores':scores,'status':'ok','provenance':query_provenance(l,x['query']),'timing':active_timing(bt[x['id']],dt)})
 write_rows(STUDY/'results/final/reconstructed-base.jsonl',out);evaluation=evaluate(g,{x['query_id']:x['ranking'] for x in out});save(STUDY/'results/final/reconstructed-base.metrics.json',evaluation)
 save(STUDY/'results/reconstruction_parity.json',{'identical_top50_count':match,'query_count':4392,'max_score_absolute_difference':maxerr,'archived_ndcg10':.8109902409310652,'reconstructed_ndcg10':evaluation['aggregate']['NDCG@10'],'status':'passed' if match==4392 else 'runtime_difference_investigate','runtime_change':'archived cu128 vs local cu126; exact same Transformers4.57.1 and tokenizer.22.2'})
 if match!=4392:raise RuntimeError('Investigate reconstruction ranking differences before claiming exact baseline parity')

def query():
 secondary='--secondary' in __import__('sys').argv;l=lock();cfg=l['secondary_query'] if secondary else l['query'];folder=CACHE/cfg['relative_cache_path'];m,tok=load(adapter=folder);s,q,g=data('test');del s;cache_name='test-adapted-queries-seed43' if secondary else 'test-adapted-queries';qids=[x['id'] for x in q];a=encode_cache(m,tok,[PREFIX+x['query'] for x in q],qids,cache_name,32768);b,bids=load_vectors('test-base-queries');assert bids==qids;docs,dids=load_vectors('test-documents');out=[];bt=encoder_times('test-base-queries');at=encoder_times(cache_name)
 for weight in sorted(set([1.,float(cfg['mixture_weight'])])):
  out=[]
  for i,x in enumerate(q):
   v=mix_query_vectors(b[i],a[i],weight)
   t=time.perf_counter();r,scores=retrieve(v,docs,dids);dt=time.perf_counter()-t;enc=(bt[x['id']] if weight<1 else 0.)+(at[x['id']] if weight>0 else 0.);out.append({'query_id':x['id'],'ranking':r,'scores':scores,'status':'ok','mixture_adapted_weight':weight,'provenance':query_provenance(l,x['query'],cfg['adapter_sha256'] if weight>0 else None),'timing':active_timing(enc,dt,query_encodings=2 if 0<weight<1 else 1)})
  name=('adapted-query' if weight==1 else 'mixture-query')+('-seed43' if secondary else '');write_rows(STUDY/'results/final'/f'{name}.jsonl',out);save(STUDY/'results/final'/f'{name}.metrics.json',evaluate(g,{x['query_id']:x['ranking'] for x in out}))

def rerank():
 from itertools import repeat
 from train_reranker import normalized_fusion
 from reranker import Renderer,admitted_inference_batch,score_sequences
 from score_reuse import ScoreReuseMismatch,records_sha256,saved_scores,score_repacked
 l=lock();cfg=l['reranker'];p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--released',action='store_true');p.add_argument('--secondary',action='store_true');p.add_argument('--combined',action='store_true');a=p.parse_args()
 if a.secondary:cfg=l['secondary_reranker']
 m,tok=load('reranker',adapter=None if a.released else CACHE/cfg['relative_cache_path']);s,q,g=data('test');del s;renderer=Renderer(tok,'test');base,bids=load_vectors('test-base-queries');assert bids==[x['id'] for x in q];docs,dids=load_vectors('test-documents')
 if a.combined:
  qc=l['secondary_query'] if a.secondary else l['query'];adapt,adaptids=load_vectors('test-adapted-queries-seed43' if a.secondary else 'test-adapted-queries');assert adaptids==bids;w=qc['mixture_weight'];base=mix_query_vectors(base,adapt,w)
 name=(('released-combined' if a.released else 'trained-combined') if a.combined else ('released-reranker' if a.released else 'trained-reranker'))+('-seed43' if a.secondary else '');path=STUDY/'results/final'/f'{name}.jsonl'
 done={r['query_id']:r['ranking'] for r in iter_rows(path)} if path.exists() else {}
 k=cfg['candidates'];weight=cfg['fusion_weight'];full=cfg['full_coverage'];budget=cfg['budget'];score_k=max(k,l.get('released_best_pipeline',cfg)['candidates']) if a.released and not a.combined else k;bt=encoder_times('test-base-queries');mix_weight=qc['mixture_weight'] if a.combined else 0.;at=encoder_times('test-adapted-queries-seed43' if a.secondary else 'test-adapted-queries') if mix_weight>0 else {}
 reuse=l.get('shape_preserving_score_reuse',{});source=reuse.get('sources',{}).get(name) if reuse.get('enabled') else None
 adapter_hash=None if a.released else cfg['adapter_sha256']
 if source:
  source_path=STUDY/'results/final'/f'{source}.jsonl';source_rows=iter_rows(source_path);source_hash=records_sha256(source_path)
  assert score_k==k==50 and budget==2048 and not full and admitted_inference_batch()==reuse['batch_size']
 else:source_rows=repeat(None,len(q))
 for i,(x,cached) in enumerate(zip(q,source_rows,strict=True)):
  qid=x['id']
  if cached is not None:
   assert cached['query_id']==qid and cached['provenance']['reranker_source_manifest_sha256']==l['reranker_source_sha256']
   assert cached['provenance']['reranker_adapter_sha256']==adapter_hash
  if qid in done:continue
  provenance=query_provenance(l,x['query'],qc['adapter_sha256'] if mix_weight>0 else None,adapter_hash,True)
  t=time.perf_counter();candidates,dense=retrieve(base[i],docs,dids,50);search=time.perf_counter()-t;rr=[];render_seconds=0;rerank_seconds=0;allspans=[]
  try:
   inputs=[];owner=[];render20=0.
   for j,sid in enumerate(candidates[:score_k]):
    t=time.perf_counter();pieces=renderer.render(x['query'],sid,budget,full);render_seconds+=time.perf_counter()-t;metas=[]
    for tokens,meta in pieces:
     inputs.append(tokens);owner.append(j);metas.append({**meta,'tokens':len(tokens),'input_sha256':digest(tokens)})
    allspans.append(metas)
    if j==19:render20=render_seconds
   tm={};reused=None
   if cached is not None:
    assert len(inputs)==len(allspans)==score_k and all(len(w)==1 for w in allspans)
    reference=saved_scores(cached,reuse['batch_size']);hashes=[w[0]['input_sha256'] for w in allspans]
    t=time.perf_counter();values,stats=score_repacked(m,tok,inputs,hashes,reference,reuse['batch_size']);rerank_seconds=time.perf_counter()-t
    checked=i%reuse['audit_stride']==0;check_seconds=None
    if checked:
     t=time.perf_counter();direct=score_sequences(m,tok,inputs);check_seconds=time.perf_counter()-t
     if values!=direct:
      save(STUDY/'results/final'/f'{name}.shape_reuse_mismatch.json',{'query_id':qid,'source':source,
           'source_sha256':source_hash,'max_absolute_score_error':max(abs(a-b) for a,b in zip(values,direct,strict=True))})
      raise ScoreReuseMismatch('Numerical reuse audit failed. Preserve evidence and stop before publishing this condition.')
    reused={**stats,'source':source,'source_sha256':source_hash,'checked_against_original_batch_execution':checked,
            'uncached_audit_rerank_seconds':check_seconds}
   else:
    t=time.perf_counter();values=score_sequences(m,tok,inputs,timing=tm);rerank_seconds=time.perf_counter()-t
   rr=[max(v for v,j in zip(values,owner) if j==i) for i in range(score_k)]
   enc=(bt[qid] if mix_weight<1 else 0.)+(at[qid] if mix_weight>0 else 0.);used_render=render20 if k==20 and score_k==50 else render_seconds;used_rerank=tm['first20_inputs_seconds'] if k==20 and score_k==50 else rerank_seconds
   order=normalized_fusion(dense[:k],rr[:k],weight);ranking=[candidates[j] for j in order]+candidates[k:];record={'query_id':qid,'ranking':ranking,'candidates':candidates,'dense_scores':dense,'reranker_scores':rr,'scored_candidate_count':score_k,'candidate_source':'selected_query' if a.combined else 'frozen_query','selections':allspans,'status':'ok','provenance':provenance,'timing':active_timing(enc,search,used_render,used_rerank,scored_superset_render_seconds=render_seconds,scored_superset_rerank_seconds=rerank_seconds,query_encodings=2 if 0<mix_weight<1 else 1)}
   if reused is not None:record['offline_score_reuse']=reused
  except ScoreReuseMismatch:
   raise
  except Exception as e:
   record={'query_id':qid,'ranking':[],'status':'failed','error_type':type(e).__name__,'candidates':candidates,'dense_scores':dense,'reranker_scores':rr,'provenance':provenance};append(path,record);raise
  append(path,record);done[qid]=record['ranking']
  if len(done)%10==0:print(name,len(done),'/',len(q),'rerank_seconds',round(rerank_seconds,2),flush=True)
 assert len(done)==4392 and set(done)==set(g);save(path.with_suffix('.metrics.json'),evaluate(g,done))

if __name__=='__main__':
 import sys
 {'baseline':baseline,'query':query,'reranker':rerank}[sys.argv[1]]()
