"""Reuse a matched checkpoint's FP16 scores only at the original tensor shape."""
from collections import defaultdict
from common import *

AUDIT='results/reranker/shape_preserving_score_reuse.json'
AUDIT_STRIDE=137
CONFIRM_STRIDE=67
REFERENCE={'released-combined':'released-reranker','trained-combined':'trained-reranker',
           'trained-combined-seed43':'trained-reranker-seed43'}
CONFIRM_REFERENCE={'released-combined-confirmation':'released-confirmation',
                   'trained-combined-confirmation':'trained-confirmation'}


class ScoreReuseMismatch(RuntimeError):
 pass


def shape_keys(lengths,hashes,batch_size):
 assert len(lengths)==len(hashes)
 for start in range(0,len(lengths),batch_size):
  lens=lengths[start:start+batch_size]
  for h in hashes[start:start+batch_size]:
   yield len(lens),max(lens),h


def keyed_scores(record,batch_size):
 if record['status']!='ok':return
 windows=record['selections'];scores=record['reranker_scores']
 assert len(windows)==len(scores)==record['scored_candidate_count']
 assert all(len(w)==1 and not w[0]['full_coverage'] for w in windows)
 keys=shape_keys([w[0]['tokens'] for w in windows],[w[0]['input_sha256'] for w in windows],batch_size)
 yield from zip(keys,scores,strict=True)


def saved_scores(record,batch_size):
 out={}
 for k,v in keyed_scores(record,batch_size):
  assert math.isfinite(v)
  if k in out and out[k]!=v:raise ScoreReuseMismatch('Conflicting cached scores at one exact input and shape')
  out[k]=v
 return out


def plan(sequences,hashes,reference,batch_size):
 keys=list(shape_keys(list(map(len,sequences)),hashes,batch_size))
 values=[reference.get(k) for k in keys];missing=defaultdict(list);work=[]
 for i,((size,length,_),v) in enumerate(zip(keys,values,strict=True)):
  if v is None:missing[size,length].append(i)
 for (size,length),indices in missing.items():
  for start in range(0,len(indices),size):
   real=indices[start:start+size]
   # Repeat a valid row for unused slots. Do not introduce all-padding attention rows.
   padded=real+[real[-1]]*(size-len(real))
   work.append((length,real,[sequences[i] for i in padded]))
 return values,work


def score_repacked(m,tok,sequences,hashes,reference,batch_size):
 from reranker import torch,batch_scores,score_sequences
 if not reference:
  return score_sequences(m,tok,sequences,batch_size),{'reused':0,'fresh_inputs':len(sequences),
           'forward_batches':math.ceil(len(sequences)/batch_size),'dummy_inputs':0}
 values,work=plan(sequences,hashes,reference,batch_size)
 reused=sum(v is not None for v in values)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
  for length,indices,batch in work:
   scored=batch_scores(m,tok,batch,padded_length=length).float().cpu().tolist()
   for i,v in zip(indices,scored):values[i]=v
 assert all(v is not None and math.isfinite(v) for v in values)
 return values,{'reused':reused,'fresh_inputs':len(sequences)-reused,'forward_batches':len(work),
                'dummy_inputs':sum(len(batch)-len(indices) for _,indices,batch in work)}


def records_sha256(path):
 import gzip
 raw=record_path(path)
 if raw.suffix!='.gz':return sha(raw)
 with gzip.open(raw,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_records(source_path,target_path,source,*,batch_size,stride,count):
 """Check every reused score against its original record, without a model."""
 source_hash=records_sha256(source_path)
 seen=successful=reused=audit=0
 for i,(x,y) in enumerate(zip(iter_rows(source_path),iter_rows(target_path),strict=True)):
  seen+=1;assert x['query_id']==y['query_id']
  for k in ('reranker_adapter_sha256','reranker_source_manifest_sha256'):
   assert x['provenance'][k]==y['provenance'][k]
  if y['status']!='ok':assert 'offline_score_reuse' not in y;continue
  successful+=1
  reference=saved_scores(x,batch_size);target_scores=list(keyed_scores(y,batch_size))
  stats=y['offline_score_reuse'];common=[(k,v) for k,v in target_scores if k in reference]
  assert stats['source']==source and stats['source_sha256']==source_hash
  assert len(target_scores)==y['scored_candidate_count']==stats['reused']+stats['fresh_inputs']
  assert len(common)==stats['reused'] and all(reference[k]==v for k,v in common)
  expected=i%stride==0
  assert stats['checked_against_original_batch_execution'] is expected
  assert (stats['uncached_audit_rerank_seconds'] is not None)==expected
  reused+=len(common);audit+=int(expected)
 assert seen==count
 return {'source':source,'source_uncompressed_sha256':source_hash,'query_count':seen,
         'successful_queries':successful,'reused_scores_exact':reused,'recorded_live_direct_checks':audit}


def verify_confirmation_records(study):
 path=study/AUDIT
 if not path.exists():return {}
 gate=read(path)
 if gate['status']!='admitted':return {}
 assert gate['confirmation_exact_check_stride']==CONFIRM_STRIDE
 root=study/'results/reranker'
 return {name:verify_records(root/source/'panel.jsonl',root/name/'panel.jsonl',source,
          batch_size=gate['batch_size'],stride=CONFIRM_STRIDE,count=512) for name,source in CONFIRM_REFERENCE.items()}


def verify_official_records(study,lock):
 contract=lock.get('shape_preserving_score_reuse',{})
 if not contract.get('enabled'):return {}
 assert sha(study/contract['audit_path'])==contract['audit_sha256']
 root=study/'results/final'
 return {name:verify_records(root/f'{source}.jsonl',root/f'{name}.jsonl',source,batch_size=contract['batch_size'],
         stride=contract['audit_stride'],count=lock['test_query_count']) for name,source in contract['sources'].items()}


def verify():
 """No labels or official inputs: compare real scores for every selected checkpoint."""
 import time
 from reranker import Renderer,admitted_inference_batch,load,score_sequences,torch
 cfg=read(STUDY/'results/reranker/SELECTION_LOCK.json')
 secondary=read(STUDY/'results/SECONDARY_SEED_LOCK.json')['reranker']
 batch_size=admitted_inference_batch()
 assert not cfg['full_coverage'] and not secondary['full_coverage']
 assert (cfg['budget'],cfg['candidates'])==(secondary['budget'],secondary['candidates'])==(2048,50)
 p=STUDY/f'results/reranker/{cfg["trial"]}/selection-{cfg["best_step"]:05d}.jsonl'
 b=STUDY/'results/reranker/combined-query/selection.jsonl'
 frozen=rows(p);adapted=rows(b);assert len(frozen)==len(adapted)==512
 matched=0;equal=True
 for x,y in zip(frozen,adapted,strict=True):
  assert x['query_id']==y['query_id']
  x,y=saved_scores(x,batch_size),saved_scores(y,batch_size)
  common=x.keys()&y.keys();matched+=len(common);equal=equal and all(x[k]==y[k] for k in common)
 s,q,_=data('train');texts={x['id']:x['query'] for x in q};del s,q
 examples=sorted(zip(frozen,adapted,strict=True),key=lambda xy:digest(xy[0]['query_id']))[:16]
 results={};expected_adapters={}
 for label,c in [('released',None),('seed42',cfg),('seed43',secondary)]:
  path=None if c is None else CACHE/c['relative_cache_path'];adapter=None if c is None else c['adapter_sha256']
  if path is not None:assert sha(path/'adapter_model.safetensors')==adapter
  expected_adapters[label]=adapter
  m,tok=load('reranker',adapter=path);renderer=Renderer(tok,'train')
  counters={'reused':0,'fresh_inputs':0,'forward_batches':0,'dummy_inputs':0}
  maximum=0.;mismatches=0;direct=[];repacked=[]
  for number,(x,y) in enumerate(examples,1):
   inputs=[];hashes=[]
   for r in (x,y):
    group=[]
    for sid in r['candidates'][:cfg['candidates']]:
     [(seq,_)]=renderer.render(texts[r['query_id']],sid,cfg['budget'],False);group.append(seq)
    group_hashes=list(map(digest,group));assert group_hashes==[w[0]['input_sha256'] for w in r['selections']]
    inputs.append(group);hashes.append(group_hashes)
   xscore=score_sequences(m,tok,inputs[0],batch_size)
   start=time.perf_counter();yscore=score_sequences(m,tok,inputs[1],batch_size);direct.append(time.perf_counter()-start)
   if label=='seed42':
    assert xscore==x['reranker_scores'] and yscore==y['reranker_scores'],'Original development scores failed exact reload'
   reference=dict(zip(shape_keys(list(map(len,inputs[0])),hashes[0],batch_size),xscore,strict=True))
   start=time.perf_counter();other,stats=score_repacked(m,tok,inputs[1],hashes[1],reference,batch_size);repacked.append(time.perf_counter()-start)
   diffs=[abs(a-b) for a,b in zip(yscore,other,strict=True)]
   maximum=max(maximum,max(diffs));mismatches+=sum(d>0 for d in diffs)
   for k,v in stats.items():counters[k]+=v
   if number%4==0:print(label,'exact score fixture',number,'/',len(examples),'mismatches',mismatches,flush=True)
  results[label]={**counters,'queries':len(examples),'score_mismatches':mismatches,'max_absolute_score_error':maximum,
                  'native_seconds':summary(direct),'reuse_and_pack_seconds':summary(repacked)}
  print(label,results[label],flush=True)
  del m,renderer;torch.cuda.empty_cache()
 use=equal and matched>0 and all(r['score_mismatches']==0 and r['reused']>0 and r['fresh_inputs']>0 and
              r['dummy_inputs']>0 and r['reuse_and_pack_seconds']['total']<=.85*r['native_seconds']['total']
              for r in results.values())
 save(STUDY/AUDIT,{'status':'admitted' if use else 'use_original_batch_execution',
                   'development_only':True,'query_ids':[x['query_id'] for x,_ in examples],
                   'development_saved_shared_exact_shape_pairs':matched,'all_saved_scores_exact':equal,
                   'development_saved_score_sources':{p.relative_to(STUDY).as_posix():sha(p),b.relative_to(STUDY).as_posix():sha(b)},
                   'model_source_sha256':sha(STUDY/'manifests/reranker_source.json'),'adapter_sha256':expected_adapters,
                   'batch_size':batch_size,'budget':cfg['budget'],'candidates':cfg['candidates'],
                   'live_checkpoints':results,'rule':'Require bit-identical scores and at least 15% less active scoring time in each checkpoint fixture. Preserve batch size and padded length for each input; group independent rows of equal shape. No model, evidence, candidate or fusion changes.',
                   'official_exact_check_stride':AUDIT_STRIDE,'confirmation_exact_check_stride':CONFIRM_STRIDE})


if __name__=='__main__':verify()
