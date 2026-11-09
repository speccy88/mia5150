"""BCE LoRA with actual frozen-retriever negatives and common evidence policy."""
from reranker import *
import argparse,random,signal
STOP_REQUESTED=False
class PanelPaused(Exception):pass
def request_stop(signum,frame):
 global STOP_REQUESTED
 STOP_REQUESTED=True

def mine():
 s,q,g=data('train');membership=read(STUDY/'manifests/split_membership.json');docs,dids=load_vectors('train-documents');b,bids=load_vectors('train-queries');bmap={x:i for i,x in enumerate(bids)};eligible=[i for i,sid in enumerate(dids) if membership['skill_partition'][sid]=='train'];d=np.array(docs[eligible]);ids=[dids[i] for i in eligible];bodymap={x['id']:digest(' '.join(__import__('unicodedata').normalize('NFKC',x['skill_md']).casefold().split())) for x in s};bodygroups={}
 for sid,h in bodymap.items():bodygroups.setdefault(h,[]).append(sid)
 out=[];rng=random.Random(42)
 for j,qid in enumerate(membership['train_query_ids']):
  positives=set(g[qid]);excluded=set(positives)
  for sid in positives:excluded.update(bodygroups[bodymap[sid]])
  top,scores=retrieve(b[bmap[qid]],d,ids,k=50);clean=[sid for sid in top if sid not in excluded];assert len(clean)>=30
  high=clean[:20];moderate=clean[20:50];rand=ids[rng.randrange(len(ids))]
  while rand in excluded:rand=ids[rng.randrange(len(ids))]
  out.append({'query_id':qid,'positive_ids':sorted(positives),'hard_ids':high,'moderate_ids':moderate,'random_id':rand,'training_gold_inserted_if_missing':sorted(positives-set(top)),'candidate_rule':'frozen full-document train-only dense top50; held-out development skills excluded'})
  if j%2000==0:print('Mined train-only negatives',j,flush=True)
 write_rows(STUDY/'manifests/reranker-negatives.jsonl',out)
 print('Mined',len(out),flush=True)

def normalized_fusion(dense_scores,rerank_scores,w):
 # Within the fixed pool, percentiles avoid adding incompatible raw score scales.
 n=len(dense_scores);dense_order=np.argsort(-np.asarray(dense_scores),kind='stable');rerank_order=np.argsort(-np.asarray(rerank_scores),kind='stable');dr=np.empty(n);rr=np.empty(n);dr[dense_order]=np.arange(n);rr[rerank_order]=np.arange(n);score=(1-w)*(1-dr/max(1,n-1))+w*(1-rr/max(1,n-1));return np.argsort(-score,kind='stable')

def score_panel(m,tok,part,tag,budget=1024,topk=50,full=False,limit=None,query_override=None,provenance=None,reuse_source=None):
 from itertools import repeat
 membership=read(STUDY/'manifests/split_membership.json');s,q,g=data('train');qm={x['id']:x for x in q};del s,q;docs,dids=load_vectors('train-documents');b,bids=load_vectors('train-queries');bmap={x:i for i,x in enumerate(bids)}
 if query_override is not None:b,bids=query_override;bmap={x:i for i,x in enumerate(bids)}
 ids=membership[part+'_query_ids'][:limit];renderer=Renderer(tok,'train');m.eval();records=[];path=STUDY/'results/reranker'/tag
 done={r['query_id']:r for r in rows(path)} if path.exists() else {}
 if reuse_source:
  from score_reuse import CONFIRM_STRIDE,ScoreReuseMismatch,records_sha256,saved_scores,score_repacked
  source=path.parent.parent/reuse_source/'panel.jsonl'
  source_hash=records_sha256(source);references=iter_rows(source);batch_size=admitted_inference_batch()
  assert part=='confirmation' and budget==2048 and topk==50 and not full and provenance
 else:references=repeat(None,len(ids))
 for number,(qid,reference) in enumerate(zip(ids,references,strict=True)):
  if STOP_REQUESTED:raise PanelPaused
  if reference is not None:assert reference['query_id']==qid and reference['provenance']==provenance
  if qid in done:records.append(done[qid]);continue
  t=time.perf_counter();cands,dense=retrieve(b[bmap[qid]],docs,dids,50);search=time.perf_counter()-t;rr=[];selections=[];render_time=0;score_time=0
  inputs=[];owner=[];render20=0.
  for j,sid in enumerate(cands[:topk]):
   if STOP_REQUESTED:raise PanelPaused
   t=time.perf_counter();pieces=renderer.render(qm[qid]['query'],sid,budget,full);render_time+=time.perf_counter()-t;spans=[]
   for tokens,meta in pieces:
    inputs.append(tokens);owner.append(j);spans.append({**meta,'input_sha256':digest(tokens),'tokens':len(tokens)})
   selections.append(spans)
   if j==19:render20=render_time
  timing={};reuse=None
  if reference is not None:
   assert len(inputs)==len(selections)==topk
   cached=saved_scores(reference,batch_size);hashes=[w[0]['input_sha256'] for w in selections]
   t=time.perf_counter();vals,stats=score_repacked(m,tok,inputs,hashes,cached,batch_size);score_time=time.perf_counter()-t
   checked=number%CONFIRM_STRIDE==0;check_seconds=None
   if checked:
    t=time.perf_counter();direct=score_sequences(m,tok,inputs);check_seconds=time.perf_counter()-t
    if vals!=direct:
     save(path.with_suffix('.shape_reuse_mismatch.json'),{'query_id':qid,'source':reuse_source,'source_sha256':source_hash,
          'max_absolute_score_error':max(abs(a-b) for a,b in zip(vals,direct,strict=True))})
     raise ScoreReuseMismatch('Confirmation numerical reuse audit failed')
   reuse={**stats,'source':reuse_source,'source_sha256':source_hash,'checked_against_original_batch_execution':checked,
          'uncached_audit_rerank_seconds':check_seconds}
  else:
   t=time.perf_counter();vals=score_sequences(m,tok,inputs,timing=timing);score_time=time.perf_counter()-t
  rr=[max(v for v,j in zip(vals,owner) if j==i) for i in range(topk)]
  record={'query_id':qid,'candidates':cands,'dense_scores':dense,'reranker_scores':rr,'scored_candidate_count':topk,'selections':selections,'timing':{'search':search,'render':render_time,'rerank':score_time,'render20':render20,'rerank20':timing.get('first20_inputs_seconds') if not full else None},'status':'ok'}
  if provenance:record['provenance']=provenance
  if reuse is not None:record['offline_score_reuse']=reuse
  append(path,record);records.append(record)
  if (number+1)%10==0:print(tag,number+1,'/',len(ids),'rerank_s',round(score_time,2),flush=True)
 results={}
 for k in (20,50):
  if k>topk:continue
  for w in (0,.25,.5,.75,1):
   ranking={r['query_id']:[r['candidates'][j] for j in normalized_fusion(r['dense_scores'][:k],r['reranker_scores'][:k],w)]+r['candidates'][k:] for r in records};ev=evaluate({i:g[i] for i in ids},ranking);ceil=[]
   for r in records:
    hits=len(g[r['query_id']]&set(r['candidates'][:k]));n=len(g[r['query_id']]);ceil.append(sum(1/math.log2(i+2) for i in range(min(hits,10)))/sum(1/math.log2(i+2) for i in range(min(n,10))))
   results[f'k{k}-w{w}']={'evaluation':ev,'ideal_candidate_ndcg10':float(np.mean(ceil)),'candidate_recall':float(np.mean([len(g[r['query_id']]&set(r['candidates'][:k]))/len(g[r['query_id']]) for r in records]))}
 save(path.with_suffix('.summary.json'),results);m.train();return results

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',default='bce-3e-6');p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=100);p.add_argument('--budget',type=int,default=1024);p.add_argument('--resume',type=Path);p.add_argument('--full-coverage',action='store_true');p.add_argument('--candidates',type=int,default=50);p.add_argument('--fixed-checkpoint-step',type=int);p.add_argument('--fixed-repeat-no-dev-scoring',action='store_true');a=p.parse_args();assert 0<a.steps<=600
 finish=a.steps if a.fixed_checkpoint_step is None else a.fixed_checkpoint_step;assert 0<finish<=a.steps
 if a.fixed_repeat_no_dev_scoring:assert a.seed==43 and a.fixed_checkpoint_step==finish
 signal.signal(signal.SIGTERM,request_stop)
 verification=read(STUDY/'results/reranker/mining_verification.json')
 assert verification['status']=='passed' and verification['source_sha256']==sha(record_path(STUDY/'manifests/reranker-negatives.jsonl'))
 membership=read(STUDY/'manifests/split_membership.json');s,q,g=data('train');qm={x['id']:x for x in q};del s,q;examples=rows(STUDY/'manifests/reranker-negatives.jsonl');random.Random(a.seed).shuffle(examples);m,tok=load('reranker',adapter=a.resume,train=True,seed=a.seed);before=fingerprint(m);renderer=Renderer(tok);params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=3e-6,weight_decay=.01);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min((step+1)/max(1,int(.1*a.steps)),max(0,(a.steps-step)/max(1,.9*a.steps))));scaler=torch.amp.GradScaler('cuda',init_scale=128);out=STUDY/'results/reranker'/a.name;folder=CACHE/'checkpoints/reranker'/a.name;out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists();assert a.resume or not(out/'steps.jsonl').exists(),'Existing trial journal; resume or use a new name';cursor=0;step0=0;best=-1;stale=0;beststep=0;overflow=0;rng=random.Random(a.seed)
 recipe={'name':a.name,'seed':a.seed,'steps':a.steps,'fixed_checkpoint_step':a.fixed_checkpoint_step,'fixed_repeat_no_dev_scoring':a.fixed_repeat_no_dev_scoring,'budget':a.budget,'candidates':a.candidates,'full_coverage':a.full_coverage,'split_sha256':sha(STUDY/'manifests/split_membership.json'),'negative_plan_sha256':sha(record_path(STUDY/'manifests/reranker-negatives.jsonl'))}
 pending_validation=False
 if a.resume:
  recovery=torch.load(a.resume/'recovery.pt',weights_only=False,map_location='cpu');opt.load_state_dict(recovery['optimizer']);sched.load_state_dict(recovery['scheduler']);scaler.load_state_dict(recovery['scaler']);state=recovery['state'];cursor=state['cursor'];step0=state['step'];best=state['best'];stale=state['stale'];beststep=state['beststep'];torch.set_rng_state(recovery['torch_rng']);torch.cuda.set_rng_state_all(recovery['cuda_rng']);rng.setstate(recovery['sampling_rng'])
  previous_recipe=dict(state['recipe']);previous_recipe.setdefault('fixed_repeat_no_dev_scoring',False)
  assert previous_recipe==recipe,'Resume with the original recipe and input manifests'
  pending_validation=state['pending_validation'];overflow=state['overflow']
  from recovery import reconcile
  reconcile(out,folder,state,'reranker')
 step=step0
 pair_plan=[]
 for e in examples:
  for sid in e['positive_ids']:pair_plan.extend([(e,sid,True),(e,None,False)])
 start=time.perf_counter();unique={e['query_id'] for e,sid,pos in pair_plan[:cursor]}
 from train_query import save_recovery
 def checkpoint(at,pending=False):
  state={'step':at,'cursor':cursor,'best':best,'stale':stale,'beststep':beststep,'recipe':recipe,'pending_validation':pending,'overflow':overflow}
  save_recovery(m,opt,sched,scaler,folder/f'step-{at:05d}',state,{'sampling_rng':rng.getstate()})
 def stop(at,pending=False):
  save(out/'checkpoint_stop.json',{'status':'checkpointed_stop','step':at,'reason':'signal','validation_pending':pending})
  raise SystemExit(75)
 def validate(at):
  nonlocal best,beststep,stale
  try:results=score_panel(m,tok,'selection',f'{a.name}/selection-{at:05d}.jsonl',a.budget,a.candidates,a.full_coverage)
  except PanelPaused:stop(at,True)
  scores={k:z['evaluation']['aggregate']['NDCG@10'] for k,z in results.items()};key=max(scores,key=lambda k:scores[k]);val=scores[key]
  if val>best+1e-12:
   best=val;beststep=at;stale=0;h=save_adapter(m,folder/'best');save(out/'best.json',{'step':at,'ndcg10':val,'configuration':key,'adapter_sha256':h})
  else:stale+=1
  checkpoint(at)
 if pending_validation:validate(step0)
 for step in (range(step0+1,finish+1) if stale<3 else ()):
  opt.zero_grad(set_to_none=True);losses=[];t=time.perf_counter()
  for micro in range(16):
   e,sid,positive=pair_plan[cursor%len(pair_plan)];qid=e['query_id'];unique.add(qid);cursor+=1
   if not positive:
    z=rng.random();pool=e['hard_ids'] if z<.6 else e['moderate_ids'] if z<.9 else [e['random_id']];sid=rng.choice(pool);assert sid not in g[qid]
   pieces=renderer.render(qm[qid]['query'],sid,a.budget,a.full_coverage)
   with torch.autocast('cuda',dtype=torch.float16):score=skill_score(m,tok,[tokens for tokens,meta in pieces])
   loss=torch.nn.functional.binary_cross_entropy_with_logits(score.float(),torch.tensor(float(positive),device='cuda'));assert torch.isfinite(loss);scaler.scale(loss/16).backward();losses.append(float(loss.detach()))
  scaler.unscale_(opt);grad=torch.nn.utils.clip_grad_norm_(params,1.);gradient_finite=bool(torch.isfinite(grad));assert not gradient_finite or grad>0;assert all(p.grad is None for n,p in m.named_parameters() if 'lora_' not in n);previous=scaler.get_scale();scaler.step(opt);scaler.update();skipped=scaler.get_scale()<previous;assert gradient_finite or skipped,'Nonfinite gradients must never update weights'
  if skipped:overflow+=1
  else:sched.step()
  assert overflow<=3;append(out/'steps.jsonl',{'step':step,'pairs':cursor,'unique_queries':len(unique),'loss':float(np.mean(losses)),'gradient_norm':float(grad) if gradient_finite else None,'gradient_finite':gradient_finite,'seconds':time.perf_counter()-t,'amp_skipped':skipped})
  if step%10==0:print(a.name,step,'loss',np.mean(losses),flush=True)
  if step==finish or (not a.fixed_checkpoint_step and step%100==0):
   # Save the completed optimizer boundary before opening the long scoring panel.
   if a.fixed_repeat_no_dev_scoring:
    checkpoint(step)
    if STOP_REQUESTED:stop(step)
   else:
    checkpoint(step,True)
    if STOP_REQUESTED:stop(step,True)
    validate(step)
    if STOP_REQUESTED:stop(step)
   if stale>=3:break
  elif STOP_REQUESTED:checkpoint(step);stop(step)
 assert fingerprint(m)==before;final=save_adapter(m,folder/'final')
 if a.fixed_repeat_no_dev_scoring:
  assert step==finish;beststep=finish;best=None;bestsha=save_adapter(m,folder/'best');assert bestsha==final
  save(out/'best.json',{'step':finish,'ndcg10':None,'configuration':'fixed_primary_policy','adapter_sha256':bestsha,'selection_inference_performed':False})
 checkpoint_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file());assert checkpoint_bytes<=1024**3;save(out/'summary.json',{'status':'complete','seed':a.seed,'steps':step,'schedule_max_steps':a.steps,'fixed_checkpoint_step':a.fixed_checkpoint_step,'pairs':cursor,'unique_query_count':len(unique),'positive_policy':'All available positives per query in sorted order, each with one matched negative; BCE mean over pairs','pair_plan_count':len(pair_plan),'training_wall_seconds':time.perf_counter()-start,'resumed_from_step':step0,'checkpoint_storage_bytes':checkpoint_bytes,'training_record_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),'best_step':beststep,'best_selection_ndcg10':best,'selection_inference_performed':not a.fixed_repeat_no_dev_scoring,'frozen_before':before,'frozen_after':fingerprint(m),'final_adapter_sha256':final,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'peak_cuda_reserved':torch.cuda.max_memory_reserved(),'overflows':overflow})
if __name__=='__main__':main()
