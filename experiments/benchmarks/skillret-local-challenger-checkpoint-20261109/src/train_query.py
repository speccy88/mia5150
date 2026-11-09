"""Anchored query-only updates against the immutable full-document index."""
from engine import *
import argparse,copy,random,signal
STOP_REQUESTED=False
def request_stop(signum,frame):
 global STOP_REQUESTED
 STOP_REQUESTED=True

def panel(m,tok,part,base,base_ids,docs,docids,budget,tag):
 membership=read(STUDY/'manifests/split_membership.json');s,q,g=data('train');qm={x['id']:x for x in q};del s,q;bmap={x:i for i,x in enumerate(base_ids)};ids=membership[part+'_query_ids'];m.eval();adapt=[];times=[]
 for i in ids:
  t=time.perf_counter()
  with torch.inference_mode():adapt.append(vector(m,tok,PREFIX+qm[i]['query'],budget).cpu().numpy())
  times.append(time.perf_counter()-t)
 a=np.asarray(adapt,np.float32);b=np.asarray([base[bmap[i]] for i in ids]);out={};rankings={}
 for w in (0,.25,.5,.75,1):
  mix=mix_query_vectors(b,a,w)
  r={}
  for i,v in zip(ids,mix):r[i]=retrieve(v,docs,docids)[0]
  out[str(w)]=evaluate({i:g[i] for i in ids},r);rankings[str(w)]=r
 save(STUDY/'results'/tag,{'part':part,'mixtures':out,'rankings':rankings,'adapted_encoding_seconds':summary(times),'adapted_vectors_sha256':digest(a.tobytes().hex())})
 m.train();return out,a

def save_recovery(m,opt,sched,scaler,path,state,extras=None):
 p=Path(path);p.mkdir(parents=True,exist_ok=True);save_adapter(m,p);tmp=p/'recovery.tmp'
 torch.save({'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),'state':state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),**(extras or {})},tmp);tmp.replace(p/'recovery.pt')

def main():
 p=argparse.ArgumentParser();p.add_argument('--lr',type=float,default=3e-6);p.add_argument('--steps',type=int,default=100);p.add_argument('--name',default='pilot-3e-6');p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--fixed-checkpoint-step',type=int);a=p.parse_args()
 signal.signal(signal.SIGTERM,request_stop)
 assert 0<a.steps<=1200
 finish=a.steps if a.fixed_checkpoint_step is None else a.fixed_checkpoint_step;assert 0<finish<=a.steps
 membership=read(STUDY/'manifests/split_membership.json');s,q,g=data('train');qm={x['id']:x for x in q};del s,q;qids=membership['train_query_ids'].copy();random.Random(a.seed).shuffle(qids);qids=qids*2
 docs,dids=load_vectors('train-documents');base,bids=load_vectors('train-queries');bmap={x:i for i,x in enumerate(bids)};eligible=[i for i,sid in enumerate(dids) if membership['skill_partition'][sid]=='train'];train_ids=[dids[i] for i in eligible];di={sid:i for i,sid in enumerate(train_ids)};assert len(eligible)>2000
 d=torch.tensor(np.array(docs[eligible]),device='cuda',dtype=torch.float32);assert not d.requires_grad;assert all(g[i]<=set(train_ids) for i in membership['train_query_ids'])
 budget=read(STUDY/'results/data_audit.json')['query_budget'];folder=CACHE/'checkpoints/query'/a.name;output=STUDY/'results/query'/a.name;output.mkdir(parents=True,exist_ok=True)
 if (output/'steps.jsonl').exists() and not a.resume:raise RuntimeError('Existing trial journal; resume a saved checkpoint or choose a new trial name')
 if (output/'summary.json').exists():raise RuntimeError('Completed trial cannot be overwritten')
 m,tok=load(adapter=a.resume,train=True,seed=a.seed);before=fingerprint(m)
 # Zero adapter must agree with base model on training-only fixtures, never test queries.
 if not a.resume:
  m.eval();errs=[]
  for qid in membership['selection_query_ids'][:4]:
   with torch.inference_mode():v=vector(m,tok,PREFIX+qm[qid]['query'],budget).cpu().numpy()
   errs.append(float(np.max(np.abs(v-base[bmap[qid]]))))
  assert max(errs)<3e-5,errs
  save(output/'zero_adapter_parity.json',{'status':'passed','max_absolute_errors':errs,'query_ids':membership['selection_query_ids'][:4]})
 m.train();params=[p for p in m.parameters() if p.requires_grad];assert all(x.dtype==torch.float32 for x in params);opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01);warm=max(1,int(.1*a.steps));sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min((step+1)/warm,max(0,(a.steps-step)/max(1,a.steps-warm))))
 scaler=torch.amp.GradScaler('cuda',init_scale=128,growth_interval=200);step0=0;cursor=0;best=-1;beststep=0;stale=0;unique=set();overflow=0
 if a.resume:
  r=torch.load(a.resume/'recovery.pt',weights_only=False,map_location='cpu');opt.load_state_dict(r['optimizer']);sched.load_state_dict(r['scheduler']);scaler.load_state_dict(r['scaler']);state=r['state'];step0=state['step'];cursor=state['cursor'];best=state['best'];beststep=state['beststep'];stale=state['stale'];unique=set(state['unique']);torch.set_rng_state(r['torch_rng']);torch.cuda.set_rng_state_all(r['cuda_rng']);np.random.set_state(r['numpy_rng']);random.setstate(r['python_rng'])
  from recovery import reconcile
  reconcile(output,folder,state,'query')
 step=step0
 start=time.perf_counter();opt.zero_grad(set_to_none=True);all_train=set(train_ids)
 for step in range(step0+1,finish+1):
  losses=[];anchors=[];ces=[];t=time.perf_counter()
  for micro in range(16):
   if cursor>=len(qids):break
   qid=qids[cursor];cursor+=1;unique.add(qid);text=PREFIX+qm[qid]['query']
   with torch.autocast('cuda',dtype=torch.float16):v=vector(m,tok,text,budget,True)
   logits=(d@v.float())/.05;pos=torch.tensor([di[x] for x in sorted(g[qid])],device='cuda');ce=-torch.log_softmax(logits,dim=0)[pos].mean();b=torch.tensor(np.array(base[bmap[qid]]),device='cuda');anchor=1-torch.dot(v.float(),b);loss=ce+anchor
   assert torch.isfinite(loss);scaler.scale(loss/16).backward();losses.append(float(loss.detach()));ces.append(float(ce.detach()));anchors.append(float(anchor.detach()))
  scaler.unscale_(opt);grad=torch.nn.utils.clip_grad_norm_(params,1.);gradient_finite=bool(torch.isfinite(grad));assert not gradient_finite or grad>0
  assert all(p.grad is None for n,p in m.named_parameters() if 'lora_' not in n)
  oldscale=scaler.get_scale();scaler.step(opt);scaler.update();skipped=scaler.get_scale()<oldscale;assert gradient_finite or skipped,'Nonfinite gradients must never update weights'
  if skipped:overflow+=1
  else:sched.step()
  opt.zero_grad(set_to_none=True);assert overflow<=3
  append(output/'steps.jsonl',{'step':step,'query_groups':cursor,'unique_queries':len(unique),'loss':sum(losses)/len(losses),'ce':sum(ces)/len(ces),'anchor':sum(anchors)/len(anchors),'gradient_norm':float(grad) if gradient_finite else None,'gradient_finite':gradient_finite,'seconds':time.perf_counter()-t,'lr':opt.param_groups[0]['lr'],'amp_skipped':skipped,'amp_scale':scaler.get_scale()})
  if step%10==0:print(a.name,step,'/',a.steps,'loss',round(sum(losses)/len(losses),4),'unique',len(unique),flush=True)
  if STOP_REQUESTED or (a.stop_after and step>=a.stop_after):
   state={'step':step,'cursor':cursor,'best':best,'beststep':beststep,'stale':stale,'unique':sorted(unique)};save_recovery(m,opt,sched,scaler,folder/f'step-{step:05d}',state);save(output/'checkpoint_stop.json',{'status':'checkpointed_stop','step':step,'reason':'signal' if STOP_REQUESTED else 'resume_parity_fixture'});raise SystemExit(75)
  if a.smoke:
   if step==4:break
   continue
  if step==finish or (not a.fixed_checkpoint_step and step%100==0) or cursor>=len(qids):
   values,panel_vectors=panel(m,tok,'selection',base,bids,docs,dids,budget,f'query/{a.name}/selection-{step:05d}.json')
   if a.fixed_checkpoint_step is not None and a.name=='secondary-seed43':
    panel_root=CACHE/'query-selected';panel_root.mkdir(exist_ok=True);tmp=panel_root/'selection-secondary-adapted.tmp.npy';np.save(tmp,panel_vectors,allow_pickle=False);tmp.replace(panel_root/'selection-secondary-adapted.npy')
    save(panel_root/'selection-secondary-ids.json',membership['selection_query_ids'])
   scores={w:z['aggregate']['NDCG@10'] for w,z in values.items()};val=max(scores.values());weight=max(scores,key=lambda w:(scores[w],-float(w)));improved=val>best+1e-12
   if improved:best=val;beststep=step;stale=0;bestsha=save_adapter(m,folder/'best');save(output/'best.json',{'step':step,'ndcg10':best,'weight':float(weight),'adapter_sha256':bestsha,'mixture_scores':scores})
   else:stale+=1
   state={'step':step,'cursor':cursor,'best':best,'beststep':beststep,'stale':stale,'unique':sorted(unique)};save_recovery(m,opt,sched,scaler,folder/f'step-{step:05d}',state)
   if stale>=3 or cursor>=len(qids):break
 after=fingerprint(m);assert before==after;finalsha=save_adapter(m,folder/'final');wall=time.perf_counter()-start
 save(output/'summary.json',{'status':'complete','seed':a.seed,'lr':a.lr,'steps':step,'schedule_max_steps':a.steps,'fixed_checkpoint_step':a.fixed_checkpoint_step,'query_groups':cursor,'unique_query_count':len(unique),'eligible_query_count':len(membership['train_query_ids']),'eligible_passes':cursor/len(membership['train_query_ids']),'training_wall_seconds_this_process':wall,'training_wall_seconds':wall,'resumed_from_step':step0,'whole_attempt_wall_evidence':'guard receipts include original and resumed attempts','best_step':beststep,'best_selection_ndcg10':best,'frozen_before':before,'frozen_after':after,'trainable_parameters':sum(p.numel() for p in params),'overflows':overflow,'successful_optimizer_updates_this_process':step-step0-overflow,'final_adapter_sha256':finalsha,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'peak_cuda_reserved':torch.cuda.max_memory_reserved(),'smoke_discard':a.smoke})
 print('COMPLETED',a.name,step,wall,flush=True)
if __name__=='__main__':main()
