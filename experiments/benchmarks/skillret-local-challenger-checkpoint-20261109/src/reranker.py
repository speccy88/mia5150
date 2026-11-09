"""Final-position yes/no rows; parity-checked scores and gradients."""
from engine import *
from render import *

def pair_score(m,tok,ids):
 x=torch.tensor([ids],device='cuda',dtype=torch.long);mask=torch.ones_like(x);base=m.get_base_model() if hasattr(m,'get_base_model') else m
 h=base.model(input_ids=x,attention_mask=mask,use_cache=False).last_hidden_state[:,-1]
 no=tok.convert_tokens_to_ids('no');yes=tok.convert_tokens_to_ids('yes');weights=base.lm_head.weight[[no,yes]]
 # Match official FP16 output head before float32 logit difference/loss.
 logits=torch.nn.functional.linear(h,weights).float();return (logits[:,1]-logits[:,0])[0]

def batch_scores(m,tok,sequences,*,padded_length=None):
 longest=max(map(len,sequences));maxlen=longest if padded_length is None else padded_length
 assert maxlen>=longest
 pad=tok.pad_token_id;ids=torch.tensor([[pad]*(maxlen-len(x))+x for x in sequences],device='cuda',dtype=torch.long);mask=torch.tensor([[0]*(maxlen-len(x))+[1]*len(x) for x in sequences],device='cuda',dtype=torch.long);base=m.get_base_model() if hasattr(m,'get_base_model') else m
 h=base.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state[:,-1];weights=base.lm_head.weight[[tok.convert_tokens_to_ids('no'),tok.convert_tokens_to_ids('yes')]];logits=torch.nn.functional.linear(h,weights).float();return logits[:,1]-logits[:,0]

@lru_cache(maxsize=1)
def admitted_inference_batch():
 path=STUDY/'results/reranker_head_parity.json'
 return int(read(path)['inference_batch_size']) if path.exists() else 4

def score_sequences(m,tok,sequences,batch_size=None,timing=None):
 batch_size=admitted_inference_batch() if batch_size is None else batch_size
 values=[];begin=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
  for start in range(0,len(sequences),batch_size):
   values.extend(batch_scores(m,tok,sequences[start:start+batch_size]).float().cpu().tolist())
   if timing is not None and len(values)==20:timing['first20_inputs_seconds']=time.perf_counter()-begin
 assert all(math.isfinite(x) for x in values);return values

def skill_score(m,tok,sequences):
 """Exact first-argmax skill score with one retained graph and dropout RNG replay."""
 if len(sequences)==1:return pair_score(m,tok,sequences[0])
 cpu_states=[];cuda_states=[];values=[]
 # Disable autocast's weight cache across no-grad search and differentiable replay.
 # A no-grad cached FP16 cast must never detach trainable FP32 adapter weights.
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
  for tokens in sequences:
   cpu_states.append(torch.get_rng_state());cuda_states.append(torch.cuda.get_rng_state());values.append(float(pair_score(m,tok,tokens)))
 after_cpu=torch.get_rng_state();after_cuda=torch.cuda.get_rng_state();winner=max(range(len(values)),key=lambda i:values[i]);torch.set_rng_state(cpu_states[winner]);torch.cuda.set_rng_state(cuda_states[winner])
 with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):score=pair_score(m,tok,sequences[winner])
 torch.set_rng_state(after_cpu);torch.cuda.set_rng_state(after_cuda)
 assert abs(float(score.detach())-values[winner])<1e-5,'Dropout recomputation mismatch'
 return score

def gradient_relative_error(params,reference):
 squared_error=0.;squared_reference=0.;dot=0.;actual_squared=0.
 for p,g in zip(params,reference):
  actual=p.grad.detach() if p.grad is not None else torch.zeros_like(p);assert torch.isfinite(actual).all() and torch.isfinite(g).all();squared_error+=float((actual-g).float().square().sum());squared_reference+=float(g.float().square().sum());dot+=float((actual.float()*g.float()).sum());actual_squared+=float(actual.float().square().sum())
 assert squared_reference>0 and actual_squared>0,'Nonzero reference and optimized gradients required'
 relative=math.sqrt(squared_error/squared_reference);cosine=dot/math.sqrt(squared_reference*actual_squared);assert relative<.03 and cosine>.999,(relative,cosine)
 return {'relative_l2_error':relative,'cosine':cosine}

def admit_budget(m,tok,ids,params,opt,scaler,budget,whole_peaks):
 suffix=tok.encode(SUFFIX,add_special_tokens=False)
 tokens=(ids[:-len(suffix)]+tok.encode(' CSV table rows '*budget,add_special_tokens=False))[:budget-len(suffix)]+suffix
 t=time.perf_counter()
 with torch.autocast('cuda',dtype=torch.float16):ss=pair_score(m,tok,tokens)
 loss=torch.nn.functional.binary_cross_entropy_with_logits(ss.float(),torch.tensor(1.,device='cuda'));scaler.scale(loss).backward();scaler.unscale_(opt);gn=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(gn) and gn>0;scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
 m.eval();t2=time.perf_counter();score_sequences(m,tok,[tokens]*4,batch_size=4);inference=time.perf_counter()-t2
 fixture=[tokens[:len(tokens)-len(suffix)-(i%5)*11]+suffix for i in range(20)];speed={};vectors={}
 for batch_size in (4,10):
  try:
   score_sequences(m,tok,fixture,batch_size=batch_size)
   whole_peaks.append(torch.cuda.max_memory_allocated());torch.cuda.reset_peak_memory_stats();elapsed=[]
   for _ in range(3):
    start=time.perf_counter();values=score_sequences(m,tok,fixture,batch_size=batch_size);elapsed.append(time.perf_counter()-start)
   vectors[batch_size]=values;speed[str(batch_size)]={'status':'passed','median_seconds_20_pairs':float(np.median(elapsed)),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'samples':elapsed}
  except torch.cuda.OutOfMemoryError:
   if batch_size==4:raise
   speed['10']={'status':'cuda_oom','peak_reserved_bytes':torch.cuda.max_memory_reserved()}
   whole_peaks.append(torch.cuda.max_memory_allocated())
  if str(batch_size) in speed and speed[str(batch_size)]['status']=='cuda_oom':torch.cuda.empty_cache()
 diff=max(abs(x-y) for x,y in zip(vectors[4],vectors[10])) if 10 in vectors else None
 m.train()
 return {'tokens':len(tokens),'forward_backward_update_seconds':t2-t,'batch4_inference_seconds':inference,'gradient_norm':float(gn),'inference_batch_profile':speed,'batch4_vs10_max_score_error':diff}

def verify():
 m,tok=load('reranker',train=True);m.eval();no=tok.convert_tokens_to_ids('no');yes=tok.convert_tokens_to_ids('yes');ids=tok.encode(PREFIX_R+f'<Instruct>: {INSTRUCTION}\n<Query>: Find CSV parsing tools\n<Document>: csv-parser | Parse CSV | Reads CSV files.'+SUFFIX,add_special_tokens=False);x=torch.tensor([ids],device='cuda');mask=torch.ones_like(x);params=[p for p in m.parameters() if p.requires_grad];before=fingerprint(m)
 with torch.autocast('cuda',dtype=torch.float16):
  # logits_to_keep=0 is the unoptimized all-token vocabulary reference on a tiny fixture.
  logits=m(input_ids=x,attention_mask=mask,use_cache=False).logits[:,-1].float();ref=(logits[:,yes]-logits[:,no])[0];ref.backward()
 grads=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params];m.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):score=pair_score(m,tok,ids);score.backward()
 error=abs(float(score.detach()-ref.detach()));assert error<.02,error
 maxerr=0;maxrel=0
 for p,g in zip(params,grads):
  actual=p.grad if p.grad is not None else torch.zeros_like(p);delta=float((actual-g).abs().max());scale=float(g.abs().max());maxerr=max(maxerr,delta);maxrel=max(maxrel,delta/max(scale,1e-5));assert torch.allclose(actual,g,atol=.03,rtol=.03),(delta,scale)
 single_gradient=gradient_relative_error(params,grads)
 assert any(float(g.abs().sum())>0 for g in grads);assert fingerprint(m)==before
 m.zero_grad(set_to_none=True)
 # The official padding semantics are preserved for heterogeneous batches.
 sequences=[ids,ids[:-20]+tok.encode(SUFFIX,add_special_tokens=False)];maxlen=max(map(len,sequences));xx=torch.tensor([[tok.pad_token_id]*(maxlen-len(z))+z for z in sequences],device='cuda');am=torch.tensor([[0]*(maxlen-len(z))+[1]*len(z) for z in sequences],device='cuda')
 with torch.autocast('cuda',dtype=torch.float16):
  raw=m(input_ids=xx,attention_mask=am,use_cache=False).logits[:,-1].float();br=raw[:,yes]-raw[:,no];br.sum().backward()
 bg=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params];m.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):bs=batch_scores(m,tok,sequences);bs.sum().backward()
 batch_error=float((bs-br).abs().max());assert batch_error<.02
 batch_grad_error=0
 for p,g in zip(params,bg):
  actual=p.grad if p.grad is not None else torch.zeros_like(p);batch_grad_error=max(batch_grad_error,float((actual-g).abs().max()));assert torch.allclose(actual,g,atol=.03,rtol=.03)
 batch_gradient=gradient_relative_error(params,bg)
 m.zero_grad(set_to_none=True)
 # Independent tiny two-window skill-max gradient parity with dropout on and AMP scaling.
 m.train();cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
 with torch.autocast('cuda',dtype=torch.float16):
  naive=torch.stack([pair_score(m,tok,z) for z in sequences]);naive_max=naive.max(dim=0).values;(naive_max*128).backward()
 naive_grads=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params];expected_cpu=torch.get_rng_state();expected_cuda=torch.cuda.get_rng_state();m.zero_grad(set_to_none=True);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
 with torch.autocast('cuda',dtype=torch.float16):cached=skill_score(m,tok,sequences);(cached*128).backward()
 assert torch.equal(torch.get_rng_state(),expected_cpu) and torch.equal(torch.cuda.get_rng_state(),expected_cuda)
 assert abs(float(cached.detach()-naive_max.detach()))<1e-5;skill_grad_error=0
 for p,g in zip(params,naive_grads):
  actual=p.grad if p.grad is not None else torch.zeros_like(p);skill_grad_error=max(skill_grad_error,float((actual-g).abs().max()));assert torch.allclose(actual,g,atol=.03,rtol=.03)
 skill_gradient=gradient_relative_error(params,naive_grads)
 m.zero_grad(set_to_none=True);m.eval()
 with torch.inference_mode(),m.disable_adapter():unadapted=pair_score(m,tok,ids)
 with torch.inference_mode():zero=pair_score(m,tok,ids)
 assert torch.equal(unadapted,zero)
 admission=[];unadmitted=[];whole_peaks=[torch.cuda.max_memory_allocated()];m.train();opt=torch.optim.AdamW(params,lr=3e-6);scaler=torch.amp.GradScaler('cuda',init_scale=128)
 for budget in (1024,2048):
  try:admission.append(admit_budget(m,tok,ids,params,opt,scaler,budget,whole_peaks))
  except torch.cuda.OutOfMemoryError:
   if budget==1024:raise
   unadmitted.append({'tokens':budget,'status':'cuda_oom','stage':'required forward/backward/update or batch-4 inference','peak_reserved_bytes':torch.cuda.max_memory_reserved()})
  if unadmitted:opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
 selected_batch=10 if len(admission)==2 and all(x['batch4_vs10_max_score_error'] is not None and x['batch4_vs10_max_score_error']<.02 and x['inference_batch_profile']['10']['peak_reserved_bytes']<7*1024**3 and x['inference_batch_profile']['10']['median_seconds_20_pairs']<.95*x['inference_batch_profile']['4']['median_seconds_20_pairs'] for x in admission) else 4
 assert fingerprint(m)==before
 assert any(bool(p.detach().ne(0).any()) for n,p in m.named_parameters() if 'lora_B' in n),'Optimizer must update zero-initialized B matrices'
 save(STUDY/'results/reranker_head_parity.json',{'status':'passed','score_error':error,'gradient_max_absolute_error':maxerr,'gradient_max_relative_error':maxrel,'single_gradient':single_gradient,'batch_gradient':batch_gradient,'window_max_gradient':skill_gradient,'padded_batch_score_error':batch_error,'padded_batch_gradient_max_error':batch_grad_error,'zero_adapter_equal':True,'optimizer_updates_verified_nonzero_B':True,'skill_max_dropout_amp_gradient_error':skill_grad_error,'skill_max_rng_replay_exact':True,'budget_admission':admission,'unadmitted_budgets':unadmitted,'inference_batch_size':selected_batch,'batch_selection_used_quality_labels':False,'batch_selection_rule':'10 only if both budgets are >=5% faster for20pairs, <.02 max logit-difference error and <7GiB CUDA reserved; otherwise4. Both divide the top20 boundary.' ,'fixture':'synthetic non-test text; full vocabulary/all tokens vs final hidden yes/no rows; eval mode dropout off','yes_id':yes,'no_id':no,'frozen_sha256':before,'tokens':len(ids),'peak_allocated':max(whole_peaks+[torch.cuda.max_memory_allocated()])})
 print('Reranker scores and gradients passed',error,maxerr,maxrel,flush=True)
if __name__=='__main__':verify()
