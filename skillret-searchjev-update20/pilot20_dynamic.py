from pilot_common import *
prior_proof=json.loads((ROOT/'runs/checkpoint-validation/two-update-proof.json').read_text())
assert prior_proof['passed'] and prior_proof['save_reload_max_abs_logit_diff']==0.0
assert json.loads((ROOT/'runs/checkpoint-validation/exit.json').read_text())['returncode']==0
model=new_model(ROOT/'models/adapter')
loaded_state=get_peft_model_state_dict(model)
original_state=load_file(str(ROOT/'models/adapter/adapter_model.safetensors'))
assert set(loaded_state)==set(original_state)
assert all(torch.equal(loaded_state[k].detach().cpu(),original_state[k].float()) for k in original_state),'released adapter weights not loaded exactly'
del loaded_state,original_state
event('released_adapter_verified')
base_before=frozen_hash(model);save('base-before.json',base_before)
adapter_before=tensors_hash((n,p) for n,p in model.named_parameters() if p.requires_grad)
save('adapter-before.json',adapter_before)
baseline,baseline_metrics=evaluate(model,'baseline')
optimizer=make_optimizer(model);scaler=torch.amp.GradScaler('cuda',init_scale=128)
update_logs=[]
start_train=time.monotonic()
def complete_update(update):
 global optimizer,scaler
 update_start=time.monotonic()
 cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
 for attempt in range(8):
  if scaler.get_scale()<1:
   raise RuntimeError('GradScaler fell below the bounded minimum scale1; no unsafe optimizer update applied.')
  torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
  model.train();optimizer.zero_grad(set_to_none=True);losses=[];micro=[]
  for i in range(8):
   row=train[((update-1)*8+i)%len(train)]
   logits=forward(model,row);loss,ce,brier=objective(logits,row['target'])
   assert torch.isfinite(loss) and torch.isfinite(logits).all(),'nonfinite forward'
   scaler.scale(loss/8).backward()
   losses.append(loss.item());micro.append(dict(query_id=row['query_id'],skill_id=row['skill_id'],y=row['y'],loss=loss.item()))
   event('microbatch',update=update,attempt=attempt+1,microbatch=i+1,loss=loss.item())
  scaler.unscale_(optimizer)
  named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
  params=[p for n,p in named]
  missing=[n for n,p in named if p.grad is None]
  assert not missing,'missing gradients: '+str(missing)
  assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
  bad=[(n,p) for n,p in named if not torch.isfinite(p.grad).all()]
  if bad:
   scale_before=scaler.get_scale()
   details=[dict(parameter=n,nan_elements=torch.isnan(p.grad).sum().item(),inf_elements=torch.isinf(p.grad).sum().item()) for n,p in bad]
   unchanged=tensors_hash(named)
   scaler.step(optimizer);scaler.update()
   assert tensors_hash(named)==unchanged,'overflow skip unexpectedly changed adapter'
   assert scaler.get_scale()==scale_before*.5
   incident=dict(update=update,attempt=attempt+1,old_scale=scale_before,new_scale=scaler.get_scale(),bad_gradients=details,optimizer_update_skipped=True,adapter_unchanged=True)
   overflow_events.append(incident);save('overflow-events.json',overflow_events);event('fp16_overflow_retry',**incident)
   continue
  norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
  assert norm.item()>0
  scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
  assert scaler.get_scale()>=scale_before
  assert all(torch.isfinite(p).all() for p in params)
  return dict(update=update,mean_loss=sum(losses)/8,gradient_norm_before_clip=norm.item(),gradients_finite=True,scale=scaler.get_scale(),seconds=time.monotonic()-update_start,attempts=attempt+1,examples=micro)
 raise RuntimeError('Nonfinite gradients persisted through bounded GradScaler retries down to scale1; no unsafe optimizer update applied.')

overflow_events=[]
for update in range(1,21):
 pre=tensors_hash((n,p) for n,p in model.named_parameters() if p.requires_grad) if update<=2 else None
 try:
  rowlog=complete_update(update)
 except Exception:
  model.save_pretrained(str(RUN/'checkpoint-last-safe'),safe_serialization=True)
  save('last-safe.json',dict(completed_optimizer_updates=len(update_logs),reason='bounded numerical validation failed; nonfinite optimizer updates skipped'))
  raise
 if pre is not None:
  post=tensors_hash((n,p) for n,p in model.named_parameters() if p.requires_grad)
  rowlog['changed_adapter_tensors']=sum(post[n]['sha256']!=pre[n]['sha256'] for n in pre)
  assert rowlog['changed_adapter_tensors']>0
 update_logs.append(rowlog);save('updates.json',update_logs);event('optimizer_update',**{k:v for k,v in rowlog.items() if k!='examples'})
 if update in (5,10,15):
  model.save_pretrained(str(RUN/'checkpoint-last-safe'),safe_serialization=True)
  save('last-safe.json',dict(completed_optimizer_updates=update))
 if update==2:
  base2=frozen_hash(model);assert base2==base_before,'frozen base changed'
  save('base-after2.json',base2)
  checkpoint=RUN/'checkpoint-update2';model.save_pretrained(str(checkpoint),safe_serialization=True)
  before_reload,_=evaluate(model,'reload-before',evaluation[:4])
  expected_adapter=get_peft_model_state_dict(model)
  expected_adapter={k:v.detach().cpu().clone() for k,v in expected_adapter.items()}
  optimizer_param_ids={id(p) for group in optimizer.param_groups for p in group['params']}
  cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
  model.load_adapter(str(checkpoint),adapter_name='reload_check',is_trainable=False,local_files_only=True)
  model.set_adapter('reload_check',inference_mode=True)
  actual_adapter=get_peft_model_state_dict(model,adapter_name='reload_check')
  assert set(actual_adapter)==set(expected_adapter)
  assert all(torch.equal(actual_adapter[k].detach().cpu(),expected_adapter[k]) for k in expected_adapter)
  after_reload,_=evaluate(model,'reload-after',evaluation[:4])
  maxdiff=max(abs(a-b) for r1,r2 in zip(before_reload,after_reload) for a,b in zip(r1['logits'],r2['logits']))
  assert maxdiff<=1e-4
  model.set_adapter('default');model.delete_adapter('reload_check')
  assert optimizer_param_ids=={id(p) for p in model.parameters() if p.requires_grad}
  current_adapter=get_peft_model_state_dict(model)
  assert all(torch.equal(current_adapter[k].detach().cpu(),expected_adapter[k]) for k in expected_adapter)
  del expected_adapter,actual_adapter,current_adapter
  torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
  assert frozen_hash(model)==base_before
  proof=dict(passed=True,optimizer_updates=2,finite_nonzero_gradients=True,adapter_changes=[x['changed_adapter_tensors'] for x in update_logs],frozen_base_identical=True,save_reload_adapter_identical=True,save_reload_max_abs_logit_diff=maxdiff,checkpoint=str(checkpoint),validation='fresh named adapter loaded from saved safetensors into same verified frozen base; independent full-base reload also passed in checkpoint-validation',independent_full_reload_proof=prior_proof,optimizer_parameter_identity_preserved=True)
  save('two-update-proof.json',proof);event('two_update_gate_passed',**proof)
checkpoint=RUN/'checkpoint-update20';model.save_pretrained(str(checkpoint),safe_serialization=True)
assert frozen_hash(model)==base_before,'frozen base changed after20'
save('adapter-after20.json',tensors_hash((n,p) for n,p in model.named_parameters() if p.requires_grad))
adapted,adapted_metrics=evaluate(model,'adapted-update20')
elapsed_train=sum(x['seconds'] for x in update_logs)
result=dict(completed_optimizer_updates=20,baseline=baseline_metrics,adapted=adapted_metrics,delta={k:adapted_metrics[k]-baseline_metrics[k] for k in ['loss','ce','brier','accuracy','auc']},training_update_seconds=elapsed_train,mean_update_seconds=elapsed_train/20,estimated_200_update_seconds=elapsed_train*10,full_training_wall_seconds=time.monotonic()-start_train,checkpoint=str(checkpoint),frozen_base_unchanged=True,peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,scope='20-update binary header relevance feasibility pilot; exposed development, no blind test')
result['overflow_retries']=len(overflow_events); result['final_grad_scaler_scale']=scaler.get_scale()
save('result.json',result);event('pilot_complete',**result)
