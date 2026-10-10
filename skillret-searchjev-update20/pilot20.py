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
for update in range(1,21):
 model.train();optimizer.zero_grad(set_to_none=True);losses=[];micro=[];update_start=time.monotonic()
 pre=tensors_hash((n,p) for n,p in model.named_parameters() if p.requires_grad) if update<=2 else None
 for i in range(8):
  row=train[((update-1)*8+i)%len(train)]
  logits=forward(model,row);loss,ce,brier=objective(logits,row['target'])
  assert torch.isfinite(loss) and torch.isfinite(logits).all(),'nonfinite forward'
  scaler.scale(loss/8).backward()
  losses.append(loss.item());micro.append(dict(query_id=row['query_id'],skill_id=row['skill_id'],y=row['y'],loss=loss.item()))
  event('microbatch',update=update,microbatch=i+1,loss=loss.item())
 scaler.unscale_(optimizer)
 params=[p for p in model.parameters() if p.requires_grad]
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),'missing/nonfinite adapter gradient'
 assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
 norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
 assert norm.item()>0,'zero adapter gradient norm'
 scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
 assert scaler.get_scale()>=scale_before,'GradScaler skipped optimizer update'
 assert all(torch.isfinite(p).all() for p in params)
 rowlog=dict(update=update,mean_loss=sum(losses)/8,gradient_norm_before_clip=norm.item(),gradients_finite=True,scale=scaler.get_scale(),seconds=time.monotonic()-update_start,examples=micro)
 if pre is not None:
  post=tensors_hash((n,p) for n,p in model.named_parameters() if p.requires_grad)
  rowlog['changed_adapter_tensors']=sum(post[n]['sha256']!=pre[n]['sha256'] for n in pre)
  assert rowlog['changed_adapter_tensors']>0
 update_logs.append(rowlog);save('updates.json',update_logs);event('optimizer_update',**{k:v for k,v in rowlog.items() if k!='examples'})
 if update==2:
  base2=frozen_hash(model);assert base2==base_before,'frozen base changed'
  save('base-after2.json',base2)
  checkpoint=RUN/'checkpoint-update2';model.save_pretrained(str(checkpoint),safe_serialization=True)
  prior_state=load_file(str(ROOT/'runs/pilot/checkpoint-update2/adapter_model.safetensors'))
  current_state=get_peft_model_state_dict(model)
  exact_match=set(prior_state)==set(current_state) and all(torch.equal(current_state[k].detach().cpu(),prior_state[k].float()) for k in prior_state)
  assert exact_match,'two-update checkpoint differs from independently reloaded smoke checkpoint'
  del prior_state,current_state
  proof=dict(prior_proof,current_run_updates=2,current_checkpoint_exactly_matches_validated_checkpoint=True,validation_run=str(ROOT/'runs/checkpoint-validation'),checkpoint=str(checkpoint))
  save('two-update-proof.json',proof);event('two_update_gate_passed',**proof)
checkpoint=RUN/'checkpoint-update20';model.save_pretrained(str(checkpoint),safe_serialization=True)
assert frozen_hash(model)==base_before,'frozen base changed after20'
save('adapter-after20.json',tensors_hash((n,p) for n,p in model.named_parameters() if p.requires_grad))
adapted,adapted_metrics=evaluate(model,'adapted-update20')
elapsed_train=sum(x['seconds'] for x in update_logs)
result=dict(completed_optimizer_updates=20,baseline=baseline_metrics,adapted=adapted_metrics,delta={k:adapted_metrics[k]-baseline_metrics[k] for k in ['loss','ce','brier','accuracy','auc']},training_update_seconds=elapsed_train,mean_update_seconds=elapsed_train/20,estimated_200_update_seconds=elapsed_train*10,full_training_wall_seconds=time.monotonic()-start_train,checkpoint=str(checkpoint),frozen_base_unchanged=True,peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,scope='20-update binary header relevance feasibility pilot; exposed development, no blind test')
save('result.json',result);event('pilot_complete',**result)
