from pilot_common import *
prior=ROOT/'runs/pilot'
checkpoint=prior/'checkpoint-update2'
model=new_model(checkpoint)
loaded=get_peft_model_state_dict(model)
saved=load_file(str(checkpoint/'adapter_model.safetensors'))
assert set(loaded)==set(saved)
assert all(torch.equal(loaded[k].detach().cpu(),saved[k].float()) for k in saved)
del loaded,saved
assert frozen_hash(model)==json.loads((prior/'base-before.json').read_text())
actual,_=evaluate(model,'reload-fresh-process',evaluation[:4])
expected=json.loads((prior/'reload-before.json').read_text())['predictions']
maxdiff=max(abs(a-b) for r1,r2 in zip(expected,actual) for a,b in zip(r1['logits'],r2['logits']))
assert maxdiff<=1e-4
updates=json.loads((prior/'updates.json').read_text())
assert len(updates)==2 and all(x['gradients_finite'] and x['gradient_norm_before_clip']>0 and x['changed_adapter_tensors']>0 for x in updates)
proof=dict(passed=True,optimizer_updates=2,finite_nonzero_gradients=True,adapter_changes=[x['changed_adapter_tensors'] for x in updates],frozen_base_identical=True,save_reload_adapter_identical=True,save_reload_max_abs_logit_diff=maxdiff,checkpoint=str(checkpoint),validation='fresh process after strict swap guard terminated same-process reload; no guard threshold relaxed',training_run=str(prior))
save('two-update-proof.json',proof);event('two_update_gate_passed',**proof)
