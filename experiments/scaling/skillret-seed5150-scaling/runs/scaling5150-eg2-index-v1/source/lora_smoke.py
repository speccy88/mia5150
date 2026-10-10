from neural import Rerank
from harness import *
import torch
from peft import LoraConfig,get_peft_model,TaskType
torch.manual_seed(5150);m=Rerank('skillret_rerank_06',256);m.model=get_peft_model(m.model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=4,lora_alpha=8,lora_dropout=0.,target_modules=['q_proj','v_proj']));m.model.enable_input_require_grads();m.model.gradient_checkpointing_enable();m.model.config.use_cache=False;m.model.train();opt=torch.optim.AdamW([p for p in m.model.parameters() if p.requires_grad],lr=2e-4);scaler=torch.amp.GradScaler('cuda');losses=[]
toy=[('Add two numbers','calculator | Adds and subtracts numbers',1.),('Add two numbers','translator | Translates English into French',0.)]
for step in range(32):
 query,doc,label=toy[step%2];opt.zero_grad(set_to_none=True);inputs=m.inputs(query,doc)
 with torch.autocast('cuda',dtype=torch.float16):
  native=m.model.get_base_model();h=native.model(**inputs).last_hidden_state[:,-1:];logits=native.lm_head(h)[0,0];loss=torch.nn.functional.binary_cross_entropy_with_logits((logits[m.yes]-logits[m.no]).float(),torch.tensor(label,device='cuda'))
 assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_([p for p in m.model.parameters() if p.requires_grad],1.);scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
assert sum(losses[-4:])/4<.1,'Toy overfit threshold not reached'
save(R/'evidence/lora-native-smoke.json',{'losses':losses,'overfit_pass':True,'steps':32,'discarded_toy_adapter':True});print(losses,flush=True)
