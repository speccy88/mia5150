"""Bounded real LoRA reranker training, with frozen disjoint dev pools."""
from harness import *
from neural import Rerank
from adaptation import partitions
import torch,numpy as np,random,time,sys
from peft import LoraConfig,get_peft_model,TaskType
torch.set_num_threads(1);torch.manual_seed(5150);random.seed(5150);np.random.seed(5150)
mode=sys.argv[1] if len(sys.argv)>1 else 'hard';assert mode in ['hard','replay'];_,train,dev=partitions(metadata_only=True);train=train[:64];dev=dev[:32];labels=qrels('train');allowed=set(json.loads((R/'manifests/training-partitions.json').read_text())['train_allowed_skill_ids'])
class LazyCorpus:
 def __init__(self):
  self.f=(R/'data/skills/train.jsonl').open('rb');self.offsets={}
  while True:
   offset=self.f.tell();line=self.f.readline()
   if not line:break
   self.offsets[json.loads(line)['id']]=offset
  import os
  os.posix_fadvise(self.f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 def __getitem__(self,d):
  self.f.seek(self.offsets[d]);return json.loads(self.f.readline())
skills=LazyCorpus()
channels={n:json.loads((R/f'results/training-bm25-clean-{n}.json').read_text())['rankings'] for n in ['header','full']}
pools={q['id']:rrf([channels[n][q['id']] for n in channels])[:20] for q in dev};save(R/f'manifests/lora-{mode}.json',{'train_query_ids':[q['id'] for q in train],'dev_query_ids':[q['id'] for q in dev],'dev_pools':pools,'pool_hash':digest(pools),'seed':5150,'max_length':512,'rank':4,'steps_cap':384,'representation':'full; token-capped at 512; preserved suffix','scope':'Public training data adaptation diagnostic; released base already saw training data'})
model=Rerank('skillret_rerank_06',max_length=512)
def evaluate_model(stage):
 result={};raw=[];model.model.eval()
 for q in dev:
  ids=pools[q['id']];start=time.perf_counter();scores,tokens=model.score(q['query'],[view(skills[d],'full') for d in ids]);result[q['id']]=[ids[i] for i in sorted(range(len(ids)),key=lambda i:(-scores[i],i))];raw.append({'id':q['id'],'candidates':ids,'scores':scores,'tokens':tokens,'seconds':time.perf_counter()-start})
 metric=evaluate({q['id']:labels[q['id']] for q in dev},result);save(R/f'results/lora-{mode}-{stage}.json',{'rankings':result,'metrics':metric,'raw':raw});print(stage,metric['aggregate'],flush=True)
evaluate_model('base')
model.model=get_peft_model(model.model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=4,lora_alpha=8,lora_dropout=0.,target_modules=['q_proj','v_proj']))
model.model.gradient_checkpointing_enable();model.model.enable_input_require_grads();model.model.config.use_cache=False
optimizer=torch.optim.AdamW([p for p in model.model.parameters() if p.requires_grad],lr=5e-5);scaler=torch.amp.GradScaler('cuda');examples=[]
for q in train:
 qid=q['id'];gold=labels[qid];pool=rrf([channels[n][qid] for n in channels]);examples.extend((q['query'],d,1.) for d in gold);examples.extend((q['query'],d,0.) for d in negatives(gold,[d for d in pool if d in allowed],4))
 if mode=='replay':
  # Labelled replay from permitted training positives and random nonpositives.
  randomneg=next(d for d in sorted(allowed,key=lambda d:digest(qid+d)) if d not in gold)
  examples.append((q['query'],next(iter(gold)),1.));examples.append((q['query'],randomneg,0.))
random.shuffle(examples);examples=examples[:384];initial={n:p.detach().cpu().clone() for n,p in model.model.named_parameters() if p.requires_grad};logs=[];model.model.train()
start=time.perf_counter()
for step,(query,doc,label) in enumerate(examples):
 optimizer.zero_grad(set_to_none=True);inputs=model.inputs(query,view(skills[doc],'full'))
 with torch.autocast('cuda',dtype=torch.float16):
  native=model.model.get_base_model();hidden=native.model(**inputs).last_hidden_state[:,-1:];logits=native.lm_head(hidden)[0,0];score=logits[model.yes].float()-logits[model.no].float();loss=torch.nn.functional.binary_cross_entropy_with_logits(score,torch.tensor(label,device='cuda'))
 assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([p for p in model.model.parameters() if p.requires_grad],1.);scaler.step(optimizer);scaler.update()
 logs.append({'step':step,'loss':float(loss.detach()),'label':label,'elapsed_seconds':time.perf_counter()-start});save(R/f'runs/lora-{mode}-training-log.json',logs)
 if step%16==0:print('train',step,float(loss),flush=True)
 if step==7:
  checkpoint=R/f'checkpoints/lora-{mode}-resume-smoke';model.model.save_pretrained(checkpoint,safe_serialization=True);torch.save({'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'step':step,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},checkpoint/'resume.pt')
  from safetensors.torch import load_file
  from peft import set_peft_model_state_dict
  expected={n:p.detach().cpu().clone() for n,p in model.model.named_parameters() if p.requires_grad}
  with torch.no_grad():
   for p in model.model.parameters():
    if p.requires_grad:p.zero_()
  set_peft_model_state_dict(model.model,load_file(str(checkpoint/'adapter_model.safetensors')))
  restored=torch.load(checkpoint/'resume.pt',weights_only=True);optimizer.load_state_dict(restored['optimizer']);scaler.load_state_dict(restored['scaler']);torch.set_rng_state(restored['rng']);torch.cuda.set_rng_state(restored['cuda_rng'])
  assert all(torch.equal(expected[n],p.detach().cpu()) for n,p in model.model.named_parameters() if p.requires_grad)
  save(checkpoint/'verification.json',{'adapter_restored_exactly':True,'optimizer_scaler_rng_restored':True,'continuing_at_step':step+1})
changed=sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in model.model.named_parameters() if p.requires_grad);assert changed>0
out=R/f'checkpoints/lora-{mode}';model.model.save_pretrained(out,safe_serialization=True);save(out/'training.json',{'examples':len(examples),'changed_parameter_tensors':changed,'seconds':time.perf_counter()-start,'seed':5150,'finite_losses':True})
# Switch wrapper back to native module for the logit scorer while retaining LoRA layers.
model.model=model.model.get_base_model();evaluate_model('adapted')
