"""Non-test loader parity, zero adapter and forward/backward admission checks."""
from engine import *

def embedding():
 m,tok=load();texts=[PREFIX+'Find a Python CSV parser',PREFIX+'Locate a tool for plotting numerical arrays and exporting figures.'];reference=[]
 old=module('archived_encoder',OLD/'run_skillret_baseline.py')
 for x in texts:
  v,dt=old.encode_one(m,tok,x,32768)
  with torch.inference_mode():w=vector(m,tok,x).cpu().numpy()
  assert np.array_equal(v,w);reference.append(v)
 theta=float(m.config.rope_theta);frozen=fingerprint(m);del m;torch.cuda.empty_cache()
 m,tok=load(train=True);m.eval();errs=[]
 for x,r in zip(texts,reference):
  with torch.inference_mode():v=vector(m,tok,x).cpu().numpy()
  errs.append(float(np.max(np.abs(v-r))));assert np.array_equal(v,r)
 params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=3e-6);scale=torch.amp.GradScaler('cuda',init_scale=128);m.train();steps=[]
 budget=read(STUDY/'results/data_audit.json')['query_budget']
 for length in sorted(set([128,1024,budget])):
  query=' '.join(['parse']*(length-30));x=tok(query,return_tensors='pt').to('cuda');before={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
  with torch.autocast('cuda',dtype=torch.float16):v=vector(m,tok,query,budget)
  target=torch.tensor(reference[0],device='cuda');loss=1-(v*target).sum();scale.scale(loss).backward();scale.unscale_(opt);grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad) and grad>0;scale.step(opt);scale.update();opt.zero_grad(set_to_none=True)
  assert any(not torch.equal(before[n],p) for n,p in m.named_parameters() if p.requires_grad);steps.append({'tokens':int(x.input_ids.shape[1]),'finite_nonzero_gradient':float(grad),'successful_update':True})
 assert fingerprint(m)==frozen
 save(STUDY/'results/embedding_parity.json',{'status':'passed','fixture':'synthetic non-test text; archived encode_one versus new loader/pooling and zero LoRA','max_abs_error':max(errs),'rope_theta':theta,'backward_checks':steps,'frozen_sha256':frozen,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'smoke_updates_discarded':True})
 print('Embedding parity and backward admission passed',steps,flush=True)
if __name__=='__main__':embedding()
