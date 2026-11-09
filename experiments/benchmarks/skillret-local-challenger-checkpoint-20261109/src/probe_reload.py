"""Diagnostic parameter-only reload fidelity, without scoring or updates."""
from engine import *
from safetensors.numpy import load_file

def main():
 checkpoint=CACHE/'checkpoints/query/resume-split/step-00002';saved=load_file(str(checkpoint/'adapter_model.safetensors'));m,tok=load(adapter=checkpoint,train=True);errors={}
 for name,p in m.named_parameters():
  if 'lora_' not in name:continue
  key=name.replace('.default.','.');assert key in saved;actual=p.detach().cpu().numpy();delta=float(np.max(np.abs(actual-saved[key])))
  if delta:errors[key]=delta
 save(STUDY/'results/query/diagnostic_reload_fidelity.json',{'status':'passed' if not errors else 'failed','checked_tensors':len(saved),'different_tensors':len(errors),'maximum_error':max(errors.values(),default=0.),'tensor_errors':errors,'training_updates':0});print('Reload diagnostic',len(saved),len(errors),max(errors.values(),default=0.),flush=True)
if __name__=='__main__':main()
