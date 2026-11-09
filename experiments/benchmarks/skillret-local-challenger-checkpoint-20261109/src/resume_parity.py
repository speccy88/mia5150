"""Real optimizer/RNG roundtrip: uninterrupted4 versus checkpoint2+resume2."""
from common import *
import subprocess,sys

def main():
 check=subprocess.run([sys.executable,str(SRC/'loader_reference.py')]);assert check.returncode==0
 script=SRC/'train_query.py';base=[sys.executable,str(script),'--steps','4','--smoke']
 for args,code in [(['--name','resume-reference-v2'],0),(['--name','resume-split-v2','--stop-after','2'],75),(['--name','resume-split-v2','--resume',str(CACHE/'checkpoints/query/resume-split-v2/step-00002')],0)]:
  p=subprocess.run(base+args);assert p.returncode==code,(args,p.returncode,code)
 from safetensors.numpy import load_file
 a=load_file(str(CACHE/'checkpoints/query/resume-reference-v2/final/adapter_model.safetensors'));b=load_file(str(CACHE/'checkpoints/query/resume-split-v2/final/adapter_model.safetensors'));assert set(a)==set(b)
 assert any(np.count_nonzero(v)>0 for k,v in a.items() if 'lora_B' in k),'Real optimizer updates must change zero-initialized B matrices'
 for k in a:assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes(),k
 ar=rows(STUDY/'results/query/resume-reference-v2/steps.jsonl');br=rows(STUDY/'results/query/resume-split-v2/steps.jsonl');assert len(ar)==len(br)==4
 for x,y in zip(ar,br):
  for k in ('loss','ce','anchor','gradient_norm','lr','query_groups','unique_queries','amp_scale'):assert x[k]==y[k],(k,x[k],y[k])
 save(STUDY/'results/query/resume_parity.json',{'status':'passed','real_optimizer_steps_per_branch':4,'interruption_checkpoint_step':2,'all_adapter_tensors_byte_identical':True,'loss_gradients_lr_scaler_and_query_coverage_identical':True,'discarded_fixture_updates_total':8,'not_used_for_model_selection':True})
 print('Actual checkpoint/reload/RNG parity passed',flush=True)
if __name__=='__main__':main()
